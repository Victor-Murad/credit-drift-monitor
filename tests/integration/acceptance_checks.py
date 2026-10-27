"""Consultas do aceite real da Etapa 3 (spec §8, critérios 1 a 5 e 10) num MLflow.

Usadas por `test_acceptance_real.py` contra o `mlflow.db` real e testadas em
`tests/test_acceptance_checks.py` com um MLflow temporário (runs sintéticos, que
provam só a lógica da consulta). Só leem o MLflow; cada checagem devolve a lista
de problemas encontrados (vazia = ok), para a falha dizer o que divergiu.

Contrato de nomes (§4.1): runs de estágio têm `run_type=stage`, `stage` e
`pipeline_run_id`; runs de lote são filhos (`mlflow.parentRunId`) com
`run_type=batch`, `batch_name`, `alert_level` (ASCII) e `alert_reasons`, e não
carregam o `pipeline_run_id`.

Extensão §4.7 (critério 10): versões `origin=retrain` com as tags do challenger e o
run `run_type=retrain` de origem; aliases `champion`/`challenger`/`previous`
coerentes com o arquivo servido; runs `run_type=audit` de promoção e rollback.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from src import config
from src.monitoring import registry
from src.monitoring.promote import AUC_METRICS
from src.monitoring.promotion import decide_promotion
from src.monitoring.retrain import (
    RETRAIN_RUN_TYPE,
    RETRAIN_VERSION_PREFIX,
    TRIGGER_ALERT,
)

PIPELINE_STAGES = ("prepare", "train", "contract_demo", "drift_demo")
# Resultado esperado com os dados reais (seed 42), §4.2 e critério de aceite 2.
EXPECTED_DRIFT_LEVELS = {
    "production_t0": "OK",
    "production_t1_data_drift": "ATENCAO",
    "production_t2_concept_drift": "CRITICO",
}
# Tabela §4.3: métricas registradas no MLflow (estágio ou lote). O `duration_ms`
# do contrato, da detecção e da pontuação vem dos eventos do log (ver
# `has_latency_events`), anexado a cada run de estágio.
HEALTH_METRICS = (
    "stage_success",
    "stage_duration_s",
    "rows_processed",
    "rows_rejected",
    "alerts_failure_count",
    "alerts_drift_count",
    "roc_auc",
    "auc_drop",
    "approval_rate",
    "default_rate_approved",
    "score_psi",
    "peak_memory_mb",
)
LATENCY_LOG_FIELD = "duration_ms"
# Teto de runs por consulta (o SQLite do MLflow aceita até 50 000).
MAX_RUNS = 50_000


@dataclass(frozen=True)
class PipelineRuns:
    """Runs de um `make pipeline`: o de cada estágio e os lotes filhos de cada um."""

    pipeline_run_id: str
    stages: dict[str, Run]
    batches: dict[str, list[Run]]

    def all_runs(self) -> list[Run]:
        lotes = [run for runs in self.batches.values() for run in runs]
        return [*self.stages.values(), *lotes]


def _batch_children(client: MlflowClient, experiment_id: str, run_id: str) -> list[Run]:
    filtro = f"tags.mlflow.parentRunId = '{run_id}' and tags.run_type = 'batch'"
    return list(client.search_runs([experiment_id], filtro, max_results=MAX_RUNS))


def latest_pipeline(client: MlflowClient, experiment_name: str) -> PipelineRuns | None:
    """O `pipeline_run_id` mais recente que tem run dos quatro estágios do pipeline.

    "Mais recente" = o do run de estágio mais novo. Grupos incompletos (um estágio
    avulso do `dvc repro`, sem `PIPELINE_RUN_ID` comum, ou o `monitor`) são pulados.
    Se um `pipeline_run_id` tiver o mesmo estágio duas vezes, vale o run mais novo.
    """
    experimento = client.get_experiment_by_name(experiment_name)
    if experimento is None:
        return None
    runs = client.search_runs(
        [experimento.experiment_id],
        "tags.run_type = 'stage'",
        max_results=MAX_RUNS,
        order_by=["attributes.start_time DESC"],
    )
    grupos: dict[str, dict[str, Run]] = {}
    for run in runs:  # do mais novo para o mais antigo
        pid = run.data.tags.get("pipeline_run_id")
        estagio = run.data.tags.get("stage")
        if pid and estagio in PIPELINE_STAGES:
            grupos.setdefault(pid, {}).setdefault(estagio, run)
    for pid, grupo in grupos.items():
        if set(grupo) == set(PIPELINE_STAGES):
            estagios = {nome: grupo[nome] for nome in PIPELINE_STAGES}
            lotes = {
                nome: _batch_children(
                    client, experimento.experiment_id, run.info.run_id
                )
                for nome, run in estagios.items()
            }
            return PipelineRuns(pid, estagios, lotes)
    return None


def failed_stages(pipeline: PipelineRuns) -> list[str]:
    """Estágios sem `stage_status=SUCCESS` ou sem `stage_success=1`."""
    return [
        nome
        for nome, run in pipeline.stages.items()
        if run.data.tags.get("stage_status") != "SUCCESS"
        or run.data.metrics.get("stage_success") != 1.0
    ]


def drift_level_problems(pipeline: PipelineRuns) -> list[str]:
    """t0/t1/t2 do `drift_demo` com o `alert_level` esperado e motivos se não `OK`."""
    lotes = {r.data.tags.get("batch_name"): r for r in pipeline.batches["drift_demo"]}
    problemas = []
    for nome, esperado in EXPECTED_DRIFT_LEVELS.items():
        run = lotes.get(nome)
        if run is None:
            problemas.append(f"{nome}: nenhum run de lote no drift_demo")
            continue
        nivel = run.data.tags.get("alert_level")
        if nivel != esperado:
            problemas.append(f"{nome}: alert_level={nivel!r}, esperado {esperado!r}")
        elif esperado != "OK" and not run.data.tags.get("alert_reasons"):
            problemas.append(f"{nome}: {esperado} sem alert_reasons")
    return problemas


def missing_health_metrics(pipeline: PipelineRuns) -> list[str]:
    """Métricas da §4.3 ausentes de todos os runs do pipeline (e a tag de status)."""
    presentes = {chave for run in pipeline.all_runs() for chave in run.data.metrics}
    faltando = [m for m in HEALTH_METRICS if m not in presentes]
    sem_status = [
        nome
        for nome, run in pipeline.stages.items()
        if "stage_status" not in run.data.tags
    ]
    faltando += [f"tag stage_status ({nome})" for nome in sem_status]
    return faltando


def _log_artifact(pipeline_run_id: str) -> str:
    return f"logs/pipeline_{pipeline_run_id}.jsonl"


def stage_logs(
    client: MlflowClient, pipeline: PipelineRuns, dst: Path
) -> dict[str, list[dict] | None]:
    """Eventos do log JSONL anexado a cada run de estágio (`None` = sem anexo)."""
    caminho = _log_artifact(pipeline.pipeline_run_id)
    logs: dict[str, list[dict] | None] = {}
    for nome, run in pipeline.stages.items():
        anexos = {a.path for a in client.list_artifacts(run.info.run_id, "logs")}
        if caminho not in anexos:
            logs[nome] = None
            continue
        baixado = client.download_artifacts(
            run.info.run_id, caminho, dst_path=str(dst / nome)
        )
        linhas = Path(baixado).read_text(encoding="utf-8").splitlines()
        logs[nome] = [json.loads(linha) for linha in linhas if linha.strip()]
    return logs


def log_problems(logs: dict[str, list[dict] | None], pipeline_run_id: str) -> list[str]:
    """Log ausente, vazio ou com linhas de outro `run_id` (o elo é o `pipeline_run_id`)."""
    problemas = []
    for nome, eventos in logs.items():
        if eventos is None:
            problemas.append(f"{nome}: sem {_log_artifact(pipeline_run_id)} anexado")
        elif not eventos:
            problemas.append(f"{nome}: log anexado vazio")
        else:
            alheias = [e for e in eventos if e.get("run_id") != pipeline_run_id]
            if alheias:
                problemas.append(
                    f"{nome}: {len(alheias)} linha(s) com run_id diferente de "
                    f"{pipeline_run_id}"
                )
    return problemas


def has_latency_events(logs: dict[str, list[dict] | None]) -> bool:
    """Algum evento anexado traz `duration_ms` (latência do contrato/detecção)."""
    return any(
        LATENCY_LOG_FIELD in evento
        for eventos in logs.values()
        for evento in eventos or []
    )


def blocked_demo_problems(pipeline: PipelineRuns) -> list[str]:
    """Lote inválido `BLOQUEADO` e intencional, contado sem derrubar o estágio."""
    estagio = pipeline.stages["contract_demo"]
    bloqueados = [
        r
        for r in pipeline.batches["contract_demo"]
        if r.data.tags.get("alert_level") == "BLOQUEADO"
    ]
    problemas = []
    if not bloqueados:
        problemas.append("contract_demo: nenhum run de lote BLOQUEADO")
    elif not any(r.data.tags.get("intentional_demo") == "true" for r in bloqueados):
        problemas.append("contract_demo: run BLOQUEADO sem intentional_demo=true")
    if estagio.data.metrics.get("alerts_failure_count", 0.0) < 1.0:
        problemas.append("contract_demo: alerts_failure_count < 1")
    if estagio.data.metrics.get("stage_success") != 1.0:
        problemas.append("contract_demo: stage_success != 1")
    if estagio.data.tags.get("intentional_demo") != "true":
        problemas.append(
            "contract_demo: run de estágio sem a tag intentional_demo=true"
        )
    return problemas


# MARK: Registry (extensão §4.7, critério 10)

# Tags que `retrain` grava em toda versão `origin=retrain` (além de `origin`, que é o
# próprio critério de seleção). `promotion_reasons` fica vazia quando elegível, por
# isso só se exige a chave.
RETRAIN_VERSION_TAGS = (
    "model_sha256",
    "bundle_model_version",
    "reference_sha256",
    "batch_name",
    "batch_sha256",
    "trigger",
    "champion_version",
    "retrain_run_id",
    "promotion_eligible",
)
# Limiares pré-registrados na ADR-0007: o run de retreino os grava como params.
# Tag `origin` que o `train` grava em cada reconstrução do baseline (`stage.py`).
ORIGIN_BASELINE = "baseline"
THRESHOLD_PARAMS = {
    "promotion_min_gain": "PROMOTION_MIN_GAIN",
    "promotion_max_regression": "PROMOTION_MAX_REGRESSION",
}


@dataclass(frozen=True)
class RegistryRuns:
    """Versões e aliases do modelo registrado e os runs do ciclo de retreino.

    `retrain_runs` vai do mais antigo ao mais novo; `audit_runs`, do mais novo ao
    mais antigo; `alert_runs` tem `None` para um alerta citado que não existe.
    """

    versions: dict[int, ModelVersion]
    aliases: dict[str, int]
    retrain_runs: dict[str, Run]
    alert_runs: dict[str, Run | None]
    audit_runs: list[Run]

    def by_origin(self, origin: str) -> dict[int, ModelVersion]:
        return {
            numero: versao
            for numero, versao in sorted(self.versions.items())
            if versao.tags.get("origin") == origin
        }


def _get_run(client: MlflowClient, run_id: str) -> Run | None:
    try:
        return client.get_run(run_id)
    except MlflowException:
        return None


def registry_state(
    client: MlflowClient, model_name: str, experiment_name: str
) -> RegistryRuns | None:
    """Estado do Registry e dos runs de retreino/auditoria; `None` sem retreino algum."""
    try:
        modelo = client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    experimento = client.get_experiment_by_name(experiment_name)
    versoes = {
        int(v.version): v for v in client.search_model_versions(f"name='{model_name}'")
    }
    retreinadas = [
        v for v in versoes.values() if v.tags.get("origin") == registry.ORIGIN_RETRAIN
    ]
    if experimento is None or not retreinadas:
        return None

    def buscar(run_type: str, ordem: str) -> list[Run]:
        return list(
            client.search_runs(
                [experimento.experiment_id],
                f"tags.run_type = '{run_type}'",
                max_results=MAX_RUNS,
                order_by=[f"attributes.start_time {ordem}"],
            )
        )

    alertas = sorted(
        {v.tags["alert_run_id"] for v in retreinadas if v.tags.get("alert_run_id")}
    )
    return RegistryRuns(
        versions=dict(sorted(versoes.items())),
        aliases={alias: int(numero) for alias, numero in modelo.aliases.items()},
        retrain_runs={r.info.run_id: r for r in buscar(RETRAIN_RUN_TYPE, "ASC")},
        alert_runs={run_id: _get_run(client, run_id) for run_id in alertas},
        audit_runs=buscar(registry.AUDIT_RUN_TYPE, "DESC"),
    )


def _retrain_run_problems(numero: int, versao: ModelVersion, run: Run) -> list[str]:
    """Run de origem: elo com a versão, limiares pré-registrados e veredito da regra."""
    problemas = []
    tags = versao.tags
    if run.data.tags.get("model_sha256") != tags.get("model_sha256"):
        problemas.append(f"v{numero}: model_sha256 da versão difere do run de retreino")
    if run.data.tags.get("challenger_version") != str(numero):
        problemas.append(f"v{numero}: run de retreino sem challenger_version={numero}")
    for param, constante in THRESHOLD_PARAMS.items():
        gravado = run.data.params.get(param)
        esperado = getattr(config, constante)
        if gravado is None or not math.isclose(float(gravado), esperado):
            problemas.append(
                f"v{numero}: param {param}={gravado} difere do pré-registrado {esperado}"
            )
    faltando = [m for m in AUC_METRICS if m not in run.data.metrics]
    problemas += [f"v{numero}: run de retreino sem a métrica {m}" for m in faltando]
    if not faltando:
        # O retreino só registra depois do contrato e da integridade (regra 1).
        aucs = {m: run.data.metrics[m] for m in AUC_METRICS}
        decisao = decide_promotion(contract_ok=True, integrity_ok=True, **aucs)
        veredito = tags.get("promotion_eligible")
        if veredito != ("true" if decisao.eligible else "false"):
            problemas.append(
                f"v{numero}: veredito promotion_eligible={veredito!r} difere da regra "
                f"recalculada (elegível={decisao.eligible}, motivos: "
                f"{', '.join(decisao.reasons) or 'nenhum'})"
            )
    return problemas


def _alert_problems(numero: int, alert_run_id: str, alerta: Run | None) -> list[str]:
    if alerta is None:
        return [f"v{numero}: alerta {alert_run_id} ausente no MLflow"]
    problemas = []
    nivel = alerta.data.tags.get("alert_level")
    if nivel != registry.CRITICAL_LEVEL:
        problemas.append(
            f"v{numero}: alerta {alert_run_id} com alert_level={nivel!r}, esperado "
            f"{registry.CRITICAL_LEVEL!r}"
        )
    if not alerta.data.tags.get(registry.RETRAIN_CONSUMED_TAG):
        problemas.append(
            f"v{numero}: alerta {alert_run_id} sem a tag {registry.RETRAIN_CONSUMED_TAG}"
        )
    return problemas


def retrain_problems(estado: RegistryRuns) -> list[str]:
    """Cada versão `origin=retrain`: tags, run de origem, veredito e alerta consumido."""
    problemas = []
    for numero, versao in estado.by_origin(registry.ORIGIN_RETRAIN).items():
        tags = versao.tags
        problemas += [
            f"v{numero}: sem a tag {t}" for t in RETRAIN_VERSION_TAGS if not tags.get(t)
        ]
        if "promotion_reasons" not in tags:
            problemas.append(f"v{numero}: sem a tag promotion_reasons")
        modelo = tags.get("bundle_model_version", "")
        if modelo and not modelo.startswith(RETRAIN_VERSION_PREFIX):
            problemas.append(
                f"v{numero}: bundle_model_version={modelo!r} sem o prefixo "
                f"{RETRAIN_VERSION_PREFIX}"
            )
        run_id = tags.get("retrain_run_id")
        run = estado.retrain_runs.get(run_id) if run_id else None
        if run_id and run is None:
            problemas.append(f"v{numero}: run de retreino {run_id} ausente")
        elif run is not None:
            problemas += _retrain_run_problems(numero, versao, run)
        if tags.get("trigger") == TRIGGER_ALERT:
            alerta_id = tags.get("alert_run_id")
            if not alerta_id:
                problemas.append(f"v{numero}: trigger=alert sem alert_run_id")
            else:
                problemas += _alert_problems(
                    numero, alerta_id, estado.alert_runs.get(alerta_id)
                )
    return problemas


def alias_problems(
    estado: RegistryRuns, served_sha: str | None, baseline_sha: str | None
) -> list[str]:
    """Aliases coerentes entre si e com os arquivos (`served_sha`: `models/champion/`).

    `baseline_sha` é o de `models/baseline_model.joblib`, que promoção e rollback
    nunca tocam: tem de ser o de alguma versão `origin=baseline`.
    """
    problemas = []
    servido = config.project_relative(config.CHAMPION_FILE)
    champion = estado.aliases.get(registry.CHAMPION_ALIAS)
    if champion is None:
        problemas.append(f"sem o alias {registry.CHAMPION_ALIAS}")
    if served_sha is None:
        problemas.append(f"sem o arquivo servido {servido}")
    elif champion is not None and (
        estado.versions[champion].tags.get("model_sha256") != served_sha
    ):
        problemas.append(f"champion v{champion}: model_sha256 difere de {servido}")
    challenger = estado.aliases.get(registry.CHALLENGER_ALIAS)
    if challenger is not None and challenger not in estado.by_origin(
        registry.ORIGIN_RETRAIN
    ):
        problemas.append(
            f"challenger v{challenger}: não é uma versão origin={registry.ORIGIN_RETRAIN}"
        )
    previous = estado.aliases.get(registry.PREVIOUS_ALIAS)
    if previous is not None and previous == champion:
        problemas.append(f"previous v{previous}: igual ao champion")
    baselines = {
        v.tags.get("model_sha256") for v in estado.by_origin(ORIGIN_BASELINE).values()
    }
    if baseline_sha not in baselines:
        problemas.append(
            f"{config.project_relative(config.MODEL_FILE)}: sha256 não é o de nenhuma "
            "versão origin=baseline (promoção e rollback nunca o tocam)"
        )
    return problemas


def _alias_text(numero: int | None, alias: str) -> str:
    return f"v{numero}" if numero is not None else f"não há {alias}"


def _audit_run_problems(run: Run) -> tuple[list[str], dict | None]:
    """Campos de quem/quando/por quê de um run de auditoria e o `details` lido."""
    rid = run.info.run_id
    tags = run.data.tags
    problemas = []
    if tags.get("action") not in registry.AUDIT_ACTIONS:
        problemas.append(f"auditoria {rid}: action={tags.get('action')!r}")
    if tags.get("status") not in registry.AUDIT_STATUSES:
        problemas.append(f"auditoria {rid}: status={tags.get('status')!r}")
    problemas += [
        f"auditoria {rid}: sem a tag {t}"
        for t in ("operator", "reason", "timestamp_utc", "version")
        if not tags.get(t)
    ]
    try:
        detalhes = json.loads(tags.get("details", ""))
    except json.JSONDecodeError:
        return [*problemas, f"auditoria {rid}: details não é JSON"], None
    if tags.get("action") == registry.ACTION_PROMOTE:
        anterior = detalhes.get("previous_version")
        if anterior is None:
            problemas.append(f"auditoria {rid}: promoção sem previous_version")
        elif anterior != detalhes.get("from_version"):
            problemas.append(
                f"auditoria {rid}: previous_version={anterior} difere de "
                f"from_version={detalhes.get('from_version')}"
            )
    return problemas, detalhes


def audit_problems(estado: RegistryRuns) -> list[str]:
    """Runs de auditoria completos; o mais recente descreve os aliases de agora."""
    problemas = []
    lidos = []
    for run in estado.audit_runs:
        encontrados, detalhes = _audit_run_problems(run)
        problemas += encontrados
        lidos.append((run, detalhes))
    if not lidos:
        return problemas
    ultimo, detalhes = lidos[0]
    status = ultimo.data.tags.get("status")
    if status != registry.STATUS_CONSISTENT:
        problemas.append(
            f"auditoria mais recente: status={status!r}, esperado "
            f"{registry.STATUS_CONSISTENT!r}"
        )
    if detalhes is None:
        return problemas
    champion = estado.aliases.get(registry.CHAMPION_ALIAS)
    previous = estado.aliases.get(registry.PREVIOUS_ALIAS)
    destino = detalhes.get("to_version")
    if destino != champion:
        problemas.append(
            f"auditoria mais recente: to_version={destino}, mas champion é "
            f"{_alias_text(champion, registry.CHAMPION_ALIAS)}"
        )
    elif detalhes.get("model_sha256") != estado.versions[champion].tags.get(
        "model_sha256"
    ):
        problemas.append(
            f"auditoria mais recente: model_sha256 difere do champion v{champion}"
        )
    if detalhes.get("previous_version") != previous:
        problemas.append(
            f"auditoria mais recente: previous_version="
            f"{detalhes.get('previous_version')}, mas previous é "
            f"{_alias_text(previous, registry.PREVIOUS_ALIAS)}"
        )
    return problemas
