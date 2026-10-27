"""Consultas do aceite real da Etapa 3 (spec §8, critérios 1 a 5) num MLflow.

Usadas por `test_acceptance_real.py` contra o `mlflow.db` real e testadas em
`tests/test_acceptance_checks.py` com um MLflow temporário (runs sintéticos, que
provam só a lógica da consulta). Só leem o MLflow; cada checagem devolve a lista
de problemas encontrados (vazia = ok), para a falha dizer o que divergiu.

Contrato de nomes (§4.1): runs de estágio têm `run_type=stage`, `stage` e
`pipeline_run_id`; runs de lote são filhos (`mlflow.parentRunId`) com
`run_type=batch`, `batch_name`, `alert_level` (ASCII) e `alert_reasons`, e não
carregam o `pipeline_run_id`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities import Run

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
