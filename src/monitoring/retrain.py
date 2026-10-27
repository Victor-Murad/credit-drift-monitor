"""Retreino do challenger guardado por alerta CRÍTICO (spec da Etapa 3, §4.7 e §4.8).

    uv run python -m src.monitoring.retrain [--alert-run ID]
    uv run python -m src.monitoring.retrain --alert-run ID --allow-reuse --reason TEXTO
    uv run python -m src.monitoring.retrain --force-no-alert --reason TEXTO

Gatilho: o run de lote `CRITICO` mais recente ainda não consumido (ou o `--alert-run`
dado, conferido por `require_critical_run`). Sem alerta, recusa; `--force-no-alert`
(sem alerta, lote `config.PRODUCTION_T2_FILE`) e `--allow-reuse` (alerta já consumido)
exigem `--reason`, que fica no run. O lote do alerta é o arquivo da tag `batch_path`,
com o sha256 conferido contra a tag `batch_sha256` (o `simulate` regera t1/t2).

Ordem, toda ANTES de treinar: champion do Registry com o sha256 conferido
(`load_bundle(expected_sha256=...)`), referência com o mesmo `reference_sha256` do
champion e contrato rotulado (`enforce_contract`) na referência e no lote.

Dados (sem vazamento): `split_reference(reference, SEED)` vem PRIMEIRO, sobre a
referência sozinha, e a validação (escolha do *threshold*) e o teste são exatamente os
do champion. O lote `CRITICO` é dividido 50/50, estratificado pelo `target`: metade
entra no treino, a outra é o *holdout*, onde champion e challenger são avaliados (o AUC
do champion é recalculado nele, não reaproveitado do lote inteiro). Nenhum `client_id`
é compartilhado entre treino, validação, teste e holdout (conferido; falha fechado).

Challenger: mesmo pipeline e hiperparâmetros do baseline (`train.fit_model`), com
`model_version` próprio (`retrain-<sha256 do pipeline[:12]>`), para o monitor distinguir
o baseline de AUC de cada modelo. Run `run_type=retrain` (filho do run de estágio
`retrain`) com as AUCs, `gain_on_holdout`, `regression_on_reference` e o veredito de
`decide_promotion`; nova versão no Registry com o alias `challenger` e as tags de
linhagem. Só depois do registro o alerta recebe `retrain_consumed`: uma falha no meio
não consome o alerta. **Nunca move `champion`**: promover é um comando humano.

Códigos de saída: 0 (challenger registrado, elegível ou não: o veredito é impresso), 2
(recusa de política: sem alerta CRÍTICO não consumido, alerta consumido sem
`--allow-reuse`, run que não é lote CRÍTICO), 3 (o contrato barrou a referência ou o
lote) e 1 (erro operacional: rastreamento desligado, MLflow indisponível, sem champion,
integridade, arquivo ausente ou divergente, argumento inválido). O champion adulterado
encerra pelo `load_bundle` ("Modelo recusado", código 1), como no `monitor`.

O retreino depende do Model Registry: com `MONITORING_TRACKING=off` falha fechado.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import config
from src.logging_utils import RUN_ID, get_logger, log_event
from src.models import predict, train
from src.models.features import model_input
from src.models.integrity import (
    ModelIntegrityError,
    file_sha256,
    verify_model_integrity,
)
from src.models.metrics import evaluate, score_distribution
from src.monitoring import registry, tracking
from src.monitoring.monitor import EXIT_OPERATIONAL_ERROR, CliParser
from src.monitoring.promotion import PromotionDecision, decide_promotion
from src.monitoring.registry import (
    BUNDLE_ARTIFACT_DIR,
    BUNDLE_METADATA_FILE,
    BUNDLE_MODEL_FILE,
    CHAMPION_ALIAS,
    RETRAIN_CONSUMED_TAG,
    RegistryStateError,
    get_alias_version,
    redact_text,
)
from src.monitoring.stage import (
    INPUT_EXAMPLE_ROWS,
    StagePreconditionError,
    champion_sha256,
    infer_model_signature,
)
from src.monitoring.tracking import (
    TRACKING_ENV_VAR,
    StageRun,
    TrackingUnavailableError,
    tracked_stage,
    tracking_errors,
)
from src.validation.gate import DataContractViolation, enforce_contract
from src.validation.io import read_batch

logger = get_logger("monitoring.retrain")

STAGE_NAME = "retrain"
RETRAIN_RUN_TYPE = "retrain"
RETRAIN_RUN_NAME = "retrain_challenger"
REFERENCE_BATCH_NAME = "reference_retrain"
# Metade do lote CRÍTICO vai ao treino; a outra é o holdout da comparação (§4.7).
CRITICAL_HOLDOUT_FRACTION = 0.5
RETRAIN_VERSION_PREFIX = "retrain-"
TRIGGER_ALERT = "alert"
TRIGGER_FORCED = "force_no_alert"
# Lido na chamada (os testes o redirecionam): relatórios do contrato do retreino.
RETRAIN_REPORTS_DIR = config.REPORTS_DIR / "retrain"

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_BLOCKED = 3


class RetrainRefusedError(RuntimeError):
    """A política recusa o retreino (sem alerta CRÍTICO elegível, código 2)."""


# MARK: divisão dos dados (sem vazamento)


def split_critical_batch(
    df: pd.DataFrame, seed: int = config.SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(treino, holdout)`: metade estratificada pelo `target` (determinístico na seed)."""
    treino, holdout = train_test_split(
        df,
        test_size=CRITICAL_HOLDOUT_FRACTION,
        stratify=df[config.TARGET_COLUMN],
        random_state=seed,
    )
    return treino, holdout


def require_disjoint(**partes: pd.DataFrame) -> None:
    """Nenhum `client_id` em mais de uma parte; senão `StagePreconditionError`."""
    ids = {nome: set(df[config.ID_COLUMN]) for nome, df in partes.items()}
    for (nome_a, a), (nome_b, b) in combinations(ids.items(), 2):
        comuns = a & b
        if comuns:
            raise StagePreconditionError(
                f"Vazamento: {len(comuns)} client_id(s) em {nome_a} e em {nome_b} ao "
                "mesmo tempo. Treino, validação, teste e holdout não podem "
                "compartilhar clientes (o lote não pode repetir clientes da "
                "referência nem a si mesmo); nenhum challenger foi treinado."
            )


def build_challenger_training(
    reference: pd.DataFrame, critical_train: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`(treino da referência + critical_train, validação, teste)` do challenger.

    `split_reference(reference, SEED)` roda PRIMEIRO, sobre a referência sozinha: a
    validação e o teste devolvidos são exatamente os do champion. Um `client_id` do
    lote que também esteja na validação ou no teste é recusado (vazamento).
    """
    treino_ref, validacao, teste = train.split_reference(reference, config.SEED)
    treino = pd.concat([treino_ref, critical_train], ignore_index=True)
    require_disjoint(treino=treino, validacao=validacao, teste=teste)
    return treino, validacao, teste


# MARK: gatilho


@dataclass(frozen=True)
class RetrainTrigger:
    """O que habilitou o retreino e o lote que ele usa."""

    trigger: str
    batch_name: str
    batch_path: Path
    expected_batch_sha256: str | None = None
    alert_run_id: str | None = None
    reused: bool = False
    reason: str | None = None


def _resolve_path(tag: str) -> Path:
    caminho = Path(tag)
    return caminho if caminho.is_absolute() else config.ROOT_DIR / caminho


def _trigger_from_run(run: Run, *, reused: bool, reason: str | None) -> RetrainTrigger:
    tags = run.data.tags
    run_id = run.info.run_id
    for chave in ("batch_path", "batch_sha256"):
        if not tags.get(chave):
            raise StagePreconditionError(
                f"Run de alerta {run_id} sem a tag {chave}: não há como achar e "
                "conferir o lote que gerou o alerta. Rode `make demo-drift` (ou o "
                "`monitor`) de novo para registrar o lote."
            )
    caminho = _resolve_path(tags["batch_path"])
    return RetrainTrigger(
        trigger=TRIGGER_ALERT,
        batch_name=tags.get("batch_name") or caminho.stem,
        batch_path=caminho,
        expected_batch_sha256=tags["batch_sha256"],
        alert_run_id=run_id,
        reused=reused,
        reason=reason,
    )


def resolve_trigger(
    *,
    alert_run: str | None,
    force_no_alert: bool,
    allow_reuse: bool,
    reason: str | None,
) -> RetrainTrigger:
    """Acha o alerta que habilita o retreino; recusa com `RetrainRefusedError`."""
    if force_no_alert:
        caminho = Path(config.PRODUCTION_T2_FILE)
        return RetrainTrigger(
            trigger=TRIGGER_FORCED,
            batch_name=caminho.stem,
            batch_path=caminho,
            reason=reason,
        )
    if alert_run is None:
        run = registry.find_unconsumed_critical_run()
        if run is None:
            raise RetrainRefusedError(
                "Nenhum alerta CRÍTICO não consumido no MLflow: o retreino só roda a "
                "partir de um lote CRÍTICO (queda de AUC). Para reaproveitar um alerta "
                "já consumido: `--alert-run <id> --allow-reuse --reason <motivo>`; "
                "para retreinar sem alerta: `--force-no-alert --reason <motivo>`."
            )
        return _trigger_from_run(run, reused=False, reason=reason)
    try:
        run = registry.require_critical_run(alert_run)
    except ValueError as exc:
        raise RetrainRefusedError(str(exc)) from exc
    consumido = run.data.tags.get(RETRAIN_CONSUMED_TAG)
    if consumido is not None and not allow_reuse:
        raise RetrainRefusedError(
            f"O alerta {alert_run} já foi consumido pelo challenger v{consumido}. "
            "Um segundo retreino sobre o mesmo alerta exige "
            "`--allow-reuse --reason <motivo>`."
        )
    return _trigger_from_run(run, reused=consumido is not None, reason=reason)


# MARK: retreino


@dataclass(frozen=True)
class RetrainResult:
    """O challenger registrado e o veredito da regra de promoção (§4.7)."""

    version: int
    model_version: str
    champion_version: int
    decision: PromotionDecision
    aucs: dict[str, float]
    holdout_rows: int
    alert_run_id: str | None


def _auc(pipeline: Any, df: pd.DataFrame) -> float:
    """ROC-AUC sem arredondar (a regra de promoção arredonda as diferenças)."""
    score = pipeline.predict_proba(model_input(df))[:, 1]
    return float(roc_auc_score(df[config.TARGET_COLUMN].to_numpy(), score))


def challenger_model_version(pipeline: Any) -> str:
    """`retrain-<sha256[:12]>` do pipeline serializado.

    É o sha do pipeline e não do arquivo: o arquivo do bundle inclui a própria versão.
    """
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    return f"{RETRAIN_VERSION_PREFIX}{digest[:12]}"


def _require_file(path: Path, o_que: str) -> Path:
    try:
        return tracking.require_path(path, o_que)
    except FileNotFoundError as exc:
        raise StagePreconditionError(f"{exc}. Rode `make pipeline` antes.") from exc


def _read_labeled(path: Path, o_que: str) -> pd.DataFrame:
    """Lote com as duas classes (ao menos 2 de cada): o split estratificado exige."""
    try:
        df = read_batch(path)
    except (ValueError, OSError) as exc:
        raise StagePreconditionError(f"{o_que} ilegível em {path}: {exc}") from exc
    alvo = config.TARGET_COLUMN
    if alvo not in df.columns:
        raise StagePreconditionError(
            f"{o_que} sem a coluna `{alvo}` ({path}): o retreino exige lote rotulado."
        )
    contagem = df[alvo].value_counts()
    if len(contagem) < 2 or contagem.min() < 2:
        raise StagePreconditionError(
            f"{o_que} sem as duas classes de `{alvo}` em quantidade suficiente "
            f"({path}): não há como dividir nem avaliar a ROC-AUC."
        )
    return df


def _validated(run: StageRun, df: pd.DataFrame, batch_name: str) -> pd.DataFrame:
    """Contrato rotulado; os relatórios vão para o run (também quando o lote é barrado)."""
    pasta = Path(RETRAIN_REPORTS_DIR)
    try:
        return enforce_contract(
            df,
            batch_name=batch_name,
            stage=STAGE_NAME,
            labeled=True,
            report_dir=pasta,
            quarantine_dir=config.QUARANTINE_DIR,
        )
    finally:
        for ext in ("html", "json"):
            relatorio = pasta / f"{batch_name}_report.{ext}"
            if relatorio.is_file():
                run.attach(relatorio, artifact_path="reports")


def _champion() -> tuple[ModelVersion, str]:
    """A versão `champion` e a tag `model_sha256` dela (sem champion: erro de domínio)."""
    esperado = champion_sha256()
    champion = get_alias_version(CHAMPION_ALIAS)
    if champion is None or champion.tags.get("model_sha256") != esperado:
        raise StagePreconditionError(
            "O alias champion mudou durante o retreino (promoção ou rollback em "
            "paralelo?): nada foi treinado; rode o retreino de novo."
        )
    return champion, esperado


def _require_same_reference(reference_path: Path, champion: ModelVersion) -> str:
    """A referência tem de ser a do champion: validação e teste dependem dela."""
    obtido = file_sha256(reference_path)
    esperado = champion.tags.get("reference_sha256")
    if obtido != esperado:
        raise StagePreconditionError(
            f"Referência {reference_path} com sha256 {obtido[:12]}... diferente da tag "
            f"reference_sha256 do champion v{champion.version} "
            f"({str(esperado)[:12]}...): a validação e o teste do challenger não seriam "
            "os do champion. Rode `make pipeline` para treinar o baseline nesta "
            "referência."
        )
    return obtido


def _require_batch_sha(gatilho: RetrainTrigger) -> str:
    obtido = file_sha256(gatilho.batch_path)
    esperado = gatilho.expected_batch_sha256
    if esperado is not None and obtido != esperado:
        raise StagePreconditionError(
            f"O lote {gatilho.batch_path} mudou desde o alerta {gatilho.alert_run_id}: "
            f"sha256 {obtido[:12]}... diferente da tag batch_sha256 "
            f"({esperado[:12]}...). O `simulate` regera t1/t2; rode `make demo-drift` "
            "para registrar um alerta sobre o lote atual."
        )
    return obtido


def _float_tags(valores: dict[str, float]) -> dict[str, str]:
    return {chave: repr(float(valor)) for chave, valor in valores.items()}


def run_retrain(run: StageRun, gatilho: RetrainTrigger) -> RetrainResult:
    """Valida, treina, avalia e registra o challenger; nunca move `champion`."""
    # 1. Champion conferido (sha256 da tag `model_sha256`) e referência dele.
    champion, esperado = _champion()
    bundle_champion = predict.load_bundle(
        predict.default_model_path(), expected_sha256=esperado
    )
    reference_path = _require_file(Path(config.REFERENCE_FILE), "Dataset de referência")
    reference_sha = _require_same_reference(reference_path, champion)
    batch_path = _require_file(gatilho.batch_path, "Lote do retreino")
    batch_sha = _require_batch_sha(gatilho)

    # 2. Contrato rotulado na referência e no lote, antes de qualquer treino.
    referencia_bruta = _read_labeled(reference_path, "Dataset de referência")
    lote = _read_labeled(batch_path, "Lote do retreino")
    referencia = _validated(run, referencia_bruta, REFERENCE_BATCH_NAME)
    lote = _validated(run, lote, gatilho.batch_name)
    run.add_rows(len(referencia) + len(lote))

    # 3. Divisão sem vazamento: o split da referência roda sobre ela sozinha (antes
    # de juntar o lote) e o holdout é conferido contra as outras três partes.
    critico_treino, holdout = split_critical_batch(lote)
    treino, validacao, teste = build_challenger_training(referencia, critico_treino)
    require_disjoint(treino=treino, validacao=validacao, teste=teste, holdout=holdout)

    # 4. Challenger: mesmo pipeline e hiperparâmetros; threshold na validação.
    inicio = time.perf_counter()
    pipeline, threshold = train.fit_model("xgboost", treino, validacao, config.SEED)
    segundos = round(time.perf_counter() - inicio, 2)

    # 5. Champion e challenger no MESMO holdout e no teste da referência.
    campeao = bundle_champion["pipeline"]
    aucs = {
        "champion_holdout_auc": _auc(campeao, holdout),
        "challenger_holdout_auc": _auc(pipeline, holdout),
        "champion_reference_auc": _auc(campeao, teste),
        "challenger_reference_auc": _auc(pipeline, teste),
    }

    model_version = challenger_model_version(pipeline)
    score_teste = pipeline.predict_proba(model_input(teste))[:, 1]
    y_teste = teste[config.TARGET_COLUMN].to_numpy()
    bundle = train.make_bundle(pipeline, threshold, model_version, config.SEED)
    metadata = train.model_metadata(
        model_version=model_version,
        seed=config.SEED,
        splits={
            "train": int(len(treino)),
            "validation": int(len(validacao)),
            "test": int(len(teste)),
            "critical_train": int(len(critico_treino)),
            "critical_holdout": int(len(holdout)),
        },
        xgb_params=None,
        models={
            "xgboost": {
                "threshold": round(threshold, 6),
                "test_metrics": evaluate(y_teste, score_teste, threshold),
                "train_seconds": segundos,
            }
        },
        # Referência do PSI do score do challenger: os scores dele no teste.
        score_reference=score_distribution(score_teste),
    )
    metadata["data_lineage"] = {
        "reference_file": config.project_relative(reference_path),
        "reference_sha256": reference_sha,
        "reference_rows": int(len(referencia)),
        "critical_batch": {
            "batch_name": gatilho.batch_name,
            "batch_path": config.project_relative(batch_path),
            "batch_sha256": batch_sha,
            "rows": int(len(lote)),
            "train_rows": int(len(critico_treino)),
            "holdout_rows": int(len(holdout)),
        },
        "trigger": gatilho.trigger,
        "alert_run_id": gatilho.alert_run_id,
        "champion": {
            "version": int(champion.version),
            "model_sha256": esperado,
            "bundle_model_version": str(bundle_champion["model_version"]),
        },
    }
    metadata["environment"] = train.environment()

    with tempfile.TemporaryDirectory(prefix="credit-drift-monitor-retrain-") as tmp:
        modelo_path = Path(tmp) / BUNDLE_MODEL_FILE
        metadata_path = Path(tmp) / BUNDLE_METADATA_FILE
        train.save_bundle(
            bundle,
            metadata,
            modelo_path,
            metadata_path,
            artifact_file=f"{BUNDLE_ARTIFACT_DIR}/{BUNDLE_MODEL_FILE}",
        )
        # Regra 1 (§4.7): integridade do artefato; o contrato já passou acima.
        verify_model_integrity(modelo_path, metadata_path)
        decisao = decide_promotion(contract_ok=True, integrity_ok=True, **aucs)
        model_sha = metadata["artifact"]["sha256"]
        versao, retrain_run_id = _register(
            run,
            gatilho,
            pipeline=pipeline,
            exemplo=referencia_bruta[config.MODEL_INPUT_COLUMNS].head(
                INPUT_EXAMPLE_ROWS
            ),
            modelo_path=modelo_path,
            metadata_path=metadata_path,
            tags={
                "bundle_model_version": model_version,
                "model_sha256": model_sha,
                "reference_sha256": reference_sha,
                "batch_name": gatilho.batch_name,
                "batch_path": config.project_relative(batch_path),
                "batch_sha256": batch_sha,
                "trigger": gatilho.trigger,
                "champion_version": str(int(champion.version)),
                **(
                    {"alert_run_id": gatilho.alert_run_id}
                    if gatilho.alert_run_id
                    else {}
                ),
            },
            decisao=decisao,
            aucs=aucs,
            metricas={
                "threshold": float(threshold),
                "train_rows": float(len(treino)),
                "critical_train_rows": float(len(critico_treino)),
                "holdout_rows": float(len(holdout)),
                "validation_rows": float(len(validacao)),
                "test_rows": float(len(teste)),
            },
            params=metadata["xgboost_params"],
        )

    # Só depois do registro: uma falha antes daqui não consome o alerta.
    if gatilho.alert_run_id is not None:
        registry.mark_alert_consumed(gatilho.alert_run_id, versao)
    log_event(
        logger,
        "retrain_decision",
        challenger_version=versao,
        model_version=model_version,
        champion_version=int(champion.version),
        eligible=decisao.eligible,
        reasons=list(decisao.reasons),
        gain_on_holdout=decisao.gain_on_holdout,
        regression_on_reference=decisao.regression_on_reference,
        mlflow_run_id=retrain_run_id,
        alert_run_id=gatilho.alert_run_id,
        **aucs,
    )
    return RetrainResult(
        version=versao,
        model_version=model_version,
        champion_version=int(champion.version),
        decision=decisao,
        aucs=aucs,
        holdout_rows=int(len(holdout)),
        alert_run_id=gatilho.alert_run_id,
    )


def _register(
    run: StageRun,
    gatilho: RetrainTrigger,
    *,
    pipeline: Any,
    exemplo: pd.DataFrame,
    modelo_path: Path,
    metadata_path: Path,
    tags: dict[str, str],
    decisao: PromotionDecision,
    aucs: dict[str, float],
    metricas: dict[str, float],
    params: dict[str, object],
) -> tuple[int, str]:
    """Run `run_type=retrain` (filho do estágio) e a versão com o alias `challenger`.

    As AUCs e o veredito vão também nas tags da versão: o comando de promoção os
    reavalia sem depender do run.
    """
    veredito = {
        "promotion_eligible": "true" if decisao.eligible else "false",
        "promotion_reasons": ";".join(decisao.reasons),
    }
    run_tags = {
        "run_type": RETRAIN_RUN_TYPE,
        "pipeline_run_id": RUN_ID,
        "alert_reused": "true" if gatilho.reused else "false",
        **tags,
        **veredito,
    }
    if gatilho.reason:
        run_tags["retrain_reason"] = redact_text(gatilho.reason)
    numeros = {
        **aucs,
        "gain_on_holdout": decisao.gain_on_holdout,
        "regression_on_reference": decisao.regression_on_reference,
    }
    with tracking_errors("Falha ao registrar o run de retreino"):
        experimento = mlflow.get_run(run.run_id).info.experiment_id
        retreino = mlflow.start_run(
            experiment_id=experimento,
            run_name=RETRAIN_RUN_NAME,
            nested=True,
            parent_run_id=run.run_id,
            tags=run_tags,
        )
    with retreino:
        retrain_run_id = retreino.info.run_id
        with tracking_errors("Falha ao registrar o run de retreino"):
            mlflow.log_params(
                {
                    **params,
                    "promotion_min_gain": config.PROMOTION_MIN_GAIN,
                    "promotion_max_regression": config.PROMOTION_MAX_REGRESSION,
                }
            )
            # NaN (AUC indefinida) não vira métrica: o veredito já reprova o eixo.
            finitos = {k: float(v) for k, v in numeros.items() if math.isfinite(v)}
            mlflow.log_metrics(
                {
                    **finitos,
                    **metricas,
                    "promotion_eligible": 1.0 if decisao.eligible else 0.0,
                }
            )
        versao = registry.register_challenger(
            run_id=retrain_run_id,
            pipeline=pipeline,
            signature=infer_model_signature(pipeline, exemplo),
            input_example=exemplo,
            bundle_path=modelo_path,
            metadata_path=metadata_path,
            tags={
                **tags,
                **veredito,
                **_float_tags(aucs),
                "retrain_run_id": retrain_run_id,
            },
        )
        with tracking_errors("Falha ao registrar o run de retreino"):
            mlflow.set_tag("challenger_version", str(versao))
    return versao, retrain_run_id


# MARK: CLI


def _parser() -> CliParser:
    parser = CliParser(
        prog="python -m src.monitoring.retrain",
        description=(
            "Treina o challenger a partir de um alerta CRÍTICO e o registra no Model "
            "Registry com o alias challenger (nunca promove). Saída: 0 challenger "
            "registrado, 2 recusa (sem alerta), 3 contrato barrou, 1 erro operacional."
        ),
    )
    parser.add_argument(
        "--alert-run",
        default=None,
        help="run de lote CRITICO a usar (padrão: o mais recente não consumido)",
    )
    parser.add_argument(
        "--force-no-alert",
        action="store_true",
        help="retreina sem alerta, com o lote t2 da demo (exige --reason)",
    )
    parser.add_argument(
        "--allow-reuse",
        action="store_true",
        help="aceita um alerta já consumido (exige --alert-run e --reason)",
    )
    parser.add_argument(
        "--reason", default=None, help="motivo da exceção (gravado no run)"
    )
    return parser


def _parse(argv: list[str] | None):
    parser = _parser()
    args = parser.parse_args(argv)
    args.reason = args.reason.strip() if args.reason else None
    if args.reason and len(args.reason) > MAX_TAG_VAL_LENGTH:
        parser.error(f"--reason passa de {MAX_TAG_VAL_LENGTH} caracteres")
    if args.force_no_alert and (args.alert_run or args.allow_reuse):
        parser.error("--force-no-alert não se combina com --alert-run/--allow-reuse")
    if args.allow_reuse and not args.alert_run:
        parser.error("--allow-reuse exige --alert-run (o alerta a reaproveitar)")
    if (args.force_no_alert or args.allow_reuse) and not args.reason:
        parser.error(
            "--force-no-alert e --allow-reuse exigem --reason (o motivo vai para o run)"
        )
    return args


def _summary(result: RetrainResult) -> str:
    d = result.decision
    veredito = "ELEGÍVEL" if d.eligible else "INELEGÍVEL"
    motivos = ", ".join(d.reasons) or "nenhum"
    a = result.aucs
    return (
        f"Challenger v{result.version} ({result.model_version}) registrado com o alias "
        f"challenger. Veredito: {veredito} à promoção (motivos: {motivos}).\n"
        f"  Holdout do lote ({result.holdout_rows} linhas): champion "
        f"{a['champion_holdout_auc']:.4f} x challenger {a['challenger_holdout_auc']:.4f}"
        f" (ganho {d.gain_on_holdout:+.4f}; mínimo {config.PROMOTION_MIN_GAIN}).\n"
        f"  Teste da referência: champion {a['champion_reference_auc']:.4f} x "
        f"challenger {a['challenger_reference_auc']:.4f} (regressão "
        f"{d.regression_on_reference:+.4f}; máximo {config.PROMOTION_MAX_REGRESSION}).\n"
        f"O champion (v{result.champion_version}) não foi alterado: promover é decisão "
        "humana (`make promote`)."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if not tracking.tracking_enabled():
        print(
            f"ERRO no retreino: rastreamento desligado ({TRACKING_ENV_VAR}=off). O "
            "retreino depende do MLflow e do Model Registry (alerta, champion e "
            "challenger) e falha fechado sem eles.",
            file=sys.stderr,
        )
        return EXIT_OPERATIONAL_ERROR
    try:
        gatilho = resolve_trigger(
            alert_run=args.alert_run,
            force_no_alert=args.force_no_alert,
            allow_reuse=args.allow_reuse,
            reason=args.reason,
        )
        with tracked_stage(STAGE_NAME) as run:
            resultado = run_retrain(run, gatilho)
    except RetrainRefusedError as erro:
        log_event(logger, "retrain_refused", level=logging.WARNING, message=str(erro))
        print(f"RECUSADO no retreino: {erro}", file=sys.stderr)
        return EXIT_REFUSED
    except DataContractViolation as erro:
        print(
            f"BLOQUEADO no retreino: {erro} Nenhum challenger foi treinado; veja os "
            f"relatórios em {config.project_relative(Path(RETRAIN_REPORTS_DIR))}.",
            file=sys.stderr,
        )
        return EXIT_BLOCKED
    except (
        TrackingUnavailableError,
        ModelIntegrityError,
        StagePreconditionError,
        RegistryStateError,
    ) as erro:
        # A mensagem já vem redigida (sem credenciais da URI); nada de traceback.
        print(f"ERRO no retreino: {erro}", file=sys.stderr)
        return EXIT_OPERATIONAL_ERROR
    print(_summary(resultado))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
