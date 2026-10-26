"""Estágios do pipeline rastreados no MLflow (spec da Etapa 3, §4.1 e §4.7).

    uv run python -m src.monitoring.stage {prepare,train,drift_demo}

Cada estágio abre um run de estágio (`tracked_stage`) e chama o `main` do módulo
de baixo, que não conhece o MLflow. O `train` cria ainda o run de treino (filho),
registra a versão no Model Registry e, só na primeira vez (sem `champion`), copia o
modelo para `models/champion/` (conferindo o sha256) e só então dá o alias
`champion`: um export falho não deixa alias, e o `train` seguinte tenta de novo.

O `drift_demo` roda a demo de drift com o champion (sha256 conferido contra a tag
`model_sha256` do alias) e, pelo gancho `on_batch`, cria um run filho por lote com
o nível de alerta (§4.2), as métricas de saúde (§4.3) e os relatórios anexados.

Falha fechada: sem MLflow acessível, imprime o erro (URI já redigida) e sai com
código 1; `MONITORING_TRACKING=off` roda o estágio sem registro.

`contract_demo` (Tarefa 8) entra em `STAGES` quando for implementado.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path

import mlflow
import pandas as pd
import pyarrow.parquet as pq
from mlflow.models import infer_signature

from src import config
from src.data import prepare
from src.drift import demo as drift_demo
from src.drift.demo import BatchObservation
from src.drift.performance import assess_auc_drop
from src.drift.psi import score_psi
from src.logging_utils import RUN_ID, get_logger, log_event
from src.models import predict, train
from src.models.integrity import ModelIntegrityError
from src.monitoring.alerts import decide_alert_level
from src.monitoring.registry import (
    CHAMPION_ALIAS,
    bootstrap_champion,
    export_champion,
    get_alias_version,
    register_model_version,
)
from src.monitoring.tracking import (
    BatchRecord,
    StageRun,
    TrackingUnavailableError,
    require_path,
    tracked_stage,
    tracking_errors,
)

logger = get_logger("monitoring.stage")

TRAINING_RUN_NAME = "train_baseline"
INPUT_EXAMPLE_ROWS = 5
_INT_COLUMNS_HINT = r"Hint: Inferred schema contains integer column"


def run_prepare(run: StageRun) -> None:
    """Referência + produção t0; volume = linhas das duas saídas."""
    prepare.main()
    linhas = sum(
        pq.read_metadata(caminho).num_rows
        for caminho in (prepare.REFERENCE_FILE, prepare.PRODUCTION_BASE_FILE)
    )
    run.add_rows(linhas)


def _training_metrics(metadata: dict) -> dict[str, float]:
    oficial = metadata["models"][metadata["official_model"]]
    metricas = {
        chave: float(valor)
        for chave, valor in oficial["test_metrics"].items()
        if valor is not None
    }
    metricas["threshold"] = float(oficial["threshold"])
    return metricas


def run_train(run: StageRun) -> None:
    """Treina o baseline e, com o rastreamento ligado, registra e faz o bootstrap."""
    bundle, metadata = train.main()
    run.add_rows(metadata["data_lineage"]["reference_rows"])
    if not run.enabled:
        return

    # Lidos do módulo na chamada: os testes redirecionam os caminhos do treino.
    model_path = require_path(train.MODEL_FILE, "Bundle do modelo")
    metadata_path = require_path(train.MODEL_METADATA_FILE, "Metadata do modelo")
    exemplo = pd.read_parquet(
        train.REFERENCE_FILE, columns=config.MODEL_INPUT_COLUMNS
    ).head(INPUT_EXAMPLE_ROWS)
    pipeline = bundle["pipeline"]
    with warnings.catch_warnings():
        # Colunas inteiras do contrato são `nullable=False`: o aviso do MLflow sobre
        # inteiros com valores ausentes não se aplica.
        warnings.filterwarnings(
            "ignore", message=_INT_COLUMNS_HINT, category=UserWarning
        )
        assinatura = infer_signature(exemplo, pipeline.predict_proba(exemplo))
    shas = {
        "reference_sha256": metadata["data_lineage"]["reference_sha256"],
        "model_sha256": metadata["artifact"]["sha256"],
    }

    with tracking_errors("Falha ao registrar o run de treino"):
        experimento = mlflow.get_run(run.run_id).info.experiment_id
        treino = mlflow.start_run(
            experiment_id=experimento,
            run_name=TRAINING_RUN_NAME,
            nested=True,
            parent_run_id=run.run_id,
            tags={
                "run_type": "training",
                "pipeline_run_id": RUN_ID,
                "bundle_model_version": bundle["model_version"],
                **shas,
            },
        )
    with treino:
        with tracking_errors("Falha ao registrar o run de treino"):
            mlflow.log_params(metadata["xgboost_params"])
            mlflow.log_metrics(_training_metrics(metadata))
        versao = register_model_version(
            run_id=treino.info.run_id,
            pipeline=pipeline,
            signature=assinatura,
            input_example=exemplo,
            bundle_path=model_path,
            metadata_path=metadata_path,
            tags={
                "origin": "baseline",
                "bundle_model_version": bundle["model_version"],
                **shas,
            },
        )

    _bootstrap(versao, shas)


class StagePreconditionError(RuntimeError):
    """Pré-condição do estágio não atendida (ex.: sem champion no Model Registry)."""


# Motivos de "performance não avaliada" (além de `single_class`, da §4.2). Lote sem
# `target` não leva motivo: `performance_evaluated=false` já o diz, e o lote
# estável sem rótulo continua `OK`.
REASON_SINGLE_CLASS = "single_class"
REASON_EMPTY_BATCH = "empty_batch"
REASON_AUC_UNDEFINED = "auc_undefined"
REASON_NO_BASELINE = "no_baseline_auc"


def _batch_auc(obs: BatchObservation) -> tuple[float | None, str | None]:
    """ROC-AUC do lote, ou `None` com o motivo de não avaliar.

    NaN nunca chega à regra: AUC não finita vira "não avaliado" (`auc_undefined`).
    """
    scored = obs.scored
    if scored.empty:
        return None, REASON_EMPTY_BATCH
    if config.TARGET_COLUMN not in scored.columns:
        return None, None
    if scored[config.TARGET_COLUMN].nunique(dropna=True) < 2:
        return None, REASON_SINGLE_CLASS
    auc = obs.metrics.get("roc_auc")
    if auc is None or not math.isfinite(auc):
        return None, REASON_AUC_UNDEFINED
    return float(auc), None


def _report_files(obs: BatchObservation) -> list[Path]:
    """Relatórios HTML/JSON do Evidently e, se houver, os do contrato do lote."""
    arquivos = [config.ROOT_DIR / p for p in obs.drift.report_paths.values()]
    contrato = obs.report_dir / "contract_gate"
    arquivos += [
        arquivo
        for arquivo in (
            contrato / f"{obs.batch_name}_report.html",
            contrato / f"{obs.batch_name}_report.json",
        )
        if arquivo.is_file()
    ]
    return arquivos


class DriftBatchObserver:
    """Gancho `on_batch` do `drift_demo`: nível de alerta e run filho por lote.

    O baseline da queda de AUC é o AUC do PRIMEIRO lote observado (t0, §4.2); o t0
    tem `auc_drop=0`. `auc_drop_vs_training` (contra o `roc_auc` de teste do
    metadata) é só informativo. O PSI do score usa `metadata["score_reference"]`.
    `target` fica fora de `drifted_columns` e do PSI por coluna; o resultado sobre
    ele vai à parte (`label_shift`, `label_shift_psi`).
    """

    def __init__(self, run: StageRun, metadata: dict) -> None:
        self._run = run
        self._score_reference = metadata["score_reference"]
        oficial = metadata["models"][metadata["official_model"]]
        self._training_auc = float(oficial["test_metrics"]["roc_auc"])
        self._baseline_auc: float | None = None
        self._seen_first = False

    def __call__(self, obs: BatchObservation) -> None:
        record = self.record(obs)
        self._run.log_batch(record)
        log_event(
            logger,
            "batch_alert",
            batch_name=record.batch_name,
            alert_level=record.decision.level.value,
            alert_reasons=list(record.decision.reasons),
            performance_evaluated=record.decision.performance_evaluated,
        )

    def record(self, obs: BatchObservation) -> BatchRecord:
        auc, skip_reason = _batch_auc(obs)
        if not self._seen_first:
            self._seen_first = True
            self._baseline_auc = auc
        auc_drop = None
        if auc is not None:
            if self._baseline_auc is None:
                skip_reason = REASON_NO_BASELINE
            else:
                auc_drop = assess_auc_drop(self._baseline_auc, auc).drop

        empty = obs.scored.empty
        batch_score_psi = (
            None
            if empty
            else score_psi(self._score_reference, obs.scored["default_probability"])
        )
        colunas = [
            c for c in obs.drift.column_results if c.column != config.TARGET_COLUMN
        ]
        drifted = [c.column for c in colunas if c.drifted]
        label = next(
            (c for c in obs.drift.column_results if c.column == config.TARGET_COLUMN),
            None,
        )
        decision = decide_alert_level(
            blocked=False,
            drifted_columns=drifted,
            score_psi=batch_score_psi,
            auc_drop=auc_drop,
            performance_skip_reason=skip_reason,
        )
        metrics = {
            "roc_auc": auc,
            "auc_drop": auc_drop,
            "auc_drop_vs_training": (
                None if auc is None else round(self._training_auc - auc, 4)
            ),
            "approval_rate": obs.metrics.get("approval_rate"),
            "default_rate_approved": obs.metrics.get("default_rate_approved"),
            "score_psi": batch_score_psi,
            "label_shift": None if label is None else float(label.drifted),
            "label_shift_psi": None if label is None else label.psi,
        }
        return BatchRecord(
            batch_name=obs.batch_name,
            decision=decision,
            n_rows=len(obs.scored),
            rows_rejected=0,
            metrics=metrics,
            psi_by_column={c.column: c.psi for c in colunas if c.psi is not None},
            n_drifted_columns=len(drifted),
            model_version=obs.model_version,
            artifacts=_report_files(obs),
            batch_path=obs.batch_path,
        )


def _champion_sha256() -> str:
    """Tag `model_sha256` do champion; sem champion, erro orientando o `train`."""
    champion = get_alias_version(CHAMPION_ALIAS)
    if champion is None:
        raise StagePreconditionError(
            f"Nenhuma versão de {config.REGISTERED_MODEL_NAME} tem o alias "
            f"`{CHAMPION_ALIAS}` no Model Registry: não há modelo de produção para "
            "monitorar. Rode `make pipeline` (ou o estágio `train`: "
            "`uv run python -m src.monitoring.stage train`) antes do drift_demo."
        )
    esperado = champion.tags.get("model_sha256")
    if not esperado:
        raise ModelIntegrityError(
            f"Versão {champion.version} (champion) sem a tag model_sha256: sem hash "
            "registrado, o modelo servido não pode ser conferido."
        )
    return esperado


def run_drift_demo(run: StageRun) -> None:
    """Demo de drift com um run filho por lote (t0, t1, t2).

    Com rastreamento, o modelo carregado tem de ter o sha256 da tag `model_sha256`
    do champion (conferido por `load_bundle` dentro de `run_demo`). Sem
    rastreamento (`MONITORING_TRACKING=off`), roda a demo como o CLI, sem Registry.
    """
    if not run.enabled:
        drift_demo.run_demo()
        return
    esperado = _champion_sha256()
    modelo = predict.default_model_path()
    metadata_path = require_path(
        modelo.with_name(config.MODEL_METADATA_FILE.name), "Metadata do modelo"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    drift_demo.run_demo(
        on_batch=DriftBatchObserver(run, metadata),
        bundle_loader=partial(predict.load_bundle, modelo, expected_sha256=esperado),
    )


def _bootstrap(versao: int, shas: dict[str, str]) -> None:
    """Sem champion: exporta (conferindo o sha256) e só então dá o alias.

    Export falho levanta antes do alias: nenhum champion aponta para um
    `models/champion/` ausente, e o `train` seguinte tenta de novo.
    """
    if get_alias_version(CHAMPION_ALIAS) is not None:
        return
    export_champion(versao)
    if bootstrap_champion(versao):
        log_event(logger, "champion_bootstrap", version=versao, **shas)


# Tabela de estágios: a Tarefa 8 acrescenta `contract_demo`.
STAGES: dict[str, Callable[[StageRun], None]] = {
    "prepare": run_prepare,
    "train": run_train,
    "drift_demo": run_drift_demo,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.monitoring.stage",
        description="Executa um estágio do pipeline com rastreamento no MLflow.",
    )
    parser.add_argument("stage", metavar="ESTAGIO", help=f"um de: {', '.join(STAGES)}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    executar = STAGES.get(args.stage)
    if executar is None:
        parser.error(
            f"estágio {args.stage!r} desconhecido ou ainda não disponível "
            f"(disponíveis: {', '.join(STAGES)})"
        )
    try:
        with tracked_stage(args.stage) as run:
            executar(run)
    except (
        TrackingUnavailableError,
        ModelIntegrityError,
        StagePreconditionError,
    ) as erro:
        # A mensagem já vem redigida (sem credenciais da URI); nada de traceback.
        print(f"ERRO no estágio {args.stage}: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
