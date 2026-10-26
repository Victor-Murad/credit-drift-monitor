"""Estágios do pipeline rastreados no MLflow (spec da Etapa 3, §4.1 e §4.7).

    uv run python -m src.monitoring.stage {prepare,train}

Cada estágio abre um run de estágio (`tracked_stage`) e chama o `main` do módulo
de baixo, que não conhece o MLflow. O `train` cria ainda o run de treino (filho),
registra a versão no Model Registry e, só na primeira vez (sem `champion`), dá o
alias `champion` e copia o modelo para `models/champion/`.

Falha fechada: sem MLflow acessível, imprime o erro (URI já redigida) e sai com
código 1; `MONITORING_TRACKING=off` roda o estágio sem registro.

`contract_demo` (Tarefa 8) e `drift_demo` (Tarefa 7) entram em `STAGES` quando
forem implementados.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable

import mlflow
import pandas as pd
import pyarrow.parquet as pq
from mlflow.models import infer_signature

from src import config
from src.data import prepare
from src.logging_utils import RUN_ID, get_logger, log_event
from src.models import train
from src.models.integrity import ModelIntegrityError
from src.monitoring.registry import (
    bootstrap_champion,
    export_champion,
    register_model_version,
)
from src.monitoring.tracking import (
    NullStageRun,
    StageRun,
    TrackingUnavailableError,
    _require_path,
    _tracking_errors,
    tracked_stage,
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
    if isinstance(run, NullStageRun):
        return

    # Lidos do módulo na chamada: os testes redirecionam os caminhos do treino.
    model_path = _require_path(train.MODEL_FILE, "Bundle do modelo")
    metadata_path = _require_path(train.MODEL_METADATA_FILE, "Metadata do modelo")
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

    with _tracking_errors("Falha ao registrar o run de treino"):
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
        with _tracking_errors("Falha ao registrar o run de treino"):
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

    if bootstrap_champion(versao):
        export_champion(versao)
        log_event(logger, "champion_bootstrap", version=versao, **shas)


# Tabela de estágios: a Tarefa 7 acrescenta `drift_demo` e a 8, `contract_demo`.
STAGES: dict[str, Callable[[StageRun], None]] = {
    "prepare": run_prepare,
    "train": run_train,
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
    except (TrackingUnavailableError, ModelIntegrityError) as erro:
        # A mensagem já vem redigida (sem credenciais da URI); nada de traceback.
        print(f"ERRO no estágio {args.stage}: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
