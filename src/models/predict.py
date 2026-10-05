"""Pontuacao de lotes com o modelo baseline -- sempre atras do gate.

Todo lote passa pelo contrato antes de chegar ao modelo. Lotes rotulados
(com `target`, ex.: feedback atrasado de producao usado na Etapa 2) sao
validados pelo contrato rotulado; os demais, pelo contrato de inferencia.

    uv run python -m src.models.predict --input data/samples/valid_batch.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import ID_COLUMN, MODEL_FILE, PRODUCTION_DIR, TARGET_COLUMN
from src.logging_utils import get_logger, log_event
from src.models.features import model_input
from src.validation.gate import DataContractViolation, enforce_contract
from src.validation.validate import read_batch

logger = get_logger("models.predict")


def load_bundle(path: Path = MODEL_FILE) -> dict:
    if not path.exists():
        sys.exit(f"Modelo nao encontrado em {path}. Rode `make train` antes.")
    return joblib.load(path)


def score_batch(
    df: pd.DataFrame,
    bundle: dict,
    *,
    batch_name: str,
    stage: str = "inference",
    **gate_options: Any,
) -> pd.DataFrame:
    """Valida o lote (bloqueia se violar o contrato) e devolve as predicoes.

    `gate_options` e repassado a `enforce_contract` (ex.: `report_dir=None`).
    """
    labeled = TARGET_COLUMN in df.columns
    validated = enforce_contract(
        df, batch_name=batch_name, stage=stage, labeled=labeled, **gate_options
    )

    started = time.perf_counter()
    proba = bundle["pipeline"].predict_proba(model_input(validated))[:, 1]
    latency_ms = (time.perf_counter() - started) * 1000

    keep = [ID_COLUMN, TARGET_COLUMN] if labeled else [ID_COLUMN]
    scored = validated[keep].copy()
    scored["default_probability"] = proba
    scored["predicted_default"] = (proba >= bundle["threshold"]).astype(int)
    scored["decision"] = np.where(
        scored["predicted_default"] == 1, "negado", "aprovado"
    )
    scored["model_version"] = bundle["model_version"]

    log_event(
        logger,
        "batch_scored",
        batch_name=batch_name,
        model_version=bundle["model_version"],
        n_rows=len(scored),
        approval_rate=round(float((scored["predicted_default"] == 0).mean()), 4),
        mean_default_probability=round(float(proba.mean()), 4),
        latency_ms=round(latency_ms, 2),
    )
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pontua um lote com o baseline.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="padrao: data/production/predictions_<lote>.parquet"
    )
    args = parser.parse_args(argv)

    bundle = load_bundle()
    try:
        scored = score_batch(read_batch(args.input), bundle, batch_name=args.input.stem)
    except DataContractViolation as exc:
        print(f"\n{exc}\nNenhuma predicao foi gerada.", file=sys.stderr)
        return 1

    output = args.output or PRODUCTION_DIR / f"predictions_{args.input.stem}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output, index=False)
    print(f"{len(scored)} predicoes gravadas em {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
