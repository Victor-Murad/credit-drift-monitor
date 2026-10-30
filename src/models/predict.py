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

from src import config
from src.config import ID_COLUMN, PRODUCTION_DIR, TARGET_COLUMN
from src.logging_utils import get_logger, log_event
from src.models.features import model_input
from src.models.integrity import (
    ModelIntegrityError,
    file_sha256,
    verify_model_integrity,
)
from src.models.portable import restore_booster
from src.validation.gate import DataContractViolation, enforce_contract
from src.validation.io import read_batch

logger = get_logger("models.predict")


def default_model_path() -> Path:
    """Champion servido (`models/champion/`) se existir; senão o bundle do treino.

    Os caminhos de `src.config` são lidos na chamada (os testes os redirecionam).
    """
    champion = Path(config.CHAMPION_FILE)
    return champion if champion.exists() else Path(config.MODEL_FILE)


def load_bundle(
    path: Path | None = None, *, expected_sha256: str | None = None
) -> dict:
    """Carrega o bundle somente se o sha256 bater com o registrado no treino.

    `path=None` usa `default_model_path()`. Com `expected_sha256` (ex.: a tag
    `model_sha256` do champion no Model Registry), o arquivo também tem de ter esse
    hash; divergência encerra com "Modelo recusado" sem desserializar nada.
    """
    path = Path(path) if path is not None else default_model_path()
    if not path.exists():
        sys.exit(f"Modelo nao encontrado em {path}. Rode `make train` antes.")
    try:
        verify_model_integrity(path)
    except ModelIntegrityError as exc:
        sys.exit(f"Modelo recusado: {exc}")
    if expected_sha256 is not None:
        obtido = file_sha256(path)
        if obtido != expected_sha256:
            sys.exit(
                f"Modelo recusado: sha256 de {path} diverge do esperado "
                f"(esperado {expected_sha256[:12]}..., obtido {obtido[:12]}...). "
                "O arquivo não é o champion registrado no Model Registry."
            )
    # joblib e pickle: so desserializamos o arquivo cujo hash foi conferido acima.
    return restore_booster(joblib.load(path))


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
