"""Leitura de lotes (CSV/Parquet) na fronteira de ingestao.

Fora de `validate.py` (CLI) para que `models.predict` leia lotes sem depender
de um modulo de linha de comando.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_batch(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato nao suportado: {path.suffix} (use .csv ou .parquet)")
