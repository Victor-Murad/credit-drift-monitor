"""Simulacao de Data Drift e Concept Drift a partir do lote de producao t0.

Parte de `data/production/production_base.parquet` (nunca visto no treino) e
gera fatias que continuam **validas no contrato Pandera** (clip nos limites).
Drift nao e dado invalido: uma parcela maior e plausivel; uma negativa nao.

    uv run python -m src.drift.simulate
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ANNUITY_SHOCK,
    CONCEPT_FLIP_RATE,
    CONCEPT_SCORE_COLUMN,
    CONCEPT_SCORE_MIN,
    EXT_SOURCE_3_SHIFT,
    INCOME_SHOCK,
    PRODUCTION_BASE_FILE,
    PRODUCTION_T1_FILE,
    PRODUCTION_T2_FILE,
    SEED,
    TARGET_COLUMN,
    set_global_seed,
)
from src.logging_utils import get_logger, log_event
from src.validation.contract import MAX_INCOME_TOTAL

logger = get_logger("drift.simulate")


@dataclass(frozen=True)
class DataDriftShocks:
    """Choques numericos aplicados as colunas de entrada."""

    annuity_multiplier: float = ANNUITY_SHOCK
    ext_source_3_shift: float = EXT_SOURCE_3_SHIFT
    income_multiplier: float = INCOME_SHOCK


DEFAULT_SHOCKS = DataDriftShocks()


def _clip_to_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que valores driftados continuem dentro das regras do contrato."""
    out = df.copy()

    if "income_total" in out.columns:
        out["income_total"] = out["income_total"].clip(upper=MAX_INCOME_TOTAL)
        # Renda deve permanecer > 0 (contrato R4).
        out.loc[out["income_total"] <= 0, "income_total"] = 0.01

    for col in ("ext_source_2", "ext_source_3"):
        if col in out.columns:
            # Mantem NaNs; so clipa valores presentes.
            mask = out[col].notna()
            out.loc[mask, col] = out.loc[mask, col].clip(0.0, 1.0)

    if "annuity_amount" in out.columns and "credit_amount" in out.columns:
        # R5: parcela > 0; R8: parcela <= credito.
        out["annuity_amount"] = out["annuity_amount"].clip(lower=0.01)
        out["annuity_amount"] = np.minimum(out["annuity_amount"], out["credit_amount"])

    return out


def apply_data_drift(
    df: pd.DataFrame,
    *,
    shocks: DataDriftShocks = DEFAULT_SHOCKS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Aplica choques nas variaveis importantes e clipa aos limites do contrato.

    `seed` e aceito para API estavel (choques sao deterministicos; nao ha
    aleatoriedade nesta funcao alem do que o caller ja tenha no DataFrame).
    """
    _ = seed  # API estavel; choques sao multiplicativos/aditivos fixos.
    out = df.copy()

    if "annuity_amount" in out.columns:
        out["annuity_amount"] = out["annuity_amount"] * shocks.annuity_multiplier

    if "ext_source_3" in out.columns:
        mask = out["ext_source_3"].notna()
        out.loc[mask, "ext_source_3"] = (
            out.loc[mask, "ext_source_3"] + shocks.ext_source_3_shift
        )

    if "income_total" in out.columns:
        out["income_total"] = out["income_total"] * shocks.income_multiplier

    return _clip_to_contract(out)


def apply_concept_drift(
    df: pd.DataFrame,
    *,
    rate: float = CONCEPT_FLIP_RATE,
    score_col: str = CONCEPT_SCORE_COLUMN,
    score_min: float = CONCEPT_SCORE_MIN,
    seed: int = SEED,
) -> pd.DataFrame:
    """Altera a relacao X -> y: flipa bons pagadores com score alto para default.

    Nao mexe nas features — so no rotulo. Requer coluna `target`.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Concept drift exige a coluna '{TARGET_COLUMN}' (lote rotulado)."
        )
    if score_col not in df.columns:
        raise ValueError(f"Coluna de score '{score_col}' ausente no DataFrame.")

    out = df.copy()
    rng = np.random.default_rng(seed)
    eligible = (
        (out[TARGET_COLUMN] == 0)
        & out[score_col].notna()
        & (out[score_col] >= score_min)
    )
    idx = out.index[eligible]
    if len(idx) == 0 or rate <= 0:
        return out

    n_flip = int(np.floor(len(idx) * rate))
    if n_flip == 0:
        return out

    chosen = rng.choice(idx.to_numpy(), size=n_flip, replace=False)
    out.loc[chosen, TARGET_COLUMN] = 1
    return out


def simulate_production_slices(
    base: pd.DataFrame | None = None,
    *,
    shocks: DataDriftShocks = DEFAULT_SHOCKS,
    concept_rate: float = CONCEPT_FLIP_RATE,
    seed: int = SEED,
    t1_path: Path = PRODUCTION_T1_FILE,
    t2_path: Path = PRODUCTION_T2_FILE,
) -> dict[str, Path]:
    """Gera t1 (data drift) e t2 (data + concept) a partir do lote t0."""
    set_global_seed(seed)

    if base is None:
        if not PRODUCTION_BASE_FILE.exists():
            sys.exit(
                f"Lote de producao t0 nao encontrado em {PRODUCTION_BASE_FILE}. "
                "Rode `make data` antes."
            )
        base = pd.read_parquet(PRODUCTION_BASE_FILE)

    t1 = apply_data_drift(base, shocks=shocks, seed=seed)
    t2 = apply_concept_drift(
        apply_data_drift(base, shocks=shocks, seed=seed),
        rate=concept_rate,
        seed=seed,
    )

    t1_path.parent.mkdir(parents=True, exist_ok=True)
    t1.to_parquet(t1_path, index=False)
    t2.to_parquet(t2_path, index=False)

    default_t0 = float(base[TARGET_COLUMN].mean()) if TARGET_COLUMN in base else None
    default_t1 = float(t1[TARGET_COLUMN].mean()) if TARGET_COLUMN in t1 else None
    default_t2 = float(t2[TARGET_COLUMN].mean()) if TARGET_COLUMN in t2 else None

    log_event(
        logger,
        "drift_simulated",
        n_rows=len(base),
        t1_path=str(t1_path),
        t2_path=str(t2_path),
        annuity_multiplier=shocks.annuity_multiplier,
        ext_source_3_shift=shocks.ext_source_3_shift,
        income_multiplier=shocks.income_multiplier,
        concept_flip_rate=concept_rate,
        default_rate_t0=None if default_t0 is None else round(default_t0, 4),
        default_rate_t1=None if default_t1 is None else round(default_t1, 4),
        default_rate_t2=None if default_t2 is None else round(default_t2, 4),
    )
    return {"t1": t1_path, "t2": t2_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simula fatias de producao com Data Drift e Concept Drift."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PRODUCTION_BASE_FILE,
        help="Lote t0 (padrao: production_base.parquet)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Arquivo nao encontrado: {args.input}", file=sys.stderr)
        return 1

    if args.input.suffix.lower() == ".csv":
        base = pd.read_csv(args.input)
    else:
        base = pd.read_parquet(args.input)

    paths = simulate_production_slices(base)
    print(f"t1 (data drift):     {paths['t1']}")
    print(f"t2 (data+concept):   {paths['t2']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
