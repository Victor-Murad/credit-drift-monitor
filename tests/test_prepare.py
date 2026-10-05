"""Ingestao do bruto -> esquema canonico -> referencia limpa e separada."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DAYS_EMPLOYED_SENTINEL, ID_COLUMN, RAW_COLUMN_MAP, TARGET_COLUMN
from src.data.prepare import (
    clean_with_contract,
    split_reference_production,
    to_canonical,
)
from src.validation.contract import LabeledCreditApplicationSchema


def _raw_from(valid: pd.DataFrame) -> pd.DataFrame:
    """Reconstroi um 'CSV bruto' do Home Credit a partir de dados canonicos."""
    inverse = {v: k for k, v in RAW_COLUMN_MAP.items()}
    raw = valid.drop(columns=["age_years", "years_employed"]).rename(columns=inverse)
    raw["DAYS_BIRTH"] = -(valid["age_years"] * 365.25).round().astype(int)
    raw["DAYS_EMPLOYED"] = np.where(
        valid["years_employed"].isna(),
        DAYS_EMPLOYED_SENTINEL,
        -(valid["years_employed"] * 365.25).round(),
    ).astype(int)
    return raw


def test_to_canonical_converts_days_and_sentinel(valid_df):
    canonical = to_canonical(_raw_from(valid_df))
    assert list(canonical.columns) == list(
        LabeledCreditApplicationSchema.to_schema().columns
    )
    assert np.allclose(canonical["age_years"], valid_df["age_years"], atol=0.01)
    assert (
        canonical["years_employed"].isna() == valid_df["years_employed"].isna()
    ).all()


def test_clean_with_contract_drops_only_bad_rows(valid_df):
    dirty = valid_df.copy()
    dirty.loc[0, "gender"] = "XNA"  # codigo de desconhecido do Home Credit
    dirty.loc[1, "income_total"] = 117_000_000.0  # outlier real do Home Credit
    dirty.loc[2, "annuity_amount"] = np.nan

    clean, summary = clean_with_contract(dirty)

    assert summary["n_rows_rejected"] == 3
    assert summary["n_rows_clean"] == len(valid_df) - 3
    assert not clean[ID_COLUMN].isin(dirty.loc[[0, 1, 2], ID_COLUMN]).any()
    LabeledCreditApplicationSchema.validate(clean)


def test_split_is_disjoint_stratified_and_reproducible(valid_df):
    ref_a, prod_a = split_reference_production(valid_df, seed=1)
    ref_b, _ = split_reference_production(valid_df, seed=1)

    assert ref_a.equals(ref_b)
    assert set(ref_a[ID_COLUMN]).isdisjoint(prod_a[ID_COLUMN])
    assert len(ref_a) + len(prod_a) == len(valid_df)
    assert abs(ref_a[TARGET_COLUMN].mean() - prod_a[TARGET_COLUMN].mean()) < 0.05
