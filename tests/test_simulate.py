"""Testes da simulacao de drift (sem Kaggle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift.simulate import (
    DataDriftShocks,
    apply_concept_drift,
    apply_data_drift,
    simulate_production_slices,
)
from src.validation.contract import MAX_INCOME_TOTAL
from src.validation.gate import APPROVED, run_contract
from tests.conftest import make_valid_df


def test_apply_data_drift_changes_important_columns(valid_df: pd.DataFrame) -> None:
    drifted = apply_data_drift(valid_df)
    assert not np.allclose(
        valid_df["annuity_amount"].to_numpy(), drifted["annuity_amount"].to_numpy()
    )
    # ext_source_3: valores nao-nulos devem cair (shift negativo), clipados em [0,1]
    mask = valid_df["ext_source_3"].notna()
    assert (
        drifted.loc[mask, "ext_source_3"] <= valid_df.loc[mask, "ext_source_3"]
    ).all()
    assert drifted.loc[mask, "ext_source_3"].between(0, 1).all()
    assert (drifted["income_total"] >= valid_df["income_total"]).all()


def test_apply_data_drift_stays_within_contract(valid_df: pd.DataFrame) -> None:
    drifted = apply_data_drift(valid_df)
    _, result, _ = run_contract(
        drifted, labeled=True, batch_name="t_data", stage="test"
    )
    assert result.status == APPROVED
    assert (drifted["annuity_amount"] <= drifted["credit_amount"]).all()
    assert (drifted["income_total"] <= MAX_INCOME_TOTAL).all()


def test_apply_data_drift_clips_extreme_shocks(valid_df: pd.DataFrame) -> None:
    shocks = DataDriftShocks(
        annuity_multiplier=100.0,
        ext_source_3_shift=-5.0,
        income_multiplier=1000.0,
    )
    drifted = apply_data_drift(valid_df, shocks=shocks)
    _, result, _ = run_contract(
        drifted, labeled=True, batch_name="t_clip", stage="test"
    )
    assert result.status == APPROVED
    mask = drifted["ext_source_3"].notna()
    assert (drifted.loc[mask, "ext_source_3"] >= 0).all()
    assert (drifted.loc[mask, "ext_source_3"] <= 1).all()
    assert (drifted["annuity_amount"] <= drifted["credit_amount"]).all()
    assert (drifted["income_total"] <= MAX_INCOME_TOTAL).all()


def test_apply_concept_drift_raises_default_rate(valid_df: pd.DataFrame) -> None:
    # Garante elegiveis: score alto e target=0
    df = valid_df.copy()
    df["ext_source_3"] = 0.8
    df["target"] = 0
    before = float(df["target"].mean())
    after_df = apply_concept_drift(df, rate=0.5, score_min=0.5, seed=42)
    after = float(after_df["target"].mean())
    assert after > before
    # Features inalteradas
    pd.testing.assert_series_equal(df["annuity_amount"], after_df["annuity_amount"])


def test_apply_concept_drift_requires_target() -> None:
    df = make_valid_df(labeled=False)
    with pytest.raises(ValueError, match="target"):
        apply_concept_drift(df)


def test_simulate_production_slices_writes_files(
    valid_df: pd.DataFrame, tmp_path
) -> None:
    t1 = tmp_path / "t1.parquet"
    t2 = tmp_path / "t2.parquet"
    paths = simulate_production_slices(valid_df, t1_path=t1, t2_path=t2, seed=42)
    assert paths["t1"].exists()
    assert paths["t2"].exists()
    t1_df = pd.read_parquet(t1)
    t2_df = pd.read_parquet(t2)
    _, r1, _ = run_contract(t1_df, labeled=True, batch_name="t1", stage="test")
    _, r2, _ = run_contract(t2_df, labeled=True, batch_name="t2", stage="test")
    assert r1.status == APPROVED
    assert r2.status == APPROVED
    assert float(t2_df["target"].mean()) >= float(t1_df["target"].mean())
