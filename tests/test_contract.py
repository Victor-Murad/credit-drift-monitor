"""Cada regra rigida do contrato precisa REPROVAR um lote que a viola."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors as pa_errors
import pytest

from src.validation.contract import (
    CreditApplicationSchema,
    LabeledCreditApplicationSchema,
    contract_columns,
)


def _failures(df: pd.DataFrame, schema=LabeledCreditApplicationSchema) -> pd.DataFrame:
    """Valida esperando falha e devolve os failure_cases do Pandera."""
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    fc = exc.value.failure_cases.copy()
    fc["column"] = fc["column"].astype(str)
    fc["check"] = fc["check"].astype(str)
    fc["failure_case"] = fc["failure_case"].astype(str)
    return fc


def _has(
    fc: pd.DataFrame, *, check: str, column: str | None = None, case: str | None = None
) -> bool:
    mask = fc["check"].str.startswith(check)
    if column is not None:
        mask &= fc["column"] == column
    if case is not None:
        mask &= fc["failure_case"] == case
    return bool(mask.any())


def _set(df: pd.DataFrame, column: str, value, rows=(0,)) -> pd.DataFrame:
    df = df.copy()
    if isinstance(value, str) or value is None:
        df[column] = df[column].astype(object)
    for r in rows:
        df.loc[r, column] = value
    return df


def test_valid_batch_passes(valid_df):
    validated = LabeledCreditApplicationSchema.validate(valid_df, lazy=True)
    assert len(validated) == len(valid_df)


def test_unlabeled_batch_passes_inference_contract(valid_unlabeled_df):
    CreditApplicationSchema.validate(valid_unlabeled_df, lazy=True)


@pytest.mark.parametrize(
    ("column", "value", "expected_check"),
    [
        ("income_total", np.nan, "not_nullable"),  # R2 renda nula
        ("contract_type", None, "not_nullable"),  # R2 categoria nula
        ("age_years", 17.9, "greater_than_or_equal_to"),  # R3 menor de idade
        ("age_years", 150.0, "less_than_or_equal_to"),  # R3 idade impossivel
        ("income_total", 0.0, "greater_than"),  # R4 renda zero
        ("income_total", 117_000_000.0, "less_than_or_equal_to"),  # R4 limite
        ("credit_amount", -1000.0, "greater_than"),  # R5 credito negativo
        ("contract_type", "Crypto loans", "isin"),  # R6 categoria desconhecida
        ("gender", "XNA", "isin"),  # R6 codigo de desconhecido
        ("ext_source_2", 1.5, "less_than_or_equal_to"),  # R7 score fora de [0,1]
    ],
)
def test_rule_violation_is_detected(valid_df, column, value, expected_check):
    fc = _failures(_set(valid_df, column, value))
    assert _has(fc, check=expected_check, column=column), fc[["column", "check"]]


def test_duplicate_client_id_is_detected(valid_df):  # R1
    bad = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
    fc = _failures(bad)
    assert _has(fc, check="field_uniqueness", column="client_id"), fc


def test_annuity_greater_than_credit_is_detected(valid_df):  # R8
    bad = valid_df.copy()
    bad.loc[0, "annuity_amount"] = bad.loc[0, "credit_amount"] * 2
    assert _has(_failures(bad), check="annuity_not_greater_than_credit")


def test_missing_employment_for_worker_is_detected(valid_df):  # R9
    bad = valid_df.copy()
    worker = bad.index[bad["income_type"] == "Working"][0]
    bad.loc[worker, "years_employed"] = np.nan
    assert _has(_failures(bad), check="employment_missing_only_if_not_employed")


def test_unexpected_column_is_rejected(valid_df):  # R10 (strict=True)
    fc = _failures(valid_df.assign(cpf="000.000.000-00"))
    assert _has(fc, check="column_in_schema", case="cpf"), fc


def test_missing_column_is_rejected(valid_df):  # R10
    fc = _failures(valid_df.drop(columns=["income_total"]))
    assert _has(fc, check="column_in_dataframe", case="income_total"), fc


def test_training_contract_requires_target(valid_unlabeled_df):
    fc = _failures(valid_unlabeled_df)
    assert _has(fc, check="column_in_dataframe", case="target"), fc


def test_wrong_type_is_rejected(valid_df):  # R11
    bad = _set(valid_df, "income_total", "abc")
    fc = _failures(bad)
    assert _has(fc, check="coerce_dtype", column="income_total", case="abc"), fc


def test_lazy_validation_reports_all_errors_at_once(valid_df):
    bad = _set(valid_df, "income_total", np.nan)
    bad = _set(bad, "age_years", 10.0, rows=(1,))
    bad = _set(bad, "contract_type", "Crypto loans", rows=(2,))
    fc = _failures(bad)
    assert {"income_total", "age_years", "contract_type"} <= set(fc["column"])


def test_contract_columns_order():
    cols = contract_columns(labeled=True)
    assert cols[0] == "client_id"
    assert cols[-1] == "target"
    assert "target" not in contract_columns(labeled=False)
