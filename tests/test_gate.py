"""O gate precisa bloquear o lote inteiro, gerar evidencias e registrar log."""

from __future__ import annotations

import json
import logging

import pytest

from src.config import ID_COLUMN
from src.validation.gate import (
    APPROVED,
    BLOCKED,
    DataContractViolation,
    enforce_contract,
    run_contract,
)
from src.validation.invalid_batch import DEFECTS, make_invalid_batch


@pytest.fixture
def quiet_logger(tmp_path):
    logger = logging.getLogger("test_gate")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def test_approved_batch_returns_validated_df(valid_df, tmp_path, quiet_logger):
    out = enforce_contract(
        valid_df,
        batch_name="ok",
        stage="test",
        labeled=True,
        report_dir=tmp_path / "reports",
        quarantine_dir=tmp_path / "quarantine",
        logger=quiet_logger,
    )
    assert len(out) == len(valid_df)
    report = json.loads((tmp_path / "reports" / "ok_report.json").read_text("utf-8"))
    assert report["status"] == APPROVED
    assert not (tmp_path / "quarantine").exists()


def test_invalid_batch_is_blocked_quarantined_and_reported(
    valid_unlabeled_df, tmp_path, quiet_logger
):
    dirty, _ = make_invalid_batch(valid_unlabeled_df)
    with pytest.raises(DataContractViolation) as exc:
        enforce_contract(
            dirty,
            batch_name="dirty",
            stage="test",
            labeled=False,
            report_dir=tmp_path / "reports",
            quarantine_dir=tmp_path / "quarantine",
            logger=quiet_logger,
        )
    result = exc.value.result
    assert result.status == BLOCKED
    assert result.n_failure_cases > 0
    assert len(list((tmp_path / "quarantine").glob("dirty_*.csv"))) == 1
    html = (tmp_path / "reports" / "dirty_report.html").read_text("utf-8")
    assert "BLOQUEADO" in html


def test_every_injected_defect_maps_to_a_violated_rule(valid_unlabeled_df):
    dirty, manifest = make_invalid_batch(valid_unlabeled_df)
    _, result, _ = run_contract(dirty, labeled=False, batch_name="dirty", stage="test")
    violated = {r["rule"].split(" - ")[0] for r in result.rules_violated}
    expected = {d["expected_rule"] for d in manifest}
    assert expected <= violated, expected - violated
    assert len(manifest) == len(DEFECTS)


def test_invalid_batch_is_deterministic(valid_unlabeled_df):
    a, _ = make_invalid_batch(valid_unlabeled_df, seed=7)
    b, _ = make_invalid_batch(valid_unlabeled_df, seed=7)
    assert a.equals(b)
    assert a[ID_COLUMN].duplicated().sum() == 8


def test_run_contract_has_no_side_effects(valid_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    validated, result, failures = run_contract(
        valid_df, labeled=True, batch_name="x", stage="test"
    )
    assert validated is not None and result.approved and failures.empty
    assert list(tmp_path.iterdir()) == []


def test_row_level_rule_is_counted_once_per_row(valid_df):
    # O Pandera reporta checagens de linha uma vez por coluna; o resumo deve
    # contar a LINHA uma unica vez.
    bad = valid_df.copy()
    bad.loc[[0, 1], "annuity_amount"] = bad.loc[[0, 1], "credit_amount"] * 2
    _, result, _ = run_contract(bad, labeled=True, batch_name="r8", stage="test")
    (r8,) = [r for r in result.rules_violated if r["rule"].startswith("R8")]
    assert r8["n_failures"] == 2
    assert r8["column"] == "<dataframe>"
    assert result.n_failure_cases == 2


def test_null_annuity_is_r2_not_r8(valid_df):
    bad = valid_df.copy()
    bad.loc[0, "annuity_amount"] = float("nan")
    _, result, _ = run_contract(bad, labeled=True, batch_name="r2", stage="test")
    rules = {r["rule"].split(" - ")[0] for r in result.rules_violated}
    assert rules == {"R2"}
