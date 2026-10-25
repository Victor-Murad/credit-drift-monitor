"""Testes da deteccao de drift (Evidently em amostra pequena)."""

from __future__ import annotations

from functools import partial

from src.drift import detect
from src.drift.detect import _default_columns, run_drift_detection
from src.drift.simulate import apply_concept_drift, apply_data_drift
from tests.conftest import make_valid_df


def test_detect_marks_data_drift_on_shocked_columns(tmp_path) -> None:
    reference = make_valid_df(n=400, seed=1)
    current = apply_data_drift(reference)

    result = run_drift_detection(
        reference,
        current,
        batch_name="synth_t1",
        columns=["annuity_amount", "ext_source_3", "income_total", "credit_amount"],
        include_target=False,
        report_dir=tmp_path,
    )

    drifted = {c.column for c in result.column_results if c.drifted}
    assert "annuity_amount" in drifted
    assert "ext_source_3" in drifted
    assert result.dataset_drift is True
    assert (tmp_path / "synth_t1_drift_report.html").exists()
    assert (tmp_path / "synth_t1_drift_report.json").exists()


def test_detect_marks_concept_drift_on_target(tmp_path) -> None:
    reference = make_valid_df(n=500, seed=2)
    # Score alto em todos para haver elegiveis ao flip.
    reference = reference.copy()
    reference["ext_source_3"] = 0.8
    reference["target"] = 0
    current = apply_concept_drift(reference, rate=0.6, score_min=0.5, seed=42)

    result = run_drift_detection(
        reference,
        current,
        batch_name="synth_concept",
        columns=["annuity_amount", "target"],
        include_target=True,
        report_dir=tmp_path,
    )

    target_row = next(c for c in result.column_results if c.column == "target")
    assert target_row.drifted is True
    assert result.concept_drift is True


def test_stable_split_has_little_or_no_drift(tmp_path) -> None:
    df = make_valid_df(n=600, seed=3)
    half = len(df) // 2
    reference = df.iloc[:half].reset_index(drop=True)
    current = df.iloc[half:].reset_index(drop=True)

    result = run_drift_detection(
        reference,
        current,
        batch_name="synth_stable",
        columns=["annuity_amount", "income_total", "credit_amount"],
        include_target=False,
        drift_share=0.5,
        report_dir=tmp_path,
    )
    # Split aleatorio da mesma geracao: tipicamente sem dataset drift.
    # Aceitamos dataset_drift False OU poucas colunas (nao todas).
    assert result.n_drifted_columns < len(result.columns_checked)


def test_default_columns_prefers_monitored_features_and_optional_target():
    df = make_valid_df(n=50, seed=5)
    with_target = _default_columns(df, include_target=True)
    without_target = _default_columns(df, include_target=False)
    assert "annuity_amount" in with_target
    assert "target" in with_target
    assert "target" not in without_target


def test_cli_reports_drift_and_missing_files(tmp_path, monkeypatch, capsys):
    reference = make_valid_df(n=400, seed=1)
    ref_path, cur_path = tmp_path / "ref.parquet", tmp_path / "cur.parquet"
    reference.to_parquet(ref_path)
    apply_data_drift(reference).to_parquet(cur_path)
    monkeypatch.setattr(
        detect,
        "run_drift_detection",
        partial(run_drift_detection, report_dir=tmp_path / "reports"),
    )

    args = ["--reference", str(ref_path), "--current", str(cur_path)]
    assert detect.main([*args, "--min-psi", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "psi=" in out and "Relatorios:" in out

    assert detect.main(["--reference", str(tmp_path / "nope.parquet")]) == 1
