"""Pipeline ponta a ponta em sandbox: bruto -> referencia -> baseline -> drift.

Roda os `main()` reais de prepare, train, validation.demo, validate, predict e
drift.demo sobre dados sinteticos. Todos os caminhos sao redirecionados para
um diretorio temporario: nada e gravado em data/, models/ ou reports/.
"""

from __future__ import annotations

import json
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import RAW_COLUMN_MAP
from src.data import prepare
from src.drift import demo as drift_demo
from src.drift import simulate
from src.models import predict, train
from src.validation import demo as contract_demo
from src.validation import validate
from src.validation.gate import enforce_contract
from tests.conftest import make_valid_df
from tests.test_prepare import _raw_from

FAST_XGB = {"n_estimators": 30, "n_jobs": 1}


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    p = SimpleNamespace(
        root=root,
        raw=root / "raw" / "application_train.csv",
        reference=root / "reference" / "reference.parquet",
        production=root / "production" / "production_base.parquet",
        t1=root / "production" / "t1.parquet",
        t2=root / "production" / "t2.parquet",
        model=root / "models" / "baseline_model.joblib",
        metadata=root / "models" / "baseline_metadata.json",
        model_reports=root / "reports" / "model",
        contract_reports=root / "reports" / "data_contract",
        drift_reports=root / "reports" / "drift",
        quarantine=root / "quarantine",
        valid_batch=root / "samples" / "valid_batch.csv",
        invalid_batch=root / "samples" / "invalid_batch.csv",
    )
    p.raw.parent.mkdir(parents=True)
    _raw_from(make_valid_df(n=4000, seed=11)).to_csv(p.raw, index=False)

    # O gate grava relatorio/quarentena em pastas default: redireciona.
    gate = partial(
        enforce_contract,
        report_dir=p.contract_reports,
        quarantine_dir=p.quarantine,
    )
    with pytest.MonkeyPatch.context() as mp:
        # prepare
        mp.setattr(prepare, "RAW_FILE", p.raw)
        mp.setattr(
            prepare,
            "load_raw",
            lambda: pd.read_csv(p.raw, usecols=list(RAW_COLUMN_MAP)),
        )
        mp.setattr(prepare, "REFERENCE_FILE", p.reference)
        mp.setattr(prepare, "PRODUCTION_BASE_FILE", p.production)
        mp.setattr(prepare, "CONTRACT_REPORTS_DIR", p.contract_reports)
        # train
        mp.setattr(train, "REFERENCE_FILE", p.reference)
        mp.setattr(train, "MODEL_FILE", p.model)
        mp.setattr(train, "MODEL_METADATA_FILE", p.metadata)
        mp.setattr(train, "MODEL_REPORTS_DIR", p.model_reports)
        mp.setattr(train, "enforce_contract", gate)
        mp.setattr(train, "XGB_PARAMS", {**train.XGB_PARAMS, **FAST_XGB})
        # predict / validate (CLIs)
        mp.setattr(predict, "load_bundle", partial(predict.load_bundle, p.model))
        mp.setattr(
            predict,
            "score_batch",
            partial(
                predict.score_batch,
                report_dir=p.contract_reports,
                quarantine_dir=p.quarantine,
            ),
        )
        mp.setattr(predict, "PRODUCTION_DIR", p.root / "production")
        mp.setattr(validate, "enforce_contract", gate)
        # demo do contrato
        mp.setattr(contract_demo, "PRODUCTION_BASE_FILE", p.production)
        mp.setattr(contract_demo, "VALID_BATCH_FILE", p.valid_batch)
        mp.setattr(contract_demo, "INVALID_BATCH_FILE", p.invalid_batch)
        mp.setattr(contract_demo, "CONTRACT_REPORTS_DIR", p.contract_reports)
        mp.setattr(contract_demo, "MODEL_FILE", p.model)
        mp.setattr(contract_demo, "load_bundle", predict.load_bundle)
        mp.setattr(contract_demo, "score_batch", predict.score_batch)
        # demo de drift
        mp.setattr(simulate, "PRODUCTION_BASE_FILE", p.production)
        mp.setattr(drift_demo, "PRODUCTION_BASE_FILE", p.production)
        mp.setattr(drift_demo, "REFERENCE_FILE", p.reference)
        mp.setattr(drift_demo, "MODEL_FILE", p.model)
        mp.setattr(drift_demo, "load_bundle", predict.load_bundle)
        mp.setattr(
            drift_demo,
            "simulate_production_slices",
            partial(simulate.simulate_production_slices, t1_path=p.t1, t2_path=p.t2),
        )
        mp.setattr(drift_demo, "DRIFT_REPORTS_DIR", p.drift_reports)
        mp.setattr(
            drift_demo,
            "run_demo",
            partial(
                drift_demo.run_demo, report_dir=p.drift_reports, evidently_sample=2000
            ),
        )

        prepare.main()
        train.main()
        p.contract_status = contract_demo.main()
        p.drift_status = drift_demo.main()
        yield p


def test_prepare_splits_reference_and_production(sandbox):
    reference = pd.read_parquet(sandbox.reference)
    production = pd.read_parquet(sandbox.production)
    assert set(reference["client_id"]).isdisjoint(production["client_id"])
    report = json.loads(
        (sandbox.contract_reports / "reference_build_report.json").read_text("utf-8")
    )
    assert report["cleaning"]["n_rows_rejected"] == 0
    assert report["splits"]["reference"]["n_rows"] == len(reference)


def test_train_persists_baseline_and_metrics(sandbox):
    assert sandbox.model.exists()
    metadata = json.loads(sandbox.metadata.read_text("utf-8"))
    assert metadata["official_model"] == "xgboost"
    assert (sandbox.model_reports / "baseline_metrics.md").exists()


def test_contract_demo_blocks_invalid_batch(sandbox):
    assert sandbox.contract_status == 0
    summary = (sandbox.contract_reports / "contract_demo_summary.md").read_text("utf-8")
    assert "**BLOQUEADO**" in summary
    assert "SUCESSO" in summary
    assert list(sandbox.quarantine.glob("invalid_batch_*.csv"))


def test_drift_demo_reports_psi_and_performance(sandbox):
    assert sandbox.drift_status == 0
    summary = json.loads(
        (sandbox.drift_reports / "drift_demo_summary.json").read_text("utf-8")
    )
    t1 = summary["drift"]["production_t1_data_drift"]
    drifted = {c["column"] for c in t1["column_results"] if c["drifted"]}
    # Em amostra sintetica pequena o PSI de cada choque varia; o de
    # ext_source_3 (-0,15 em score uniforme) e sempre forte.
    assert "ext_source_3" in drifted
    assert summary["min_psi"] == 0.10
    t0 = summary["drift"]["production_t0"]
    assert t0["n_drifted_columns"] == 0
    assert all(c["psi"] is not None for c in t1["column_results"])
    assert summary["performance"]["production_t0"]["drop"] == 0
    assert set(summary["performance"]) == set(summary["slices"])

    markdown = (sandbox.drift_reports / "drift_demo_summary.md").read_text("utf-8")
    assert "Degradação de performance" in markdown
    assert (
        sandbox.drift_reports / "production_t1_data_drift_drift_report.html"
    ).exists()


def test_validate_cli_exit_codes(sandbox, capsys):
    assert validate.main(["--input", str(sandbox.valid_batch)]) == 0
    assert validate.main(["--input", str(sandbox.invalid_batch)]) == 1
    assert "Relatorio" in capsys.readouterr().err


def test_predict_cli_scores_valid_and_refuses_invalid(sandbox, capsys):
    output = sandbox.root / "predictions.parquet"
    assert (
        predict.main(["--input", str(sandbox.valid_batch), "--output", str(output)])
        == 0
    )
    scored = pd.read_parquet(output)
    assert {"default_probability", "decision"} <= set(scored.columns)

    assert predict.main(["--input", str(sandbox.invalid_batch)]) == 1
    assert "Nenhuma predicao foi gerada" in capsys.readouterr().err


def test_read_batch_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Formato nao suportado"):
        validate.read_batch(tmp_path / "lote.xlsx")
