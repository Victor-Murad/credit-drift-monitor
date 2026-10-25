"""Treino do baseline: pipeline funcional, reprodutivel e atras do gate."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.features import add_ratio_features, model_input
from src.models.metrics import evaluate, ks_statistic
from src.models.predict import score_batch
from src.models.train import train_models
from src.validation.gate import DataContractViolation
from tests.conftest import make_valid_df

FAST_XGB = {"n_estimators": 20, "n_jobs": 1}


@pytest.fixture(scope="module")
def trained():
    reference = make_valid_df(n=1500, seed=3)
    return train_models(reference, seed=42, xgb_params=FAST_XGB, importance_sample=300)


def test_ratio_features_are_finite(valid_df):
    X = add_ratio_features(model_input(valid_df))
    ratios = X[["credit_to_income", "annuity_to_income", "credit_to_annuity"]]
    assert np.isfinite(ratios.to_numpy()).all()


def test_ks_statistic_bounds():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


def test_evaluate_business_metrics():
    y = np.array([0, 0, 0, 1])
    m = evaluate(y, np.array([0.1, 0.2, 0.7, 0.9]), threshold=0.5)
    assert m["approval_rate"] == 0.5
    assert m["default_rate_approved"] == 0.0


def test_train_models_learns_signal(trained):
    bundle, metadata = trained
    assert set(metadata["models"]) == {"xgboost", "logistic_regression"}
    assert metadata["models"]["xgboost"]["test_metrics"]["roc_auc"] > 0.6
    assert 0 < bundle["threshold"] < 1
    top_features = [f["feature"] for f in metadata["raw_feature_importance"][:3]]
    assert "ext_source_2" in top_features  # sinal plantado no sintetico
    assert "gender" not in bundle["input_columns"]  # so auditoria


def test_training_is_reproducible(trained):
    reference = make_valid_df(n=1500, seed=3)
    bundle_b, _ = train_models(
        reference, seed=42, xgb_params=FAST_XGB, importance_sample=300
    )
    batch = model_input(make_valid_df(n=50, seed=9))
    proba_a = trained[0]["pipeline"].predict_proba(batch)[:, 1]
    proba_b = bundle_b["pipeline"].predict_proba(batch)[:, 1]
    assert np.allclose(proba_a, proba_b)


def test_score_batch_accepts_valid_and_blocks_invalid(trained):
    bundle, _ = trained
    batch = make_valid_df(n=40, seed=11, labeled=False)
    no_artifacts = {"report_dir": None, "quarantine_dir": None}

    scored = score_batch(batch, bundle, batch_name="pytest_ok", **no_artifacts)
    assert len(scored) == 40
    assert scored["default_probability"].between(0, 1).all()
    assert set(scored["decision"]) <= {"aprovado", "negado"}

    bad = batch.copy()
    bad.loc[0, "age_years"] = 12.0
    with pytest.raises(DataContractViolation):
        score_batch(bad, bundle, batch_name="pytest_bad", **no_artifacts)
