"""PSI (tamanho do efeito) e degradacao de performance (concept drift)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import PSI_MODERATE, PSI_SIGNIFICANT
from src.drift.detect import run_drift_detection
from src.drift.performance import assess_auc_drop
from src.drift.psi import population_stability_index, psi_level, score_psi
from src.models.metrics import score_distribution
from tests.conftest import make_valid_df


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(size=5000))
    cur = pd.Series(rng.normal(size=5000))
    assert population_stability_index(ref, cur) < 0.02


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(size=5000))
    small = population_stability_index(ref, ref + 0.1)
    large = population_stability_index(ref, ref + 1.0)
    assert small < large
    assert psi_level(large) == "significant"


def test_psi_counts_missing_rate_change():
    ref = pd.Series(np.linspace(0, 1, 1000))
    cur = ref.copy()
    cur.iloc[:300] = np.nan
    assert population_stability_index(ref, cur) > 0.1


def test_psi_categorical_and_binary_target():
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    cur = pd.Series(["a"] * 40 + ["b"] * 40 + ["c"] * 20)
    assert psi_level(population_stability_index(ref, cur)) == "significant"

    target_ref = pd.Series([0] * 92 + [1] * 8)
    target_cur = pd.Series([0] * 92 + [1] * 8)
    assert population_stability_index(target_ref, target_cur) == pytest.approx(0.0)


def test_psi_requires_reference_values():
    with pytest.raises(ValueError, match="PSI indefinido"):
        population_stability_index(
            pd.Series([np.nan, np.nan], dtype=float), pd.Series([1.0, 2.0])
        )


@pytest.mark.parametrize(
    ("psi", "level"),
    [(0.0, "stable"), (0.1, "moderate"), (0.24, "moderate"), (0.25, "significant")],
)
def test_psi_level_thresholds(psi, level):
    assert psi_level(psi) == level


def test_auc_drop_flags_degradation_only_above_threshold():
    mild = assess_auc_drop(0.753, 0.722)
    severe = assess_auc_drop(0.753, 0.573)
    assert mild.drop == pytest.approx(0.031)
    assert mild.degraded is False
    assert severe.degraded is True
    assert severe.to_dict()["metric"] == "roc_auc"


def test_score_distribution_formato():
    scores = np.random.default_rng(0).uniform(size=2000)
    dist = score_distribution(scores)
    assert len(dist["edges"]) == 9
    assert len(dist["shares"]) == 10
    assert sum(dist["shares"]) == pytest.approx(1)
    assert all(a < b for a, b in zip(dist["edges"], dist["edges"][1:], strict=False))


def test_score_distribution_scores_constantes():
    dist = score_distribution(np.full(100, 0.3))
    assert len(dist["shares"]) >= 1
    assert len(dist["shares"]) == len(dist["edges"]) + 1
    assert sum(dist["shares"]) == pytest.approx(1)


def test_score_psi_identico_e_zero():
    scores = np.random.default_rng(1).uniform(size=3000)
    assert score_psi(score_distribution(scores), scores) < 1e-9


def test_score_psi_deslocado_alto():
    scores = np.random.default_rng(2).uniform(size=3000)
    shifted = np.clip(scores + 0.3, 0, 1)
    assert score_psi(score_distribution(scores), shifted) > PSI_SIGNIFICANT


def test_score_psi_constante_e_finito():
    dist = score_distribution(np.full(100, 0.3))
    psi = score_psi(dist, np.full(100, 0.8))
    assert math.isfinite(psi)


def test_detection_reports_psi_and_min_psi_filters_noise():
    reference = make_valid_df(n=20_000, seed=4)
    current = make_valid_df(n=20_000, seed=5)
    # Deslocamento de 3%: com n=20 mil o KS acusa, mas o efeito (PSI) e minimo.
    current["annuity_amount"] = current["annuity_amount"] * 1.03
    kwargs = {
        "batch_name": "noise",
        "columns": ["annuity_amount"],
        "include_target": False,
        "report_dir": None,
    }

    default = run_drift_detection(reference, current, **kwargs)
    filtered = run_drift_detection(reference, current, min_psi=PSI_MODERATE, **kwargs)

    col = default.column_results[0]
    assert col.drifted is True
    assert col.psi is not None and col.psi < PSI_MODERATE
    assert col.psi_level == "stable"
    assert filtered.column_results[0].drifted is False
    assert filtered.n_drifted_columns == 0
    assert filtered.dataset_drift is False
