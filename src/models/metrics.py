"""Metricas de avaliacao do modelo de credito.

Fora de `train.py` para que o monitoramento (drift/performance) reutilize as
mesmas metricas sem depender do modulo de treino.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS de credito: maxima separacao entre as distribuicoes de bons e maus."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def ks_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Ponto de corte que maximiza o KS (estatistica J de Youden)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    approved = y_pred == 0
    auc = roc_auc_score(y_true, y_score)
    return {
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        # Visao de negocio: quanto aprovamos e qual a inadimplencia da
        # carteira aprovada (vs. taxa base).
        "approval_rate": round(float(approved.mean()), 4),
        "default_rate_approved": round(float(y_true[approved].mean()), 4)
        if approved.any()
        else None,
        "default_rate_base": round(float(y_true.mean()), 4),
    }
