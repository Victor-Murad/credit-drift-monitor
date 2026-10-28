"""Degradacao de performance do modelo -- sinal confiavel de Concept Drift.

O teste estatistico sobre `target` (detect.py) so enxerga mudanca na
PREVALENCIA do rotulo (label/prior shift). Concept drift de verdade e uma
mudanca em P(y|X): o modelo passa a errar mais. Quando ha rotulos (feedback
atrasado), a queda de ROC-AUC frente ao baseline mede isso diretamente.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.config import AUC_DROP_ALERT


@dataclass(frozen=True)
class PerformanceDrift:
    metric: str
    baseline: float
    current: float
    drop: float
    threshold: float
    degraded: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_auc_drop(
    baseline_auc: float, current_auc: float, *, threshold: float = AUC_DROP_ALERT
) -> PerformanceDrift:
    """Compara o ROC-AUC corrente com o baseline; degradado se queda >= limiar."""
    drop = round(baseline_auc - current_auc, 4)
    return PerformanceDrift(
        metric="roc_auc",
        baseline=baseline_auc,
        current=current_auc,
        drop=drop,
        threshold=threshold,
        degraded=drop >= threshold,
    )
