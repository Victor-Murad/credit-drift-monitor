"""Regra pura de nível de alerta (spec da Etapa 3, §4.2).

Sem I/O e sem dependência de MLflow: recebe os sinais já calculados e devolve o nível,
os motivos estáveis e a ação do runbook. Precedência: BLOQUEADO > CRÍTICO > ATENÇÃO > OK.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from src.config import AUC_DROP_ALERT, PSI_MODERATE


class AlertLevel(StrEnum):
    """Nível de alerta. Os valores são ASCII (tags e métricas); `label` é para exibição."""

    OK = "OK"
    ATENCAO = "ATENCAO"
    CRITICO = "CRITICO"
    BLOQUEADO = "BLOQUEADO"

    @property
    def label(self) -> str:
        return _LABELS.get(self, self.value)


_LABELS = {AlertLevel.ATENCAO: "ATENÇÃO", AlertLevel.CRITICO: "CRÍTICO"}

# Ação associada de cada nível (coluna "Ação associada (runbook)" da §4.2 da spec).
ACTIONS: dict[AlertLevel, str] = {
    AlertLevel.BLOQUEADO: (
        "Inspecionar o relatório do contrato e a quarentena; corrigir a fonte; não pontuar."
    ),
    AlertLevel.CRITICO: (
        "Tratar como concept drift: investigar e avaliar retreino com validação humana."
    ),
    AlertLevel.ATENCAO: (
        "Investigar a coluna; um data drift sozinho não prova piora do modelo."
    ),
    AlertLevel.OK: "Nenhuma ação necessária.",
}

REASON_CONTRACT_BLOCKED = "contract_blocked"
REASON_AUC_DROP = f"auc_drop>={AUC_DROP_ALERT:.2f}"
REASON_SCORE_PSI = f"score_psi>={PSI_MODERATE:.2f}"
REASON_DRIFT_PREFIX = "drift:"


@dataclass(frozen=True)
class AlertDecision:
    """Resultado da regra: nível, motivos estáveis, se a performance foi avaliada e a ação."""

    level: AlertLevel
    reasons: tuple[str, ...]
    performance_evaluated: bool
    action: str


def decide_alert_level(
    *,
    blocked: bool,
    drifted_columns: Sequence[str],
    score_psi: float | None,
    auc_drop: float | None,
    performance_skip_reason: str | None = None,
) -> AlertDecision:
    """Decide o nível de alerta de um lote.

    `drifted_columns` não inclui `target` (o chamador o remove). `auc_drop=None` significa
    "não avaliado" (lote sem rótulo ou `single_class`): o nível máximo passa a ser ATENÇÃO.
    """
    if blocked:
        return _decision(AlertLevel.BLOQUEADO, (REASON_CONTRACT_BLOCKED,), False)

    performance_evaluated = auc_drop is not None
    reasons: list[str] = []
    if performance_skip_reason and not performance_evaluated:
        reasons.append(performance_skip_reason)

    critical = auc_drop is not None and auc_drop >= AUC_DROP_ALERT
    if critical:
        reasons.append(REASON_AUC_DROP)
    reasons.extend(f"{REASON_DRIFT_PREFIX}{column}" for column in drifted_columns)
    psi_alert = score_psi is not None and score_psi >= PSI_MODERATE
    if psi_alert:
        reasons.append(REASON_SCORE_PSI)

    if critical:
        level = AlertLevel.CRITICO
    elif drifted_columns or psi_alert:
        level = AlertLevel.ATENCAO
    else:
        level = AlertLevel.OK
    return _decision(level, tuple(reasons), performance_evaluated)


def _decision(
    level: AlertLevel, reasons: tuple[str, ...], performance_evaluated: bool
) -> AlertDecision:
    return AlertDecision(level, reasons, performance_evaluated, ACTIONS[level])
