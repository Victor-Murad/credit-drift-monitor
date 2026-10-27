"""Regra pura de promoção champion/challenger (spec da Etapa 3, §4.7).

Sem I/O e sem dependência de MLflow: recebe as ROC-AUC já calculadas e os resultados do
contrato e da integridade e devolve a elegibilidade, os motivos estáveis e os números que
decidiram. Quem chama (retreino e promoção) registra a decisão; esta função nunca move alias.

Fronteira: os limiares são inclusivos (ganho exatamente `min_gain` e regressão exatamente
`max_regression` são elegíveis). Como `0.70 - 0.68 == 0.01999999999999991` e
`0.75 - 0.73 == 0.020000000000000018` em ponto flutuante, as diferenças e os limiares são
arredondados para `_COMPARISON_DECIMALS` casas antes de comparar. 1e-9 fica muito acima do
erro de representação de um float (~1e-16) e abaixo da resolução da AUC de um holdout real
(1/(positivos x negativos), ~4,5e-9 para o holdout do t2), então só absorve ruído numérico.

AUC não finita (NaN/inf) em qualquer lado deixa a diferença indefinida (NaN): o eixo
correspondente reprova com o seu motivo estável, sem levantar exceção (falha fechada).
"""

import math
from dataclasses import dataclass

from src.config import PROMOTION_MAX_REGRESSION, PROMOTION_MIN_GAIN

REASON_CONTRACT_FAILED = "contract_failed"
REASON_INTEGRITY_FAILED = "integrity_failed"
REASON_GAIN_BELOW_MIN = "gain_below_min"
REASON_REGRESSION_ABOVE_MAX = "regression_above_max"

_COMPARISON_DECIMALS = 9


@dataclass(frozen=True)
class PromotionDecision:
    """Resultado da regra: elegibilidade, motivos estáveis e as diferenças comparadas.

    `gain_on_holdout` = AUC do challenger - AUC do champion no holdout do lote `CRÍTICO`.
    `regression_on_reference` = AUC do champion - AUC do challenger no teste da referência
    (positivo é queda). Ambos já arredondados como na comparação; NaN quando indefinidos.
    """

    eligible: bool
    reasons: tuple[str, ...]
    gain_on_holdout: float
    regression_on_reference: float


def decide_promotion(
    *,
    contract_ok: bool,
    integrity_ok: bool,
    champion_holdout_auc: float,
    challenger_holdout_auc: float,
    champion_reference_auc: float,
    challenger_reference_auc: float,
    min_gain: float = PROMOTION_MIN_GAIN,
    max_regression: float = PROMOTION_MAX_REGRESSION,
) -> PromotionDecision:
    """Decide se o challenger é elegível a `champion` (as três regras da §4.7).

    Os motivos se acumulam na ordem das regras (contrato, integridade, ganho, regressão), e
    os números são calculados mesmo quando o contrato ou a integridade falham.
    """
    gain = _difference(challenger_holdout_auc, champion_holdout_auc)
    regression = _difference(champion_reference_auc, challenger_reference_auc)

    reasons: list[str] = []
    if not contract_ok:
        reasons.append(REASON_CONTRACT_FAILED)
    if not integrity_ok:
        reasons.append(REASON_INTEGRITY_FAILED)
    # Escrito como "não passa" para que NaN (na diferença ou no limiar) reprove.
    if not gain >= _rounded(min_gain):
        reasons.append(REASON_GAIN_BELOW_MIN)
    if not regression <= _rounded(max_regression):
        reasons.append(REASON_REGRESSION_ABOVE_MAX)
    return PromotionDecision(not reasons, tuple(reasons), gain, regression)


def _difference(minuend: float, subtrahend: float) -> float:
    if not (math.isfinite(minuend) and math.isfinite(subtrahend)):
        return math.nan
    return _rounded(minuend - subtrahend)


def _rounded(value: float) -> float:
    # round() devolve NaN/inf inalterados, então limiares não finitos seguem reprovando.
    return round(value, _COMPARISON_DECIMALS)
