"""Testes da regra pura de promoção champion/challenger (spec da Etapa 3, §4.7)."""

import math

import pytest

from src.config import PROMOTION_MAX_REGRESSION, PROMOTION_MIN_GAIN
from src.monitoring.promotion import PromotionDecision, decide_promotion


def _decide(
    *,
    contract_ok: bool = True,
    integrity_ok: bool = True,
    champion_holdout_auc: float = 0.60,
    challenger_holdout_auc: float = 0.63,
    champion_reference_auc: float = 0.75,
    challenger_reference_auc: float = 0.745,
    **limiares: float,
) -> PromotionDecision:
    """Cenário elegível por padrão (ganho 0,03; regressão 0,005); cada teste muda um eixo."""
    return decide_promotion(
        contract_ok=contract_ok,
        integrity_ok=integrity_ok,
        champion_holdout_auc=champion_holdout_auc,
        challenger_holdout_auc=challenger_holdout_auc,
        champion_reference_auc=champion_reference_auc,
        challenger_reference_auc=challenger_reference_auc,
        **limiares,
    )


def test_limiares_padrao_vem_do_config():
    assert PROMOTION_MIN_GAIN == 0.02
    assert PROMOTION_MAX_REGRESSION == 0.02


def test_elegivel():
    decision = _decide()

    assert decision.eligible is True
    assert decision.reasons == ()
    assert decision.gain_on_holdout == pytest.approx(0.03)
    assert decision.regression_on_reference == pytest.approx(0.005)


def test_ganho_insuficiente():
    abaixo = _decide(champion_holdout_auc=0.60, challenger_holdout_auc=0.619)
    no_limiar = _decide(champion_holdout_auc=0.60, challenger_holdout_auc=0.62)

    assert abaixo.eligible is False
    assert abaixo.reasons == ("gain_below_min",)
    assert abaixo.gain_on_holdout == pytest.approx(0.019)
    assert no_limiar.eligible is True
    assert no_limiar.reasons == ()


@pytest.mark.parametrize(
    ("champion", "challenger"),
    [
        (0.68, 0.70),  # 0.70 - 0.68 = 0.01999999999999991 em float
        (0.51, 0.53),  # 0.53 - 0.51 = 0.020000000000000018
        (0.73, 0.75),  # 0.75 - 0.73 = 0.020000000000000018
        (0.1, 0.12),  # 0.12 - 0.1 = 0.01999999999999999
    ],
)
def test_ganho_exatamente_no_limiar_com_erro_de_float_e_elegivel(champion, challenger):
    decision = _decide(champion_holdout_auc=champion, challenger_holdout_auc=challenger)

    assert decision.eligible is True
    assert decision.reasons == ()
    assert decision.gain_on_holdout == 0.02


def test_ganho_logo_abaixo_do_limiar_continua_inelegivel():
    # A tolerância só absorve erro de representação (casas além da 9ª), não ganho real.
    decision = _decide(champion_holdout_auc=0.60, challenger_holdout_auc=0.619999)

    assert decision.eligible is False
    assert decision.reasons == ("gain_below_min",)


def test_ganho_negativo_e_inelegivel():
    decision = _decide(champion_holdout_auc=0.62, challenger_holdout_auc=0.60)

    assert decision.reasons == ("gain_below_min",)
    assert decision.gain_on_holdout == pytest.approx(-0.02)


def test_regressao_na_referencia():
    acima = _decide(champion_reference_auc=0.75, challenger_reference_auc=0.729)
    no_limiar = _decide(champion_reference_auc=0.75, challenger_reference_auc=0.73)

    assert acima.eligible is False
    assert acima.reasons == ("regression_above_max",)
    assert acima.regression_on_reference == pytest.approx(0.021)
    assert no_limiar.eligible is True
    assert no_limiar.reasons == ()


@pytest.mark.parametrize(
    ("champion", "challenger"),
    [
        (0.70, 0.68),  # 0.70 - 0.68 = 0.01999999999999991
        (0.53, 0.51),  # 0.53 - 0.51 = 0.020000000000000018 (passaria do limiar)
        (0.75, 0.73),  # 0.75 - 0.73 = 0.020000000000000018 (passaria do limiar)
    ],
)
def test_regressao_exatamente_no_limiar_com_erro_de_float_e_elegivel(
    champion, challenger
):
    decision = _decide(
        champion_reference_auc=champion, challenger_reference_auc=challenger
    )

    assert decision.eligible is True
    assert decision.reasons == ()
    assert decision.regression_on_reference == 0.02


def test_melhora_na_referencia_nao_e_regressao():
    decision = _decide(champion_reference_auc=0.74, challenger_reference_auc=0.76)

    assert decision.eligible is True
    assert decision.regression_on_reference == pytest.approx(-0.02)


def test_contrato_falhando():
    decision = _decide(contract_ok=False)

    assert decision.eligible is False
    assert decision.reasons == ("contract_failed",)


def test_integridade_falhando():
    decision = _decide(integrity_ok=False)

    assert decision.eligible is False
    assert decision.reasons == ("integrity_failed",)


def test_varios_motivos_acumulam():
    decision = _decide(
        contract_ok=False,
        integrity_ok=False,
        champion_holdout_auc=0.60,
        challenger_holdout_auc=0.61,
        champion_reference_auc=0.75,
        challenger_reference_auc=0.70,
    )

    assert decision.eligible is False
    # Ordem estável: a mesma das regras da §4.7.
    assert decision.reasons == (
        "contract_failed",
        "integrity_failed",
        "gain_below_min",
        "regression_above_max",
    )
    # Os números são calculados mesmo quando o contrato ou a integridade falham.
    assert decision.gain_on_holdout == pytest.approx(0.01)
    assert decision.regression_on_reference == pytest.approx(0.05)


@pytest.mark.parametrize("valor", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("campo", ["champion_holdout_auc", "challenger_holdout_auc"])
def test_auc_nao_finita_no_holdout_e_inelegivel(campo, valor):
    decision = _decide(**{campo: valor})

    assert decision.eligible is False
    assert decision.reasons == ("gain_below_min",)
    assert math.isnan(decision.gain_on_holdout)


@pytest.mark.parametrize("valor", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "campo", ["champion_reference_auc", "challenger_reference_auc"]
)
def test_auc_nao_finita_na_referencia_e_inelegivel(campo, valor):
    decision = _decide(**{campo: valor})

    assert decision.eligible is False
    assert decision.reasons == ("regression_above_max",)
    assert math.isnan(decision.regression_on_reference)


def test_limiar_nao_finito_falha_fechado():
    decision = _decide(min_gain=math.nan, max_regression=math.nan)

    assert decision.eligible is False
    assert decision.reasons == ("gain_below_min", "regression_above_max")


def test_limiares_explicitos_substituem_os_padroes():
    # Ganho 0,03 e regressão 0,005: reprova com limiares mais exigentes.
    decision = _decide(min_gain=0.05, max_regression=0.001)

    assert decision.reasons == ("gain_below_min", "regression_above_max")


def test_decisao_e_imutavel():
    decision = _decide()

    with pytest.raises(AttributeError):
        decision.eligible = False  # type: ignore[misc]
