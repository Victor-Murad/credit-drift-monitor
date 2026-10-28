"""Testes da regra pura de nível de alerta (spec da Etapa 3, §4.2)."""

from src.monitoring.alerts import ACTIONS, AlertLevel, decide_alert_level


def test_bloqueado_vence_tudo():
    decision = decide_alert_level(
        blocked=True,
        drifted_columns=["annuity_amount", "income"],
        score_psi=0.40,
        auc_drop=0.30,
    )

    assert decision.level is AlertLevel.BLOQUEADO
    assert decision.reasons == ("contract_blocked",)
    assert decision.performance_evaluated is False


def test_critico_por_queda_de_auc():
    decision = decide_alert_level(
        blocked=False, drifted_columns=[], score_psi=0.02, auc_drop=0.177
    )

    assert decision.level is AlertLevel.CRITICO
    assert decision.performance_evaluated is True
    assert "auc_drop>=0.05" in decision.reasons


def test_critico_no_limiar_exato_e_nao_critico_logo_abaixo():
    no_limiar = decide_alert_level(
        blocked=False, drifted_columns=[], score_psi=0.0, auc_drop=0.05
    )
    abaixo = decide_alert_level(
        blocked=False, drifted_columns=[], score_psi=0.0, auc_drop=0.0499
    )

    assert no_limiar.level is AlertLevel.CRITICO
    assert abaixo.level is not AlertLevel.CRITICO


def test_atencao_por_coluna():
    decision = decide_alert_level(
        blocked=False,
        drifted_columns=["annuity_amount"],
        score_psi=0.03,
        auc_drop=0.027,
    )

    assert decision.level is AlertLevel.ATENCAO
    assert "drift:annuity_amount" in decision.reasons
    assert decision.performance_evaluated is True


def test_atencao_por_psi_do_score():
    no_limiar = decide_alert_level(
        blocked=False, drifted_columns=[], score_psi=0.10, auc_drop=0.0
    )
    abaixo = decide_alert_level(
        blocked=False, drifted_columns=[], score_psi=0.0999, auc_drop=0.0
    )

    assert no_limiar.level is AlertLevel.ATENCAO
    assert "score_psi>=0.10" in no_limiar.reasons
    assert abaixo.level is AlertLevel.OK


def test_ok():
    decision = decide_alert_level(
        blocked=False, drifted_columns=[], score_psi=0.02, auc_drop=0.0
    )

    assert decision.level is AlertLevel.OK
    assert decision.reasons == ()
    assert decision.performance_evaluated is True
    assert decision.action


def test_sem_rotulo_nunca_passa_de_atencao():
    decision = decide_alert_level(
        blocked=False, drifted_columns=[], score_psi=0.40, auc_drop=None
    )

    assert decision.level is AlertLevel.ATENCAO
    assert decision.performance_evaluated is False


def test_classe_unica_registra_motivo():
    decision = decide_alert_level(
        blocked=False,
        drifted_columns=[],
        score_psi=0.40,
        auc_drop=None,
        performance_skip_reason="single_class",
    )

    assert "single_class" in decision.reasons
    assert decision.level is AlertLevel.ATENCAO
    assert decision.performance_evaluated is False


def test_label_de_exibicao_com_acento():
    assert AlertLevel.ATENCAO.label == "ATENÇÃO"
    assert AlertLevel.CRITICO.label == "CRÍTICO"
    assert AlertLevel.OK.label == "OK"
    assert AlertLevel.BLOQUEADO.label == "BLOQUEADO"


def test_toda_decisao_tem_acao():
    decisions = [
        decide_alert_level(
            blocked=True, drifted_columns=[], score_psi=None, auc_drop=None
        ),
        decide_alert_level(
            blocked=False, drifted_columns=[], score_psi=0.0, auc_drop=0.2
        ),
        decide_alert_level(
            blocked=False, drifted_columns=["x"], score_psi=0.0, auc_drop=0.0
        ),
        decide_alert_level(
            blocked=False, drifted_columns=[], score_psi=0.0, auc_drop=0.0
        ),
    ]

    assert {d.level for d in decisions} == set(AlertLevel)
    assert all(d.action.strip() for d in decisions)
    assert set(ACTIONS) == set(AlertLevel)
