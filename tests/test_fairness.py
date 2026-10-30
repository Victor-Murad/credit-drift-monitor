"""Auditoria de vies (Etapa 4): metricas por grupo, resumo e CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.governance import fairness
from tests.conftest import make_valid_df


class _ScoreByExtSource:
    """Pipeline de mentira: risco alto quando o score de bureau e baixo."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        risk = 1 - X["ext_source_3"].fillna(0.5).to_numpy()
        return np.column_stack([1 - risk, risk])


def _bundle() -> dict:
    return {
        "pipeline": _ScoreByExtSource(),
        "threshold": 0.5,
        "model_version": "teste",
    }


def _scored() -> pd.DataFrame:
    # Grupo A: 4 linhas, 3 aprovadas; grupo B: 4 linhas, 1 aprovada.
    return pd.DataFrame(
        {
            "grp": ["A"] * 4 + ["B"] * 4,
            TARGET_COLUMN: [0, 0, 1, 1, 0, 0, 1, 1],
            "predicted_default": [0, 0, 0, 1, 0, 1, 1, 1],
            "default_probability": [0.1, 0.2, 0.3, 0.9, 0.2, 0.6, 0.7, 0.8],
        }
    )


def test_group_metrics_calcula_taxas_por_grupo():
    a, b = fairness.group_metrics(_scored(), "grp")

    assert (a["group"], a["n"], a["share"]) == ("A", 4, 0.5)
    assert a["approval_rate"] == 0.75
    assert a["good_payer_approval_rate"] == 1.0
    assert a["bad_payer_approval_rate"] == 0.5
    assert a["default_rate_approved"] == pytest.approx(1 / 3, abs=1e-4)
    assert a["roc_auc"] == 1.0
    assert b["approval_rate"] == 0.25
    assert b["good_payer_approval_rate"] == 0.5
    assert b["default_rate_approved"] == 0.0


def test_summarize_razao_de_impacto_e_diferencas():
    resumo = fairness.summarize(fairness.group_metrics(_scored(), "grp"), 1)

    assert resumo["disparate_impact_ratio"] == pytest.approx(0.25 / 0.75, abs=1e-4)
    assert resumo["statistical_parity_difference"] == 0.5
    assert resumo["equal_opportunity_difference"] == 0.5
    assert resumo["needs_review"] is True


def test_summarize_ignora_grupos_pequenos():
    resumo = fairness.summarize(fairness.group_metrics(_scored(), "grp"), 5)

    assert resumo["groups_compared"] == []
    assert resumo["groups_below_min_size"] == ["A", "B"]
    assert resumo["disparate_impact_ratio"] is None
    assert resumo["needs_review"] is False


def test_grupo_com_uma_classe_nao_tem_auc():
    scored = _scored()
    scored.loc[scored["grp"] == "B", TARGET_COLUMN] = 0
    _, b = fairness.group_metrics(scored, "grp")

    assert b["roc_auc"] is None
    assert b["bad_payer_approval_rate"] is None


def test_age_band_intervalos_fechados_a_esquerda():
    faixas = fairness.age_band(pd.Series([18.0, 24.99, 25.0, 59.9, 60.0, 99.0]))
    assert faixas.tolist() == ["18-24", "18-24", "25-34", "45-59", "60+", "60+"]


def test_audit_cobre_genero_e_idade_sem_usar_genero_no_modelo():
    lote = make_valid_df(n=400, seed=3)
    relatorio = fairness.audit(lote, _bundle(), batch_name="sintetico")

    assert relatorio["n_rows"] == 400
    assert set(relatorio["attributes"]) == {"gender", "age_band"}
    grupos = {g["group"] for g in relatorio["attributes"]["gender"]["groups"]}
    assert grupos <= {"F", "M"}
    assert sum(g["n"] for g in relatorio["attributes"]["age_band"]["groups"]) == 400


def test_audit_exige_lote_rotulado():
    with pytest.raises(ValueError, match="rotulado"):
        fairness.audit(
            make_valid_df(n=50, labeled=False), _bundle(), batch_name="sem_rotulo"
        )


def test_cli_grava_json_e_markdown(tmp_path, monkeypatch, capsys):
    lote = tmp_path / "lote.parquet"
    make_valid_df(n=300, seed=5).to_parquet(lote, index=False)
    monkeypatch.setattr(fairness, "load_bundle", _bundle)

    assert fairness.main(["--batch", str(lote), "--out-dir", str(tmp_path)]) == 0

    dados = json.loads((tmp_path / "lote_fairness.json").read_text(encoding="utf-8"))
    assert dados["model_version"] == "teste"
    markdown = (tmp_path / "lote_fairness.md").read_text(encoding="utf-8")
    assert "## `gender`" in markdown and "## `age_band`" in markdown
    assert "Razão de impacto disparatado" in capsys.readouterr().out


def test_cli_falha_sem_lote_ou_sem_rotulo(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fairness, "load_bundle", _bundle)
    assert fairness.main(["--batch", str(tmp_path / "sumiu.parquet")]) == 1

    sem_rotulo = tmp_path / "sem_rotulo.parquet"
    make_valid_df(n=50, labeled=False).to_parquet(sem_rotulo, index=False)
    assert fairness.main(["--batch", str(sem_rotulo), "--out-dir", str(tmp_path)]) == 1
    assert "rotulado" in capsys.readouterr().err
