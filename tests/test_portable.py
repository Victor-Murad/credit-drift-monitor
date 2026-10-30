"""Bundle portavel: o booster vai no formato de modelo do XGBoost, nao no pickle."""

from __future__ import annotations

import joblib
import numpy as np

from src.models.features import model_input
from src.models.portable import PORTABLE_BOOSTER_KEY, restore_booster, to_portable
from src.models.predict import load_bundle
from src.models.train import make_bundle, save_bundle, train_models
from tests.conftest import make_valid_df

FAST_XGB = {"n_estimators": 20, "n_jobs": 1}


def _bundle() -> dict:
    bundle, _ = train_models(
        make_valid_df(n=900, seed=5), xgb_params=FAST_XGB, importance_sample=200
    )
    return bundle


def test_to_portable_nao_altera_o_bundle_em_memoria():
    bundle = _bundle()
    portavel = to_portable(bundle)

    assert PORTABLE_BOOSTER_KEY in portavel
    assert PORTABLE_BOOSTER_KEY not in bundle
    assert hasattr(dict(bundle["pipeline"].steps)["model"], "_Booster")
    assert not hasattr(dict(portavel["pipeline"].steps)["model"], "_Booster")


def test_pickle_gravado_nao_carrega_o_estado_interno_do_booster(tmp_path):
    # O estado interno (XGBoosterSerializeToBuffer) e o que nao e portavel entre
    # sistemas operacionais; o arquivo gravado so pode ter o formato de modelo.
    arquivo = tmp_path / "m.joblib"
    joblib.dump(to_portable(_bundle()), arquivo)

    bruto = joblib.load(arquivo)
    assert not hasattr(dict(bruto["pipeline"].steps)["model"], "_Booster")
    assert bruto[PORTABLE_BOOSTER_KEY].startswith(b"{")


def test_ida_e_volta_preserva_as_predicoes(tmp_path):
    bundle = _bundle()
    lote = model_input(make_valid_df(n=200, seed=8, labeled=False))
    esperado = bundle["pipeline"].predict_proba(lote)[:, 1]

    modelo = tmp_path / "baseline_model.joblib"
    meta = tmp_path / "baseline_metadata.json"
    save_bundle(bundle, {"model_version": "t"}, modelo, meta)
    carregado = load_bundle(modelo)

    assert PORTABLE_BOOSTER_KEY not in carregado
    obtido = carregado["pipeline"].predict_proba(lote)[:, 1]
    np.testing.assert_array_equal(obtido, esperado)


def test_bundle_antigo_sem_a_chave_carrega_como_antes():
    bundle = make_bundle(_bundle()["pipeline"], 0.1, "antigo")
    assert restore_booster(bundle) is bundle
