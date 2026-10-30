"""Persistencia portavel do XGBoost entre sistemas operacionais.

O pickle de um `xgboost.Booster` grava o estado interno do objeto em memoria
(`XGBoosterSerializeToBuffer`), que o XGBoost nao garante entre plataformas:
um bundle treinado no macOS falha no Windows com `input stream corrupted`.
O formato de *modelo* (`save_raw("json")`) e o formato portavel documentado.

O bundle gravado em disco leva o booster nesse formato, na chave
`PORTABLE_BOOSTER_KEY`, e o classificador vai no pickle sem o booster.
`restore_booster` reconstroi o booster ao carregar. Bundles antigos, sem a
chave, continuam carregando como antes.
"""

from __future__ import annotations

import copy

from xgboost import Booster, XGBClassifier

MODEL_STEP = "model"
PORTABLE_BOOSTER_KEY = "xgb_booster_json"


def to_portable(bundle: dict) -> dict:
    """Copia do bundle pronta para gravar; o bundle em memoria nao e alterado."""
    pipeline = bundle["pipeline"]
    classifier = dict(pipeline.steps).get(MODEL_STEP)
    if not isinstance(classifier, XGBClassifier) or not hasattr(classifier, "_Booster"):
        return bundle

    sem_booster = copy.copy(classifier)
    del sem_booster._Booster
    pipeline_leve = copy.copy(pipeline)
    pipeline_leve.steps = [
        (nome, sem_booster if nome == MODEL_STEP else passo)
        for nome, passo in pipeline.steps
    ]
    booster_json = bytes(classifier.get_booster().save_raw(raw_format="json"))
    return {**bundle, "pipeline": pipeline_leve, PORTABLE_BOOSTER_KEY: booster_json}


def restore_booster(bundle: dict) -> dict:
    """Reconstroi o booster de um bundle gravado por `to_portable` (in-place)."""
    booster_json = bundle.pop(PORTABLE_BOOSTER_KEY, None)
    if booster_json is None:
        return bundle
    booster = Booster()
    booster.load_model(bytearray(booster_json))
    dict(bundle["pipeline"].steps)[MODEL_STEP]._Booster = booster
    return bundle
