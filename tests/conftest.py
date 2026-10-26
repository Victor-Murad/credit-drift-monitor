"""Fixtures: dados sinteticos aderentes ao contrato.

Os testes nao dependem do dataset do Kaggle (licenca nao permite versiona-lo
e o CI nao tem credenciais), entao geramos lotes sinteticos com o mesmo
esquema e um sinal aprendivel (ext_source baixo -> mais default).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Antes de qualquer import de `src`: os testes nao escrevem no log real do
# pipeline (logs/pipeline.jsonl), que sera centralizado na Etapa 3, nem tocam
# o mlflow.db/mlartifacts reais (os padroes abaixo apontam para um diretorio
# temporario da sessao).
os.environ["PIPELINE_LOG_FILE"] = str(
    Path(tempfile.gettempdir()) / "credit-drift-monitor-tests.jsonl"
)
os.environ["MLFLOW_ARTIFACT_ROOT"] = str(
    Path(tempfile.mkdtemp(prefix="credit-drift-monitor-tests-")) / "mlartifacts"
)
os.environ["MLFLOW_TRACKING_URI"] = (
    f"sqlite:///{Path(os.environ['MLFLOW_ARTIFACT_ROOT']).parent / 'mlflow.db'}"
)

import numpy as np  # noqa: E402
import pandas as pd
import pytest

from src import config
from src.validation import contract as c


def make_valid_df(n: int = 300, seed: int = 0, labeled: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income_type = rng.choice(c.INCOME_TYPES[:4], size=n, p=[0.5, 0.2, 0.2, 0.1])
    not_employed = np.isin(income_type, c.NOT_EMPLOYED_INCOME_TYPES)
    age = rng.uniform(21, 68, size=n).round(2)
    credit = rng.uniform(50_000, 1_500_000, size=n).round(1)
    children = rng.integers(0, 4, size=n)
    ext2 = rng.uniform(0, 1, size=n)
    ext3 = rng.uniform(0, 1, size=n)
    ext3[rng.random(n) < 0.15] = np.nan

    df = pd.DataFrame(
        {
            "client_id": np.arange(100_001, 100_001 + n),
            "contract_type": rng.choice(c.CONTRACT_TYPES, size=n, p=[0.9, 0.1]),
            "gender": rng.choice(c.GENDERS, size=n),
            "own_car": rng.choice(c.YES_NO, size=n),
            "own_realty": rng.choice(c.YES_NO, size=n),
            "income_type": income_type,
            "education_type": rng.choice(c.EDUCATION_TYPES[:3], size=n),
            "family_status": rng.choice(c.FAMILY_STATUSES[:3], size=n),
            "housing_type": rng.choice(c.HOUSING_TYPES[:3], size=n),
            "region_rating": rng.choice(c.REGION_RATINGS, size=n),
            "children_count": children,
            "family_members": children + rng.integers(1, 3, size=n),
            "age_years": age,
            "years_employed": np.where(
                not_employed, np.nan, rng.uniform(0, 15, size=n).round(2)
            ),
            "income_total": rng.uniform(30_000, 500_000, size=n).round(1),
            "credit_amount": credit,
            "annuity_amount": (credit / rng.uniform(10, 40, size=n)).round(1),
            "goods_price": (credit * 0.9).round(1),
            "ext_source_2": ext2,
            "ext_source_3": ext3,
        }
    )
    if labeled:
        logit = -1.0 - 3.0 * (ext2 - 0.5) - 0.02 * (age - 40)
        df["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return make_valid_df()


@pytest.fixture
def valid_unlabeled_df() -> pd.DataFrame:
    return make_valid_df(labeled=False)


@pytest.fixture(scope="session")
def _banco_modelo_mlflow(tmp_path_factory) -> Path:
    """Banco SQLite do MLflow ja migrado, criado uma vez por sessao.

    A migracao leva ~2,4 s; copiar o arquivo para cada teste e quase instantaneo.
    """
    from mlflow import MlflowClient

    caminho = tmp_path_factory.mktemp("mlflow-modelo") / "modelo.db"
    uri = f"sqlite:///{caminho}"
    MlflowClient(tracking_uri=uri, registry_uri=uri).search_experiments()
    return caminho


# Variaveis que o proprio MLflow grava no os.environ (set_tracking_uri,
# set_registry_uri, set_experiment, start_run) para subprocessos herdarem.
MLFLOW_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_REGISTRY_URI",
    "MLFLOW_EXPERIMENT_ID",
    "MLFLOW_RUN_ID",
)


@contextmanager
def isolar_estado_mlflow() -> Iterator[None]:
    """Desfaz o estado global do MLflow ao sair: runs ativos, URIs, experimento, ambiente.

    Sem isso, um teste que configurou um banco inexistente (ou uma URI com senha)
    deixaria o `os.environ` e o cliente fluente apontando para ela nos seguintes.
    """
    import mlflow
    from mlflow.tracking import fluent

    antes = {nome: os.environ.get(nome) for nome in MLFLOW_ENV_VARS}
    try:
        yield
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(None)
        mlflow.set_registry_uri(None)
        fluent._active_experiment_id = None
        for nome, valor in antes.items():
            if valor is None:
                os.environ.pop(nome, None)
            else:
                os.environ[nome] = valor


@pytest.fixture
def mlflow_isolado() -> Iterator[None]:
    with isolar_estado_mlflow():
        yield


@pytest.fixture
def tracking_env(tmp_path, monkeypatch, mlflow_isolado, _banco_modelo_mlflow) -> str:
    """Aponta o MLflow para um banco novo e isolado em `tmp_path`; devolve o URI."""
    banco = tmp_path / "mlflow.db"
    shutil.copy(_banco_modelo_mlflow, banco)
    uri = f"sqlite:///{banco}"
    artefatos = tmp_path / "mlartifacts"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(artefatos))
    # `src.config` le o ambiente na importacao: reatribui as constantes tambem.
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", artefatos)
    return uri
