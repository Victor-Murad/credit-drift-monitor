"""Isolamento do MLflow nos testes: nenhum teste toca o mlflow.db real."""

from __future__ import annotations

from pathlib import Path

from mlflow import MlflowClient

from src import config


def test_padrao_da_sessao_nao_aponta_para_o_banco_real():
    assert str(config.ROOT_DIR) not in config.MLFLOW_TRACKING_URI
    assert not config.MLFLOW_ARTIFACT_ROOT.is_relative_to(config.ROOT_DIR)


def test_tracking_env_entrega_banco_migrado_e_isolado(tracking_env, tmp_path):
    assert config.MLFLOW_TRACKING_URI == tracking_env
    assert config.MLFLOW_ARTIFACT_ROOT == tmp_path / "mlartifacts"
    assert Path(tracking_env.removeprefix("sqlite:///")).parent == tmp_path
    client = MlflowClient(tracking_uri=tracking_env, registry_uri=tracking_env)
    client.create_registered_model("m")
    versao = client.create_model_version("m", source="file:///x")
    client.set_registered_model_alias("m", "champion", versao.version)
    assert client.get_model_version_by_alias("m", "champion").version == versao.version


def test_tracking_env_nao_vaza_entre_testes(tracking_env):
    client = MlflowClient(tracking_uri=tracking_env, registry_uri=tracking_env)
    assert client.search_registered_models() == []
