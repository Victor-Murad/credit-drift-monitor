"""Model Registry: versões, aliases e cópia do champion (spec da Etapa 3, §4.7)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.models.integrity import ModelIntegrityError
from src.monitoring import registry
from src.monitoring.registry import (
    bootstrap_champion,
    export_champion,
    get_alias_version,
    register_model_version,
    set_alias,
)
from src.monitoring.tracking import configure_tracking


@pytest.fixture
def champion_dir(tmp_path, monkeypatch) -> Path:
    """`models/champion/` redirecionado para `tmp_path`."""
    pasta = tmp_path / "models" / "champion"
    monkeypatch.setattr(config, "CHAMPION_DIR", pasta)
    monkeypatch.setattr(config, "CHAMPION_FILE", pasta / "baseline_model.joblib")
    monkeypatch.setattr(
        config, "CHAMPION_METADATA_FILE", pasta / "baseline_metadata.json"
    )
    return pasta


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registrar(tmp_path: Path, nome: str, seed: int = 0) -> tuple[int, str]:
    """Treina um modelo minúsculo, grava bundle + metadata e registra uma versão."""
    from tests.conftest import make_valid_df

    df = make_valid_df(n=60, seed=seed)
    X = df[["ext_source_2", "age_years"]]
    pipeline = Pipeline([("model", LogisticRegression())]).fit(X, df["target"])
    pasta = tmp_path / nome
    pasta.mkdir()
    bundle = pasta / "modelo.joblib"
    joblib.dump({"pipeline": pipeline, "model_version": "baseline-v1"}, bundle)
    sha = _sha256(bundle)
    metadata = pasta / "meta.json"
    metadata.write_text(json.dumps({"artifact": {"sha256": sha}}), encoding="utf-8")
    exemplo = X.head(3)
    assinatura = infer_signature(exemplo, pipeline.predict_proba(exemplo))
    configure_tracking()
    with mlflow.start_run(run_name=nome, tags={"run_type": "training"}) as run:
        versao = register_model_version(
            run_id=run.info.run_id,
            pipeline=pipeline,
            signature=assinatura,
            input_example=exemplo,
            bundle_path=bundle,
            metadata_path=metadata,
            tags={"origin": "baseline", "model_sha256": sha},
        )
    return versao, sha


def test_registry_vazio_nao_tem_alias(tracking_env):
    assert get_alias_version("champion") is None


def test_register_model_version_cria_versoes_sequenciais(tracking_env, tmp_path):
    v1, sha = _registrar(tmp_path, "a")
    v2, _ = _registrar(tmp_path, "b", seed=1)
    assert (v1, v2) == (1, 2)
    client = MlflowClient(tracking_uri=tracking_env, registry_uri=tracking_env)
    versao = client.get_model_version(config.REGISTERED_MODEL_NAME, "1")
    assert versao.tags["origin"] == "baseline"
    assert versao.tags["model_sha256"] == sha
    artefatos = sorted(
        a.path
        for a in client.list_artifacts(versao.run_id, registry.BUNDLE_ARTIFACT_DIR)
    )
    assert artefatos == [
        "bundle/baseline_metadata.json",
        "bundle/baseline_model.joblib",
    ]


def test_register_model_version_exige_o_run_ativo(tracking_env, tmp_path):
    configure_tracking()
    with pytest.raises(ValueError, match="run de treino ativo"):
        register_model_version(
            run_id="outro",
            pipeline=None,
            signature=None,
            input_example=None,
            bundle_path=tmp_path,
            metadata_path=tmp_path,
            tags={},
        )


def test_set_alias_e_get_alias_version(tracking_env, tmp_path):
    _registrar(tmp_path, "a")
    _registrar(tmp_path, "b", seed=1)
    set_alias("challenger", 2)
    assert int(get_alias_version("challenger").version) == 2
    assert get_alias_version("champion") is None


def test_bootstrap_champion_so_na_primeira_vez(tracking_env, tmp_path):
    _registrar(tmp_path, "a")
    _registrar(tmp_path, "b", seed=1)
    assert bootstrap_champion(1) is True
    assert bootstrap_champion(2) is False
    assert int(get_alias_version("champion").version) == 1


def test_export_champion_copia_bundle_conferindo_sha256(
    tracking_env, tmp_path, champion_dir
):
    versao, sha = _registrar(tmp_path, "a")
    export_champion(versao)
    assert _sha256(config.CHAMPION_FILE) == sha
    metadata = json.loads(config.CHAMPION_METADATA_FILE.read_text(encoding="utf-8"))
    assert metadata["artifact"]["sha256"] == sha
    # Nada de temporários deixados para trás.
    assert sorted(p.name for p in champion_dir.iterdir()) == [
        "baseline_metadata.json",
        "baseline_model.joblib",
    ]


def test_export_champion_recusa_hash_divergente(tracking_env, tmp_path, champion_dir):
    versao, _ = _registrar(tmp_path, "a")
    client = MlflowClient(tracking_uri=tracking_env, registry_uri=tracking_env)
    client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(versao), "model_sha256", "0" * 64
    )
    with pytest.raises(ModelIntegrityError, match="sha256"):
        export_champion(versao)
    assert not config.CHAMPION_FILE.exists()
    assert not config.CHAMPION_METADATA_FILE.exists()
    assert list(champion_dir.iterdir()) == []


def test_export_champion_sem_tag_de_hash_falha_fechado(
    tracking_env, tmp_path, champion_dir
):
    versao, _ = _registrar(tmp_path, "a")
    client = MlflowClient(tracking_uri=tracking_env, registry_uri=tracking_env)
    client.delete_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(versao), "model_sha256"
    )
    with pytest.raises(ModelIntegrityError, match="model_sha256"):
        export_champion(versao)
    assert not config.CHAMPION_FILE.exists()
