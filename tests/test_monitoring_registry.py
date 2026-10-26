"""Model Registry: versões, aliases e cópia do champion (spec da Etapa 3, §4.7)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.models.integrity import ModelIntegrityError, verify_model_integrity
from src.monitoring import registry
from src.monitoring.registry import (
    bootstrap_champion,
    export_champion,
    get_alias_version,
    register_model_version,
    set_alias,
)
from src.monitoring.tracking import TrackingUnavailableError, configure_tracking
from tests.conftest import cliente_mlflow, make_valid_df, sha256_de

_SEM_HASH = object()


def _registrar(
    tmp_path: Path, nome: str, seed: int = 0, metadata: object = None
) -> tuple[int, str]:
    """Treina um modelo minúsculo, grava bundle + metadata e registra uma versão.

    `metadata=_SEM_HASH` grava um metadata sem `artifact.sha256`.
    """
    df = make_valid_df(n=60, seed=seed)
    X = df[["ext_source_2", "age_years"]]
    pipeline = Pipeline([("model", LogisticRegression())]).fit(X, df["target"])
    pasta = tmp_path / nome
    pasta.mkdir()
    bundle = pasta / "modelo.joblib"
    joblib.dump({"pipeline": pipeline, "model_version": "baseline-v1"}, bundle)
    sha = sha256_de(bundle)
    conteudo = {"outro": 1} if metadata is _SEM_HASH else {"artifact": {"sha256": sha}}
    arquivo_metadata = pasta / "meta.json"
    arquivo_metadata.write_text(json.dumps(conteudo), encoding="utf-8")
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
            metadata_path=arquivo_metadata,
            tags={"origin": "baseline", "model_sha256": sha},
        )
    return versao, sha


def test_registry_vazio_nao_tem_alias(tracking_env):
    assert get_alias_version("champion") is None


def test_register_model_version_cria_versoes_sequenciais(tracking_env, tmp_path):
    v1, sha = _registrar(tmp_path, "a")
    v2, _ = _registrar(tmp_path, "b", seed=1)
    assert (v1, v2) == (1, 2)
    client = cliente_mlflow(tracking_env)
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


def test_register_model_version_falha_do_mlflow_nao_vaza_credencial(
    tracking_env, tmp_path, monkeypatch
):
    def _falhar(*args, **kwargs):
        raise MlflowException("API request to http://admin:segredo123@host failed")

    monkeypatch.setattr(mlflow, "register_model", _falhar)
    with pytest.raises(TrackingUnavailableError) as erro:
        _registrar(tmp_path, "a")
    mensagem = str(erro.value)
    assert "Falha ao registrar a versão do modelo" in mensagem
    assert "MONITORING_TRACKING=off" in mensagem
    assert "segredo123" not in mensagem
    assert "admin:" not in mensagem


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
    assert sha256_de(config.CHAMPION_FILE) == sha
    metadata = json.loads(config.CHAMPION_METADATA_FILE.read_text(encoding="utf-8"))
    assert metadata["artifact"]["sha256"] == sha
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)
    # Nada de temporários deixados para trás.
    assert sorted(p.name for p in champion_dir.iterdir()) == [
        "baseline_metadata.json",
        "baseline_model.joblib",
    ]


def test_export_champion_recusa_hash_divergente(tracking_env, tmp_path, champion_dir):
    versao, _ = _registrar(tmp_path, "a")
    cliente_mlflow(tracking_env).set_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(versao), "model_sha256", "0" * 64
    )
    with pytest.raises(ModelIntegrityError, match="sha256"):
        export_champion(versao)
    assert list(champion_dir.iterdir()) == []


def test_export_champion_sem_tag_de_hash_falha_fechado(
    tracking_env, tmp_path, champion_dir
):
    versao, _ = _registrar(tmp_path, "a")
    cliente_mlflow(tracking_env).delete_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(versao), "model_sha256"
    )
    with pytest.raises(ModelIntegrityError, match="model_sha256"):
        export_champion(versao)
    assert not config.CHAMPION_FILE.exists()


def test_export_champion_metadata_sem_hash_falha_fechado(
    tracking_env, tmp_path, champion_dir
):
    versao, _ = _registrar(tmp_path, "a", metadata=_SEM_HASH)
    with pytest.raises(ModelIntegrityError, match=r"artifact\.sha256"):
        export_champion(versao)
    assert list(champion_dir.iterdir()) == []


def test_export_champion_versao_sem_run_de_origem(tracking_env, tmp_path, champion_dir):
    _registrar(tmp_path, "a")
    client = cliente_mlflow(tracking_env)
    v1 = client.get_model_version(config.REGISTERED_MODEL_NAME, "1")
    orfa = client.create_model_version(
        config.REGISTERED_MODEL_NAME,
        source=(tmp_path / "a").as_uri(),
        tags={"model_sha256": v1.tags["model_sha256"]},
    )
    assert not orfa.run_id
    with pytest.raises(ModelIntegrityError, match="sem run de origem"):
        export_champion(int(orfa.version))
    assert not config.CHAMPION_FILE.exists()


def _replace_que_falha(alvo_que_falha):
    original = os.replace

    def _replace(origem, destino):
        if Path(destino) == Path(alvo_que_falha()):
            raise PermissionError(13, "Permission denied", str(destino))
        return original(origem, destino)

    return _replace


@pytest.mark.parametrize("onde", ["replace", "sha256", "mkdir"])
def test_export_champion_oserror_vira_erro_de_dominio(
    tracking_env, tmp_path, champion_dir, monkeypatch, onde
):
    versao, _ = _registrar(tmp_path, "a")
    if onde == "replace":
        monkeypatch.setattr(
            registry.os, "replace", _replace_que_falha(lambda: config.CHAMPION_FILE)
        )
    elif onde == "sha256":

        def _ilegivel(path):
            raise FileNotFoundError(2, "No such file or directory", str(path))

        monkeypatch.setattr(registry, "file_sha256", _ilegivel)
    else:
        # O "diretório" do champion é um arquivo: mkdir falha.
        champion_dir.parent.mkdir(parents=True)
        champion_dir.write_text("não sou diretório", encoding="utf-8")

    with pytest.raises(ModelIntegrityError) as erro:
        export_champion(versao)
    assert "Falha ao exportar o champion" in str(erro.value)
    assert str(champion_dir) in str(erro.value)
    assert not config.CHAMPION_FILE.exists()
    assert not config.CHAMPION_METADATA_FILE.exists()


def test_export_champion_troca_o_modelo_por_ultimo(
    tracking_env, tmp_path, champion_dir, monkeypatch
):
    """Entre os dois `os.replace` o par em disco é um que a integridade recusa."""
    v1, sha1 = _registrar(tmp_path, "a")
    v2, sha2 = _registrar(tmp_path, "b", seed=1)
    assert sha1 != sha2
    export_champion(v1)
    original = os.replace
    trocas: list[str] = []

    def _replace(origem, destino):
        destino = Path(destino)
        if destino.parent == champion_dir:
            trocas.append(destino.name)
            if destino == config.CHAMPION_FILE:
                # Janela: metadata já é o da v2, o modelo ainda é o da v1.
                with pytest.raises(ModelIntegrityError, match="diverge"):
                    verify_model_integrity(
                        config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE
                    )
        return original(origem, destino)

    monkeypatch.setattr(registry.os, "replace", _replace)
    export_champion(v2)
    assert trocas == ["baseline_metadata.json", "baseline_model.joblib"]
    assert sha256_de(config.CHAMPION_FILE) == sha2
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)


def test_export_champion_falha_na_troca_do_modelo_restaura_o_par(
    tracking_env, tmp_path, champion_dir, monkeypatch
):
    v1, sha1 = _registrar(tmp_path, "a")
    v2, _ = _registrar(tmp_path, "b", seed=1)
    export_champion(v1)
    monkeypatch.setattr(
        registry.os, "replace", _replace_que_falha(lambda: config.CHAMPION_FILE)
    )
    with pytest.raises(ModelIntegrityError, match="Falha ao exportar o champion"):
        export_champion(v2)
    metadata = json.loads(config.CHAMPION_METADATA_FILE.read_text(encoding="utf-8"))
    assert metadata["artifact"]["sha256"] == sha1
    assert sha256_de(config.CHAMPION_FILE) == sha1
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)
    assert sorted(p.name for p in champion_dir.iterdir()) == [
        "baseline_metadata.json",
        "baseline_model.joblib",
    ]


def test_requisito_ausente_falha_com_mensagem(tracking_env, tmp_path, monkeypatch):
    monkeypatch.setattr(
        registry, "_MODEL_REQUIREMENTS", ("scikit-learn", "pacote-que-nao-existe")
    )
    with pytest.raises(TrackingUnavailableError, match="pacote-que-nao-existe"):
        _registrar(tmp_path, "a")
