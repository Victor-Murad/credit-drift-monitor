"""Estágios rastreados `prepare` e `train` (spec da Etapa 3, §4.1 e §4.7)."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest

from src import config
from src.config import RAW_COLUMN_MAP
from src.data import prepare
from src.models import train
from src.monitoring import registry, stage
from src.monitoring.registry import get_alias_version
from src.validation.gate import enforce_contract
from tests.conftest import cliente_mlflow, make_valid_df, sha256_de
from tests.test_prepare import _raw_from

FAST_XGB = {"n_estimators": 20, "n_jobs": 1}


@pytest.fixture
def caminhos(tmp_path, monkeypatch, champion_dir) -> SimpleNamespace:
    """Redireciona prepare/train/champion/log para `tmp_path` (nada do repositório)."""
    p = SimpleNamespace(
        raw=tmp_path / "raw" / "application_train.csv",
        reference=tmp_path / "reference" / "reference.parquet",
        production=tmp_path / "production" / "production_base.parquet",
        model=tmp_path / "models" / "baseline_model.joblib",
        metadata=tmp_path / "models" / "baseline_metadata.json",
        champion=tmp_path / "models" / "champion",
        model_reports=tmp_path / "reports" / "model",
        contract_reports=tmp_path / "reports" / "data_contract",
        quarantine=tmp_path / "quarantine",
        log=tmp_path / "pipeline.jsonl",
    )
    monkeypatch.setattr(config, "PIPELINE_LOG_FILE", p.log)
    # prepare
    monkeypatch.setattr(prepare, "RAW_FILE", p.raw)
    monkeypatch.setattr(
        prepare, "load_raw", lambda: pd.read_csv(p.raw, usecols=list(RAW_COLUMN_MAP))
    )
    monkeypatch.setattr(prepare, "REFERENCE_FILE", p.reference)
    monkeypatch.setattr(prepare, "PRODUCTION_BASE_FILE", p.production)
    monkeypatch.setattr(prepare, "CONTRACT_REPORTS_DIR", p.contract_reports)
    # train
    monkeypatch.setattr(train, "REFERENCE_FILE", p.reference)
    monkeypatch.setattr(train, "MODEL_FILE", p.model)
    monkeypatch.setattr(train, "MODEL_METADATA_FILE", p.metadata)
    monkeypatch.setattr(train, "MODEL_REPORTS_DIR", p.model_reports)
    monkeypatch.setattr(
        train,
        "enforce_contract",
        partial(
            enforce_contract, report_dir=p.contract_reports, quarantine_dir=p.quarantine
        ),
    )
    monkeypatch.setattr(train, "XGB_PARAMS", {**train.XGB_PARAMS, **FAST_XGB})
    assert champion_dir == p.champion  # champion servido (fixture do conftest)
    return p


@pytest.fixture
def referencia(caminhos) -> pd.DataFrame:
    df = make_valid_df(n=800, seed=3)
    caminhos.reference.parent.mkdir(parents=True)
    df.to_parquet(caminhos.reference, index=False)
    return df


def _runs(uri: str, filtro: str):
    client = cliente_mlflow(uri)
    experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    return client.search_runs([experimento.experiment_id], filter_string=filtro)


def _run_de_treino(uri: str):
    (treino,) = _runs(uri, "tags.run_type = 'training'")
    return treino


def test_stage_train_registra_run_de_treino(tracking_env, caminhos, referencia):
    assert stage.main(["train"]) == 0

    (estagio,) = _runs(tracking_env, "tags.run_type = 'stage' and tags.stage = 'train'")
    assert estagio.info.status == "FINISHED"
    assert estagio.data.metrics["stage_success"] == 1
    assert estagio.data.metrics["rows_processed"] == len(referencia)

    treino = _run_de_treino(tracking_env)
    assert treino.data.tags["mlflow.parentRunId"] == estagio.info.run_id
    params = treino.data.params
    assert params["n_estimators"] == "20"
    assert params["max_depth"] == str(train.XGB_PARAMS["max_depth"])
    assert 0 < treino.data.metrics["roc_auc"] <= 1
    assert 0 <= treino.data.metrics["ks"] <= 1
    assert treino.data.tags["reference_sha256"] == sha256_de(caminhos.reference)
    assert treino.data.tags["model_sha256"] == sha256_de(caminhos.model)


def test_run_de_treino_anexa_bundle_e_metadata(tracking_env, caminhos, referencia):
    assert stage.main(["train"]) == 0
    treino = _run_de_treino(tracking_env)
    client = cliente_mlflow(tracking_env)
    bundle = sorted(a.path for a in client.list_artifacts(treino.info.run_id, "bundle"))
    assert bundle == ["bundle/baseline_metadata.json", "bundle/baseline_model.joblib"]

    versao = get_alias_version("champion")
    modelo = mlflow.models.get_model_info(versao.source)
    assert modelo.signature is not None
    assert modelo.saved_input_example_info is not None
    carregado = mlflow.pyfunc.load_model(versao.source)
    lote = make_valid_df(n=7, seed=9)[config.MODEL_INPUT_COLUMNS]
    proba = carregado.predict(lote)
    assert proba.shape == (7, 2)  # pyfunc_predict_fn="predict_proba"
    assert ((proba >= 0) & (proba <= 1)).all()


def test_train_bootstrap_registra_v1_como_champion(tracking_env, caminhos, referencia):
    assert stage.main(["train"]) == 0
    champion = get_alias_version("champion")
    assert champion.version == 1  # MLflow 3.x devolve `int` (o plano dizia "1")
    assert champion.tags["origin"] == "baseline"
    assert champion.tags["bundle_model_version"] == train.MODEL_VERSION
    assert champion.tags["reference_sha256"] == sha256_de(caminhos.reference)
    arquivo = caminhos.champion / "baseline_model.joblib"
    assert arquivo.exists()
    assert sha256_de(arquivo) == champion.tags["model_sha256"]
    assert (caminhos.champion / "baseline_metadata.json").exists()


def test_train_repetido_nao_move_champion(tracking_env, caminhos, referencia):
    assert stage.main(["train"]) == 0
    arquivo = caminhos.champion / "baseline_model.joblib"
    antes = arquivo.read_bytes()
    assert stage.main(["train"]) == 0

    client = cliente_mlflow(tracking_env)
    v2 = client.get_model_version(config.REGISTERED_MODEL_NAME, "2")
    assert v2.tags["origin"] == "baseline"
    assert get_alias_version("champion").version == 1
    assert arquivo.read_bytes() == antes


def _quebrar_export(monkeypatch, como: str) -> None:
    if como == "hash":
        # Hash do bundle baixado diverge da tag `model_sha256`.
        monkeypatch.setattr(registry, "file_sha256", lambda path: "0" * 64)
        return
    original = os.replace

    def _replace(origem, destino):
        if Path(destino) == Path(config.CHAMPION_FILE):
            raise PermissionError(13, "Permission denied", str(destino))
        return original(origem, destino)

    monkeypatch.setattr(registry.os, "replace", _replace)


@pytest.mark.parametrize("como", ["hash", "oserror"])
def test_export_falho_nao_da_alias_e_proximo_train_recupera(
    tracking_env, caminhos, referencia, capsys, como
):
    with pytest.MonkeyPatch.context() as mp:
        _quebrar_export(mp, como)
        assert stage.main(["train"]) == 1
    saida = capsys.readouterr().err
    assert "ERRO no estágio train" in saida
    assert "Traceback" not in saida
    assert get_alias_version("champion") is None
    assert not (caminhos.champion / "baseline_model.joblib").exists()
    assert not (caminhos.champion / "baseline_metadata.json").exists()

    # Com o export funcionando, o train seguinte faz o bootstrap (v2).
    assert stage.main(["train"]) == 0
    champion = get_alias_version("champion")
    assert champion.version == 2
    arquivo = caminhos.champion / "baseline_model.joblib"
    assert sha256_de(arquivo) == champion.tags["model_sha256"]


def test_stage_prepare_registra_volume(tracking_env, caminhos):
    caminhos.raw.parent.mkdir(parents=True)
    _raw_from(make_valid_df(n=500, seed=11)).to_csv(caminhos.raw, index=False)
    assert stage.main(["prepare"]) == 0

    (estagio,) = _runs(
        tracking_env, "tags.run_type = 'stage' and tags.stage = 'prepare'"
    )
    linhas = len(pd.read_parquet(caminhos.reference)) + len(
        pd.read_parquet(caminhos.production)
    )
    assert estagio.data.metrics["rows_processed"] == linhas
    assert estagio.data.metrics["stage_success"] == 1


@pytest.mark.parametrize(
    "uri",
    [
        "sqlite:////proc/inexistente/mlflow.db",
        "postgresql://admin:segredo123@host.invalid/db",
    ],
)
def test_stage_falha_fechada_sem_mlflow(
    tmp_path, monkeypatch, mlflow_isolado, caminhos, capsys, uri
):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    monkeypatch.delenv("MONITORING_TRACKING", raising=False)
    caminhos.raw.parent.mkdir(parents=True)
    _raw_from(make_valid_df(n=200, seed=11)).to_csv(caminhos.raw, index=False)

    assert stage.main(["prepare"]) != 0
    saida = capsys.readouterr()
    assert "MONITORING_TRACKING=off" in saida.err
    assert "segredo123" not in saida.err + saida.out
    assert "Traceback" not in saida.err
    assert not caminhos.reference.exists(), "o estágio não pode rodar sem o MLflow"


def test_stage_tracking_off_executa_sem_mlflow(
    tmp_path, monkeypatch, mlflow_isolado, caminhos, referencia
):
    banco = tmp_path / "mlflow" / "mlflow.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{banco}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    monkeypatch.setenv("MONITORING_TRACKING", "off")

    assert stage.main(["train"]) == 0
    assert caminhos.model.exists()
    assert not banco.exists()
    assert not (tmp_path / "mlartifacts").exists()
    # Sem Registry não há champion para copiar.
    assert not caminhos.champion.exists()


@pytest.mark.parametrize("nome", ["drift_demo", "contract_demo", "inexistente"])
def test_stage_desconhecido_e_erro_de_uso(nome, capsys):
    with pytest.raises(SystemExit) as saida:
        stage.main([nome])
    assert saida.value.code == 2
    assert "ainda não disponível" in capsys.readouterr().err


def test_train_main_devolve_bundle_e_metadata(caminhos, referencia):
    bundle, metadata = train.main()
    assert bundle["model_version"] == train.MODEL_VERSION
    assert metadata["artifact"]["sha256"] == sha256_de(caminhos.model)
    assert metadata["data_lineage"]["reference_sha256"] == sha256_de(caminhos.reference)
