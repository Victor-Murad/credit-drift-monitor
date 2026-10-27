"""Retreino do challenger guardado por alerta CRÍTICO (spec da Etapa 3, §4.7 e §8, item 10).

Dados SINTÉTICOS (sem Kaggle) e MLflow real num banco temporário: o `train` roda uma vez
no módulo (bootstrap do champion) e o `monitor` registra um lote rotulado degradado
(`CRITICO`, o "t2" destes testes) e um lote estável sem rótulo (`OK`). Cada teste que
grava no Registry trabalha numa cópia do banco (`com_alerta`/`sem_alerta`). O aceite
com os dados reais é a Tarefa 17.
"""

from __future__ import annotations

import json
import shutil
from functools import partial
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import config
from src.models import predict, train
from src.models.features import model_input
from src.models.train import split_reference
from src.monitoring import monitor, registry, retrain, stage
from src.monitoring.tracking import TrackingUnavailableError
from src.validation.gate import enforce_contract
from tests.conftest import (
    cliente_mlflow,
    isolar_estado_mlflow,
    make_valid_df,
    sha256_de,
)

FAST_XGB = {"n_estimators": 20, "n_jobs": 1}
N_LOTE = 1500
# Os lotes de produção não repetem clientes da referência (como no `prepare` real).
OFFSET_PRODUCAO = 500_000


def _lote_producao(*, seed: int, rotulado: bool = True, degradado: bool = False):
    df = make_valid_df(n=N_LOTE, seed=seed, labeled=rotulado)
    df["client_id"] = df["client_id"] + OFFSET_PRODUCAO
    if degradado:
        df["target"] = 1 - df["target"]
    return df


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory, _banco_modelo_mlflow):
    root = tmp_path_factory.mktemp("retrain")
    p = SimpleNamespace(
        root=root,
        banco=root / "mlflow.db",
        banco_sem_alerta=root / "mlflow_sem_alerta.db",
        reference=root / "reference" / "reference.parquet",
        model=root / "models" / "baseline_model.joblib",
        metadata=root / "models" / "baseline_metadata.json",
        champion=root / "models" / "champion",
        t2=root / "production" / "production_t2_concept_drift.parquet",
        estavel=root / "production" / "production_estavel.parquet",
        quarantine=root / "quarantine",
    )
    shutil.copy(_banco_modelo_mlflow, p.banco)
    p.uri = f"sqlite:///{p.banco}"
    p.reference.parent.mkdir(parents=True)
    make_valid_df(n=3000, seed=3).to_parquet(p.reference, index=False)
    p.t2.parent.mkdir(parents=True)
    _lote_producao(seed=5, degradado=True).to_parquet(p.t2, index=False)
    _lote_producao(seed=6, rotulado=False).to_parquet(p.estavel, index=False)

    with isolar_estado_mlflow(), pytest.MonkeyPatch.context() as mp:
        mp.delenv("MONITORING_TRACKING", raising=False)
        mp.setenv("MLFLOW_TRACKING_URI", p.uri)
        mp.setenv("MLFLOW_ARTIFACT_ROOT", str(root / "mlartifacts"))
        mp.setattr(config, "MLFLOW_TRACKING_URI", p.uri)
        mp.setattr(config, "MLFLOW_ARTIFACT_ROOT", root / "mlartifacts")
        mp.setattr(config, "PIPELINE_LOG_FILE", root / "pipeline.jsonl")
        mp.setattr(config, "MODEL_FILE", p.model)
        mp.setattr(config, "CHAMPION_DIR", p.champion)
        mp.setattr(config, "CHAMPION_FILE", p.champion / "baseline_model.joblib")
        mp.setattr(
            config, "CHAMPION_METADATA_FILE", p.champion / "baseline_metadata.json"
        )
        mp.setattr(config, "REFERENCE_FILE", p.reference)
        mp.setattr(config, "QUARANTINE_DIR", p.quarantine)
        mp.setattr(config, "PRODUCTION_T2_FILE", p.t2)
        mp.setattr(monitor, "MONITOR_REPORTS_DIR", root / "reports" / "monitor")
        mp.setattr(retrain, "RETRAIN_REPORTS_DIR", root / "reports" / "retrain")
        # train
        mp.setattr(train, "REFERENCE_FILE", p.reference)
        mp.setattr(train, "MODEL_FILE", p.model)
        mp.setattr(train, "MODEL_METADATA_FILE", p.metadata)
        mp.setattr(train, "MODEL_REPORTS_DIR", root / "reports" / "model")
        mp.setattr(
            train,
            "enforce_contract",
            partial(
                enforce_contract,
                report_dir=root / "reports" / "data_contract",
                quarantine_dir=p.quarantine,
            ),
        )
        mp.setattr(train, "XGB_PARAMS", {**train.XGB_PARAMS, **FAST_XGB})

        assert stage.main(["train"]) == 0
        shutil.copy(p.banco, p.banco_sem_alerta)
        assert monitor.main(["--batch", str(p.t2), "--baseline-auc", "0.75"]) == 2
        assert monitor.main(["--batch", str(p.estavel)]) == 0
        p.alerta = _lote(p.uri, p.t2.stem).info.run_id
        p.lote_ok = _lote(p.uri, p.estavel.stem).info.run_id
        yield p


def _runs(uri: str, filtro: str):
    client = cliente_mlflow(uri)
    experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    return client.search_runs(
        [experimento.experiment_id],
        filter_string=filtro,
        order_by=["attributes.start_time ASC"],
    )


def _lote(uri: str, nome: str):
    (lote,) = _runs(uri, f"tags.run_type = 'batch' and tags.batch_name = '{nome}'")
    return lote


def _retrain_runs(uri: str):
    return _runs(uri, "tags.run_type = 'retrain'")


def _copiar_banco(origem, tmp_path, monkeypatch) -> str:
    banco = tmp_path / "mlflow.db"
    shutil.copy(origem, banco)
    uri = f"sqlite:///{banco}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture
def com_alerta(sandbox, tmp_path, monkeypatch, mlflow_isolado) -> str:
    """Cópia do banco com o champion e o alerta CRÍTICO: runs gravados aqui não vazam."""
    return _copiar_banco(sandbox.banco, tmp_path, monkeypatch)


@pytest.fixture
def sem_alerta(sandbox, tmp_path, monkeypatch, mlflow_isolado) -> str:
    """Cópia do banco logo depois do `train`: champion, nenhum run de lote."""
    return _copiar_banco(sandbox.banco_sem_alerta, tmp_path, monkeypatch)


def _champion_version(uri: str) -> int:
    nome = config.REGISTERED_MODEL_NAME
    return int(cliente_mlflow(uri).get_model_version_by_alias(nome, "champion").version)


def _t2_validado(sandbox) -> pd.DataFrame:
    return enforce_contract(
        pd.read_parquet(sandbox.t2),
        batch_name="pytest_t2",
        stage="pytest",
        labeled=True,
        report_dir=None,
        quarantine_dir=None,
    )


# MARK: divisão dos dados (funções puras)


def test_split_estratificado_e_sem_vazamento():
    lote = _lote_producao(seed=7)

    treino, holdout = retrain.split_critical_batch(lote)

    assert set(treino["client_id"]).isdisjoint(holdout["client_id"])
    assert len(treino) + len(holdout) == len(lote)
    assert abs(len(treino) - len(holdout)) <= 1
    assert treino["target"].mean() == pytest.approx(holdout["target"].mean(), abs=0.01)
    de_novo, holdout_de_novo = retrain.split_critical_batch(lote)
    pd.testing.assert_frame_equal(treino, de_novo)
    pd.testing.assert_frame_equal(holdout, holdout_de_novo)


def test_challenger_nao_treina_com_linhas_do_teste_da_referencia():
    referencia = make_valid_df(n=1500, seed=3)
    critico_treino, _ = retrain.split_critical_batch(_lote_producao(seed=8))

    treino, validacao, teste = retrain.build_challenger_training(
        referencia, critico_treino
    )

    treino_ref, val_ref, test_ref = split_reference(referencia, config.SEED)
    pd.testing.assert_frame_equal(validacao, val_ref)
    pd.testing.assert_frame_equal(teste, test_ref)
    assert set(treino["client_id"]).isdisjoint(test_ref["client_id"])
    assert set(treino["client_id"]).isdisjoint(val_ref["client_id"])
    assert set(treino["client_id"]) == set(treino_ref["client_id"]) | set(
        critico_treino["client_id"]
    )
    assert len(treino) == len(treino_ref) + len(critico_treino)


def test_challenger_recusa_lote_com_clientes_da_validacao_ou_do_teste():
    referencia = make_valid_df(n=1500, seed=3)
    _, val_ref, test_ref = split_reference(referencia, config.SEED)
    contaminado = pd.concat([_lote_producao(seed=8).head(50), test_ref.head(3)])

    with pytest.raises(stage.StagePreconditionError, match="client_id"):
        retrain.build_challenger_training(referencia, contaminado)


# MARK: gatilho (alerta CRÍTICO ou flag explícito)


def test_sem_alerta_critico_recusa(sem_alerta, capsys):
    assert retrain.main([]) == retrain.EXIT_REFUSED
    erro = capsys.readouterr().err
    assert "CRÍTICO" in erro
    assert "--force-no-alert" in erro
    assert "Traceback" not in erro
    assert not _retrain_runs(sem_alerta)

    motivo = "demonstração: retreino sem alerta, decidido pelo time"
    assert retrain.main(["--force-no-alert", "--reason", motivo]) == retrain.EXIT_OK

    (run,) = _retrain_runs(sem_alerta)
    assert run.data.tags["retrain_reason"] == motivo
    assert run.data.tags["trigger"] == "force_no_alert"
    assert "alert_run_id" not in run.data.tags


@pytest.mark.parametrize(
    "argv",
    [
        ["--force-no-alert"],
        ["--force-no-alert", "--reason", "   "],
        ["--alert-run", "abc", "--allow-reuse"],
        ["--allow-reuse", "--reason", "sem --alert-run"],
        ["--force-no-alert", "--alert-run", "abc", "--reason", "os dois"],
    ],
)
def test_flags_de_excecao_exigem_motivo_e_combinacao_valida(sandbox, argv, capsys):
    with pytest.raises(SystemExit) as saida:
        retrain.main(argv)
    # Erro de uso é operacional (1): o 2 é a recusa de política.
    assert saida.value.code == 1
    assert "erro" in capsys.readouterr().err


def test_alerta_consumido_recusa_sem_flag(sandbox, com_alerta, capsys):
    cliente_mlflow(com_alerta).set_tag(sandbox.alerta, "retrain_consumed", "99")

    assert retrain.main([]) == retrain.EXIT_REFUSED
    assert "--force-no-alert" in capsys.readouterr().err
    assert retrain.main(["--alert-run", sandbox.alerta]) == retrain.EXIT_REFUSED
    erro = capsys.readouterr().err
    assert "consumido" in erro
    assert "--allow-reuse" in erro
    assert not _retrain_runs(com_alerta)

    motivo = "reaproveitar o alerta depois de corrigir o lote"
    argv = ["--alert-run", sandbox.alerta, "--allow-reuse", "--reason", motivo]
    assert retrain.main(argv) == retrain.EXIT_OK

    (run,) = _retrain_runs(com_alerta)
    assert run.data.tags["retrain_reason"] == motivo
    assert run.data.tags["alert_reused"] == "true"
    versao = run.data.tags["challenger_version"]
    alerta = cliente_mlflow(com_alerta).get_run(sandbox.alerta)
    assert alerta.data.tags["retrain_consumed"] == versao


def _alertas_t2(sandbox, uri: str):
    filtro = f"tags.run_type = 'batch' and tags.batch_name = '{sandbox.t2.stem}'"
    return _runs(uri, filtro)


def _monitorar_t2(sandbox) -> None:
    """Outro alerta CRÍTICO do mesmo t2 (como um segundo `make pipeline`)."""
    argv = ["--batch", str(sandbox.t2), "--baseline-auc", "0.75"]
    assert monitor.main(argv) == 2


def test_dois_alertas_do_mesmo_lote_e_dois_retreinos_seguidos(
    sandbox, com_alerta, capsys
):
    _monitorar_t2(sandbox)
    alertas = _alertas_t2(sandbox, com_alerta)
    assert len(alertas) == 2
    assert len({a.data.tags["batch_sha256"] for a in alertas}) == 1
    cliente = cliente_mlflow(com_alerta)
    nome = config.REGISTERED_MODEL_NAME

    assert retrain.main([]) == retrain.EXIT_OK
    versao = cliente.get_model_version_by_alias(nome, "challenger").version
    for alerta in alertas:
        tags = cliente.get_run(alerta.info.run_id).data.tags
        assert tags["retrain_consumed"] == str(versao)
    capsys.readouterr()

    # O segundo retreino sobre o mesmo lote exige o flag explícito (§4.7).
    assert retrain.main([]) == retrain.EXIT_REFUSED
    assert "--allow-reuse" in capsys.readouterr().err
    for alerta in alertas:
        argv = ["--alert-run", alerta.info.run_id]
        assert retrain.main(argv) == retrain.EXIT_REFUSED
        erro = capsys.readouterr().err
        assert "consumido" in erro
        assert "--allow-reuse" in erro

    # Um alerta novo do mesmo lote, gravado depois do retreino, também não habilita.
    _monitorar_t2(sandbox)
    assert retrain.main([]) == retrain.EXIT_REFUSED
    assert len(_retrain_runs(com_alerta)) == 1
    assert cliente.get_model_version_by_alias(nome, "challenger").version == versao


def test_alerta_de_outro_champion_recusa_sem_flag(sandbox, com_alerta, capsys):
    # Alerta emitido contra um modelo que já não é o champion (ex.: antes de um
    # `make promote`): o champion atual nunca ficou CRÍTICO.
    antigo = "retrain-000000000000"
    cliente_mlflow(com_alerta).set_tag(sandbox.alerta, "model_version", antigo)

    assert retrain.main([]) == retrain.EXIT_REFUSED
    erro = capsys.readouterr().err
    assert train.MODEL_VERSION in erro
    assert "--allow-reuse" in erro
    assert retrain.main(["--alert-run", sandbox.alerta]) == retrain.EXIT_REFUSED
    erro = capsys.readouterr().err
    assert antigo in erro
    assert train.MODEL_VERSION in erro
    assert "--allow-reuse" in erro
    assert not _retrain_runs(com_alerta)

    motivo = "o lote ainda representa a produção; retreinar sobre o alerta antigo"
    argv = ["--alert-run", sandbox.alerta, "--allow-reuse", "--reason", motivo]
    assert retrain.main(argv) == retrain.EXIT_OK

    (run,) = _retrain_runs(com_alerta)
    assert run.data.tags["alert_reused"] == "true"
    assert run.data.tags["alert_model_version"] == antigo
    assert run.data.tags["retrain_reason"] == motivo


def test_alert_run_de_lote_nao_critico_recusa(sandbox, com_alerta, capsys):
    assert retrain.main(["--alert-run", sandbox.lote_ok]) == retrain.EXIT_REFUSED
    assert "CRITICO" in capsys.readouterr().err
    assert retrain.main(["--alert-run", "nao-existe"]) == retrain.EXIT_REFUSED
    assert "não encontrado" in capsys.readouterr().err
    assert not _retrain_runs(com_alerta)


# MARK: retreino com o alerta


def test_retrain_nao_move_champion(sandbox, com_alerta):
    antes = _champion_version(com_alerta)
    sha_champion = sha256_de(config.CHAMPION_FILE)
    sha_baseline = sha256_de(sandbox.model)

    assert retrain.main([]) == retrain.EXIT_OK

    assert _champion_version(com_alerta) == antes
    assert sha256_de(config.CHAMPION_FILE) == sha_champion
    assert sha256_de(sandbox.model) == sha_baseline
    nome = config.REGISTERED_MODEL_NAME
    cliente = cliente_mlflow(com_alerta)
    challenger = cliente.get_model_version_by_alias(nome, "challenger")
    assert int(challenger.version) != antes
    aliases = cliente.get_registered_model(nome).aliases
    assert "previous" not in aliases


def test_retrain_marca_alerta_consumido_e_registra_tags(
    sandbox, com_alerta, tmp_path, capsys
):
    cliente = cliente_mlflow(com_alerta)
    nome = config.REGISTERED_MODEL_NAME
    champion = cliente.get_model_version_by_alias(nome, "champion")
    alerta = cliente.get_run(sandbox.alerta)

    assert retrain.main([]) == retrain.EXIT_OK

    challenger = cliente.get_model_version_by_alias(nome, "challenger")
    tags = challenger.tags
    assert cliente.get_run(sandbox.alerta).data.tags["retrain_consumed"] == str(
        challenger.version
    )
    assert tags["origin"] == "retrain"
    assert tags["alert_run_id"] == sandbox.alerta
    assert tags["reference_sha256"] == champion.tags["reference_sha256"]
    assert tags["batch_name"] == sandbox.t2.stem
    assert tags["batch_sha256"] == alerta.data.tags["batch_sha256"]
    assert tags["batch_sha256"] == sha256_de(sandbox.t2)
    assert tags["champion_version"] == str(champion.version)
    assert tags["bundle_model_version"].startswith("retrain-")
    assert tags["bundle_model_version"] != train.MODEL_VERSION
    assert tags["promotion_eligible"] in {"true", "false"}
    for chave in (
        "champion_holdout_auc",
        "challenger_holdout_auc",
        "champion_reference_auc",
        "challenger_reference_auc",
    ):
        assert 0 <= float(tags[chave]) <= 1

    (run,) = _retrain_runs(com_alerta)
    assert challenger.run_id == run.info.run_id
    assert tags["retrain_run_id"] == run.info.run_id
    assert run.data.tags["model_sha256"] == tags["model_sha256"]
    assert run.data.tags["trigger"] == "alert"
    assert run.data.tags["alert_reused"] == "false"
    metricas = run.data.metrics
    for chave in (
        "gain_on_holdout",
        "regression_on_reference",
        "champion_holdout_auc",
        "challenger_holdout_auc",
        "champion_reference_auc",
        "challenger_reference_auc",
        "promotion_eligible",
    ):
        assert chave in metricas
    assert metricas["gain_on_holdout"] == pytest.approx(
        metricas["challenger_holdout_auc"] - metricas["champion_holdout_auc"], abs=1e-8
    )
    assert run.data.params["promotion_min_gain"] == str(config.PROMOTION_MIN_GAIN)
    pai = cliente.get_run(run.data.tags["mlflow.parentRunId"])
    assert pai.data.tags["run_type"] == "stage"
    assert pai.data.tags["stage"] == "retrain"

    # O bundle anexado carrega a versão própria e o score_reference do challenger.
    # joblib (pickle) só em artefatos que o próprio teste acabou de gerar.
    baixado = cliente.download_artifacts(run.info.run_id, "bundle", str(tmp_path))
    bundle = joblib.load(f"{baixado}/baseline_model.joblib")
    with open(f"{baixado}/baseline_metadata.json", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert bundle["model_version"] == tags["bundle_model_version"]
    assert metadata["model_version"] == tags["bundle_model_version"]
    assert metadata["artifact"]["sha256"] == tags["model_sha256"]
    assert len(metadata["score_reference"]["shares"]) >= 2
    assert metadata["data_lineage"]["alert_run_id"] == sandbox.alerta

    saida = capsys.readouterr().out
    assert f"v{challenger.version}" in saida
    assert "make promote" in saida
    assert "ELEGÍVEL" in saida


def test_retrain_avalia_champion_e_challenger_no_mesmo_holdout(
    sandbox, com_alerta, tmp_path
):
    assert retrain.main([]) == retrain.EXIT_OK

    (run,) = _retrain_runs(com_alerta)
    metricas = run.data.metrics
    _, holdout = retrain.split_critical_batch(_t2_validado(sandbox))
    alvo = holdout["target"].to_numpy()
    campeao = predict.load_bundle(config.CHAMPION_FILE)["pipeline"]
    baixado = cliente_mlflow(com_alerta).download_artifacts(
        run.info.run_id, "bundle", str(tmp_path)
    )
    desafiante = joblib.load(f"{baixado}/baseline_model.joblib")["pipeline"]

    auc_campeao = roc_auc_score(alvo, campeao.predict_proba(model_input(holdout))[:, 1])
    auc_desafiante = roc_auc_score(
        alvo, desafiante.predict_proba(model_input(holdout))[:, 1]
    )
    assert metricas["holdout_rows"] == len(holdout)
    assert metricas["champion_holdout_auc"] == pytest.approx(auc_campeao, abs=1e-9)
    assert metricas["challenger_holdout_auc"] == pytest.approx(auc_desafiante, abs=1e-9)
    # Não é o AUC do lote inteiro (registrado pelo monitor no run do alerta).
    auc_lote = _lote(com_alerta, sandbox.t2.stem).data.metrics["roc_auc"]
    assert metricas["champion_holdout_auc"] != pytest.approx(auc_lote, abs=1e-6)


# MARK: falha fechada


def test_retrain_falha_fechada_com_tracking_off(
    sandbox, tmp_path, monkeypatch, mlflow_isolado, capsys
):
    banco = tmp_path / "mlflow" / "mlflow.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{banco}")
    monkeypatch.setenv("MONITORING_TRACKING", "off")

    assert retrain.main([]) == retrain.EXIT_OPERATIONAL_ERROR
    assert retrain.main(["--force-no-alert", "--reason", "x"]) == 1

    erro = capsys.readouterr().err
    assert "MONITORING_TRACKING=off" in erro
    assert "Registry" in erro
    assert not banco.exists()


def test_retrain_falha_fechada_com_mlflow_indisponivel(
    sandbox, tmp_path, monkeypatch, mlflow_isolado, capsys
):
    # Diretório que não pode ser criado (o mesmo dos testes do adaptador).
    monkeypatch.setattr(
        config, "MLFLOW_TRACKING_URI", "sqlite:////proc/inexistente/mlflow.db"
    )

    assert retrain.main([]) == retrain.EXIT_OPERATIONAL_ERROR

    erro = capsys.readouterr().err
    assert "falha fechada" in erro
    assert "Traceback" not in erro


def _sem_treino(monkeypatch):
    def _proibido(*args, **kwargs):
        raise AssertionError("o challenger não pode ser treinado neste cenário")

    monkeypatch.setattr(train, "fit_model", _proibido)


def test_retrain_contrato_falhando_aborta_antes_de_treinar(
    sandbox, sem_alerta, tmp_path, monkeypatch, capsys
):
    invalido = _lote_producao(seed=5, degradado=True)
    invalido.loc[invalido.index[:3], "age_years"] = 12.0
    caminho = tmp_path / "production_t2_invalido.parquet"
    invalido.to_parquet(caminho, index=False)
    monkeypatch.setattr(config, "PRODUCTION_T2_FILE", caminho)
    _sem_treino(monkeypatch)

    argv = ["--force-no-alert", "--reason", "lote inválido"]
    assert retrain.main(argv) == retrain.EXIT_BLOCKED

    assert "BLOQUEADO" in capsys.readouterr().err
    assert not _retrain_runs(sem_alerta)
    aliases = (
        cliente_mlflow(sem_alerta)
        .get_registered_model(config.REGISTERED_MODEL_NAME)
        .aliases
    )
    assert "challenger" not in aliases
    assert (retrain.RETRAIN_REPORTS_DIR / f"{caminho.stem}_report.json").is_file()


def test_retrain_champion_adulterado_aborta_antes_de_treinar(
    sandbox, com_alerta, tmp_path, monkeypatch
):
    pasta = tmp_path / "champion"
    shutil.copytree(sandbox.champion, pasta)
    with (pasta / "baseline_model.joblib").open("ab") as fh:
        fh.write(b"adulterado")
    monkeypatch.setattr(config, "CHAMPION_FILE", pasta / "baseline_model.joblib")
    _sem_treino(monkeypatch)

    with pytest.raises(SystemExit) as saida:
        retrain.main([])

    assert "Modelo recusado" in str(saida.value.code)
    assert not _retrain_runs(com_alerta)
    alerta = cliente_mlflow(com_alerta).get_run(sandbox.alerta)
    assert "retrain_consumed" not in alerta.data.tags


def test_retrain_referencia_diferente_da_do_champion_recusa(
    sandbox, com_alerta, tmp_path, monkeypatch, capsys
):
    outra = tmp_path / "reference.parquet"
    make_valid_df(n=3000, seed=4).to_parquet(outra, index=False)
    monkeypatch.setattr(config, "REFERENCE_FILE", outra)
    _sem_treino(monkeypatch)

    assert retrain.main([]) == retrain.EXIT_OPERATIONAL_ERROR

    assert "reference_sha256" in capsys.readouterr().err
    assert not _retrain_runs(com_alerta)


def test_retrain_lote_alterado_desde_o_alerta_recusa(
    sandbox, com_alerta, monkeypatch, capsys
):
    cliente_mlflow(com_alerta).set_tag(sandbox.alerta, "batch_sha256", "0" * 64)
    _sem_treino(monkeypatch)

    assert retrain.main([]) == retrain.EXIT_OPERATIONAL_ERROR

    assert "batch_sha256" in capsys.readouterr().err
    assert not _retrain_runs(com_alerta)


def test_alerta_sem_batch_path_falha_fechado(sandbox, com_alerta, monkeypatch, capsys):
    cliente_mlflow(com_alerta).delete_tag(sandbox.alerta, "batch_path")
    _sem_treino(monkeypatch)

    assert retrain.main([]) == retrain.EXIT_OPERATIONAL_ERROR

    assert "batch_path" in capsys.readouterr().err
    assert not _retrain_runs(com_alerta)


@pytest.mark.parametrize(
    ("lote", "trecho"),
    [
        ("sem_rotulo", "rotulado"),
        ("uma_classe", "duas classes"),
        ("ausente", "não encontrado"),
    ],
)
def test_lote_inutilizavel_para_retreino_falha_fechado(
    sandbox, sem_alerta, tmp_path, monkeypatch, lote, trecho, capsys
):
    caminho = tmp_path / f"{lote}.parquet"
    if lote == "sem_rotulo":
        _lote_producao(seed=5, rotulado=False).to_parquet(caminho, index=False)
    elif lote == "uma_classe":
        _lote_producao(seed=5).assign(target=0).to_parquet(caminho, index=False)
    monkeypatch.setattr(config, "PRODUCTION_T2_FILE", caminho)
    _sem_treino(monkeypatch)

    argv = ["--force-no-alert", "--reason", "lote de teste"]
    assert retrain.main(argv) == retrain.EXIT_OPERATIONAL_ERROR

    assert trecho in capsys.readouterr().err
    assert not _retrain_runs(sem_alerta)


def test_falha_no_registro_nao_consome_alerta(sandbox, com_alerta, monkeypatch, capsys):
    def _registro_falho(**kwargs):
        raise TrackingUnavailableError("Falha simulada ao registrar o challenger")

    monkeypatch.setattr(registry, "register_challenger", _registro_falho)

    assert retrain.main([]) == retrain.EXIT_OPERATIONAL_ERROR

    assert "Falha simulada" in capsys.readouterr().err
    alerta = cliente_mlflow(com_alerta).get_run(sandbox.alerta)
    assert "retrain_consumed" not in alerta.data.tags
    aliases = (
        cliente_mlflow(com_alerta)
        .get_registered_model(config.REGISTERED_MODEL_NAME)
        .aliases
    )
    assert "challenger" not in aliases
