"""Comando `monitor`: avaliação de UM lote (spec da Etapa 3, §4.2, §4.3 e §4.8).

Dados SINTÉTICOS (sem Kaggle) e MLflow real num banco temporário: o `train` roda uma vez
no módulo (bootstrap do champion) e cada teste avalia um lote gravado em `tmp_path`, com
um nome próprio (o `batch_name` é o nome do arquivo). O aceite com os dados reais é a
Tarefa 11.
"""

from __future__ import annotations

import shutil
from functools import partial
from types import SimpleNamespace

import pytest

from src import config
from src.models import train
from src.monitoring import monitor, stage
from src.validation.gate import enforce_contract
from tests.conftest import cliente_mlflow, isolar_estado_mlflow, make_valid_df

FAST_XGB = {"n_estimators": 20, "n_jobs": 1}
N_LOTE = 1500


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory, _banco_modelo_mlflow):
    root = tmp_path_factory.mktemp("monitor")
    p = SimpleNamespace(
        root=root,
        banco=root / "mlflow.db",
        reference=root / "reference" / "reference.parquet",
        model=root / "models" / "baseline_model.joblib",
        metadata=root / "models" / "baseline_metadata.json",
        champion=root / "models" / "champion",
        reports=root / "reports" / "monitor",
        contract_reports=root / "reports" / "data_contract",
        quarantine=root / "quarantine",
    )
    shutil.copy(_banco_modelo_mlflow, p.banco)
    p.uri = f"sqlite:///{p.banco}"
    p.reference.parent.mkdir(parents=True)
    make_valid_df(n=3000, seed=3).to_parquet(p.reference, index=False)

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
        mp.setattr(monitor, "MONITOR_REPORTS_DIR", p.reports)
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
                report_dir=p.contract_reports,
                quarantine_dir=p.quarantine,
            ),
        )
        mp.setattr(train, "XGB_PARAMS", {**train.XGB_PARAMS, **FAST_XGB})

        assert stage.main(["train"]) == 0
        yield p


def _gravar(pasta, nome: str, *, rotulado: bool = True, seed: int = 5, **ajustes):
    """Lote sintético estável em `pasta/<nome>.parquet` (o `batch_name` é `nome`)."""
    df = make_valid_df(n=N_LOTE, seed=seed, labeled=rotulado)
    for coluna, valor in ajustes.items():
        df[coluna] = valor(df) if callable(valor) else valor
    caminho = pasta / f"{nome}.parquet"
    df.to_parquet(caminho, index=False)
    return caminho


def _runs(uri: str, filtro: str):
    client = cliente_mlflow(uri)
    experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    return client.search_runs(
        [experimento.experiment_id],
        filter_string=filtro,
        order_by=["attributes.start_time ASC"],
    )


def _lote(uri: str, nome: str):
    """O run de lote (único) com `batch_name=nome` e o run de estágio pai."""
    (lote,) = _runs(uri, f"tags.run_type = 'batch' and tags.batch_name = '{nome}'")
    pai = cliente_mlflow(uri).get_run(lote.data.tags["mlflow.parentRunId"])
    return lote, pai


def test_monitor_lote_sem_rotulo_ok(sandbox, tmp_path, capsys):
    caminho = _gravar(tmp_path, "lote_sem_rotulo", rotulado=False)

    assert monitor.main(["--batch", str(caminho)]) == 0

    assert "OK" in capsys.readouterr().out
    lote, estagio = _lote(sandbox.uri, "lote_sem_rotulo")
    tags, metricas = lote.data.tags, lote.data.metrics
    assert tags["alert_level"] == "OK"
    assert tags["performance_evaluated"] == "false"
    assert tags["intentional_demo"] == "false"
    assert tags["model_version"] == train.MODEL_VERSION
    assert len(tags["batch_sha256"]) == 64
    assert tags["batch_path"].endswith("lote_sem_rotulo.parquet")
    assert "roc_auc" not in metricas and "auc_drop" not in metricas
    assert metricas["score_psi"] < config.PSI_MODERATE
    assert metricas["n_rows"] == N_LOTE
    assert "psi_ext_source_3" in metricas
    assert "approval_rate" in metricas
    # Relatório do Evidently anexado ao run do lote.
    artefatos = {
        a.path
        for a in cliente_mlflow(sandbox.uri).list_artifacts(lote.info.run_id, "reports")
    }
    assert "reports/lote_sem_rotulo_drift_report.html" in artefatos
    # Run de estágio `monitor` com o `pipeline_run_id`.
    assert estagio.data.tags["run_type"] == "stage"
    assert estagio.data.tags["stage"] == "monitor"
    assert estagio.data.tags["stage_status"] == "SUCCESS"
    assert estagio.data.tags["pipeline_run_id"]
    assert estagio.data.metrics["rows_processed"] == N_LOTE
    assert estagio.data.metrics["alerts_failure_count"] == 0


def test_monitor_lote_invalido_e_bloqueado(sandbox, tmp_path, capsys):
    # R3: idade fora do intervalo do contrato.
    caminho = _gravar(tmp_path, "lote_invalido", rotulado=False, age_years=5.0)

    assert monitor.main(["--batch", str(caminho)]) == 3

    assert "BLOQUEADO" in capsys.readouterr().out
    lote, estagio = _lote(sandbox.uri, "lote_invalido")
    tags, metricas = lote.data.tags, lote.data.metrics
    assert tags["alert_level"] == "BLOQUEADO"
    assert tags["intentional_demo"] == "false"
    assert tags["performance_evaluated"] == "false"
    motivos = tags["alert_reasons"].split(";")
    assert motivos and all(m.startswith("R") for m in motivos)
    # Nenhuma pontuação: nem score, nem AUC, nem drift.
    assert not {"score_psi", "roc_auc", "approval_rate"} & set(metricas)
    assert metricas["rows_rejected"] == N_LOTE
    assert estagio.data.metrics["alerts_failure_count"] == 1
    assert estagio.data.metrics["rows_rejected"] == N_LOTE
    assert estagio.data.tags["stage_status"] == "SUCCESS"
    assert "intentional_demo" not in estagio.data.tags
    artefatos = {
        a.path
        for a in cliente_mlflow(sandbox.uri).list_artifacts(lote.info.run_id, "reports")
    }
    assert "reports/lote_invalido_report.html" in artefatos


def test_monitor_rotulado_sem_baseline_falha_fechado(sandbox, tmp_path, capsys):
    caminho = _gravar(tmp_path, "lote_rotulado_sem_baseline")

    assert monitor.main(["--batch", str(caminho)]) == 1

    saida = capsys.readouterr().err
    assert "make demo-drift" in saida
    assert "--baseline-auc" in saida
    assert "Traceback" not in saida
    assert not _runs(
        sandbox.uri,
        "tags.run_type = 'batch' and tags.batch_name = 'lote_rotulado_sem_baseline'",
    )


def test_monitor_rotulado_degradado_e_critico(sandbox, tmp_path):
    # Rótulos invertidos: o AUC do lote despenca (muito mais que 0,05 abaixo de 0,75).
    caminho = _gravar(tmp_path, "lote_degradado", target=lambda d: 1 - d["target"])

    assert monitor.main(["--batch", str(caminho), "--baseline-auc", "0.75"]) == 2

    lote, estagio = _lote(sandbox.uri, "lote_degradado")
    tags, metricas = lote.data.tags, lote.data.metrics
    assert tags["alert_level"] == "CRITICO"
    assert tags["performance_evaluated"] == "true"
    assert "auc_drop>=0.05" in tags["alert_reasons"].split(";")
    assert metricas["auc_drop"] == pytest.approx(0.75 - metricas["roc_auc"], abs=1e-4)
    assert estagio.data.metrics["alerts_drift_count"] == 1


def test_monitor_classe_unica_nao_quebra(sandbox, tmp_path):
    """Review Focus 1: rótulo de uma só classe não é exceção nem exige baseline."""
    caminho = _gravar(tmp_path, "lote_classe_unica", target=0)

    assert monitor.main(["--batch", str(caminho)]) == 0

    lote, _ = _lote(sandbox.uri, "lote_classe_unica")
    tags = lote.data.tags
    assert tags["performance_evaluated"] == "false"
    assert "single_class" in tags["alert_reasons"].split(";")
    assert tags["alert_level"] in {"OK", "ATENCAO"}
    assert "roc_auc" not in lote.data.metrics


def test_monitor_no_labeled_ignora_o_rotulo(sandbox, tmp_path):
    caminho = _gravar(tmp_path, "lote_rotulo_ignorado")

    assert monitor.main(["--batch", str(caminho), "--no-labeled"]) == 0

    lote, _ = _lote(sandbox.uri, "lote_rotulo_ignorado")
    assert lote.data.tags["performance_evaluated"] == "false"
    assert "psi_target" not in lote.data.metrics
    assert "label_shift" not in lote.data.metrics


def test_monitor_labeled_sem_coluna_target_e_erro(sandbox, tmp_path, capsys):
    caminho = _gravar(tmp_path, "lote_sem_target", rotulado=False)

    assert monitor.main(["--batch", str(caminho), "--labeled"]) == 1

    assert "target" in capsys.readouterr().err


@pytest.fixture
def banco_copiado(sandbox, tmp_path, monkeypatch, mlflow_isolado):
    """Cópia do banco do sandbox (com o champion): runs gravados aqui não vazam."""
    banco = tmp_path / "mlflow.db"
    shutil.copy(sandbox.banco, banco)
    uri = f"sqlite:///{banco}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    return uri


def test_monitor_baseline_vem_do_run_production_t0(banco_copiado, tmp_path):
    pasta_t0 = tmp_path / "t0"
    pasta_t0.mkdir()
    t0 = _gravar(pasta_t0, "production_t0")
    assert monitor.main(["--batch", str(t0), "--baseline-auc", "0.75"]) in {0, 2}
    (run_t0,) = _runs(
        banco_copiado, "tags.run_type = 'batch' and tags.batch_name = 'production_t0'"
    )
    auc_t0 = run_t0.data.metrics["roc_auc"]

    degradado = _gravar(tmp_path, "lote_vs_t0", target=lambda d: 1 - d["target"])
    assert monitor.main(["--batch", str(degradado)]) == 2

    lote, _ = _lote(banco_copiado, "lote_vs_t0")
    metricas = lote.data.metrics
    assert metricas["auc_drop"] == pytest.approx(auc_t0 - metricas["roc_auc"], abs=1e-4)


def test_monitor_baseline_ignora_t0_de_outro_modelo(banco_copiado, tmp_path, capsys):
    pasta_t0 = tmp_path / "t0"
    pasta_t0.mkdir()
    t0 = _gravar(pasta_t0, "production_t0")
    assert monitor.main(["--batch", str(t0), "--baseline-auc", "0.75"]) in {0, 2}
    client = cliente_mlflow(banco_copiado)
    (run_t0,) = _runs(
        banco_copiado, "tags.run_type = 'batch' and tags.batch_name = 'production_t0'"
    )
    client.set_tag(run_t0.info.run_id, "model_version", "outro-modelo")
    capsys.readouterr()

    rotulado = _gravar(tmp_path, "lote_sem_t0_do_modelo")
    assert monitor.main(["--batch", str(rotulado)]) == 1
    assert "make demo-drift" in capsys.readouterr().err


def test_monitor_sem_champion_falha_com_orientacao(
    sandbox, tracking_env, tmp_path, capsys
):
    caminho = _gravar(tmp_path, "lote_sem_champion", rotulado=False)

    assert monitor.main(["--batch", str(caminho)]) == 1

    saida = capsys.readouterr().err
    assert "champion" in saida
    assert "make pipeline" in saida
    assert "Traceback" not in saida


def test_monitor_lote_inexistente_e_erro_operacional(sandbox, tmp_path, capsys):
    assert monitor.main(["--batch", str(tmp_path / "nao_existe.parquet")]) == 1
    assert "nao_existe.parquet" in capsys.readouterr().err


def test_monitor_lote_vazio_e_erro_operacional(sandbox, tmp_path, capsys):
    caminho = tmp_path / "lote_vazio.parquet"
    make_valid_df(n=N_LOTE, labeled=False).iloc[0:0].to_parquet(caminho, index=False)

    assert monitor.main(["--batch", str(caminho)]) == 1

    assert "vazio" in capsys.readouterr().err
    assert not _runs(
        sandbox.uri, "tags.run_type = 'batch' and tags.batch_name = 'lote_vazio'"
    )


@pytest.mark.parametrize("valor", ["abc", "nan", "0", "1.5"])
def test_monitor_baseline_auc_invalido_sai_com_1(sandbox, tmp_path, valor, capsys):
    caminho = tmp_path / "qualquer.parquet"
    with pytest.raises(SystemExit) as saida:
        monitor.main(["--batch", str(caminho), "--baseline-auc", valor])
    # Erro de uso é operacional (1): o 2 é reservado ao nível CRÍTICO.
    assert saida.value.code == 1
    assert "--baseline-auc" in capsys.readouterr().err


def test_monitor_tracking_off_roda_sem_mlflow(
    sandbox, tmp_path, monkeypatch, mlflow_isolado, capsys
):
    banco = tmp_path / "mlflow" / "mlflow.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{banco}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    monkeypatch.setenv("MONITORING_TRACKING", "off")

    sem_rotulo = _gravar(tmp_path, "lote_off", rotulado=False)
    assert monitor.main(["--batch", str(sem_rotulo)]) == 0

    degradado = _gravar(tmp_path, "lote_off_rotulado", target=lambda d: 1 - d["target"])
    capsys.readouterr()
    assert monitor.main(["--batch", str(degradado)]) == 1
    assert "--baseline-auc" in capsys.readouterr().err
    assert monitor.main(["--batch", str(degradado), "--baseline-auc", "0.75"]) == 2

    assert not banco.exists()
    assert not (tmp_path / "mlartifacts").exists()
