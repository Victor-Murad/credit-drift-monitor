"""Gancho de lote da demo do contrato e estágio `contract_demo` (spec da Etapa 3, §4.1-§4.2).

Dados SINTÉTICOS (sem Kaggle) e MLflow real num banco temporário. O lote válido da demo
(500 linhas, sem rótulo) é APROVADO e pontuado; o inválido é BLOQUEADO de propósito
(`intentional_demo=true`) e conta em `alerts_failure_count` sem derrubar o estágio.
O aceite com os dados reais é a Tarefa 11.
"""

from __future__ import annotations

import shutil
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest

from src import config
from src.models import predict, train
from src.monitoring import stage
from src.monitoring.alerts import AlertLevel
from src.validation import demo as contract_demo
from src.validation.gate import enforce_contract
from tests.conftest import cliente_mlflow, isolar_estado_mlflow, make_valid_df

FAST_XGB = {"n_estimators": 20, "n_jobs": 1}
N_PRODUCAO = 2000
LOTE_VALIDO = "valid_batch"
LOTE_INVALIDO = "invalid_batch"


# --------------------------------------------------------------------------
# Gancho `on_batch` de validation.demo.main
# --------------------------------------------------------------------------
@pytest.fixture
def demo_sintetica(tmp_path, monkeypatch):
    """`validation.demo` com todos os caminhos em `tmp_path` e o treino já feito."""
    raiz = tmp_path / "demo"
    producao = raiz / "production_base.parquet"
    producao.parent.mkdir(parents=True)
    make_valid_df(n=N_PRODUCAO, seed=5).to_parquet(producao, index=False)
    reference = raiz / "reference.parquet"
    make_valid_df(n=3000, seed=3).to_parquet(reference, index=False)
    modelo = raiz / "baseline_model.joblib"
    relatorios = raiz / "reports"
    quarentena = raiz / "quarantine"

    gate = partial(enforce_contract, report_dir=relatorios, quarantine_dir=quarentena)
    monkeypatch.setattr(train, "REFERENCE_FILE", reference)
    monkeypatch.setattr(train, "MODEL_FILE", modelo)
    monkeypatch.setattr(train, "MODEL_METADATA_FILE", raiz / "baseline_metadata.json")
    monkeypatch.setattr(train, "MODEL_REPORTS_DIR", raiz / "model_reports")
    monkeypatch.setattr(train, "enforce_contract", gate)
    monkeypatch.setattr(train, "XGB_PARAMS", {**train.XGB_PARAMS, **FAST_XGB})
    train.main()

    monkeypatch.setattr(contract_demo, "PRODUCTION_BASE_FILE", producao)
    monkeypatch.setattr(contract_demo, "VALID_BATCH_FILE", raiz / "valid_batch.csv")
    monkeypatch.setattr(contract_demo, "INVALID_BATCH_FILE", raiz / "invalid_batch.csv")
    monkeypatch.setattr(contract_demo, "CONTRACT_REPORTS_DIR", relatorios)
    monkeypatch.setattr(contract_demo, "MODEL_FILE", modelo)
    monkeypatch.setattr(
        contract_demo,
        "load_bundle",
        partial(predict.load_bundle, modelo),
    )
    monkeypatch.setattr(
        contract_demo,
        "score_batch",
        partial(predict.score_batch, report_dir=relatorios, quarantine_dir=quarentena),
    )
    return SimpleNamespace(raiz=raiz, relatorios=relatorios)


def test_demo_on_batch_recebe_os_dois_lotes_na_ordem(demo_sintetica):
    chamadas = []

    status = contract_demo.main(on_batch=lambda *args: chamadas.append(args))

    assert status == 0
    assert [(c[0], c[1]) for c in chamadas] == [
        (LOTE_VALIDO, "APROVADO"),
        (LOTE_INVALIDO, "BLOQUEADO"),
    ]
    (_, _, resultado_valido, n_valido), (_, _, resultado_invalido, n_invalido) = (
        chamadas
    )
    assert resultado_valido is None
    assert n_valido == contract_demo.BATCH_SIZE
    assert resultado_invalido.rules_violated
    # O lote inválido tem linhas a mais que o válido (duplicatas injetadas).
    assert resultado_invalido.n_rows > contract_demo.BATCH_SIZE
    assert n_invalido == 0


def test_demo_sem_on_batch_continua_igual(demo_sintetica):
    assert contract_demo.main() == 0
    resumo = (demo_sintetica.relatorios / "contract_demo_summary.md").read_text("utf-8")
    assert "**BLOQUEADO**" in resumo


# --------------------------------------------------------------------------
# Observador (unidade, sem MLflow)
# --------------------------------------------------------------------------
class _RunFalso:
    """Captura os `BatchRecord` em vez de abrir runs no MLflow."""

    enabled = True

    def __init__(self) -> None:
        self.records = []

    def log_batch(self, record):
        self.records.append(record)
        return None


def test_lote_vazio_nao_gera_run_com_metricas():
    """Review Focus 2: 0 linhas pontuadas e sem resultado não vira lote `OK`."""
    run = _RunFalso()
    observador = stage.ContractBatchObserver(run)

    with pytest.raises(stage.StageFailedError, match=LOTE_VALIDO):
        observador(LOTE_VALIDO, "APROVADO", None, 0)

    assert run.records == []


def test_bloqueado_sem_resultado_do_contrato_e_erro_explicito():
    run = _RunFalso()
    observador = stage.ContractBatchObserver(run)

    with pytest.raises(stage.StageFailedError, match=LOTE_INVALIDO):
        observador(LOTE_INVALIDO, "BLOQUEADO", None, 0)
    with pytest.raises(stage.StageFailedError, match="ABORTADO"):
        observador(LOTE_INVALIDO, "ABORTADO", None, 0)

    assert run.records == []


def test_aprovado_registra_so_o_resultado_do_contrato():
    run = _RunFalso()
    observador = stage.ContractBatchObserver(run)

    observador(LOTE_VALIDO, "APROVADO", None, 500)

    (record,) = run.records
    assert record.decision.level is AlertLevel.OK
    assert record.decision.performance_evaluated is False
    assert record.n_rows == 500
    assert record.rows_rejected == 0
    assert record.intentional_demo is False
    assert not any(k in record.metrics for k in ("roc_auc", "auc_drop", "score_psi"))


# --------------------------------------------------------------------------
# Estágio completo em sandbox (train + contract_demo com MLflow temporário)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sandbox(tmp_path_factory, _banco_modelo_mlflow):
    root = tmp_path_factory.mktemp("contract-stage")
    p = SimpleNamespace(
        root=root,
        banco=root / "mlflow.db",
        reference=root / "reference" / "reference.parquet",
        production=root / "production" / "production_base.parquet",
        model=root / "models" / "baseline_model.joblib",
        metadata=root / "models" / "baseline_metadata.json",
        champion=root / "models" / "champion",
        contract_reports=root / "reports" / "data_contract",
        quarantine=root / "quarantine",
        valid_batch=root / "samples" / "valid_batch.csv",
        invalid_batch=root / "samples" / "invalid_batch.csv",
    )
    shutil.copy(_banco_modelo_mlflow, p.banco)
    p.uri = f"sqlite:///{p.banco}"
    p.reference.parent.mkdir(parents=True)
    p.production.parent.mkdir(parents=True)
    make_valid_df(n=3000, seed=3).to_parquet(p.reference, index=False)
    make_valid_df(n=N_PRODUCAO, seed=5).to_parquet(p.production, index=False)

    gate = partial(
        enforce_contract, report_dir=p.contract_reports, quarantine_dir=p.quarantine
    )
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
        # train
        mp.setattr(train, "REFERENCE_FILE", p.reference)
        mp.setattr(train, "MODEL_FILE", p.model)
        mp.setattr(train, "MODEL_METADATA_FILE", p.metadata)
        mp.setattr(train, "MODEL_REPORTS_DIR", root / "reports" / "model")
        mp.setattr(train, "enforce_contract", gate)
        mp.setattr(train, "XGB_PARAMS", {**train.XGB_PARAMS, **FAST_XGB})
        # demo do contrato
        mp.setattr(contract_demo, "PRODUCTION_BASE_FILE", p.production)
        mp.setattr(contract_demo, "VALID_BATCH_FILE", p.valid_batch)
        mp.setattr(contract_demo, "INVALID_BATCH_FILE", p.invalid_batch)
        mp.setattr(contract_demo, "CONTRACT_REPORTS_DIR", p.contract_reports)
        mp.setattr(contract_demo, "MODEL_FILE", p.model)
        mp.setattr(
            contract_demo,
            "score_batch",
            partial(
                predict.score_batch,
                report_dir=p.contract_reports,
                quarantine_dir=p.quarantine,
            ),
        )

        assert stage.main(["contract_demo"]) == 1  # ainda sem champion
        assert stage.main(["train"]) == 0
        p.status = stage.main(["contract_demo"])
        yield p


def _runs(uri: str, filtro: str):
    client = cliente_mlflow(uri)
    experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    return client.search_runs(
        [experimento.experiment_id],
        filter_string=filtro,
        order_by=["attributes.start_time ASC"],
    )


def _estagio(uri: str):
    """O run do estágio `contract_demo` que terminou bem (o único da fixture)."""
    (estagio,) = _runs(
        uri,
        "tags.run_type = 'stage' and tags.stage = 'contract_demo' "
        "and tags.stage_status = 'SUCCESS'",
    )
    return estagio


def _filhos(uri: str) -> dict:
    estagio = _estagio(uri)
    filhos = _runs(
        uri,
        f"tags.run_type = 'batch' and tags.mlflow.parentRunId = '{estagio.info.run_id}'",
    )
    return {f.data.tags["batch_name"]: f for f in filhos}


def test_contract_demo_gera_run_bloqueado(sandbox):
    assert sandbox.status == 0
    filhos = _filhos(sandbox.uri)
    assert list(filhos) == [LOTE_VALIDO, LOTE_INVALIDO]  # ordem de início

    valido = filhos[LOTE_VALIDO].data
    assert valido.tags["alert_level"] == "OK"
    assert valido.tags["intentional_demo"] == "false"
    assert valido.tags["performance_evaluated"] == "false"

    invalido = filhos[LOTE_INVALIDO].data
    assert invalido.tags["alert_level"] == "BLOQUEADO"
    assert invalido.tags["alert_label"] == "BLOQUEADO"
    assert invalido.tags["intentional_demo"] == "true"
    assert invalido.tags["alert_action"].startswith("Inspecionar o relatório")
    # Motivos = regras violadas do contrato (R2, R3...), separadas por ";".
    motivos = invalido.tags["alert_reasons"].split(";")
    assert len(motivos) >= 2
    assert all(motivo[0] == "R" for motivo in motivos)
    for filho in filhos.values():
        assert filho.data.tags["model_version"] == train.MODEL_VERSION
        assert len(filho.data.tags["batch_sha256"]) == 64
        assert "roc_auc" not in filho.data.metrics
        assert "score_psi" not in filho.data.metrics


def test_contract_demo_anexa_relatorio_do_contrato_a_cada_lote(sandbox):
    client = cliente_mlflow(sandbox.uri)
    for nome, filho in _filhos(sandbox.uri).items():
        artefatos = {
            a.path for a in client.list_artifacts(filho.info.run_id, "reports")
        }
        assert f"reports/{nome}_report.html" in artefatos
        assert f"reports/{nome}_report.json" in artefatos


def test_contract_demo_estagio_continua_com_sucesso(sandbox):
    estagio = _estagio(sandbox.uri)
    assert estagio.info.status == "FINISHED"
    assert estagio.data.metrics["stage_success"] == 1
    assert estagio.data.tags["stage_status"] == "SUCCESS"
    assert estagio.data.tags["intentional_demo"] == "true"


def test_bloqueado_incrementa_alerts_failure_count_no_run_do_lote(sandbox):
    """Spec §4.1: o bloqueio intencional é um alerta real, mesmo com a demo."""
    metricas = _estagio(sandbox.uri).data.metrics
    assert metricas["alerts_failure_count"] == 1
    assert metricas["alerts_drift_count"] == 0
    # rows_rejected = linhas do lote BLOQUEADO (o gate é tudo ou nada).
    linhas_invalido = len(pd.read_csv(sandbox.invalid_batch))
    assert linhas_invalido > contract_demo.BATCH_SIZE
    assert metricas["rows_rejected"] == linhas_invalido
    assert metricas["rows_processed"] == contract_demo.BATCH_SIZE + linhas_invalido
    filhos = _filhos(sandbox.uri)
    assert filhos[LOTE_INVALIDO].data.metrics["rows_rejected"] == linhas_invalido
    assert filhos[LOTE_INVALIDO].data.metrics["n_rows"] == linhas_invalido
    assert filhos[LOTE_VALIDO].data.metrics["rows_rejected"] == 0
    assert filhos[LOTE_VALIDO].data.metrics["n_rows"] == contract_demo.BATCH_SIZE


def test_contract_demo_sem_champion_falha_com_orientacao(sandbox, tracking_env, capsys):
    # Banco novo e vazio (`tracking_env`): não há champion registrado.
    assert stage.main(["contract_demo"]) == 1
    saida = capsys.readouterr().err
    assert "ERRO no estágio contract_demo" in saida
    assert "champion" in saida
    assert "make pipeline" in saida
    assert "Traceback" not in saida


def test_contract_demo_falha_da_demo_derruba_o_estagio(sandbox, monkeypatch, capsys):
    # Nenhum defeito é "detectado": a demo devolve 1 e o estágio tem de falhar.
    monkeypatch.setattr(contract_demo, "was_detected", lambda *_: False)
    antes = len(_runs(sandbox.uri, "tags.stage = 'contract_demo'"))

    assert stage.main(["contract_demo"]) == 1

    assert "ERRO no estágio contract_demo" in capsys.readouterr().err
    novos = _runs(sandbox.uri, "tags.stage = 'contract_demo'")[antes:]
    (falho,) = novos
    assert falho.data.tags["stage_status"] == "FAILED"
    assert falho.data.metrics["stage_success"] == 0


def test_contract_demo_tracking_off_roda_sem_mlflow(
    sandbox, tmp_path, monkeypatch, mlflow_isolado
):
    banco = tmp_path / "mlflow" / "mlflow.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{banco}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    monkeypatch.setenv("MONITORING_TRACKING", "off")

    assert stage.main(["contract_demo"]) == 0
    assert not banco.exists()
    assert not (tmp_path / "mlartifacts").exists()
