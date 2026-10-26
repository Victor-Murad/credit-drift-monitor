"""Adaptador do MLflow: runs de estágio e de lote (spec da Etapa 3, §4.1, §4.3, §4.4)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src import config
from src.monitoring import tracking
from src.monitoring.alerts import ACTIONS, AlertDecision, AlertLevel
from src.monitoring.tracking import (
    BatchRecord,
    NullStageRun,
    TrackingUnavailableError,
    configure_tracking,
    redact_uri,
    tracked_stage,
    tracking_enabled,
)


@pytest.fixture(autouse=True)
def _sem_estado_mlflow(mlflow_isolado):
    """Nenhum run ativo, URI, experimento ou variável do MLflow vaza entre testes."""
    yield


@pytest.fixture
def log_file(tmp_path, monkeypatch) -> Path:
    """Log JSONL do pipeline isolado por teste (lido por `tracked_stage` na chamada)."""
    caminho = tmp_path / "pipeline.jsonl"
    monkeypatch.setattr(config, "PIPELINE_LOG_FILE", caminho)
    return caminho


def _client(uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _decision(level: AlertLevel, reasons: tuple[str, ...] = (), evaluated=True):
    return AlertDecision(level, reasons, evaluated, ACTIONS[level])


def _record(name: str, level: AlertLevel, **kwargs) -> BatchRecord:
    padrao = {
        "batch_name": name,
        "decision": _decision(level),
        "n_rows": 100,
        "rows_rejected": 100 if level is AlertLevel.BLOQUEADO else 0,
        "metrics": {},
        "psi_by_column": {},
        "n_drifted_columns": 0,
        "model_version": "1",
    }
    padrao.update(kwargs)
    return BatchRecord(**padrao)


def _escrever_evento(log_file: Path, run_id: str, event: str) -> None:
    linha = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "level": "INFO",
        "event": event,
    }
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(linha) + "\n")


def test_tracking_enabled_le_o_ambiente_na_chamada(monkeypatch):
    monkeypatch.delenv("MONITORING_TRACKING", raising=False)
    assert tracking_enabled()
    monkeypatch.setenv("MONITORING_TRACKING", "off")
    assert not tracking_enabled()
    monkeypatch.setenv("MONITORING_TRACKING", "on")
    assert tracking_enabled()


def test_configure_tracking_cria_experimento_com_artifact_root(tracking_env):
    experiment_id = configure_tracking()
    experimento = _client(tracking_env).get_experiment(experiment_id)
    assert experimento.name == config.MLFLOW_EXPERIMENT
    assert experimento.artifact_location == config.MLFLOW_ARTIFACT_ROOT.as_uri()
    # Idempotente: a segunda chamada reaproveita o experimento.
    assert configure_tracking() == experiment_id


def test_stage_sucesso_grava_metricas_e_tags(tracking_env, log_file, tmp_path):
    avulso = tmp_path / "resumo.json"
    avulso.write_text("{}", encoding="utf-8")
    pasta = tmp_path / "relatorios"
    pasta.mkdir()
    (pasta / "a.html").write_text("<html></html>", encoding="utf-8")
    with tracked_stage("train", pipeline_run_id="pipe-1") as run:
        run.add_rows(10, rejected=2)
        run.set_metric("custom", 1.5)
        run.attach(avulso)
        run.attach(pasta, artifact_path="reports")
    client = _client(tracking_env)
    raiz = {a.path for a in client.list_artifacts(run.run_id)}
    assert {"resumo.json", "reports", "logs"} <= raiz
    assert [
        a.path for a in client.list_artifacts(run.run_id, "reports/relatorios")
    ] == ["reports/relatorios/a.html"]
    dados = client.get_run(run.run_id).data
    assert dados.metrics["stage_success"] == 1
    assert dados.metrics["stage_duration_s"] > 0
    assert dados.metrics["peak_memory_mb"] > 0
    assert dados.metrics["rows_processed"] == 10
    assert dados.metrics["rows_rejected"] == 2
    assert dados.metrics["alerts_failure_count"] == 0
    assert dados.metrics["alerts_drift_count"] == 0
    assert dados.metrics["custom"] == 1.5
    assert dados.tags["run_type"] == "stage"
    assert dados.tags["pipeline_run_id"] == "pipe-1"
    assert dados.tags["stage"] == "train"
    assert dados.tags["stage_status"] == "SUCCESS"
    assert mlflow.active_run() is None


@pytest.mark.parametrize(
    "erro", [ValueError("falhou de propósito"), SystemExit("saída de propósito")]
)
def test_stage_falha_grava_e_repropaga(tracking_env, log_file, erro):
    with pytest.raises(type(erro)) as capturada:
        with tracked_stage("prepare", pipeline_run_id="pipe-2") as run:
            raise erro
    assert capturada.value is erro
    info = _client(tracking_env).get_run(run.run_id)
    assert info.info.status == "FAILED"
    assert info.data.metrics["stage_success"] == 0
    assert info.data.metrics["alerts_failure_count"] >= 1
    assert info.data.tags["stage_status"] == "FAILED"
    assert mlflow.active_run() is None


def test_log_batch_cria_run_filho_com_nivel_e_motivos(tracking_env, log_file, tmp_path):
    lote = tmp_path / "lote.parquet"
    lote.write_bytes(b"conteudo do lote")
    relatorio = tmp_path / "relatorio.html"
    relatorio.write_text("<html></html>", encoding="utf-8")
    record = _record(
        "production_t2",
        AlertLevel.CRITICO,
        decision=_decision(
            AlertLevel.CRITICO, ("auc_drop>=0.05", "drift:ext_source_3")
        ),
        metrics={"roc_auc": 0.57, "auc_drop": 0.18, "score_psi": 0.03},
        psi_by_column={"ext_source_3": 0.31},
        n_drifted_columns=1,
        artifacts=[relatorio],
        batch_path=lote,
    )
    with tracked_stage("drift_demo", pipeline_run_id="pipe-3") as run:
        filho_id = run.log_batch(record)

    client = _client(tracking_env)
    filho = client.get_run(filho_id)
    tags, metricas = filho.data.tags, filho.data.metrics
    assert tags["mlflow.parentRunId"] == run.run_id
    assert tags["run_type"] == "batch"
    assert tags["alert_level"] == "CRITICO"
    assert tags["alert_label"] == "CRÍTICO"
    assert tags["alert_reasons"] == "auc_drop>=0.05;drift:ext_source_3"
    assert tags["batch_name"] == "production_t2"
    assert tags["batch_sha256"] == hashlib.sha256(b"conteudo do lote").hexdigest()
    assert tags["batch_path"].endswith("lote.parquet")
    assert tags["model_version"] == "1"
    assert tags["performance_evaluated"] == "true"
    assert tags["intentional_demo"] == "false"
    assert metricas["roc_auc"] == 0.57
    assert metricas["auc_drop"] == 0.18
    assert metricas["score_psi"] == 0.03
    assert metricas["psi_ext_source_3"] == 0.31
    assert metricas["n_drifted_columns"] == 1
    assert metricas["n_rows"] == 100
    artefatos = [a.path for a in client.list_artifacts(filho_id, "reports")]
    assert artefatos == ["reports/relatorio.html"]

    experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    achados = client.search_runs(
        [experimento.experiment_id], filter_string="tags.alert_level = 'CRITICO'"
    )
    assert [r.info.run_id for r in achados] == [filho_id]


def test_contadores_de_alerta(tracking_env, log_file):
    with tracked_stage("drift_demo", pipeline_run_id="pipe-4") as run:
        run.log_batch(_record("t1", AlertLevel.ATENCAO))
        run.log_batch(_record("t2", AlertLevel.CRITICO))
        run.log_batch(_record("invalido", AlertLevel.BLOQUEADO, n_rows=8))
        run.log_batch(_record("t0", AlertLevel.OK))
    metricas = _client(tracking_env).get_run(run.run_id).data.metrics
    assert metricas["alerts_drift_count"] == 2
    assert metricas["alerts_failure_count"] == 1
    assert metricas["rows_processed"] == 308
    assert metricas["rows_rejected"] == 8
    assert metricas["stage_success"] == 1


def test_bloqueio_intencional_conta_como_alerta_sem_falhar_o_estagio(
    tracking_env, log_file
):
    with tracked_stage("contract_demo", pipeline_run_id="pipe-5") as run:
        filho_id = run.log_batch(
            _record(
                "invalid_batch",
                AlertLevel.BLOQUEADO,
                decision=_decision(AlertLevel.BLOQUEADO, ("contract_blocked",), False),
                intentional_demo=True,
            )
        )
    client = _client(tracking_env)
    estagio = client.get_run(run.run_id).data
    assert estagio.metrics["alerts_failure_count"] == 1
    assert estagio.metrics["stage_success"] == 1
    assert estagio.tags["stage_status"] == "SUCCESS"
    assert estagio.tags["intentional_demo"] == "true"
    filho = client.get_run(filho_id).data.tags
    assert filho["alert_level"] == "BLOQUEADO"
    assert filho["intentional_demo"] == "true"
    assert filho["performance_evaluated"] == "false"


def test_log_jsonl_anexado_ao_run(tracking_env, log_file, tmp_path):
    with tracked_stage("prepare", pipeline_run_id="pipe-6"):
        _escrever_evento(log_file, "pipe-6", "evento_do_prepare")
    with tracked_stage("train", pipeline_run_id="pipe-6") as run:
        _escrever_evento(log_file, "pipe-6", "evento_do_train")
        _escrever_evento(log_file, "outro-run", "evento_alheio")

    baixado = _client(tracking_env).download_artifacts(
        run.run_id, "logs/pipeline_pipe-6.jsonl", dst_path=str(tmp_path / "baixado")
    )
    eventos = [
        json.loads(linha)
        for linha in Path(baixado).read_text(encoding="utf-8").splitlines()
    ]
    assert [e["event"] for e in eventos] == ["evento_do_train"]


def test_reexecucao_com_mesmo_pipeline_run_id_acumula_runs(tracking_env, log_file):
    ids = []
    for _ in range(2):
        with tracked_stage("train", pipeline_run_id="pipe-7") as run:
            ids.append(run.run_id)
    client = _client(tracking_env)
    experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    achados = client.search_runs(
        [experimento.experiment_id],
        filter_string="tags.pipeline_run_id = 'pipe-7' and tags.run_type = 'stage'",
    )
    assert sorted(r.info.run_id for r in achados) == sorted(ids)
    assert len(set(ids)) == 2


@pytest.mark.parametrize(
    ("uri", "esperado"),
    [
        (
            "postgresql://admin:segredo123@host/db",
            "postgresql://***@host/db",
        ),
        ("http://admin:segredo123@host:5000/", "http://***@host:5000/"),
        ("https://user@host/api", "https://***@host/api"),
        ("https://host/api?token=segredo123&x=1#frag", "https://host/api?***"),
        ("https://host/api#segredo123", "https://host/api#***"),
        # Senha com caracteres reservados sem percent-encoding.
        ("postgresql://u:pa/segredo123@h/db", "postgresql://***@h/db"),
        ("postgresql://u:p@segredo123@h/db", "postgresql://***@h/db"),
        # Ambíguas (`@` depois do primeiro `?`/`#`): só o esquema.
        ("postgresql://u:p?segredo123@h/db", "postgresql://***"),
        ("postgresql://u:p#segredo123@h/db", "postgresql://***"),
        ("https://host/api?token=abc@SECRET9", "https://***"),
        ("http://admin:pw@host/x?email=a@SECRET9", "http://***"),
        ("https://host/p#frag@SECRET9", "https://***"),
        ("postgresql://h/db?password=p@ssword", "postgresql://***"),
        ("http://[::1]:5000/", "http://[::1]:5000/"),
        ("http://host:5000/x?a=1", "http://host:5000/x?***"),
        (
            "postgresql://h/db?password=segredo123&sslpassword=segredo123",
            "postgresql://h/db?***",
        ),
        ("https://h/api?token=segredo123", "https://h/api?***"),
        ("http://u:segredo123@h/x?token=segredo123", "http://***@h/x?***"),
        ("sqlite:////abs/mlflow.db", "sqlite:////abs/mlflow.db"),
        ("sqlite:///rel/mlflow.db", "sqlite:///rel/mlflow.db"),
        # Sem `esquema://`: falha fechada (só o esquema, ou nada).
        ("/caminho/segredo123/mlruns", "***"),
        ("databricks:segredo123", "databricks:***"),
        ("segredo123", "***"),
        ("", "***"),
    ],
)
def test_redact_uri_remove_credenciais(uri, esperado):
    redigida = redact_uri(uri)
    assert redigida == esperado
    for segredo in ("segredo123", "SECRET9", "ssword", "admin:", "pw@"):
        assert segredo not in redigida


@pytest.mark.parametrize(
    "uri",
    [
        # Depende do ambiente: sem `psycopg` instalado dá ImportError; com ele, o
        # host `.invalid` não resolve e dá SQLAlchemyError. Os dois são convertidos.
        "postgresql://admin:segredo123@host.invalid/db",
        # Depende de nada escutar na porta 9 local e de
        # MLFLOW_HTTP_REQUEST_MAX_RETRIES=0 (definido abaixo) para falhar rápido. A
        # MlflowException do cliente repete a URI crua no texto.
        "http://admin:segredo123@127.0.0.1:9/",
    ],
)
def test_falha_de_tracking_nao_vaza_credenciais(tmp_path, monkeypatch, log_file, uri):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    monkeypatch.delenv("MONITORING_TRACKING", raising=False)
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    eventos: list[dict] = []

    def _registrar(_logger, event, level=None, message=None, **fields):
        eventos.append({"event": event, "message": message, **fields})

    monkeypatch.setattr(tracking, "log_event", _registrar)

    with pytest.raises(TrackingUnavailableError) as erro:
        with tracked_stage("train"):
            pass

    persistido = str(erro.value) + json.dumps(eventos, ensure_ascii=False)
    assert "segredo123" not in persistido
    assert "admin:" not in persistido
    assert "MONITORING_TRACKING=off" in str(erro.value)
    assert redact_uri(uri) in str(erro.value)
    assert [
        e["tracking_uri"] for e in eventos if e["event"] == "tracking_unavailable"
    ] == [redact_uri(uri)]
    assert mlflow.active_run() is None


def _quebrar_mlflow(monkeypatch) -> None:
    def _sem_banco():
        raise OSError("banco sumiu")

    monkeypatch.setattr(tracking, "_client", _sem_banco)


def test_registro_final_falha_com_estagio_ok_levanta(
    tracking_env, log_file, monkeypatch
):
    with pytest.raises(TrackingUnavailableError, match="MONITORING_TRACKING=off"):
        with tracked_stage("train", pipeline_run_id="pipe-8") as run:
            _quebrar_mlflow(monkeypatch)
    assert mlflow.active_run() is None
    # O registro final não aconteceu: o run não pode ficar como FINISHED.
    assert _client(tracking_env).get_run(run.run_id).info.status == "FAILED"


def _quebrar_end_run(monkeypatch) -> None:
    def _falhar(self, run_id, status=None, end_time=None):
        raise MlflowException("API request to http://admin:segredo123@host failed")

    monkeypatch.setattr(MlflowClient, "set_terminated", _falhar)


def _capturar_eventos(monkeypatch) -> list[dict]:
    eventos: list[dict] = []

    def _registrar(_logger, event, level=None, message=None, **fields):
        eventos.append({"event": event, "message": message, **fields})

    monkeypatch.setattr(tracking, "log_event", _registrar)
    return eventos


def test_end_run_falha_com_estagio_ok_levanta_erro_redigido(
    tracking_env, log_file, monkeypatch
):
    eventos = _capturar_eventos(monkeypatch)
    with pytest.raises(TrackingUnavailableError) as erro:
        with tracked_stage("train", pipeline_run_id="pipe-10"):
            _quebrar_end_run(monkeypatch)
    assert "MONITORING_TRACKING=off" in str(erro.value)
    persistido = str(erro.value) + json.dumps(eventos, ensure_ascii=False)
    assert "segredo123" not in persistido
    assert "admin:" not in persistido
    assert mlflow.active_run() is None


def test_end_run_falha_nao_mascara_systemexit_do_estagio(
    tracking_env, log_file, monkeypatch
):
    eventos = _capturar_eventos(monkeypatch)
    saida = SystemExit(3)
    with pytest.raises(SystemExit) as capturada:
        with tracked_stage("train", pipeline_run_id="pipe-11"):
            _quebrar_end_run(monkeypatch)
            raise saida
    assert capturada.value is saida
    assert capturada.value.code == 3
    assert "stage_failure_not_recorded" in [e["event"] for e in eventos]
    assert "segredo123" not in json.dumps(eventos, ensure_ascii=False)
    assert mlflow.active_run() is None


def test_attach_de_arquivo_inexistente_nao_culpa_o_mlflow(
    tracking_env, log_file, tmp_path
):
    with pytest.raises(FileNotFoundError, match="não encontrado") as erro:
        with tracked_stage("train", pipeline_run_id="pipe-12") as run:
            run.attach(tmp_path / "sumiu.json")
    assert not isinstance(erro.value, TrackingUnavailableError)
    assert "MONITORING_TRACKING" not in str(erro.value)
    assert _client(tracking_env).get_run(run.run_id).info.status == "FAILED"


@pytest.mark.parametrize("campo", ["artifacts", "batch_path"])
def test_log_batch_com_caminho_inexistente_nao_culpa_o_mlflow(
    tracking_env, log_file, tmp_path, campo
):
    sumido = tmp_path / "sumiu.parquet"
    valor = [sumido] if campo == "artifacts" else sumido
    record = _record("t0", AlertLevel.OK, **{campo: valor})
    with pytest.raises(FileNotFoundError, match="não encontrado") as erro:
        with tracked_stage("drift_demo", pipeline_run_id="pipe-13") as run:
            run.log_batch(record)
    assert "MONITORING_TRACKING" not in str(erro.value)
    client = _client(tracking_env)
    filhos = client.search_runs(
        [client.get_run(run.run_id).info.experiment_id],
        filter_string="tags.run_type = 'batch'",
    )
    assert filhos == []


def test_estado_global_do_mlflow_e_restaurado(tmp_path, monkeypatch, log_file):
    """A ordem dos testes não importa: o ambiente volta ao que era antes."""
    from tests.conftest import MLFLOW_ENV_VARS, isolar_estado_mlflow

    antes = {nome: os.environ.get(nome) for nome in MLFLOW_ENV_VARS}
    uri_ruim = "sqlite:////proc/inexistente/mlflow.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri_ruim)
    monkeypatch.delenv("MONITORING_TRACKING", raising=False)
    with isolar_estado_mlflow():
        with pytest.raises(TrackingUnavailableError):
            configure_tracking()
        # Sem o isolamento, o próprio MLflow deixaria isto no ambiente.
        assert os.environ["MLFLOW_TRACKING_URI"] == uri_ruim
    assert {nome: os.environ.get(nome) for nome in MLFLOW_ENV_VARS} == antes
    assert mlflow.get_tracking_uri() != uri_ruim


def test_registro_final_falha_nao_mascara_erro_do_estagio(
    tracking_env, log_file, monkeypatch
):
    erro = ValueError("erro original do estágio")
    with pytest.raises(ValueError) as capturada:
        with tracked_stage("train", pipeline_run_id="pipe-9"):
            _quebrar_mlflow(monkeypatch)
            raise erro
    assert capturada.value is erro
    assert mlflow.active_run() is None


def test_tracking_off_nao_cria_banco(tmp_path, monkeypatch, log_file):
    banco = tmp_path / "mlflow" / "mlflow.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{banco}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    monkeypatch.setenv("MONITORING_TRACKING", "off")
    with tracked_stage("train") as run:
        assert isinstance(run, NullStageRun)
        assert run.run_id is None
        run.add_rows(5, rejected=1)
        run.set_metric("x", 1.0)
        run.attach(tmp_path / "qualquer.txt")
        assert run.log_batch(_record("t0", AlertLevel.OK)) is None
    assert not banco.exists()
    assert not banco.parent.exists()
    assert not (tmp_path / "mlartifacts").exists()


def _uri_sem_permissao(tmp_path: Path) -> str:
    if os.geteuid() == 0:
        pytest.skip("root ignora permissões de diretório")
    bloqueado = tmp_path / "bloqueado"
    bloqueado.mkdir()
    bloqueado.chmod(0o500)
    return f"sqlite:///{bloqueado}/inexistente/mlflow.db"


@pytest.mark.parametrize("origem", ["inexistente", "sem_permissao"])
def test_mlflow_inacessivel_falha_fechado(tmp_path, monkeypatch, log_file, origem):
    if origem == "inexistente":
        uri = "sqlite:////proc/inexistente/mlflow.db"
    else:
        uri = _uri_sem_permissao(tmp_path)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    monkeypatch.delenv("MONITORING_TRACKING", raising=False)
    banco_padrao = config.ROOT_DIR / "mlflow.db"
    existia = banco_padrao.exists()
    executou = False

    try:
        with pytest.raises(TrackingUnavailableError) as erro:
            with tracked_stage("train"):
                executou = True
    finally:
        if origem == "sem_permissao":
            (tmp_path / "bloqueado").chmod(0o700)

    mensagem = str(erro.value)
    assert uri in mensagem
    assert "MONITORING_TRACKING=off" in mensagem
    assert not executou, "o estágio não pode rodar sem o MLflow"
    assert banco_padrao.exists() == existia
    assert not list(tmp_path.rglob("mlflow.db"))
    assert not (tmp_path / "mlartifacts").exists()
    assert mlflow.active_run() is None
