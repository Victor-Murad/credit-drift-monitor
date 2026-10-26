"""Adaptador do MLflow: run de estágio (pai) e runs de lote (filhos aninhados).

Spec da Etapa 3, §4.1 (estrutura e contrato de nomes), §4.3 (métricas de saúde) e
§4.4 (log JSONL do `run_id` anexado ao run). O adaptador só registra: o nível de
alerta chega pronto em `AlertDecision` (regra pura em `alerts.py`).

Falha fechada (§3): sem MLflow acessível, `tracked_stage` levanta
`TrackingUnavailableError` antes de executar o estágio. Só `MONITORING_TRACKING=off`
desliga o rastreamento, e aí o estágio recebe um `NullStageRun`.

As constantes `src.config.MLFLOW_*` e `PIPELINE_LOG_FILE` são lidas na chamada (não
na importação), para que os testes possam redirecioná-las.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, RunTag
from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from src import config
from src.logging_utils import RUN_ID, get_logger, log_event
from src.monitoring.alerts import AlertDecision, AlertLevel
from src.monitoring.logs import extract_run_events
from src.monitoring.resources import peak_memory_mb

logger = get_logger("monitoring.tracking")

TRACKING_ENV_VAR = "MONITORING_TRACKING"
# ImportError: URI de banco cujo driver não está instalado (ex.: `postgresql://`
# sem `psycopg`) também é "MLflow inacessível" e deve falhar fechado.
_TRACKING_ERRORS = (MlflowException, SQLAlchemyError, OSError, ImportError)
_DRIFT_LEVELS = frozenset({AlertLevel.ATENCAO, AlertLevel.CRITICO})
_REASON_SEPARATOR = ";"
_REDACTED = "***"
_URI_WITH_AUTHORITY = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*://)(.*)$", re.DOTALL)
_URI_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")


class TrackingUnavailableError(RuntimeError):
    """O MLflow não está acessível ou o registro falhou (falha fechada)."""


def redact_uri(uri: str) -> str:
    """URI sem credenciais, para mensagens, logs e tags. Redigir a mais é seguro.

    - Tudo entre `esquema://` e a ÚLTIMA `@` vira `***@` (a senha pode ter `/`,
      `?`, `#` ou `@` sem percent-encoding, o que engana o `urlsplit`).
    - A *query string* inteira (chaves e valores) vira `?***`; o fragmento, `#***`.
    - Falha fechada: o que não tem a forma `esquema://` volta só como
      `esquema:***` (se houver esquema) ou `***`, nunca como texto cru.

    URIs sem nada a esconder, como `sqlite:///caminho/mlflow.db`, voltam iguais.
    """
    com_autoridade = _URI_WITH_AUTHORITY.match(uri)
    if com_autoridade is None:
        so_esquema = _URI_SCHEME.match(uri)
        return f"{so_esquema.group(1)}:{_REDACTED}" if so_esquema else _REDACTED
    esquema, resto = com_autoridade.groups()
    if "@" in resto:
        resto = f"{_REDACTED}@{resto.rpartition('@')[2]}"
    corte = min((i for i in (resto.find("?"), resto.find("#")) if i >= 0), default=-1)
    if corte >= 0:
        resto = f"{resto[:corte]}{resto[corte]}{_REDACTED}"
    return f"{esquema}{resto}"


def tracking_enabled() -> bool:
    """Falso só quando `MONITORING_TRACKING=off` (lido no momento da chamada)."""
    return os.environ.get(TRACKING_ENV_VAR, "").strip().lower() != "off"


def _describe_error(exc: BaseException) -> str:
    """Tipo do erro e, para `OSError`, o `strerror` (sem caminho nem URI)."""
    if isinstance(exc, OSError) and exc.strerror:
        return f"{type(exc).__name__} ({exc.strerror})"
    if isinstance(exc, ImportError) and exc.name:
        return f"{type(exc).__name__} (driver ausente: {exc.name})"
    return type(exc).__name__


@contextmanager
def _tracking_errors(contexto: str) -> Iterator[None]:
    """Converte erros de conexão/permissão do MLflow em `TrackingUnavailableError`.

    A mensagem e o evento de log levam só a URI redigida e uma descrição curta do
    erro: o texto do driver/cliente pode repetir a URI crua (com credenciais) e por
    isso fica apenas na cadeia `__cause__`, que não é persistida.
    """
    uri = redact_uri(config.MLFLOW_TRACKING_URI)
    try:
        yield
    except _TRACKING_ERRORS as exc:
        mensagem = (
            f"{contexto} (MLflow em {uri}): {_describe_error(exc)}. O pipeline não "
            f"segue sem registro (falha fechada); para rodar sem rastreamento, "
            f"defina {TRACKING_ENV_VAR}=off."
        )
        log_event(
            logger,
            "tracking_unavailable",
            level=logging.ERROR,
            message=mensagem,
            tracking_uri=uri,
            error=type(exc).__name__,
        )
        raise TrackingUnavailableError(mensagem) from exc


def _client() -> MlflowClient:
    uri = config.MLFLOW_TRACKING_URI
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def configure_tracking() -> str:
    """Aponta o MLflow para `MLFLOW_TRACKING_URI`, garante o experimento e checa a saúde.

    Cria o experimento `MLFLOW_EXPERIMENT` com `artifact_location` em
    `MLFLOW_ARTIFACT_ROOT` se ele não existir e faz um `search_runs(max_results=1)`
    para provar leitura do banco. Devolve o `experiment_id`.
    """
    with _tracking_errors("MLflow indisponível"):
        uri = config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        client = _client()
        experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
        if experimento is None:
            experiment_id = client.create_experiment(
                config.MLFLOW_EXPERIMENT,
                artifact_location=Path(config.MLFLOW_ARTIFACT_ROOT).absolute().as_uri(),
            )
        else:
            experiment_id = experimento.experiment_id
        client.search_runs([experiment_id], max_results=1)
        mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


@dataclass
class BatchRecord:
    """O que um run de lote registra (§4.1). Campos extras têm padrão.

    `metrics`: roc_auc, auc_drop, auc_drop_vs_training, approval_rate,
    default_rate_approved, score_psi (as ausentes ou `None` não são logadas).
    `batch_sha256`, se omitido, é calculado a partir de `batch_path`.
    """

    batch_name: str
    decision: AlertDecision
    n_rows: int
    rows_rejected: int
    metrics: dict[str, float | None]
    psi_by_column: dict[str, float]
    n_drifted_columns: int
    model_version: str
    artifacts: list[Path] = field(default_factory=list)
    intentional_demo: bool = False
    batch_path: Path | None = None
    batch_sha256: str | None = None


def _bool_tag(valor: bool) -> str:
    return "true" if valor else "false"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _batch_tags(record: BatchRecord) -> dict[str, str]:
    decision = record.decision
    tags = {
        "run_type": "batch",
        "alert_level": decision.level.value,
        "alert_label": decision.level.label,
        "alert_reasons": _REASON_SEPARATOR.join(decision.reasons),
        "alert_action": decision.action,
        "batch_name": record.batch_name,
        "model_version": str(record.model_version),
        "performance_evaluated": _bool_tag(decision.performance_evaluated),
        "intentional_demo": _bool_tag(record.intentional_demo),
    }
    sha256 = record.batch_sha256
    if record.batch_path is not None:
        tags["batch_path"] = config.project_relative(record.batch_path)
        if sha256 is None and record.batch_path.is_file():
            sha256 = _sha256(record.batch_path)
    if sha256 is not None:
        tags["batch_sha256"] = sha256
    return tags


def _batch_metrics(record: BatchRecord) -> dict[str, float]:
    metricas = {k: float(v) for k, v in record.metrics.items() if v is not None}
    metricas.update(
        {f"psi_{coluna}": float(v) for coluna, v in record.psi_by_column.items()}
    )
    metricas["n_drifted_columns"] = float(record.n_drifted_columns)
    metricas["n_rows"] = float(record.n_rows)
    metricas["rows_rejected"] = float(record.rows_rejected)
    return metricas


def _require_path(path: Path, o_que: str) -> Path:
    """Confere um caminho local antes de falar com o MLflow."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{o_que} não encontrado: {path}")
    return path


def _log_path(run_id: str, path: Path, artifact_path: str | None) -> None:
    client = _client()
    if path.is_dir():
        destino = f"{artifact_path}/{path.name}" if artifact_path else path.name
        client.log_artifacts(run_id, str(path), artifact_path=destino)
    else:
        client.log_artifact(run_id, str(path), artifact_path=artifact_path)


class StageRun:
    """Run de estágio no MLflow. `NullStageRun` tem a mesma interface, sem efeito."""

    def __init__(self, run_id: str, experiment_id: str, stage: str) -> None:
        self.run_id: str | None = run_id
        self._experiment_id = experiment_id
        self._stage = stage
        self._rows_processed = 0
        self._rows_rejected = 0
        self._failure_alerts = 0
        self._drift_alerts = 0
        self._intentional_demo = False

    def add_rows(self, processed: int, rejected: int = 0) -> None:
        """Soma volume ao estágio. `log_batch` já soma os lotes que recebe."""
        self._rows_processed += processed
        self._rows_rejected += rejected

    def set_metric(self, key: str, value: float) -> None:
        """Métrica avulsa no run de estágio."""
        with _tracking_errors("Falha ao registrar métrica"):
            _client().log_metric(self.run_id, key, float(value))

    def attach(self, path: Path, *, artifact_path: str | None = None) -> None:
        """Anexa um arquivo (ou diretório) ao run de estágio.

        Caminho inexistente é erro local (`FileNotFoundError`), não do MLflow.
        """
        path = _require_path(path, "Artefato")
        with _tracking_errors("Falha ao anexar artefato"):
            _log_path(self.run_id, path, artifact_path)

    def log_batch(self, record: BatchRecord) -> str | None:
        """Cria o run filho do lote e atualiza os contadores do estágio.

        `n_rows` entra em `rows_processed`; um lote `BLOQUEADO` soma `n_rows` em
        `rows_rejected` e conta em `alerts_failure_count`, mesmo com
        `intentional_demo=True` (o alerta é real, §4.1). `ATENÇÃO`/`CRÍTICO`
        contam em `alerts_drift_count`. Devolve o `run_id` do filho.

        `batch_path` e `artifacts` são conferidos (e o hash do lote calculado) antes
        de abrir o run: caminho inexistente é `FileNotFoundError`, sem run filho.
        """
        artefatos = [_require_path(a, "Artefato do lote") for a in record.artifacts]
        if record.batch_path is not None:
            _require_path(record.batch_path, "Lote")
        tags = _batch_tags(record)
        metricas = _batch_metrics(record)
        with (
            _tracking_errors("Falha ao registrar o lote"),
            mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=record.batch_name,
                nested=True,
                parent_run_id=self.run_id,
                tags=tags,
            ) as filho,
        ):
            mlflow.log_metrics(metricas)
            for artefato in artefatos:
                _log_path(filho.info.run_id, artefato, "reports")

        level = record.decision.level
        self._rows_processed += record.n_rows
        if level is AlertLevel.BLOQUEADO:
            self._failure_alerts += 1
            self._rows_rejected += record.n_rows
        elif level in _DRIFT_LEVELS:
            self._drift_alerts += 1
        self._intentional_demo = self._intentional_demo or record.intentional_demo
        return filho.info.run_id

    def _finish(
        self,
        *,
        succeeded: bool,
        duration_s: float,
        pipeline_run_id: str,
        since: datetime,
    ) -> None:
        """Métricas e tags finais do estágio e o log JSONL da janela do estágio."""
        status = "SUCCESS" if succeeded else "FAILED"
        log_event(
            logger,
            "stage_finished",
            level=logging.INFO if succeeded else logging.ERROR,
            stage=self._stage,
            status=status,
            duration_s=round(duration_s, 3),
            mlflow_run_id=self.run_id,
        )
        metricas = {
            "stage_success": 1.0 if succeeded else 0.0,
            "stage_duration_s": duration_s,
            "peak_memory_mb": peak_memory_mb(),
            "rows_processed": float(self._rows_processed),
            "rows_rejected": float(self._rows_rejected),
            "alerts_failure_count": float(
                self._failure_alerts + (0 if succeeded else 1)
            ),
            "alerts_drift_count": float(self._drift_alerts),
        }
        tags = {"stage_status": status}
        if self._intentional_demo:
            tags["intentional_demo"] = "true"
        agora = int(time.time() * 1000)
        with _tracking_errors("Falha ao finalizar o run do estágio"):
            client = _client()
            client.log_batch(
                self.run_id,
                metrics=[Metric(k, v, agora, 0) for k, v in metricas.items()],
                tags=[RunTag(k, v) for k, v in tags.items()],
            )
            self._attach_log(pipeline_run_id, since)

    def _attach_log(self, pipeline_run_id: str, since: datetime) -> None:
        linhas = extract_run_events(
            config.PIPELINE_LOG_FILE,
            pipeline_run_id,
            since=since,
            until=datetime.now(UTC),
        )
        with tempfile.TemporaryDirectory(prefix="credit-drift-monitor-log-") as tmp:
            arquivo = Path(tmp) / f"pipeline_{pipeline_run_id}.jsonl"
            arquivo.write_text(
                "".join(f"{linha}\n" for linha in linhas), encoding="utf-8"
            )
            _client().log_artifact(self.run_id, str(arquivo), artifact_path="logs")


class NullStageRun(StageRun):
    """Rastreamento desligado (`MONITORING_TRACKING=off`): tudo é no-op."""

    def __init__(self) -> None:
        # Não chama `super().__init__`: não há run, experimento nem contadores.
        self.run_id = None

    def add_rows(self, processed: int, rejected: int = 0) -> None:
        return None

    def set_metric(self, key: str, value: float) -> None:
        return None

    def attach(self, path: Path, *, artifact_path: str | None = None) -> None:
        return None

    def log_batch(self, record: BatchRecord) -> str | None:
        return None


@contextmanager
def tracked_stage(stage: str, *, pipeline_run_id: str = RUN_ID) -> Iterator[StageRun]:
    """Abre o run de estágio (tag `run_type=stage`) e o fecha com as métricas de saúde.

    A checagem de saúde do MLflow acontece antes de o estágio rodar. A exceção (ou
    `SystemExit`) do estágio sai intacta; o run fica `FAILED`, com
    `stage_success=0` e o estágio contado em `alerts_failure_count`. Se o estágio
    terminou bem e o registro final falhar, levanta `TrackingUnavailableError`.
    """
    if not tracking_enabled():
        log_event(
            logger,
            "tracking_disabled",
            level=logging.WARNING,
            message=f"Rastreamento desligado ({TRACKING_ENV_VAR}=off): estágio {stage} sem MLflow",
            stage=stage,
        )
        yield NullStageRun()
        return

    experiment_id = configure_tracking()
    since = datetime.now(UTC)
    with _tracking_errors("Falha ao abrir o run do estágio"):
        ativo = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=stage,
            tags={
                "run_type": "stage",
                "pipeline_run_id": pipeline_run_id,
                "stage": stage,
            },
        )
    run = StageRun(ativo.info.run_id, experiment_id, stage)
    log_event(logger, "stage_started", stage=stage, mlflow_run_id=run.run_id)
    inicio = time.perf_counter()
    succeeded = False
    try:
        yield run
        succeeded = True
    finally:
        _close_stage(
            run, succeeded, time.perf_counter() - inicio, pipeline_run_id, since
        )


def _close_stage(
    run: StageRun,
    succeeded: bool,
    duration_s: float,
    pipeline_run_id: str,
    since: datetime,
) -> None:
    """Fecha o run de estágio sem mascarar a exceção original do estágio.

    Registro final e encerramento passam por `_tracking_errors` (mensagem redigida).
    Estágio OK com registro falho: o run termina `FAILED` e a primeira
    `TrackingUnavailableError` é levantada. Estágio que já falhou: a exceção dele
    sai intacta e a falha do registro vai só para o log.
    """
    erro: TrackingUnavailableError | None = None
    registrado = False
    try:
        try:
            run._finish(
                succeeded=succeeded,
                duration_s=duration_s,
                pipeline_run_id=pipeline_run_id,
                since=since,
            )
            registrado = True
        except TrackingUnavailableError as exc:
            erro = exc
    finally:
        status = "FINISHED" if succeeded and registrado else "FAILED"
        erro_ao_encerrar = _end_stage_runs(run, status)
        erro = erro or erro_ao_encerrar
    if erro is None:
        return
    if succeeded:
        raise erro
    log_event(
        logger,
        "stage_failure_not_recorded",
        level=logging.ERROR,
        stage=run._stage,
        mlflow_run_id=run.run_id,
    )


def _end_stage_runs(run: StageRun, status: str) -> TrackingUnavailableError | None:
    """Encerra o run do estágio (e um filho que tenha ficado aberto).

    Não toca num run que o chamador já tivesse aberto antes. O `end_run` do MLflow
    tira o run da pilha antes de gravar o término, então o laço avança mesmo se a
    gravação falhar; a primeira falha é devolvida (já redigida), não levantada.
    """
    primeiro: TrackingUnavailableError | None = None
    while (ativo := mlflow.active_run()) is not None:
        try:
            with _tracking_errors("Falha ao encerrar o run do estágio"):
                mlflow.end_run(status=status)
        except TrackingUnavailableError as exc:
            primeiro = primeiro or exc
        if ativo.info.run_id == run.run_id:
            break
    return primeiro
