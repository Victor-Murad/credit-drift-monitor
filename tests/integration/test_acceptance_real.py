"""Aceite REAL do núcleo da Etapa 3 (spec §8, critérios 1 a 5) no `mlflow.db` real.

Roda só na máquina do usuário, depois de `make pipeline` com o dataset do Kaggle.
Sem o `mlflow.db` na raiz do repositório ou sem `data/raw/application_train.csv`
(o caso do CI, que também define `CI`), fica SKIPPED. Lê o MLflow só pelo cliente,
sem gravar nada, e confere o `pipeline_run_id` completo mais recente.

O `conftest` aponta `MLFLOW_TRACKING_URI` para um banco temporário antes de
importar `src`; por isso o URI real é montado aqui a partir de `ROOT_DIR`.
"""

from __future__ import annotations

import os

import pytest
from mlflow import MlflowClient

from src import config
from tests.integration import acceptance_checks as checks

REAL_DB = config.ROOT_DIR / "mlflow.db"
REAL_URI = f"sqlite:///{REAL_DB}"


def _motivo_para_pular() -> str | None:
    if os.environ.get("CI"):
        return "aceite real não roda em CI"
    if not REAL_DB.is_file():
        return f"sem mlflow.db real em {REAL_DB} (rode `make pipeline`)"
    if not config.RAW_FILE.is_file():
        return f"sem o dataset do Kaggle em {config.RAW_FILE}"
    return None


MOTIVO = _motivo_para_pular()
pytestmark = pytest.mark.skipif(MOTIVO is not None, reason=MOTIVO or "")


@pytest.fixture(scope="module")
def client() -> MlflowClient:
    return MlflowClient(tracking_uri=REAL_URI, registry_uri=REAL_URI)


@pytest.fixture(scope="module")
def pipeline(client) -> checks.PipelineRuns:
    encontrado = checks.latest_pipeline(client, config.MLFLOW_EXPERIMENT)
    assert encontrado is not None, (
        f"Nenhum pipeline_run_id com os estágios {checks.PIPELINE_STAGES} no "
        f"experimento {config.MLFLOW_EXPERIMENT}: rode `make pipeline`."
    )
    return encontrado


def test_pipeline_rodou_todos_os_estagios_com_sucesso(pipeline):
    assert checks.failed_stages(pipeline) == [], pipeline.pipeline_run_id


def test_t0_t1_t2_com_os_niveis_esperados(pipeline):
    assert checks.drift_level_problems(pipeline) == [], pipeline.pipeline_run_id


def test_cada_metrica_de_saude_aparece_em_algum_run(pipeline):
    assert checks.missing_health_metrics(pipeline) == [], pipeline.pipeline_run_id


def test_log_jsonl_anexado_a_cada_run_de_estagio(client, pipeline, tmp_path_factory):
    logs = checks.stage_logs(client, pipeline, tmp_path_factory.mktemp("logs"))
    assert checks.log_problems(logs, pipeline.pipeline_run_id) == []
    assert checks.has_latency_events(logs), (
        f"nenhum evento com {checks.LATENCY_LOG_FIELD} nos logs anexados"
    )


def test_lote_invalido_bloqueado_e_intencional(pipeline):
    assert checks.blocked_demo_problems(pipeline) == [], pipeline.pipeline_run_id
