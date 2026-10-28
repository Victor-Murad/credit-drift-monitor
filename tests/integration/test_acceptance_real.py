"""Aceite REAL da Etapa 3 (spec §8, critérios 1 a 5 e 10) no `mlflow.db` real.

Roda só na máquina do usuário, depois de `make pipeline` com o dataset do Kaggle; a
parte do Registry (critério 10), depois de `make retrain` e, para a auditoria, de
`make promote`/`make rollback`. Sem retreino ou sem auditoria, essa parte PULA.
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
from src.models.integrity import file_sha256
from tests.integration import acceptance_checks as checks

REAL_DB = config.ROOT_DIR / "mlflow.db"
REAL_URI = f"sqlite:///{REAL_DB}"


def _motivo_para_pular() -> str | None:
    if os.environ.get("CI"):
        return "aceite real não roda em CI"
    if not REAL_DB.is_file():
        return f"sem mlflow.db real em {REAL_DB} (rode `make pipeline`)"
    if REAL_DB.stat().st_size == 0:
        # O cliente criaria o esquema num arquivo vazio; este teste só lê.
        return f"mlflow.db vazio em {REAL_DB} (rode `make pipeline`)"
    if not config.RAW_FILE.is_file():
        return f"sem o dataset do Kaggle em {config.RAW_FILE}"
    return None


MOTIVO = _motivo_para_pular()
pytestmark = pytest.mark.skipif(MOTIVO is not None, reason=MOTIVO or "")


@pytest.fixture(scope="module")
def client() -> MlflowClient:
    return MlflowClient(tracking_uri=REAL_URI, registry_uri=REAL_URI)


def pipeline_completo_ou_pular(client: MlflowClient) -> checks.PipelineRuns:
    """O pipeline completo mais recente; sem nenhum (ex.: só `make data`), PULA.

    Pular não esconde divergência: havendo pipeline completo, toda asserção que
    divergir é falha dura.
    """
    encontrado = checks.latest_pipeline(client, config.MLFLOW_EXPERIMENT)
    if encontrado is None:
        pytest.skip(
            f"sem pipeline completo (nenhum pipeline_run_id com {checks.PIPELINE_STAGES} "
            f"no experimento {config.MLFLOW_EXPERIMENT}); rode `make pipeline`"
        )
    return encontrado


@pytest.fixture(scope="module")
def pipeline(client) -> checks.PipelineRuns:
    return pipeline_completo_ou_pular(client)


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


# MARK: Extensão §4.7 (critério 10), depois de `make retrain` e `make promote`


def registro_ou_pular(client: MlflowClient) -> checks.RegistryRuns:
    """Registry com ao menos uma versão `origin=retrain`; sem nenhuma, PULA."""
    estado = checks.registry_state(
        client, config.REGISTERED_MODEL_NAME, config.MLFLOW_EXPERIMENT
    )
    if estado is None:
        pytest.skip(
            f"sem retreino no Registry ({config.REGISTERED_MODEL_NAME} sem versão "
            "origin=retrain); rode `make retrain`"
        )
    return estado


def auditorias_ou_pular(estado: checks.RegistryRuns) -> checks.RegistryRuns:
    """Ao menos um run de auditoria; sem promoção nem rollback, PULA."""
    if not estado.audit_runs:
        pytest.skip("sem promoção nem rollback (nenhum run run_type=audit)")
    return estado


@pytest.fixture(scope="module")
def registro(client) -> checks.RegistryRuns:
    return registro_ou_pular(client)


def _sha256(caminho) -> str | None:
    return file_sha256(caminho) if caminho.is_file() else None


def test_versoes_de_retreino_com_tags_e_run_de_origem(registro):
    assert checks.retrain_problems(registro) == []


def test_aliases_coerentes_com_os_arquivos(registro):
    problemas = checks.alias_problems(
        registro, _sha256(config.CHAMPION_FILE), _sha256(config.MODEL_FILE)
    )
    assert problemas == [], registro.aliases


def test_runs_de_auditoria_de_promocao_e_rollback(registro):
    assert checks.audit_problems(auditorias_ou_pular(registro)) == []
