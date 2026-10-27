"""Lógica das consultas do aceite real (`tests/integration/acceptance_checks.py`).

Os runs aqui são SINTÉTICOS, montados pelo adaptador real (`tracked_stage` e
`log_batch`) num MLflow temporário (`tracking_env`). Isto prova só que as consultas
leem o contrato de nomes da §4.1 como o adaptador o grava; **não é evidência de
aceite**. A evidência é `tests/integration/test_acceptance_real.py` sobre o
`mlflow.db` real, depois de `make pipeline` com o dataset do Kaggle.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src import config
from src.monitoring.alerts import ACTIONS, AlertDecision, AlertLevel
from src.monitoring.tracking import BatchRecord, tracked_stage
from tests.conftest import cliente_mlflow
from tests.integration import acceptance_checks as checks
from tests.integration import test_acceptance_real as aceite_real

T0, T1, T2 = checks.EXPECTED_DRIFT_LEVELS
NIVEIS_ESPERADOS = {
    T0: AlertLevel.OK,
    T1: AlertLevel.ATENCAO,
    T2: AlertLevel.CRITICO,
}
METRICAS_DE_LOTE = {
    "roc_auc": 0.74,
    "auc_drop": 0.0,
    "approval_rate": 0.8,
    "default_rate_approved": 0.05,
    "score_psi": 0.01,
}


@pytest.fixture
def log_file(tmp_path, monkeypatch) -> Path:
    caminho = tmp_path / "pipeline.jsonl"
    monkeypatch.setattr(config, "PIPELINE_LOG_FILE", caminho)
    return caminho


def _evento(log_file: Path, run_id: str, event: str, **campos) -> None:
    linha = {"timestamp": datetime.now(UTC).isoformat(), "run_id": run_id}
    linha.update(event=event, **campos)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(linha) + "\n")


def _record(nome: str, nivel: AlertLevel, **extra) -> BatchRecord:
    motivos = () if nivel is AlertLevel.OK else (f"motivo_{nivel.value.lower()}",)
    padrao = {
        "batch_name": nome,
        "decision": AlertDecision(nivel, motivos, True, ACTIONS[nivel]),
        "n_rows": 100,
        "rows_rejected": 0,
        "metrics": {},
        "psi_by_column": {},
        "n_drifted_columns": 0,
        "model_version": "1",
    }
    padrao.update(extra)
    return BatchRecord(**padrao)


def _pipeline(
    log_file: Path,
    pid: str,
    *,
    niveis: dict[str, AlertLevel] = NIVEIS_ESPERADOS,
    metricas: dict[str, float] = METRICAS_DE_LOTE,
    bloqueio_intencional: bool = True,
    estagios: tuple[str, ...] = checks.PIPELINE_STAGES,
) -> None:
    """Runs sintéticos de um `make pipeline` com o `pipeline_run_id` dado."""
    for estagio in estagios:
        with tracked_stage(estagio, pipeline_run_id=pid) as run:
            _evento(log_file, pid, f"evento_{estagio}")
            if estagio == "contract_demo":
                run.log_batch(_record("valid_batch", AlertLevel.OK))
                run.log_batch(
                    _record(
                        "invalid_batch",
                        AlertLevel.BLOQUEADO,
                        rows_rejected=100,
                        intentional_demo=bloqueio_intencional,
                    )
                )
            elif estagio == "drift_demo":
                _evento(log_file, pid, "drift_detection", duration_ms=12.5)
                for nome, nivel in niveis.items():
                    run.log_batch(_record(nome, nivel, metrics=dict(metricas)))
    # Ordem por `start_time` (ms): o pipeline seguinte começa depois.
    time.sleep(0.01)


@pytest.fixture
def client(tracking_env):
    return cliente_mlflow(tracking_env)


def _mais_recente(client) -> checks.PipelineRuns:
    pipeline = checks.latest_pipeline(client, config.MLFLOW_EXPERIMENT)
    assert pipeline is not None
    return pipeline


def test_sem_pipeline_completo_devolve_none(client, log_file):
    assert checks.latest_pipeline(client, config.MLFLOW_EXPERIMENT) is None
    _pipeline(log_file, "so-dois", estagios=("prepare", "train"))
    assert checks.latest_pipeline(client, config.MLFLOW_EXPERIMENT) is None


def test_pipeline_completo_mais_recente_ignora_estagios_avulsos(client, log_file):
    _pipeline(log_file, "antigo")
    _pipeline(log_file, "novo")
    # Depois do pipeline: um estágio avulso (ex.: `dvc repro` sem PIPELINE_RUN_ID).
    _pipeline(log_file, "avulso", estagios=("drift_demo",))

    pipeline = _mais_recente(client)

    assert pipeline.pipeline_run_id == "novo"
    assert list(pipeline.stages) == list(checks.PIPELINE_STAGES)
    assert {r.data.tags["batch_name"] for r in pipeline.batches["drift_demo"]} == set(
        NIVEIS_ESPERADOS
    )
    assert len(pipeline.batches["contract_demo"]) == 2
    assert pipeline.batches["prepare"] == []
    assert checks.failed_stages(pipeline) == []


def test_niveis_esperados_passam(client, log_file):
    _pipeline(log_file, "pipe")
    assert checks.drift_level_problems(_mais_recente(client)) == []


def test_nivel_divergente_e_apontado(client, log_file):
    _pipeline(log_file, "pipe", niveis={**NIVEIS_ESPERADOS, T1: AlertLevel.OK})
    problemas = checks.drift_level_problems(_mais_recente(client))
    assert len(problemas) == 1
    assert T1 in problemas[0]
    assert "ATENCAO" in problemas[0]


def test_lote_ausente_e_apontado(client, log_file):
    _pipeline(log_file, "pipe", niveis={T0: AlertLevel.OK, T1: AlertLevel.ATENCAO})
    assert checks.drift_level_problems(_mais_recente(client)) == [
        f"{T2}: nenhum run de lote no drift_demo"
    ]


def test_metricas_de_saude_presentes(client, log_file):
    _pipeline(log_file, "pipe")
    assert checks.missing_health_metrics(_mais_recente(client)) == []


def test_metrica_de_saude_ausente_e_apontada(client, log_file):
    _pipeline(log_file, "pipe", metricas={"roc_auc": 0.7})
    faltando = checks.missing_health_metrics(_mais_recente(client))
    assert faltando == [
        "auc_drop",
        "approval_rate",
        "default_rate_approved",
        "score_psi",
    ]


def test_log_jsonl_de_cada_estagio(client, log_file, tmp_path):
    _pipeline(log_file, "pipe")
    pipeline = _mais_recente(client)

    logs = checks.stage_logs(client, pipeline, tmp_path / "logs")

    assert list(logs) == list(checks.PIPELINE_STAGES)
    assert checks.log_problems(logs, pipeline.pipeline_run_id) == []
    assert checks.has_latency_events(logs)
    assert [e["event"] for e in logs["prepare"]] == ["evento_prepare"]


def test_log_ausente_vazio_ou_de_outro_run_e_apontado():
    logs = {
        "prepare": None,
        "train": [],
        "contract_demo": [{"run_id": "outro", "event": "x"}],
        "drift_demo": [{"run_id": "pipe", "event": "y"}],
    }
    assert checks.log_problems(logs, "pipe") == [
        "prepare: sem logs/pipeline_pipe.jsonl anexado",
        "train: log anexado vazio",
        "contract_demo: 1 linha(s) com run_id diferente de pipe",
    ]
    assert not checks.has_latency_events(logs)


def test_bloqueio_intencional_registrado(client, log_file):
    _pipeline(log_file, "pipe")
    assert checks.blocked_demo_problems(_mais_recente(client)) == []


def test_bloqueio_sem_intentional_demo_e_apontado(client, log_file):
    _pipeline(log_file, "pipe", bloqueio_intencional=False)
    problemas = checks.blocked_demo_problems(_mais_recente(client))
    assert problemas == [
        "contract_demo: run BLOQUEADO sem intentional_demo=true",
        "contract_demo: run de estágio sem a tag intentional_demo=true",
    ]


# Decisões de `test_acceptance_real.py`: sem pipeline completo ele PULA (não quebra
# a suíte); com pipeline completo e um nível divergente ele FALHA.


def test_aceite_real_pula_sem_pipeline_completo(client, log_file):
    _pipeline(log_file, "so-data", estagios=("prepare",))
    with pytest.raises(pytest.skip.Exception, match="sem pipeline completo"):
        aceite_real.pipeline_completo_ou_pular(client)


def test_aceite_real_falha_com_pipeline_completo_e_t2_errado(client, log_file):
    _pipeline(log_file, "pipe", niveis={**NIVEIS_ESPERADOS, T2: AlertLevel.ATENCAO})
    pipeline = aceite_real.pipeline_completo_ou_pular(client)
    assert pipeline.pipeline_run_id == "pipe"
    with pytest.raises(AssertionError, match=T2):
        aceite_real.test_t0_t1_t2_com_os_niveis_esperados(pipeline)


@pytest.fixture
def ambiente_real(tmp_path, monkeypatch) -> Path:
    """`mlflow.db` e CSV "reais" redirecionados para `tmp_path`, fora do CI."""
    monkeypatch.delenv("CI", raising=False)
    csv = tmp_path / "application_train.csv"
    csv.write_text("SK_ID_CURR\n1\n", encoding="utf-8")
    monkeypatch.setattr(config, "RAW_FILE", csv)
    banco = tmp_path / "mlflow.db"
    monkeypatch.setattr(aceite_real, "REAL_DB", banco)
    return banco


def test_aceite_real_pula_com_mlflow_db_vazio(ambiente_real):
    ambiente_real.write_bytes(b"")
    motivo = aceite_real._motivo_para_pular()
    assert motivo is not None
    assert "vazio" in motivo


def test_aceite_real_roda_com_mlflow_db_e_csv(ambiente_real):
    ambiente_real.write_bytes(b"SQLite format 3\x00")
    assert aceite_real._motivo_para_pular() is None
