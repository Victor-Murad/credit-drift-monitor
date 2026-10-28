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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mlflow import MlflowClient

from src import config
from src.monitoring.alerts import ACTIONS, AlertDecision, AlertLevel
from src.monitoring.promotion import REASON_GAIN_BELOW_MIN, decide_promotion
from src.monitoring.registry import log_audit_run
from src.monitoring.tracking import BatchRecord, configure_tracking, tracked_stage
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


# MARK: Registry (extensão §4.7, critério 10)
#
# Versões e runs SINTÉTICOS no MLflow temporário: v1 baseline, um alerta CRÍTICO e
# v2 retreinada a partir dele, com o contrato de nomes que `retrain` e `promote`
# gravam. Provam só a lógica das consultas, nunca o aceite.

SHA_BASELINE = "a" * 64
SHA_RETREINO = "b" * 64
AUCS_ELEGIVEIS = {
    "champion_holdout_auc": 0.57,
    "challenger_holdout_auc": 0.70,
    "champion_reference_auc": 0.75,
    "challenger_reference_auc": 0.74,
}


def _run(client, experimento: str, tags: dict[str, str], **metricas: float) -> str:
    run = client.create_run(experimento, tags=tags)
    for nome, valor in metricas.items():
        client.log_metric(run.info.run_id, nome, valor)
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def _versao_registrada(
    client, tmp_path: Path, run_id: str, tags: dict[str, str]
) -> int:
    origem = tmp_path / "fontes" / run_id
    origem.mkdir(parents=True)
    versao = client.create_model_version(
        config.REGISTERED_MODEL_NAME, str(origem), run_id=run_id, tags=tags
    )
    return int(versao.version)


@dataclass
class Registro:
    client: MlflowClient
    experimento: str
    alerta: str
    retreino: str
    v1: int
    v2: int


@pytest.fixture
def registro(client, tmp_path) -> Registro:
    """v1 baseline champion; v2 challenger elegível treinado do alerta CRÍTICO."""
    experimento = configure_tracking()
    client.create_registered_model(config.REGISTERED_MODEL_NAME)
    treino = _run(client, experimento, {"run_type": "training"})
    v1 = _versao_registrada(
        client,
        tmp_path,
        treino,
        {
            "origin": "baseline",
            "model_sha256": SHA_BASELINE,
            "bundle_model_version": "baseline-v1",
            "reference_sha256": "c" * 64,
        },
    )
    alerta = _run(
        client,
        experimento,
        {"run_type": "batch", "alert_level": "CRITICO", "batch_name": T2},
    )
    decisao = decide_promotion(contract_ok=True, integrity_ok=True, **AUCS_ELEGIVEIS)
    tags = {
        "bundle_model_version": "retrain-bbbbbbbbbbbb",
        "model_sha256": SHA_RETREINO,
        "reference_sha256": "c" * 64,
        "batch_name": T2,
        "batch_path": "data/production/t2.parquet",
        "batch_sha256": "d" * 64,
        "trigger": "alert",
        "champion_version": str(v1),
        "alert_run_id": alerta,
        "alert_model_version": "baseline-v1",
        "promotion_eligible": "true",
        "promotion_reasons": "",
    }
    retreino = _run(
        client,
        experimento,
        {"run_type": "retrain", **tags},
        **AUCS_ELEGIVEIS,
        gain_on_holdout=decisao.gain_on_holdout,
        regression_on_reference=decisao.regression_on_reference,
        promotion_eligible=1.0,
    )
    client.log_param(retreino, "promotion_min_gain", config.PROMOTION_MIN_GAIN)
    client.log_param(
        retreino, "promotion_max_regression", config.PROMOTION_MAX_REGRESSION
    )
    v2 = _versao_registrada(
        client,
        tmp_path,
        retreino,
        {**tags, "origin": "retrain", "retrain_run_id": retreino},
    )
    client.set_tag(retreino, "challenger_version", str(v2))
    client.set_tag(alerta, "retrain_consumed", str(v2))
    client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", str(v1))
    client.set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, "challenger", str(v2)
    )
    return Registro(client, experimento, alerta, retreino, v1, v2)


def _estado(registro: Registro) -> checks.RegistryRuns:
    estado = checks.registry_state(
        registro.client, config.REGISTERED_MODEL_NAME, config.MLFLOW_EXPERIMENT
    )
    assert estado is not None
    return estado


def _alias(registro: Registro, alias: str, versao: int | None) -> None:
    nome = config.REGISTERED_MODEL_NAME
    if versao is None:
        registro.client.delete_registered_model_alias(nome, alias)
    else:
        registro.client.set_registered_model_alias(nome, alias, str(versao))


def _promover(registro: Registro, **detalhes) -> None:
    """O que `make promote` grava: aliases movidos e o run de auditoria."""
    _alias(registro, "previous", registro.v1)
    _alias(registro, "champion", registro.v2)
    padrao = {
        "from_version": registro.v1,
        "to_version": registro.v2,
        "previous_version": registro.v1,
        "model_sha256": SHA_RETREINO,
    }
    log_audit_run(
        action="promote",
        version=registro.v2,
        reason="challenger elegível",
        details={**padrao, **detalhes},
        operator="analista",
        status="CONSISTENT",
    )
    time.sleep(0.01)


def _reverter(registro: Registro) -> None:
    """O que `make rollback` grava: champion de volta, previous consumido."""
    _alias(registro, "champion", registro.v1)
    _alias(registro, "previous", None)
    registro.client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(registro.v2), "rolled_back", "true"
    )
    log_audit_run(
        action="rollback",
        version=registro.v1,
        reason="demonstração",
        details={
            "from_version": registro.v2,
            "to_version": registro.v1,
            "previous_version": None,
            "model_sha256": SHA_BASELINE,
        },
        operator="analista",
        status="CONSISTENT",
    )
    time.sleep(0.01)


def test_registry_sem_retreino_devolve_none(client):
    assert (
        checks.registry_state(
            client, config.REGISTERED_MODEL_NAME, config.MLFLOW_EXPERIMENT
        )
        is None
    )
    experimento = configure_tracking()
    client.create_registered_model(config.REGISTERED_MODEL_NAME)
    treino = _run(client, experimento, {"run_type": "training"})
    client.create_model_version(
        config.REGISTERED_MODEL_NAME,
        "file:///x",
        run_id=treino,
        tags={"origin": "baseline"},
    )
    assert (
        checks.registry_state(
            client, config.REGISTERED_MODEL_NAME, config.MLFLOW_EXPERIMENT
        )
        is None
    )


def test_registry_le_versoes_aliases_e_runs(registro):
    estado = _estado(registro)
    assert set(estado.versions) == {registro.v1, registro.v2}
    assert estado.aliases == {"champion": registro.v1, "challenger": registro.v2}
    assert list(estado.retrain_runs) == [registro.retreino]
    assert list(estado.alert_runs) == [registro.alerta]
    assert estado.audit_runs == []


def test_retreino_consistente_passa(registro):
    assert checks.retrain_problems(_estado(registro)) == []


def test_retreino_sem_tag_obrigatoria_e_apontado(registro):
    registro.client.delete_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(registro.v2), "reference_sha256"
    )
    assert checks.retrain_problems(_estado(registro)) == [
        f"v{registro.v2}: sem a tag reference_sha256"
    ]


def test_retreino_com_model_version_do_baseline_e_apontado(registro):
    registro.client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME,
        str(registro.v2),
        "bundle_model_version",
        "baseline-v1",
    )
    problemas = checks.retrain_problems(_estado(registro))
    assert problemas == [
        f"v{registro.v2}: bundle_model_version='baseline-v1' sem o prefixo retrain-"
    ]


def test_retreino_com_sha_divergente_do_run_e_apontado(registro):
    registro.client.set_tag(registro.retreino, "model_sha256", "e" * 64)
    problemas = checks.retrain_problems(_estado(registro))
    assert problemas == [
        f"v{registro.v2}: model_sha256 da versão difere do run de retreino"
    ]


def test_retreino_com_veredito_divergente_da_regra_e_apontado(registro):
    # A regra reprova (ganho 0,01 < 0,02), mas a tag diz elegível.
    registro.client.log_metric(registro.retreino, "challenger_holdout_auc", 0.58)
    assert checks.retrain_problems(_estado(registro)) == [
        f"v{registro.v2}: veredito promotion_eligible='true' difere da regra "
        f"recalculada (elegível=False, motivos: {REASON_GAIN_BELOW_MIN})"
    ]


def test_retreino_com_limiar_diferente_do_pre_registrado_e_apontado(
    registro, monkeypatch
):
    # O run gravou 0,02; um `config` com outro mínimo denuncia limiar mexido depois.
    monkeypatch.setattr(config, "PROMOTION_MIN_GAIN", 0.05)
    assert checks.retrain_problems(_estado(registro)) == [
        f"v{registro.v2}: param promotion_min_gain=0.02 difere do pré-registrado 0.05"
    ]


def test_retreino_com_alerta_nao_consumido_e_apontado(registro):
    registro.client.delete_tag(registro.alerta, "retrain_consumed")
    assert checks.retrain_problems(_estado(registro)) == [
        f"v{registro.v2}: alerta {registro.alerta} sem a tag retrain_consumed"
    ]


def test_alerta_que_nao_e_critico_e_apontado(registro):
    registro.client.set_tag(registro.alerta, "alert_level", "ATENCAO")
    assert checks.retrain_problems(_estado(registro)) == [
        f"v{registro.v2}: alerta {registro.alerta} com alert_level='ATENCAO', esperado 'CRITICO'"
    ]


def test_aliases_coerentes_com_o_arquivo_servido_passam(registro):
    estado = _estado(registro)
    assert checks.alias_problems(estado, SHA_BASELINE, SHA_BASELINE) == []


def test_champion_divergente_do_arquivo_servido_e_apontado(registro):
    problemas = checks.alias_problems(_estado(registro), SHA_RETREINO, SHA_BASELINE)
    assert problemas == [
        f"champion v{registro.v1}: model_sha256 difere de {config.project_relative(config.CHAMPION_FILE)}"
    ]


def test_sem_champion_ou_arquivo_servido_e_apontado(registro):
    _alias(registro, "champion", None)
    problemas = checks.alias_problems(_estado(registro), None, SHA_BASELINE)
    assert problemas == [
        "sem o alias champion",
        f"sem o arquivo servido {config.project_relative(config.CHAMPION_FILE)}",
    ]


def test_challenger_fora_de_retreino_e_previous_igual_ao_champion_sao_apontados(
    registro,
):
    _alias(registro, "challenger", registro.v1)
    _alias(registro, "previous", registro.v1)
    problemas = checks.alias_problems(_estado(registro), SHA_BASELINE, SHA_BASELINE)
    assert problemas == [
        f"challenger v{registro.v1}: não é uma versão origin=retrain",
        f"previous v{registro.v1}: igual ao champion",
    ]


def test_baseline_de_models_alterado_e_apontado(registro):
    problemas = checks.alias_problems(_estado(registro), SHA_BASELINE, SHA_RETREINO)
    assert problemas == [
        f"{config.project_relative(config.MODEL_FILE)}: sha256 não é o de nenhuma "
        "versão origin=baseline (promoção e rollback nunca o tocam)"
    ]


def test_auditoria_de_promocao_e_rollback_passa(registro):
    _promover(registro)
    assert checks.audit_problems(_estado(registro)) == []
    _reverter(registro)
    estado = _estado(registro)
    assert [r.data.tags["action"] for r in estado.audit_runs] == ["rollback", "promote"]
    assert checks.audit_problems(estado) == []


def test_auditoria_mais_recente_divergente_dos_aliases_e_apontada(registro):
    _promover(registro)
    _alias(registro, "champion", registro.v1)  # mudado por fora do comando
    problemas = checks.audit_problems(_estado(registro))
    assert problemas == [
        f"auditoria mais recente: to_version={registro.v2}, mas champion é v{registro.v1}",
    ]


def test_auditoria_inconsistente_ou_promocao_sem_previous_e_apontada(registro):
    _promover(registro, previous_version=None)
    run = _estado(registro).audit_runs[0]
    registro.client.set_tag(run.info.run_id, "status", "INCONSISTENT")
    problemas = checks.audit_problems(_estado(registro))
    assert problemas == [
        f"auditoria {run.info.run_id}: promoção sem previous_version",
        "auditoria mais recente: status='INCONSISTENT', esperado 'CONSISTENT'",
        f"auditoria mais recente: previous_version=None, mas previous é v{registro.v1}",
    ]


def test_aceite_real_pula_sem_retreino(client):
    with pytest.raises(pytest.skip.Exception, match="sem retreino"):
        aceite_real.registro_ou_pular(client)


def test_aceite_real_pula_auditoria_sem_promocao(registro):
    with pytest.raises(pytest.skip.Exception, match="sem promoção"):
        aceite_real.auditorias_ou_pular(_estado(registro))


def test_aceite_real_falha_com_retreino_divergente(registro):
    registro.client.delete_tag(registro.alerta, "retrain_consumed")
    estado = aceite_real.registro_ou_pular(registro.client)
    with pytest.raises(AssertionError, match="retrain_consumed"):
        aceite_real.test_versoes_de_retreino_com_tags_e_run_de_origem(estado)
