"""Gancho de lote da demo de drift e estágio `drift_demo` (spec da Etapa 3, §4.1-§4.4).

Tudo roda sobre dados SINTÉTICOS pequenos (sem Kaggle): t0 estável, t1 com as
colunas deslocadas por `apply_data_drift` e t2 com os rótulos invertidos (AUC
derrubado). Estes testes conferem a fiação; não substituem o aceite real com os
dados da demo (Tarefa 11).
"""

from __future__ import annotations

import json
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from src import config
from src.drift import demo as drift_demo
from src.drift import simulate
from src.drift.demo import BatchObservation
from src.drift.detect import ColumnDrift, DriftResult
from src.models import predict, train
from src.models.integrity import file_sha256
from src.monitoring import stage
from src.monitoring.alerts import ACTIONS, AlertLevel
from src.validation.gate import enforce_contract
from tests.conftest import cliente_mlflow, isolar_estado_mlflow, make_valid_df

FAST_XGB = {"n_estimators": 20, "n_jobs": 1}
EVIDENTLY_SAMPLE = 1000
N_PRODUCAO = 2000
LOTES = ["production_t0", "production_t1_data_drift", "production_t2_concept_drift"]


# --------------------------------------------------------------------------
# load_bundle
# --------------------------------------------------------------------------
def _gravar_bundle(pasta: Path, versao: str = "t") -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    modelo = pasta / "baseline_model.joblib"
    joblib.dump({"model_version": versao}, modelo)
    (pasta / "baseline_metadata.json").write_text(
        json.dumps({"artifact": {"sha256": file_sha256(modelo)}}), encoding="utf-8"
    )
    return modelo


def test_load_bundle_recusa_hash_divergente(tmp_path):
    modelo = _gravar_bundle(tmp_path)
    with pytest.raises(SystemExit, match="Modelo recusado"):
        predict.load_bundle(modelo, expected_sha256="0" * 64)
    assert predict.load_bundle(modelo)["model_version"] == "t"
    sha = file_sha256(modelo)
    assert predict.load_bundle(modelo, expected_sha256=sha)["model_version"] == "t"


def test_load_bundle_prefere_champion_dir(tmp_path, monkeypatch, champion_dir):
    treino = _gravar_bundle(tmp_path / "models", versao="treino")
    monkeypatch.setattr(config, "MODEL_FILE", treino)
    assert predict.load_bundle()["model_version"] == "treino"

    _gravar_bundle(champion_dir, versao="champion")
    assert predict.load_bundle()["model_version"] == "champion"


# --------------------------------------------------------------------------
# Observador do estágio (unidade, sem MLflow)
# --------------------------------------------------------------------------
class _RunFalso:
    """Captura os `BatchRecord` em vez de abrir runs no MLflow."""

    enabled = True

    def __init__(self) -> None:
        self.records = []

    def log_batch(self, record):
        self.records.append(record)
        return None


METADATA_FALSO = {
    "official_model": "xgboost",
    "models": {"xgboost": {"test_metrics": {"roc_auc": 0.75}}},
    "score_reference": {"edges": [0.25, 0.5, 0.75], "shares": [0.25] * 4},
}


def _drift(nome: str, colunas: dict[str, bool]) -> DriftResult:
    resultados = [
        ColumnDrift(
            column=coluna,
            method="ks",
            statistic=0.001 if driftou else 0.9,
            threshold=0.05,
            drifted=driftou,
            psi=0.5 if driftou else 0.01,
        )
        for coluna, driftou in colunas.items()
    ]
    n = sum(colunas.values())
    return DriftResult(
        batch_name=nome,
        n_rows_reference=100,
        n_rows_current=100,
        columns_checked=list(colunas),
        n_drifted_columns=n,
        drifted_share=n / len(colunas),
        dataset_drift=False,
        column_results=resultados,
        concept_drift=colunas.get("target", False),
    )


def _observacao(
    tmp_path, nome: str, *, colunas: dict[str, bool], rotulado: bool = True
) -> BatchObservation:
    scored = pd.DataFrame({"default_probability": [0.1, 0.3, 0.6, 0.9] * 25})
    metrics: dict = {}
    if rotulado:
        scored["target"] = [0, 0, 1, 1] * 25
        metrics = {"roc_auc": 0.8, "approval_rate": 0.5, "default_rate_approved": 0.1}
    return BatchObservation(
        batch_name=nome,
        scored=scored,
        metrics=metrics,
        drift=_drift(nome, colunas),
        report_dir=tmp_path,
        model_version="v-teste",
    )


def test_target_nao_conta_como_coluna_driftada(tmp_path, monkeypatch):
    recebidas = []
    original = stage.decide_alert_level

    def _espiao(**kwargs):
        recebidas.append(list(kwargs["drifted_columns"]))
        return original(**kwargs)

    monkeypatch.setattr(stage, "decide_alert_level", _espiao)
    run = _RunFalso()
    observador = stage.DriftBatchObserver(run, METADATA_FALSO)
    observador(
        _observacao(
            tmp_path, "production_t0", colunas={"ext_source_3": False, "target": True}
        )
    )

    assert recebidas == [[]]
    (record,) = run.records
    assert record.n_drifted_columns == 0
    assert "drift:target" not in record.decision.reasons
    assert "target" not in record.psi_by_column
    assert record.metrics["label_shift"] == 1.0
    assert record.decision.level.value == "OK"


@pytest.mark.parametrize(
    ("driftou", "nivel"), [(True, "ATENCAO"), (False, "OK")], ids=["drift", "estavel"]
)
def test_lote_sem_rotulo_nao_avalia_performance(tmp_path, driftou, nivel):
    run = _RunFalso()
    observador = stage.DriftBatchObserver(run, METADATA_FALSO)
    observador(
        _observacao(
            tmp_path,
            "lote_sem_rotulo",
            colunas={"ext_source_3": driftou},
            rotulado=False,
        )
    )

    (record,) = run.records
    assert record.decision.performance_evaluated is False
    assert record.decision.level.value == nivel
    assert record.metrics.get("roc_auc") is None
    assert record.metrics.get("auc_drop") is None


def test_auc_nao_finita_nao_sai_ok(tmp_path):
    run = _RunFalso()
    observador = stage.DriftBatchObserver(run, METADATA_FALSO)
    obs = _observacao(tmp_path, "production_t0", colunas={"ext_source_3": False})
    obs.metrics["roc_auc"] = float("nan")
    observador(obs)

    (record,) = run.records
    assert record.decision.performance_evaluated is False
    assert "auc_undefined" in record.decision.reasons
    assert record.decision.level.value == "ATENCAO"
    assert record.metrics["roc_auc"] is None
    assert record.metrics["auc_drop"] is None


def test_sem_baseline_no_t0_nao_avalia_t1(tmp_path):
    run = _RunFalso()
    observador = stage.DriftBatchObserver(run, METADATA_FALSO)
    t0 = _observacao(tmp_path, "production_t0", colunas={"ext_source_3": False})
    t0.scored["target"] = 0  # t0 de classe única: sem AUC de baseline
    observador(t0)
    observador(_observacao(tmp_path, "production_t1", colunas={"ext_source_3": False}))

    primeiro, segundo = run.records
    assert "single_class" in primeiro.decision.reasons
    assert "no_baseline_auc" in segundo.decision.reasons
    assert segundo.decision.performance_evaluated is False
    assert segundo.metrics["roc_auc"] == 0.8
    assert segundo.metrics["auc_drop"] is None


def test_lote_de_classe_unica_registra_single_class(tmp_path):
    run = _RunFalso()
    observador = stage.DriftBatchObserver(run, METADATA_FALSO)
    obs = _observacao(tmp_path, "production_t0", colunas={"ext_source_3": False})
    obs.scored["target"] = 0
    observador(obs)

    (record,) = run.records
    assert record.decision.performance_evaluated is False
    assert "single_class" in record.decision.reasons


def test_lote_vazio_nao_avalia_score_nem_performance(tmp_path):
    run = _RunFalso()
    observador = stage.DriftBatchObserver(run, METADATA_FALSO)
    obs = _observacao(tmp_path, "production_t0", colunas={"ext_source_3": False})
    vazio = BatchObservation(
        batch_name=obs.batch_name,
        scored=obs.scored.iloc[0:0],
        metrics={},
        drift=obs.drift,
        report_dir=obs.report_dir,
        model_version=obs.model_version,
    )
    observador(vazio)

    (record,) = run.records
    assert record.n_rows == 0
    assert record.metrics.get("score_psi") is None
    assert "empty_batch" in record.decision.reasons
    # Falha cautelosa: lote vazio não é "tudo bem".
    assert record.decision.level.value == "ATENCAO"
    assert record.decision.action == ACTIONS[AlertLevel.ATENCAO]


# --------------------------------------------------------------------------
# Estágio completo em sandbox (train + drift_demo com MLflow temporário)
# --------------------------------------------------------------------------
def _fatias_controladas(p: SimpleNamespace):
    def _simular(*_args, **_kwargs) -> dict[str, Path]:
        base = pd.read_parquet(p.production)
        # t1: colunas deslocadas (choques da Etapa 2), rótulo intacto.
        simulate.apply_data_drift(base).to_parquet(p.t1, index=False)
        # t2: mesmas features do t0, rótulos invertidos (AUC derrubado).
        t2 = base.copy()
        t2["target"] = 1 - t2["target"]
        t2.to_parquet(p.t2, index=False)
        p.simulacoes += 1
        return {"t1": p.t1, "t2": p.t2}

    return _simular


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory, _banco_modelo_mlflow):
    root = tmp_path_factory.mktemp("drift-stage")
    p = SimpleNamespace(
        root=root,
        banco=root / "mlflow.db",
        reference=root / "reference" / "reference.parquet",
        production=root / "production" / "production_base.parquet",
        t1=root / "production" / "t1.parquet",
        t2=root / "production" / "t2.parquet",
        model=root / "models" / "baseline_model.joblib",
        metadata=root / "models" / "baseline_metadata.json",
        champion=root / "models" / "champion",
        drift_reports=root / "reports" / "drift",
        contract_reports=root / "reports" / "data_contract",
        quarantine=root / "quarantine",
        simulacoes=0,
    )
    shutil.copy(_banco_modelo_mlflow, p.banco)
    p.uri = f"sqlite:///{p.banco}"
    p.reference.parent.mkdir(parents=True)
    p.production.parent.mkdir(parents=True)
    make_valid_df(n=3000, seed=3).to_parquet(p.reference, index=False)
    make_valid_df(n=N_PRODUCAO, seed=5).to_parquet(p.production, index=False)

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
        # demo de drift
        mp.setattr(drift_demo, "PRODUCTION_BASE_FILE", p.production)
        mp.setattr(drift_demo, "REFERENCE_FILE", p.reference)
        mp.setattr(drift_demo, "simulate_production_slices", _fatias_controladas(p))
        mp.setattr(
            drift_demo,
            "run_demo",
            partial(
                drift_demo.run_demo,
                report_dir=p.drift_reports,
                evidently_sample=EVIDENTLY_SAMPLE,
            ),
        )

        assert stage.main(["train"]) == 0
        p.status = stage.main(["drift_demo"])
        yield p


def _runs(uri: str, filtro: str):
    client = cliente_mlflow(uri)
    experimento = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    return client.search_runs(
        [experimento.experiment_id],
        filter_string=filtro,
        order_by=["attributes.start_time ASC"],
    )


def _estagio_drift(uri: str):
    """O run do estágio `drift_demo` da fixture (o único que terminou bem)."""
    (estagio,) = _runs(
        uri,
        "tags.run_type = 'stage' and tags.stage = 'drift_demo' "
        "and tags.stage_status = 'SUCCESS'",
    )
    return estagio


def _filhos(uri: str) -> dict:
    estagio = _estagio_drift(uri)
    filhos = _runs(
        uri,
        f"tags.run_type = 'batch' and tags.mlflow.parentRunId = '{estagio.info.run_id}'",
    )
    return {f.data.tags["batch_name"]: f for f in filhos}


def test_run_demo_chama_on_batch_na_ordem(sandbox):
    vistos: list[BatchObservation] = []
    drift_demo.run_demo(
        on_batch=vistos.append,
        bundle_loader=partial(predict.load_bundle, sandbox.model),
    )
    assert [o.batch_name for o in vistos] == LOTES
    for obs in vistos:
        assert "default_probability" in obs.scored.columns
        assert len(obs.scored) == N_PRODUCAO
        assert obs.drift.batch_name == obs.batch_name
        assert obs.model_version == train.MODEL_VERSION
        assert "roc_auc" in obs.metrics


def test_stage_drift_demo_classifica_lotes(sandbox):
    assert sandbox.status == 0
    estagio = _estagio_drift(sandbox.uri)
    assert estagio.info.status == "FINISHED"
    filhos = _filhos(sandbox.uri)
    assert list(filhos) == LOTES  # ordem de início: t0, t1, t2
    niveis = [filhos[n].data.tags["alert_level"] for n in LOTES]
    assert niveis == ["OK", "ATENCAO", "CRITICO"]
    rotulos = [filhos[n].data.tags["alert_label"] for n in LOTES]
    assert rotulos == ["OK", "ATENÇÃO", "CRÍTICO"]
    assert estagio.data.metrics["rows_processed"] == 3 * N_PRODUCAO
    assert estagio.data.metrics["alerts_drift_count"] == 2
    assert estagio.data.metrics["alerts_failure_count"] == 0
    for filho in filhos.values():
        assert filho.data.tags["performance_evaluated"] == "true"
        assert filho.data.tags["model_version"] == train.MODEL_VERSION
        assert "drift:target" not in filho.data.tags["alert_reasons"]
    t1 = filhos["production_t1_data_drift"].data
    assert "drift:ext_source_3" in t1.tags["alert_reasons"].split(";")


def test_stage_drift_demo_baseline_e_o_auc_do_t0(sandbox):
    filhos = _filhos(sandbox.uri)
    t0 = filhos["production_t0"].data.metrics
    t2 = filhos["production_t2_concept_drift"].data.metrics
    assert t0["auc_drop"] == 0
    assert t2["auc_drop"] == pytest.approx(t0["roc_auc"] - t2["roc_auc"], abs=1e-4)
    assert t2["auc_drop"] >= config.AUC_DROP_ALERT


def test_run_filho_traz_relatorios_e_psi_do_score(sandbox):
    filhos = _filhos(sandbox.uri)
    client = cliente_mlflow(sandbox.uri)
    metadata = json.loads(sandbox.metadata.read_text(encoding="utf-8"))
    auc_treino = metadata["models"]["xgboost"]["test_metrics"]["roc_auc"]
    for nome, filho in filhos.items():
        artefatos = {
            a.path for a in client.list_artifacts(filho.info.run_id, "reports")
        }
        assert f"reports/{nome}_drift_report.html" in artefatos
        assert f"reports/{nome}_drift_report.json" in artefatos
        metricas = filho.data.metrics
        assert "score_psi" in metricas
        assert metricas["auc_drop_vs_training"] == pytest.approx(
            auc_treino - metricas["roc_auc"], abs=1e-4
        )
        assert "psi_ext_source_3" in metricas
        assert "psi_target" not in metricas
        assert metricas["n_rows"] == N_PRODUCAO
        assert len(filho.data.tags["batch_sha256"]) == 64
    # t0 não é regerado por outros testes: o hash tem de ser o do arquivo.
    t0 = filhos["production_t0"].data
    assert Path(config.ROOT_DIR / t0.tags["batch_path"]) == sandbox.production
    assert t0.tags["batch_sha256"] == file_sha256(sandbox.production)
    assert t0.metrics["score_psi"] < config.PSI_MODERATE


def test_stage_drift_demo_sem_champion_falha_com_orientacao(
    sandbox, tracking_env, capsys
):
    antes = sandbox.simulacoes
    assert stage.main(["drift_demo"]) == 1
    saida = capsys.readouterr().err
    assert "champion" in saida
    assert "make pipeline" in saida
    assert "Traceback" not in saida
    assert sandbox.simulacoes == antes, "a demo não pode rodar sem champion"


def test_stage_drift_demo_recusa_champion_adulterado(sandbox, tmp_path, monkeypatch):
    adulterado = tmp_path / "champion"
    shutil.copytree(sandbox.champion, adulterado)
    modelo = adulterado / "baseline_model.joblib"
    modelo.write_bytes(modelo.read_bytes() + b"x")
    metadata = adulterado / "baseline_metadata.json"
    dados = json.loads(metadata.read_text(encoding="utf-8"))
    dados["artifact"]["sha256"] = file_sha256(
        modelo
    )  # par coerente, mas não é o champion
    metadata.write_text(json.dumps(dados), encoding="utf-8")
    monkeypatch.setattr(config, "CHAMPION_DIR", adulterado)
    monkeypatch.setattr(config, "CHAMPION_FILE", modelo)
    monkeypatch.setattr(config, "CHAMPION_METADATA_FILE", metadata)

    with pytest.raises(SystemExit, match="Modelo recusado"):
        stage.main(["drift_demo"])


def test_stage_drift_demo_tracking_off_roda_sem_mlflow(
    sandbox, tmp_path, monkeypatch, mlflow_isolado
):
    banco = tmp_path / "mlflow" / "mlflow.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{banco}")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlartifacts")
    monkeypatch.setenv("MONITORING_TRACKING", "off")
    antes = sandbox.simulacoes

    assert stage.main(["drift_demo"]) == 0
    assert sandbox.simulacoes == antes + 1
    assert not banco.exists()
    assert not (tmp_path / "mlartifacts").exists()


def test_run_demo_padrao_aceita_so_o_champion(sandbox, tmp_path, monkeypatch):
    # Sem o bundle do treino, o carregador padrão usa `models/champion/`.
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "ausente.joblib")
    vistos: list[BatchObservation] = []
    drift_demo.run_demo(on_batch=vistos.append)
    assert [o.batch_name for o in vistos] == LOTES


def test_run_demo_sem_modelo_nenhum_sai_antes_de_simular(
    sandbox, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "ausente.joblib")
    monkeypatch.setattr(config, "CHAMPION_FILE", tmp_path / "champion.joblib")
    antes = sandbox.simulacoes
    with pytest.raises(SystemExit, match="Modelo nao encontrado"):
        drift_demo.run_demo()
    assert sandbox.simulacoes == antes


def _champion_com_metadata(sandbox, tmp_path, monkeypatch, estragar) -> None:
    """Copia o champion para `tmp_path` e estraga o metadata do jeito pedido."""
    pasta = tmp_path / "champion"
    shutil.copytree(sandbox.champion, pasta)
    metadata = pasta / "baseline_metadata.json"
    if estragar == "ausente":
        metadata.unlink()
    elif estragar == "corrompido":
        metadata.write_text("{ isto não é json", encoding="utf-8")
    elif estragar == "sem_score_reference":
        dados = json.loads(metadata.read_text(encoding="utf-8"))
        del dados["score_reference"]
        metadata.write_text(json.dumps(dados), encoding="utf-8")
    monkeypatch.setattr(config, "CHAMPION_DIR", pasta)
    monkeypatch.setattr(config, "CHAMPION_FILE", pasta / "baseline_model.joblib")
    monkeypatch.setattr(config, "CHAMPION_METADATA_FILE", metadata)


@pytest.mark.parametrize("estragar", ["ausente", "corrompido", "sem_score_reference"])
def test_stage_drift_demo_metadata_invalido_orienta_sem_traceback(
    sandbox, tmp_path, monkeypatch, capsys, estragar
):
    _champion_com_metadata(sandbox, tmp_path, monkeypatch, estragar)
    antes = sandbox.simulacoes

    assert stage.main(["drift_demo"]) == 1
    saida = capsys.readouterr().err
    assert "Metadata do modelo" in saida
    assert "make pipeline" in saida
    assert "Traceback" not in saida
    assert sandbox.simulacoes == antes, "a demo não pode rodar sem metadata válido"
