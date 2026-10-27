"""Promoção e rollback humanos do champion, com auditoria (spec da Etapa 3, §4.7).

Os modelos são minúsculos (regressão logística em 60 linhas sintéticas) e as AUCs do
challenger são métricas controladas no run de origem: o que se testa é a ordem da
promoção, a reavaliação da regra, a conferência do arquivo e a trilha de auditoria.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.models import predict
from src.models.integrity import verify_model_integrity
from src.monitoring import monitor, promote, registry
from src.monitoring.registry import (
    export_champion,
    get_alias_version,
    register_challenger,
    register_model_version,
    set_alias,
)
from src.monitoring.stage import champion_sha256
from src.monitoring.tracking import TrackingUnavailableError, configure_tracking
from tests.conftest import cliente_mlflow, make_valid_df, sha256_de

ELEGIVEL = {
    "champion_holdout_auc": 0.57,
    "challenger_holdout_auc": 0.62,
    "champion_reference_auc": 0.75,
    "challenger_reference_auc": 0.745,
}
# Ganho de 0,01 no holdout: abaixo do mínimo de 0,02.
INELEGIVEL = {**ELEGIVEL, "challenger_holdout_auc": 0.58}


def _versao(
    tmp_path: Path,
    nome: str,
    seed: int,
    *,
    aucs: dict[str, float] | None = None,
    champion_version: int | None = 1,
) -> tuple[int, str]:
    """Registra uma versão com bundle real; `aucs` a torna um challenger (run de retreino)."""
    df = make_valid_df(n=60, seed=seed)
    X = df[["ext_source_2", "age_years"]]
    pipeline = Pipeline([("model", LogisticRegression())]).fit(X, df["target"])
    retreino = aucs is not None
    model_version = f"retrain-{nome}" if retreino else "baseline-v1"
    pasta = tmp_path / "versoes" / nome
    pasta.mkdir(parents=True)
    bundle = pasta / "modelo.joblib"
    joblib.dump(
        {"pipeline": pipeline, "threshold": 0.5, "model_version": model_version}, bundle
    )
    sha = sha256_de(bundle)
    metadata = pasta / "meta.json"
    metadata.write_text(json.dumps({"artifact": {"sha256": sha}}), encoding="utf-8")
    exemplo = X.head(3)
    run_tags = {"run_type": "retrain" if retreino else "training", "model_sha256": sha}
    if retreino and champion_version is not None:
        run_tags["champion_version"] = str(champion_version)
    tags = {"model_sha256": sha, "bundle_model_version": model_version}
    if not retreino:
        tags["origin"] = "baseline"
    configure_tracking()
    with mlflow.start_run(run_name=nome, tags=run_tags) as run:
        if retreino:
            mlflow.log_metrics(aucs)
        versao = (register_challenger if retreino else register_model_version)(
            run_id=run.info.run_id,
            pipeline=pipeline,
            signature=infer_signature(exemplo, pipeline.predict_proba(exemplo)),
            input_example=exemplo,
            bundle_path=bundle,
            metadata_path=metadata,
            tags=tags,
        )
    return versao, sha


@dataclass
class Cenario:
    tmp_path: Path
    models_dir: Path
    baseline_file: Path
    baseline_sha: str
    sha1: str


@pytest.fixture
def cenario(tracking_env, tmp_path, champion_dir, monkeypatch) -> Cenario:
    """v1 = baseline, champion servido em `models/champion/`; `models/` do treino à parte."""
    monkeypatch.setattr(registry.getpass, "getuser", lambda: "analista")
    models_dir = champion_dir.parent
    models_dir.mkdir(parents=True)
    baseline = models_dir / "baseline_model.joblib"
    baseline.write_bytes(b"artefato do treino: promocao e rollback nunca o tocam")
    monkeypatch.setattr(config, "MODEL_FILE", baseline)
    v1, sha1 = _versao(tmp_path, "baseline", seed=0)
    export_champion(v1)
    set_alias("champion", v1)
    return Cenario(tmp_path, models_dir, baseline, sha256_de(baseline), sha1)


def _alias(nome: str) -> int | None:
    versao = get_alias_version(nome)
    return None if versao is None else int(versao.version)


def _snapshot(pasta: Path) -> dict[str, str]:
    return {
        str(p.relative_to(pasta)): sha256_de(p)
        for p in sorted(pasta.rglob("*"))
        if p.is_file()
    }


def _auditorias(uri: str) -> list:
    cliente = cliente_mlflow(uri)
    experimento = cliente.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    runs = cliente.search_runs(
        [experimento.experiment_id],
        filter_string="tags.run_type = 'audit'",
        order_by=["attributes.start_time ASC"],
    )
    return list(runs)


def _servido_confere() -> dict:
    """O que `drift_demo`/`monitor` fazem: champion do Registry contra o arquivo."""
    return predict.load_bundle(
        predict.default_model_path(), expected_sha256=champion_sha256()
    )


# MARK: promoção


def test_promote_reavalia_e_recusa_inelegivel(cenario, tracking_env, capsys):
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=INELEGIVEL)
    # Tags da versão editadas para "parecer" elegível: a regra usa as métricas do run.
    cliente = cliente_mlflow(tracking_env)
    for chave, valor in (
        ("promotion_eligible", "true"),
        ("challenger_holdout_auc", "0.99"),
    ):
        cliente.set_model_version_tag(
            config.REGISTERED_MODEL_NAME, str(v2), chave, valor
        )
    antes = _snapshot(cenario.models_dir)

    assert promote.main([]) == promote.EXIT_REFUSED

    erro = capsys.readouterr().err
    assert "RECUSADO" in erro
    assert "gain_below_min" in erro
    assert "--override" in erro
    assert (_alias("champion"), _alias("previous"), _alias("challenger")) == (
        1,
        None,
        v2,
    )
    assert _snapshot(cenario.models_dir) == antes
    assert _auditorias(tracking_env) == []


def test_promote_override_exige_motivo_e_audita(cenario, tracking_env, capsys):
    v2, sha2 = _versao(cenario.tmp_path, "challenger", seed=1, aucs=INELEGIVEL)
    with pytest.raises(SystemExit) as saida:
        promote.main(["--override"])
    assert saida.value.code == monitor.EXIT_OPERATIONAL_ERROR
    assert "--reason" in capsys.readouterr().err
    assert _alias("champion") == 1

    motivo = "Comitê de crédito aceitou o ganho de 0,01 (ata 42)"
    assert promote.main(["--override", "--reason", motivo]) == promote.EXIT_OK

    assert (_alias("champion"), _alias("previous")) == (v2, 1)
    assert sha256_de(config.CHAMPION_FILE) == sha2
    (auditoria,) = _auditorias(tracking_env)
    tags = auditoria.data.tags
    assert tags["reason"] == motivo
    assert tags["status"] == "CONSISTENT"
    detalhes = json.loads(tags["details"])
    assert detalhes["override"] is True
    assert detalhes["eligible"] is False
    assert detalhes["reasons"] == ["gain_below_min"]
    assert detalhes["from_version"] == 1


def test_promote_elegivel_move_aliases_e_exporta_modelo(cenario, tracking_env, capsys):
    v2, sha2 = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)

    assert promote.main(["--version", str(v2)]) == promote.EXIT_OK

    assert (_alias("champion"), _alias("previous")) == (v2, 1)
    tag = get_alias_version("champion").tags["model_sha256"]
    assert sha256_de(config.CHAMPION_FILE) == tag == sha2
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)
    assert sha256_de(cenario.baseline_file) == cenario.baseline_sha
    assert _servido_confere()["model_version"] == "retrain-challenger"
    assert "CONSISTENT" in capsys.readouterr().out
    # Nada de temporário deixado em models/champion/.
    assert sorted(p.name for p in config.CHAMPION_DIR.iterdir()) == [
        "baseline_metadata.json",
        "baseline_model.joblib",
    ]


# `registry.os` é o próprio módulo `os`: guarda o `replace` real antes dos patches.
_OS_REPLACE = os.replace


def _replace_que_falha_no_modelo(original=_OS_REPLACE):
    def _replace(origem, destino):
        if Path(destino) == Path(config.CHAMPION_FILE):
            raise PermissionError(13, "Permission denied", str(destino))
        return original(origem, destino)

    return _replace


def test_promote_falha_no_meio_registra_inconsistente(
    cenario, tracking_env, monkeypatch, capsys
):
    """Falha na troca do arquivo, depois de mover `previous` e `champion`, sem reversão.

    O MLflow cai logo depois de mover os aliases: a reversão não consegue devolvê-los,
    o run de auditoria fica `INCONSISTENT` e o `load_bundle` recusa o estado partido.
    """
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    original = registry.set_alias
    chamadas: list[tuple[str, int]] = []

    def _set_alias(alias, version):
        chamadas.append((alias, int(version)))
        if len(chamadas) > 2:  # previous e champion passam; a reversão não
            raise TrackingUnavailableError("MLflow indisponível (simulado)")
        original(alias, version)

    monkeypatch.setattr(registry, "set_alias", _set_alias)
    monkeypatch.setattr(registry.os, "replace", _replace_que_falha_no_modelo())

    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR

    assert chamadas[:2] == [("previous", 1), ("champion", v2)]
    erro = capsys.readouterr().err
    assert "INCONSISTENT" in erro
    assert "rollback" in erro
    (auditoria,) = _auditorias(tracking_env)
    assert auditoria.data.tags["status"] == "INCONSISTENT"
    detalhes = json.loads(auditoria.data.tags["details"])
    assert detalhes["failed_step"] == "replace_file"
    assert detalhes["revert_failures"]
    assert detalhes["served_problems"]
    # Estado partido: Registry diz v2, arquivo é o da v1 => recusado (fail-closed).
    # O `previous` não é tocado enquanto o `champion` não volta: o rollback recupera.
    assert (_alias("champion"), _alias("previous")) == (v2, 1)
    assert sha256_de(config.CHAMPION_FILE) == cenario.sha1
    with pytest.raises(SystemExit, match="Modelo recusado"):
        _servido_confere()
    assert sha256_de(cenario.baseline_file) == cenario.baseline_sha

    # MLflow e disco de volta (só as duas falhas simuladas; as fixtures continuam).
    monkeypatch.setattr(registry, "set_alias", original)
    monkeypatch.setattr(registry.os, "replace", _OS_REPLACE)
    assert promote.main(["rollback", "--reason", "Recuperar a promoção partida"]) == 0
    assert (_alias("champion"), _alias("previous")) == (1, None)
    assert _servido_confere()["model_version"] == "baseline-v1"


def test_promote_interrompido_tambem_reverte_e_audita(
    cenario, tracking_env, monkeypatch
):
    """Interrupção fora das falhas previstas (Ctrl+C na troca): reverte e audita."""
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)

    def _interromper(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(registry.StagedChampion, "install", _interromper)
    antes = _snapshot(cenario.models_dir)

    with pytest.raises(KeyboardInterrupt):
        promote.main([])

    assert (_alias("champion"), _alias("previous")) == (1, None)
    assert _snapshot(cenario.models_dir) == antes
    (auditoria,) = _auditorias(tracking_env)
    assert auditoria.data.tags["status"] == "INCONSISTENT"
    detalhes = json.loads(auditoria.data.tags["details"])
    assert (detalhes["failed_step"], detalhes["error"]) == (
        "replace_file",
        "interrompido",
    )
    assert _servido_confere()["model_version"] == "baseline-v1"


def test_promote_falha_ao_mover_champion_reverte_previous(
    cenario, tracking_env, monkeypatch, capsys
):
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    original = registry.set_alias
    falhou: list[bool] = []

    def _set_alias(alias, version):
        if alias == "champion" and not falhou:
            falhou.append(True)
            raise TrackingUnavailableError("MLflow indisponível (simulado)")
        original(alias, version)

    monkeypatch.setattr(registry, "set_alias", _set_alias)
    antes = _snapshot(cenario.models_dir)

    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR

    # `previous` revertido (não existia), champion e arquivo intactos.
    assert (_alias("champion"), _alias("previous")) == (1, None)
    assert _snapshot(cenario.models_dir) == antes
    (auditoria,) = _auditorias(tracking_env)
    assert auditoria.data.tags["status"] == "INCONSISTENT"
    detalhes = json.loads(auditoria.data.tags["details"])
    assert detalhes["failed_step"] == "move_aliases"
    assert detalhes["revert_failures"] == []
    assert detalhes["served_problems"] == []
    assert "INCONSISTENT" in capsys.readouterr().err
    assert _servido_confere()["model_version"] == "baseline-v1"


def test_promote_falha_depois_da_troca_reexporta_o_arquivo_anterior(
    cenario, tracking_env, monkeypatch
):
    """Falha logo depois de o arquivo ser trocado: aliases e arquivo voltam à v1."""
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    install_real = registry.StagedChampion.install
    falhou: list[bool] = []

    def _install_e_falha(self):
        install_real(self)
        if not falhou:  # só a troca da promoção; a reexportação da v1 passa
            falhou.append(True)
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(registry.StagedChampion, "install", _install_e_falha)

    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR

    assert (_alias("champion"), _alias("previous")) == (1, None)
    assert sha256_de(config.CHAMPION_FILE) == cenario.sha1
    (auditoria,) = _auditorias(tracking_env)
    detalhes = json.loads(auditoria.data.tags["details"])
    assert "arquivo=v1" in detalhes["reverted"]
    assert detalhes["served_problems"] == []
    assert _servido_confere()["model_version"] == "baseline-v1"


def test_promote_falha_sem_mlflow_nem_para_auditar_vai_ao_log(
    cenario, tracking_env, monkeypatch, capsys
):
    _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    original = registry.set_alias

    def _set_alias(alias, version):
        if alias == "champion":
            raise TrackingUnavailableError("MLflow indisponível (simulado)")
        original(alias, version)

    def _sem_auditoria(**kwargs):
        raise TrackingUnavailableError("Falha ao registrar o run de auditoria")

    monkeypatch.setattr(registry, "set_alias", _set_alias)
    monkeypatch.setattr(registry, "log_audit_run", _sem_auditoria)

    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR

    erro = capsys.readouterr().err
    assert "INCONSISTENT" in erro
    assert "auditoria falhou" in erro


def test_promote_champion_mudou_durante_o_comando_nao_altera_nada(
    cenario, tracking_env, monkeypatch, capsys
):
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    v3, _ = _versao(cenario.tmp_path, "outro", seed=2, aucs=ELEGIVEL)
    avaliar = promote.evaluate_candidate

    def _avaliar_e_alguem_promove(version):
        avaliacao = avaliar(version)
        set_alias("champion", v3)  # promoção em paralelo
        return avaliacao

    monkeypatch.setattr(promote, "evaluate_candidate", _avaliar_e_alguem_promove)
    antes = _snapshot(cenario.models_dir)

    assert promote.main(["--version", str(v2)]) == monitor.EXIT_OPERATIONAL_ERROR

    assert "mudou durante" in capsys.readouterr().err
    assert (_alias("champion"), _alias("previous")) == (v3, None)
    assert _snapshot(cenario.models_dir) == antes
    assert _auditorias(tracking_env) == []


def test_promote_auditoria_final_falha_avisa_que_a_troca_foi_feita(
    cenario, tracking_env, monkeypatch, capsys
):
    v2, sha2 = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)

    def _sem_auditoria(**kwargs):
        raise TrackingUnavailableError("Falha ao registrar o run de auditoria")

    monkeypatch.setattr(registry, "log_audit_run", _sem_auditoria)

    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR

    erro = capsys.readouterr().err
    assert "concluída e reconferida (CONSISTENT)" in erro
    assert "não foi gravado" in erro
    assert (_alias("champion"), _alias("previous")) == (v2, 1)
    assert sha256_de(config.CHAMPION_FILE) == sha2


def test_promote_sem_champion_falha_com_mensagem(tracking_env, tmp_path, capsys):
    _versao(tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR
    assert "make pipeline" in capsys.readouterr().err


def test_reconferencia_aponta_arquivo_ausente_metadata_e_registry(
    cenario, tracking_env, monkeypatch
):
    assert promote.served_state_problems(1, None) == []

    def _ilegivel(path):
        raise PermissionError(13, "Permission denied", str(path))

    with monkeypatch.context() as m:
        m.setattr(promote, "file_sha256", _ilegivel)
        assert any("ilegível" in p for p in promote.served_state_problems(1, None))
    config.CHAMPION_METADATA_FILE.write_text("{}", encoding="utf-8")
    assert any("artifact.sha256" in p for p in promote.served_state_problems(1, None))
    config.CHAMPION_FILE.unlink()
    assert any("ausente" in p for p in promote.served_state_problems(1, None))

    def _indisponivel(alias):
        raise TrackingUnavailableError("MLflow indisponível (simulado)")

    monkeypatch.setattr(promote, "get_alias_version", _indisponivel)
    (problema,) = promote.served_state_problems(1, None)
    assert "reconferir" in problema


def test_promote_divergencia_na_reconferencia_encerra_com_erro(
    cenario, tracking_env, monkeypatch, capsys
):
    """Troca que "passa" sem trocar o arquivo: a reconferência pega e grava INCONSISTENT."""
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    monkeypatch.setattr(registry.StagedChampion, "install", lambda self: None)

    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR

    (auditoria,) = _auditorias(tracking_env)
    assert auditoria.data.tags["status"] == "INCONSISTENT"
    assert json.loads(auditoria.data.tags["details"])["served_problems"]
    assert "INCONSISTENT" in capsys.readouterr().err
    assert _alias("champion") == v2
    with pytest.raises(SystemExit, match="Modelo recusado"):
        _servido_confere()


def test_export_atomico_nao_deixa_arquivo_parcial(
    cenario, tracking_env, monkeypatch, capsys
):
    """Falha no meio da cópia do bundle: `models/` fica como estava, aliases também."""
    _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    antes = _snapshot(cenario.models_dir)

    def _download_parcial(self, run_id, path, dst_path=None):
        parcial = Path(dst_path) / path / registry.BUNDLE_MODEL_FILE
        parcial.parent.mkdir(parents=True, exist_ok=True)
        parcial.write_bytes(b"metade de um modelo")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(MlflowClient, "download_artifacts", _download_parcial)

    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR

    # Erro de disco no download local passa por `tracking_errors` (mensagem redigida).
    assert "Falha ao baixar o bundle" in capsys.readouterr().err
    assert _snapshot(cenario.models_dir) == antes
    assert sorted(p.name for p in config.CHAMPION_DIR.iterdir()) == [
        "baseline_metadata.json",
        "baseline_model.joblib",
    ]
    assert (_alias("champion"), _alias("previous")) == (1, None)
    # Nada foi trocado: não há transição a auditar.
    assert _auditorias(tracking_env) == []


def test_promote_tag_de_hash_editada_falha_sem_tocar_em_nada(
    cenario, tracking_env, capsys
):
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    cliente_mlflow(tracking_env).set_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(v2), "model_sha256", "0" * 64
    )
    antes = _snapshot(cenario.models_dir)

    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR

    assert "model_sha256" in capsys.readouterr().err
    assert _snapshot(cenario.models_dir) == antes
    assert (_alias("champion"), _alias("previous")) == (1, None)


def test_promote_challenger_comparado_com_outro_champion_e_inelegivel(
    cenario, tracking_env, capsys
):
    _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL, champion_version=7)

    assert promote.main([]) == promote.EXIT_REFUSED

    assert "champion_mismatch" in capsys.readouterr().err
    assert _alias("champion") == 1


@pytest.mark.parametrize(
    ("argv", "trecho"),
    [
        ([], "challenger"),  # sem alias challenger
        (["--version", "1"], "já é o champion"),
        (["--version", "99"], "Versão 99"),
    ],
)
def test_promote_sem_candidato_valido_falha_com_mensagem(
    cenario, tracking_env, capsys, argv, trecho
):
    assert promote.main(argv) == monitor.EXIT_OPERATIONAL_ERROR
    assert trecho in capsys.readouterr().err
    assert _alias("champion") == 1


def test_promote_sem_rastreamento_falha_fechado(monkeypatch, capsys):
    monkeypatch.setenv("MONITORING_TRACKING", "off")
    assert promote.main([]) == monitor.EXIT_OPERATIONAL_ERROR
    assert promote.main(["rollback"]) == monitor.EXIT_OPERATIONAL_ERROR
    assert "MONITORING_TRACKING=off" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--version", "0"],
        ["--version", "x"],
        ["--override", "--reason", "x" * 9000],
        ["rollback", "--reason", "x" * 9000],
    ],
)
def test_promote_versao_invalida_e_erro_de_uso(argv):
    with pytest.raises(SystemExit) as saida:
        promote.main(argv)
    assert saida.value.code == monitor.EXIT_OPERATIONAL_ERROR


# MARK: rollback


def test_rollback_restaura_arquivo_e_alias(cenario, tracking_env):
    v2, sha2 = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    assert promote.main([]) == promote.EXIT_OK
    assert sha256_de(config.CHAMPION_FILE) == sha2

    assert promote.main(["rollback", "--reason", "Regressão observada"]) == 0

    assert (_alias("champion"), _alias("previous")) == (1, None)
    assert sha256_de(config.CHAMPION_FILE) == cenario.sha1
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)
    assert _servido_confere()["model_version"] == "baseline-v1"
    assert sha256_de(cenario.baseline_file) == cenario.baseline_sha


def test_promote_depois_do_rollback_exige_override(cenario, tracking_env, capsys):
    """promote -> rollback -> promote: a versão revertida não volta como elegível.

    O alias `challenger` continua na v2 e as AUCs do run de retreino seguem válidas
    contra a v1, mas a v2 foi revertida por regressão: só volta com `--override`.
    """
    v2, sha2 = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    assert promote.main([]) == promote.EXIT_OK
    assert promote.main(["rollback", "--reason", "Regressão observada"]) == 0
    capsys.readouterr()
    antes = _snapshot(cenario.models_dir)

    for argv in ([], ["--version", str(v2)]):
        assert promote.main(argv) == promote.EXIT_REFUSED
        erro = capsys.readouterr().err
        assert "rolled_back" in erro
        assert "--override" in erro
    assert (_alias("champion"), _alias("previous"), _alias("challenger")) == (
        1,
        None,
        v2,
    )
    assert _snapshot(cenario.models_dir) == antes
    assert len(_auditorias(tracking_env)) == 2  # só a promoção e o rollback

    motivo = "Correção da regressão confirmada (chamado 7)"
    assert promote.main(["--override", "--reason", motivo]) == promote.EXIT_OK
    assert (_alias("champion"), _alias("previous")) == (v2, 1)
    assert sha256_de(config.CHAMPION_FILE) == sha2
    detalhes = json.loads(_auditorias(tracking_env)[-1].data.tags["details"])
    assert (detalhes["eligible"], detalhes["reasons"]) == (False, ["rolled_back"])


def test_rollback_sem_previous_falha(cenario, tracking_env, capsys):
    antes = _snapshot(cenario.models_dir)

    assert promote.main(["rollback"]) == monitor.EXIT_OPERATIONAL_ERROR

    assert "previous" in capsys.readouterr().err
    assert _alias("champion") == 1
    assert _snapshot(cenario.models_dir) == antes
    assert _auditorias(tracking_env) == []


def test_rollback_falha_na_troca_devolve_os_aliases(
    cenario, tracking_env, monkeypatch, capsys
):
    v2, sha2 = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    assert promote.main([]) == promote.EXIT_OK
    monkeypatch.setattr(registry.os, "replace", _replace_que_falha_no_modelo())

    assert promote.main(["rollback"]) == monitor.EXIT_OPERATIONAL_ERROR

    # Aliases de volta ao estado antes do rollback (o previous consumido é refeito).
    assert (_alias("champion"), _alias("previous")) == (v2, 1)
    assert sha256_de(config.CHAMPION_FILE) == sha2
    assert _servido_confere()["model_version"] == "retrain-challenger"
    rollback = _auditorias(tracking_env)[-1].data.tags
    assert (rollback["action"], rollback["status"]) == ("rollback", "INCONSISTENT")
    assert "INCONSISTENT" in capsys.readouterr().err


# MARK: auditoria e integração com o champion servido


def test_promote_e_rollback_geram_run_de_auditoria(cenario, tracking_env):
    v2, sha2 = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    assert promote.main([]) == 0
    assert promote.main(["rollback"]) == 0

    promocao, rollback = (r.data.tags for r in _auditorias(tracking_env))
    assert (promocao["action"], promocao["version"]) == ("promote", str(v2))
    assert (rollback["action"], rollback["version"]) == ("rollback", "1")
    for tags in (promocao, rollback):
        assert tags["run_type"] == "audit"
        assert tags["status"] == "CONSISTENT"
        assert tags["operator"] == "analista"
        assert tags["reason"].strip()
        assert tags["timestamp_utc"].endswith("+00:00")
    detalhes = json.loads(promocao["details"])
    assert detalhes["eligible"] is True
    assert detalhes["override"] is False
    assert detalhes["model_sha256"] == sha2
    assert detalhes["challenger_holdout_auc"] == ELEGIVEL["challenger_holdout_auc"]
    assert json.loads(rollback["details"])["from_version"] == v2


def test_load_bundle_com_champion_divergente_falha(
    cenario, tracking_env, tmp_path, monkeypatch
):
    """Arquivo de `models/champion/` da v1 e alias `champion` na v2: o monitor recusa."""
    v2, _ = _versao(cenario.tmp_path, "challenger", seed=1, aucs=ELEGIVEL)
    set_alias("champion", v2)  # Registry mudou sem o arquivo acompanhar
    referencia = tmp_path / "reference.parquet"
    make_valid_df(n=50, seed=3).to_parquet(referencia)
    monkeypatch.setattr(config, "REFERENCE_FILE", referencia)
    lote = tmp_path / "lote.csv"
    make_valid_df(n=50, seed=4, labeled=False).to_csv(lote, index=False)

    with pytest.raises(SystemExit) as saida:
        monitor.main(["--batch", str(lote)])
    assert "Modelo recusado" in str(saida.value.code)
    with pytest.raises(SystemExit, match="Modelo recusado"):
        _servido_confere()
