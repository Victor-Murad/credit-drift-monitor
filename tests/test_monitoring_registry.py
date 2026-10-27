"""Model Registry: versões, aliases e cópia do champion (spec da Etapa 3, §4.7)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.models.integrity import ModelIntegrityError, verify_model_integrity
from src.monitoring import registry
from src.monitoring.registry import (
    RegistryStateError,
    alert_consumed_by,
    bootstrap_champion,
    delete_alias,
    export_champion,
    find_unconsumed_critical_run,
    get_alias_version,
    log_audit_run,
    mark_alert_consumed,
    promote_alias,
    register_challenger,
    register_model_version,
    require_critical_run,
    require_version,
    rollback_alias,
    set_alias,
    staged_champion,
)
from src.monitoring.tracking import TrackingUnavailableError, configure_tracking
from tests.conftest import cliente_mlflow, make_valid_df, sha256_de

_SEM_HASH = object()


def _registrar(
    tmp_path: Path,
    nome: str,
    seed: int = 0,
    metadata: object = None,
    *,
    registrar=register_model_version,
    model_version: str = "baseline-v1",
    tags: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Treina um modelo minúsculo, grava bundle + metadata e registra uma versão.

    `metadata=_SEM_HASH` grava um metadata sem `artifact.sha256`. `registrar` troca
    a função de registro (ex.: `register_challenger`); `tags` substitui as padrão,
    com `{sha}` trocado pelo sha256 do bundle.
    """
    df = make_valid_df(n=60, seed=seed)
    X = df[["ext_source_2", "age_years"]]
    pipeline = Pipeline([("model", LogisticRegression())]).fit(X, df["target"])
    pasta = tmp_path / nome
    pasta.mkdir()
    bundle = pasta / "modelo.joblib"
    joblib.dump({"pipeline": pipeline, "model_version": model_version}, bundle)
    sha = sha256_de(bundle)
    conteudo = {"outro": 1} if metadata is _SEM_HASH else {"artifact": {"sha256": sha}}
    arquivo_metadata = pasta / "meta.json"
    arquivo_metadata.write_text(json.dumps(conteudo), encoding="utf-8")
    exemplo = X.head(3)
    assinatura = infer_signature(exemplo, pipeline.predict_proba(exemplo))
    if tags is None:
        tags = {"origin": "baseline", "model_sha256": sha}
    else:
        tags = {chave: valor.replace("{sha}", sha) for chave, valor in tags.items()}
    configure_tracking()
    with mlflow.start_run(run_name=nome, tags={"run_type": "training"}) as run:
        versao = registrar(
            run_id=run.info.run_id,
            pipeline=pipeline,
            signature=assinatura,
            input_example=exemplo,
            bundle_path=bundle,
            metadata_path=arquivo_metadata,
            tags=tags,
        )
    return versao, sha


def test_registry_vazio_nao_tem_alias(tracking_env):
    assert get_alias_version("champion") is None


def test_register_model_version_cria_versoes_sequenciais(tracking_env, tmp_path):
    v1, sha = _registrar(tmp_path, "a")
    v2, _ = _registrar(tmp_path, "b", seed=1)
    assert (v1, v2) == (1, 2)
    client = cliente_mlflow(tracking_env)
    versao = client.get_model_version(config.REGISTERED_MODEL_NAME, "1")
    assert versao.tags["origin"] == "baseline"
    assert versao.tags["model_sha256"] == sha
    artefatos = sorted(
        a.path
        for a in client.list_artifacts(versao.run_id, registry.BUNDLE_ARTIFACT_DIR)
    )
    assert artefatos == [
        "bundle/baseline_metadata.json",
        "bundle/baseline_model.joblib",
    ]


def test_register_model_version_exige_o_run_ativo(tracking_env, tmp_path):
    configure_tracking()
    with pytest.raises(ValueError, match="run de treino ativo"):
        register_model_version(
            run_id="outro",
            pipeline=None,
            signature=None,
            input_example=None,
            bundle_path=tmp_path,
            metadata_path=tmp_path,
            tags={},
        )


def test_register_model_version_falha_do_mlflow_nao_vaza_credencial(
    tracking_env, tmp_path, monkeypatch
):
    def _falhar(*args, **kwargs):
        raise MlflowException("API request to http://admin:segredo123@host failed")

    monkeypatch.setattr(mlflow, "register_model", _falhar)
    with pytest.raises(TrackingUnavailableError) as erro:
        _registrar(tmp_path, "a")
    mensagem = str(erro.value)
    assert "Falha ao registrar a versão do modelo" in mensagem
    assert "MONITORING_TRACKING=off" in mensagem
    assert "segredo123" not in mensagem
    assert "admin:" not in mensagem


def test_set_alias_e_get_alias_version(tracking_env, tmp_path):
    _registrar(tmp_path, "a")
    _registrar(tmp_path, "b", seed=1)
    set_alias("challenger", 2)
    assert int(get_alias_version("challenger").version) == 2
    assert get_alias_version("champion") is None


def test_bootstrap_champion_so_na_primeira_vez(tracking_env, tmp_path):
    _registrar(tmp_path, "a")
    _registrar(tmp_path, "b", seed=1)
    assert bootstrap_champion(1) is True
    assert bootstrap_champion(2) is False
    assert int(get_alias_version("champion").version) == 1


def test_export_champion_copia_bundle_conferindo_sha256(
    tracking_env, tmp_path, champion_dir
):
    versao, sha = _registrar(tmp_path, "a")
    export_champion(versao)
    assert sha256_de(config.CHAMPION_FILE) == sha
    metadata = json.loads(config.CHAMPION_METADATA_FILE.read_text(encoding="utf-8"))
    assert metadata["artifact"]["sha256"] == sha
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)
    # Nada de temporários deixados para trás.
    assert sorted(p.name for p in champion_dir.iterdir()) == [
        "baseline_metadata.json",
        "baseline_model.joblib",
    ]


def test_export_champion_recusa_hash_divergente(tracking_env, tmp_path, champion_dir):
    versao, _ = _registrar(tmp_path, "a")
    cliente_mlflow(tracking_env).set_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(versao), "model_sha256", "0" * 64
    )
    with pytest.raises(ModelIntegrityError, match="sha256"):
        export_champion(versao)
    assert list(champion_dir.iterdir()) == []


def test_export_champion_sem_tag_de_hash_falha_fechado(
    tracking_env, tmp_path, champion_dir
):
    versao, _ = _registrar(tmp_path, "a")
    cliente_mlflow(tracking_env).delete_model_version_tag(
        config.REGISTERED_MODEL_NAME, str(versao), "model_sha256"
    )
    with pytest.raises(ModelIntegrityError, match="model_sha256"):
        export_champion(versao)
    assert not config.CHAMPION_FILE.exists()


def test_export_champion_metadata_sem_hash_falha_fechado(
    tracking_env, tmp_path, champion_dir
):
    versao, _ = _registrar(tmp_path, "a", metadata=_SEM_HASH)
    with pytest.raises(ModelIntegrityError, match=r"artifact\.sha256"):
        export_champion(versao)
    assert list(champion_dir.iterdir()) == []


def test_export_champion_versao_sem_run_de_origem(tracking_env, tmp_path, champion_dir):
    _registrar(tmp_path, "a")
    client = cliente_mlflow(tracking_env)
    v1 = client.get_model_version(config.REGISTERED_MODEL_NAME, "1")
    orfa = client.create_model_version(
        config.REGISTERED_MODEL_NAME,
        source=(tmp_path / "a").as_uri(),
        tags={"model_sha256": v1.tags["model_sha256"]},
    )
    assert not orfa.run_id
    with pytest.raises(ModelIntegrityError, match="sem run de origem"):
        export_champion(int(orfa.version))
    assert not config.CHAMPION_FILE.exists()


def _replace_que_falha(alvo_que_falha):
    original = os.replace

    def _replace(origem, destino):
        if Path(destino) == Path(alvo_que_falha()):
            raise PermissionError(13, "Permission denied", str(destino))
        return original(origem, destino)

    return _replace


@pytest.mark.parametrize("onde", ["replace", "sha256", "mkdir"])
def test_export_champion_oserror_vira_erro_de_dominio(
    tracking_env, tmp_path, champion_dir, monkeypatch, onde
):
    versao, _ = _registrar(tmp_path, "a")
    if onde == "replace":
        monkeypatch.setattr(
            registry.os, "replace", _replace_que_falha(lambda: config.CHAMPION_FILE)
        )
    elif onde == "sha256":

        def _ilegivel(path):
            raise FileNotFoundError(2, "No such file or directory", str(path))

        monkeypatch.setattr(registry, "file_sha256", _ilegivel)
    else:
        # O "diretório" do champion é um arquivo: mkdir falha.
        champion_dir.parent.mkdir(parents=True)
        champion_dir.write_text("não sou diretório", encoding="utf-8")

    with pytest.raises(ModelIntegrityError) as erro:
        export_champion(versao)
    assert "Falha ao exportar o champion" in str(erro.value)
    assert str(champion_dir) in str(erro.value)
    assert not config.CHAMPION_FILE.exists()
    assert not config.CHAMPION_METADATA_FILE.exists()


def test_export_champion_troca_o_modelo_por_ultimo(
    tracking_env, tmp_path, champion_dir, monkeypatch
):
    """Entre os dois `os.replace` o par em disco é um que a integridade recusa."""
    v1, sha1 = _registrar(tmp_path, "a")
    v2, sha2 = _registrar(tmp_path, "b", seed=1)
    assert sha1 != sha2
    export_champion(v1)
    original = os.replace
    trocas: list[str] = []

    def _replace(origem, destino):
        destino = Path(destino)
        if destino.parent == champion_dir:
            trocas.append(destino.name)
            if destino == config.CHAMPION_FILE:
                # Janela: metadata já é o da v2, o modelo ainda é o da v1.
                with pytest.raises(ModelIntegrityError, match="diverge"):
                    verify_model_integrity(
                        config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE
                    )
        return original(origem, destino)

    monkeypatch.setattr(registry.os, "replace", _replace)
    export_champion(v2)
    assert trocas == ["baseline_metadata.json", "baseline_model.joblib"]
    assert sha256_de(config.CHAMPION_FILE) == sha2
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)


def test_export_champion_falha_na_troca_do_modelo_restaura_o_par(
    tracking_env, tmp_path, champion_dir, monkeypatch
):
    v1, sha1 = _registrar(tmp_path, "a")
    v2, _ = _registrar(tmp_path, "b", seed=1)
    export_champion(v1)
    monkeypatch.setattr(
        registry.os, "replace", _replace_que_falha(lambda: config.CHAMPION_FILE)
    )
    with pytest.raises(ModelIntegrityError, match="Falha ao exportar o champion"):
        export_champion(v2)
    metadata = json.loads(config.CHAMPION_METADATA_FILE.read_text(encoding="utf-8"))
    assert metadata["artifact"]["sha256"] == sha1
    assert sha256_de(config.CHAMPION_FILE) == sha1
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)
    assert sorted(p.name for p in champion_dir.iterdir()) == [
        "baseline_metadata.json",
        "baseline_model.joblib",
    ]


def _conteudo(pasta: Path) -> dict[str, str]:
    """Nome → sha256 dos arquivos; o temporário `.export-*` aparece só pelo nome."""
    return {
        p.name: sha256_de(p) if p.is_file() else "<dir>"
        for p in sorted(pasta.iterdir())
    }


def test_staged_champion_confere_antes_e_so_troca_no_install(
    tracking_env, tmp_path, champion_dir
):
    """O bundle é baixado e conferido sem tocar em `models/champion/` até o install."""
    v1, sha1 = _registrar(tmp_path, "a")
    v2, sha2 = _registrar(tmp_path, "b", seed=1)
    export_champion(v1)
    antes = _conteudo(champion_dir)
    with staged_champion(v2) as preparado:
        assert (preparado.version, preparado.model_sha256) == (v2, sha2)
        assert sha256_de(preparado.model) == sha2
        # Conferido e pronto, mas o champion servido ainda é o da v1.
        assert {
            n: s for n, s in _conteudo(champion_dir).items() if not n.startswith(".")
        } == antes
        preparado.install()
    assert sha256_de(config.CHAMPION_FILE) == sha2
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)
    assert sorted(_conteudo(champion_dir)) == [
        "baseline_metadata.json",
        "baseline_model.joblib",
    ]
    assert sha1 != sha2


def test_staged_champion_sem_install_nao_toca_models(
    tracking_env, tmp_path, champion_dir
):
    v1, _ = _registrar(tmp_path, "a")
    v2, _ = _registrar(tmp_path, "b", seed=1)
    export_champion(v1)
    antes = _conteudo(champion_dir)
    with staged_champion(v2):
        pass
    assert _conteudo(champion_dir) == antes


def test_staged_champion_install_com_oserror_vira_erro_de_dominio(
    tracking_env, tmp_path, champion_dir, monkeypatch
):
    v1, sha1 = _registrar(tmp_path, "a")
    v2, _ = _registrar(tmp_path, "b", seed=1)
    export_champion(v1)
    monkeypatch.setattr(
        registry.os, "replace", _replace_que_falha(lambda: config.CHAMPION_FILE)
    )
    with staged_champion(v2) as preparado:
        with pytest.raises(ModelIntegrityError, match="Falha ao exportar o champion"):
            preparado.install()
    assert sha256_de(config.CHAMPION_FILE) == sha1
    verify_model_integrity(config.CHAMPION_FILE, config.CHAMPION_METADATA_FILE)


def test_requisito_ausente_falha_com_mensagem(tracking_env, tmp_path, monkeypatch):
    monkeypatch.setattr(
        registry, "_MODEL_REQUIREMENTS", ("scikit-learn", "pacote-que-nao-existe")
    )
    with pytest.raises(TrackingUnavailableError, match="pacote-que-nao-existe"):
        _registrar(tmp_path, "a")


# MARK: challenger


def _tags_challenger(lote: str = "run-do-lote") -> dict[str, str]:
    return {
        "model_sha256": "{sha}",
        "reference_sha256": "f" * 64,
        "bundle_model_version": "retrain-{sha}",
        "alert_run_id": lote,
        "alert_batch_name": "production_t2",
    }


def _challenger(tmp_path: Path, nome: str, seed: int, **extra) -> tuple[int, str]:
    return _registrar(
        tmp_path,
        nome,
        seed=seed,
        registrar=register_challenger,
        model_version=f"retrain-{nome}",
        tags=extra.pop("tags", None) or _tags_challenger(),
        **extra,
    )


def test_register_challenger_move_alias_e_grava_tags(tracking_env, tmp_path):
    _registrar(tmp_path, "baseline")
    set_alias("champion", 1)
    v2, _ = _challenger(tmp_path, "c1", seed=1)
    assert int(get_alias_version("challenger").version) == v2 == 2
    v3, sha3 = _challenger(tmp_path, "c2", seed=2)
    # O alias se move para o challenger novo; o champion não é tocado.
    assert int(get_alias_version("challenger").version) == v3 == 3
    assert int(get_alias_version("champion").version) == 1
    tags = get_alias_version("challenger").tags
    assert tags["origin"] == "retrain"
    assert tags["model_sha256"] == sha3
    assert tags["bundle_model_version"] == f"retrain-{sha3}"
    assert tags["reference_sha256"] == "f" * 64
    assert tags["alert_run_id"] == "run-do-lote"
    assert tags["alert_batch_name"] == "production_t2"


@pytest.mark.parametrize(
    ("tags", "erro"),
    [
        ({"bundle_model_version": "retrain-x"}, "model_sha256"),
        ({"model_sha256": "{sha}"}, "bundle_model_version"),
        (
            {"model_sha256": "{sha}", "bundle_model_version": "baseline-v1"},
            "baseline-v1",
        ),
        ({"model_sha256": " ", "bundle_model_version": "retrain-x"}, "model_sha256"),
    ],
)
def test_register_challenger_recusa_tags_invalidas_sem_criar_versao(
    tracking_env, tmp_path, tags, erro
):
    with pytest.raises(ValueError, match=erro):
        _challenger(tmp_path, "c1", seed=1, tags=tags)
    assert get_alias_version("challenger") is None
    modelos = cliente_mlflow(tracking_env).search_registered_models()
    assert modelos == []


# MARK: alerta crítico e consumo


def _run_de_lote(
    cliente, experimento: str, inicio: int, status: str = "FINISHED", **tags
) -> str:
    base = {"run_type": "batch", "alert_level": "CRITICO", "batch_name": "t2"}
    base.update(tags)
    run = cliente.create_run(experimento, start_time=inicio, tags=base)
    cliente.set_terminated(run.info.run_id, status)
    return run.info.run_id


def test_find_unconsumed_critical_ignora_consumidos_e_nao_criticos(
    tracking_env, monkeypatch
):
    experimento = configure_tracking()
    cliente = cliente_mlflow(tracking_env)
    _run_de_lote(cliente, experimento, 1_000, alert_level="OK")
    _run_de_lote(cliente, experimento, 2_000, alert_level="ATENCAO")
    _run_de_lote(cliente, experimento, 3_000, alert_level="BLOQUEADO")
    antigo = _run_de_lote(cliente, experimento, 4_000)
    esperado = _run_de_lote(cliente, experimento, 5_000)
    _run_de_lote(cliente, experimento, 6_000, retrain_consumed="2")
    _run_de_lote(cliente, experimento, 7_000, run_type="stage")
    _run_de_lote(cliente, experimento, 8_000, status="FAILED")

    filtros: list[str] = []
    original = MlflowClient.search_runs

    def _espiao(self, *args, **kwargs):
        filtros.append(kwargs.get("filter_string", ""))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MlflowClient, "search_runs", _espiao)
    encontrado = find_unconsumed_critical_run()
    assert encontrado is not None
    assert encontrado.info.run_id == esperado
    # O filtro de `retrain_consumed` é feito no cliente (tag ausente na maioria).
    assert filtros
    assert all("retrain_consumed" not in filtro for filtro in filtros)

    mark_alert_consumed(esperado, 3)
    assert cliente.get_run(esperado).data.tags["retrain_consumed"] == "3"
    assert find_unconsumed_critical_run().info.run_id == antigo
    mark_alert_consumed(antigo, 3)
    assert find_unconsumed_critical_run() is None


def test_consumo_vale_para_todos_os_alertas_do_mesmo_lote(tracking_env):
    # `make pipeline` duas vezes: o `simulate` é determinístico, então os dois alertas
    # do t2 têm o mesmo batch_sha256. Um retreino consome o LOTE, não só um run.
    experimento = configure_tracking()
    cliente = cliente_mlflow(tracking_env)
    lote, outro_lote = "a" * 64, "b" * 64
    a1 = _run_de_lote(cliente, experimento, 1_000, batch_sha256=lote)
    a2 = _run_de_lote(cliente, experimento, 2_000, batch_sha256=lote)
    falho = _run_de_lote(cliente, experimento, 2_500, "FAILED", batch_sha256=lote)
    outro = _run_de_lote(cliente, experimento, 500, batch_sha256=outro_lote)
    assert find_unconsumed_critical_run().info.run_id == a2
    assert alert_consumed_by(cliente.get_run(a1)) is None

    mark_alert_consumed(a2, 3)

    assert cliente.get_run(a2).data.tags["retrain_consumed"] == "3"
    assert cliente.get_run(a1).data.tags["retrain_consumed"] == "3"
    assert "retrain_consumed" not in cliente.get_run(falho).data.tags
    assert "retrain_consumed" not in cliente.get_run(outro).data.tags
    assert find_unconsumed_critical_run().info.run_id == outro

    # Alerta novo sobre um lote que já gerou challenger (outro `make pipeline`).
    a3 = _run_de_lote(cliente, experimento, 9_000, batch_sha256=lote)
    assert alert_consumed_by(cliente.get_run(a3)) == "3"
    assert find_unconsumed_critical_run().info.run_id == outro
    assert alert_consumed_by(cliente.get_run(outro)) is None


def test_find_unconsumed_critical_filtra_pelo_model_version(tracking_env):
    experimento = configure_tracking()
    cliente = cliente_mlflow(tracking_env)
    do_champion = _run_de_lote(cliente, experimento, 1_000, model_version="baseline-v1")
    _run_de_lote(cliente, experimento, 2_000, model_version="retrain-abc")

    achado = find_unconsumed_critical_run(model_version="baseline-v1")
    assert achado.info.run_id == do_champion
    assert find_unconsumed_critical_run(model_version="retrain-xyz") is None


def test_find_unconsumed_critical_desempata_pelo_run_id(tracking_env):
    experimento = configure_tracking()
    cliente = cliente_mlflow(tracking_env)
    ids = [_run_de_lote(cliente, experimento, 5_000) for _ in range(3)]
    assert find_unconsumed_critical_run().info.run_id == max(ids)


def test_find_unconsumed_critical_pagina_a_busca(tracking_env, monkeypatch):
    experimento = configure_tracking()
    cliente = cliente_mlflow(tracking_env)
    novo = _run_de_lote(cliente, experimento, 9_000)
    for inicio in range(1_000, 4_000, 1_000):
        _run_de_lote(cliente, experimento, inicio)
    monkeypatch.setattr(registry, "SEARCH_PAGE_SIZE", 2)
    assert find_unconsumed_critical_run().info.run_id == novo


def test_find_unconsumed_critical_sem_experimento_ou_sem_alerta(tracking_env):
    assert find_unconsumed_critical_run() is None
    experimento = configure_tracking()
    _run_de_lote(cliente_mlflow(tracking_env), experimento, 1_000, alert_level="OK")
    assert find_unconsumed_critical_run() is None


def test_require_critical_run_recusa_inexistente_e_nao_critico(tracking_env):
    experimento = configure_tracking()
    cliente = cliente_mlflow(tracking_env)
    critico = _run_de_lote(cliente, experimento, 1_000)
    atencao = _run_de_lote(cliente, experimento, 2_000, alert_level="ATENCAO")
    estagio = _run_de_lote(cliente, experimento, 3_000, run_type="stage")
    assert require_critical_run(critico).info.run_id == critico
    with pytest.raises(ValueError, match="ATENCAO"):
        require_critical_run(atencao)
    with pytest.raises(ValueError, match="não é um run de lote"):
        require_critical_run(estagio)
    with pytest.raises(ValueError, match="não encontrado"):
        require_critical_run("0" * 32)
    # Id malformado é "não encontrado", não "MLflow indisponível".
    with pytest.raises(ValueError, match="não encontrado"):
        require_critical_run("x' or 1=1 --")
    # Marcar consumo exige o mesmo: um run que não é alerta crítico fica intocado.
    with pytest.raises(ValueError, match="ATENCAO"):
        mark_alert_consumed(atencao, 2)
    assert "retrain_consumed" not in cliente.get_run(atencao).data.tags


def test_require_critical_run_recusa_lote_cujo_run_falhou(tracking_env):
    experimento = configure_tracking()
    falho = _run_de_lote(cliente_mlflow(tracking_env), experimento, 1_000, "FAILED")
    with pytest.raises(ValueError, match="FAILED"):
        require_critical_run(falho)


def test_consultas_com_outro_erro_do_mlflow_saem_redigidas(tracking_env, monkeypatch):
    """Só "não existe" vira erro de domínio; o resto é MLflow indisponível."""

    def _falhar(*args, **kwargs):
        raise MlflowException("API request to http://admin:segredo123@host failed")

    configure_tracking()
    monkeypatch.setattr(MlflowClient, "get_run", _falhar)
    monkeypatch.setattr(MlflowClient, "get_model_version", _falhar)
    for operacao in (lambda: require_critical_run("0" * 32), lambda: promote_alias(1)):
        with pytest.raises(TrackingUnavailableError) as erro:
            operacao()
        assert "segredo123" not in str(erro.value)


# MARK: promoção e rollback dos aliases


def _tres_versoes(tmp_path: Path) -> None:
    for seed, nome in enumerate(("a", "b", "c")):
        _registrar(tmp_path, nome, seed=seed)


def _alias(nome: str) -> int | None:
    versao = get_alias_version(nome)
    return None if versao is None else int(versao.version)


def test_promote_move_aliases_e_define_previous(tracking_env, tmp_path):
    _tres_versoes(tmp_path)
    # Sem champion: só dá o alias, sem previous.
    promote_alias(1)
    assert (_alias("champion"), _alias("previous")) == (1, None)
    promote_alias(2)
    assert (_alias("champion"), _alias("previous")) == (2, 1)
    # Promover a versão que já é champion não apaga nem move o previous.
    promote_alias(2)
    assert (_alias("champion"), _alias("previous")) == (2, 1)
    promote_alias(3)
    assert (_alias("champion"), _alias("previous")) == (3, 2)


def test_promote_versao_inexistente_falha_sem_mexer_nos_aliases(tracking_env, tmp_path):
    _tres_versoes(tmp_path)
    promote_alias(1)
    with pytest.raises(RegistryStateError, match="Versão 99"):
        promote_alias(99)
    assert (_alias("champion"), _alias("previous")) == (1, None)


def test_promote_sem_modelo_registrado_falha_com_erro_de_dominio(tracking_env):
    configure_tracking()
    with pytest.raises(RegistryStateError, match="não existe"):
        promote_alias(1)


def test_rollback_devolve_previous_e_falha_sem_previous(tracking_env, tmp_path):
    _tres_versoes(tmp_path)
    promote_alias(1)
    with pytest.raises(RegistryStateError, match="previous"):
        rollback_alias()
    assert _alias("champion") == 1
    promote_alias(2)
    assert rollback_alias() == 1
    # O previous é consumido: um segundo rollback não re-promove a versão revertida.
    assert (_alias("champion"), _alias("previous")) == (1, None)
    with pytest.raises(RegistryStateError, match="previous"):
        rollback_alias()
    assert _alias("champion") == 1


def test_delete_alias_remove_e_ignora_alias_ausente(tracking_env, tmp_path):
    _tres_versoes(tmp_path)
    set_alias("previous", 1)
    delete_alias("previous")
    assert _alias("previous") is None
    # Alias ausente: nada a fazer, sem erro.
    delete_alias("previous")
    assert _alias("previous") is None


def test_require_version_devolve_a_versao_ou_erro_de_dominio(tracking_env, tmp_path):
    _tres_versoes(tmp_path)
    assert int(require_version(2).version) == 2
    with pytest.raises(RegistryStateError, match="Versão 99"):
        require_version(99)


def test_rollback_sem_modelo_registrado_falha_com_erro_de_dominio(tracking_env):
    configure_tracking()
    with pytest.raises(RegistryStateError, match="previous"):
        rollback_alias()


def test_alias_falha_do_mlflow_nao_vaza_credencial(tracking_env, monkeypatch):
    def _falhar(*args, **kwargs):
        raise MlflowException("API request to http://admin:segredo123@host failed")

    monkeypatch.setattr(MlflowClient, "search_registered_models", _falhar)
    for operacao in (lambda: promote_alias(1), rollback_alias):
        with pytest.raises(TrackingUnavailableError) as erro:
            operacao()
        assert "segredo123" not in str(erro.value)


# MARK: auditoria


_AGORA = datetime(2026, 10, 27, 9, 30, 15, tzinfo=timezone(timedelta(hours=-3)))


def test_audit_run_registra_quem_quando_por_que(tracking_env, monkeypatch):
    monkeypatch.setattr(registry.getpass, "getuser", lambda: "analista")
    run_id = log_audit_run(
        action="promote",
        version=2,
        reason="Ganho de 0,03 no holdout de t2",
        details={
            "auc_holdout": {"champion": 0.57, "challenger": 0.60},
            "eligible": True,
        },
        clock=lambda: _AGORA,
    )
    run = cliente_mlflow(tracking_env).get_run(run_id)
    tags = run.data.tags
    assert run.info.status == "FINISHED"
    assert tags["run_type"] == "audit"
    assert tags["action"] == "promote"
    assert tags["version"] == "2"
    assert tags["model_name"] == config.REGISTERED_MODEL_NAME
    assert tags["reason"] == "Ganho de 0,03 no holdout de t2"
    assert tags["operator"] == "analista"
    # Hora em UTC, ISO 8601, mesmo que o relógio venha em outro fuso.
    assert tags["timestamp_utc"] == "2026-10-27T12:30:15+00:00"
    assert json.loads(tags["details"]) == {
        "auc_holdout": {"champion": 0.57, "challenger": 0.60},
        "eligible": True,
    }

    explicito = log_audit_run(
        action="rollback",
        version=1,
        reason="Regressão observada",
        operator="revisora",
        details={},
        clock=lambda: _AGORA,
    )
    assert cliente_mlflow(tracking_env).get_run(explicito).data.tags["operator"] == (
        "revisora"
    )


def test_audit_run_nao_vaza_segredo(tracking_env):
    run_id = log_audit_run(
        action="promote",
        version=2,
        reason="Override pedido em http://admin:segredo123@host/painel",
        operator="analista",
        details={
            "tracking_uri": "postgresql://user:segredo456@db:5432/mlflow",
            "api_token": "tok-segredo789",
            "aninhado": {"Password": "segredo000", "lista": ["https://u:segredo111@h"]},
        },
    )
    tags = cliente_mlflow(tracking_env).get_run(run_id).data.tags
    texto = json.dumps(tags)
    for segredo in (
        "segredo123",
        "segredo456",
        "segredo789",
        "segredo000",
        "segredo111",
    ):
        assert segredo not in texto
    detalhes = json.loads(tags["details"])
    assert detalhes["tracking_uri"] == "postgresql://***@db:5432/mlflow"
    assert detalhes["api_token"] == "***"
    assert "http://***@host/painel" in tags["reason"]


@pytest.mark.parametrize(
    ("argumentos", "erro"),
    [
        ({"reason": ""}, "motivo"),
        ({"reason": "   "}, "motivo"),
        ({"action": "apagar"}, "apagar"),
        ({"details": {"x": object()}}, "JSON"),
        ({"details": {"x": "a" * 9000}}, "details"),
        ({"clock": lambda: datetime(2026, 10, 27, 9, 0)}, "fuso"),
    ],
)
def test_audit_run_recusa_entrada_invalida_sem_criar_run(
    tracking_env, argumentos, erro
):
    padrao = {
        "action": "promote",
        "version": 2,
        "reason": "motivo válido",
        "operator": "analista",
        "details": {},
    }
    with pytest.raises(ValueError, match=erro):
        log_audit_run(**{**padrao, **argumentos})
    experimento = configure_tracking()
    assert cliente_mlflow(tracking_env).search_runs([experimento]) == []


def test_audit_run_grava_o_status_do_estado_servido(tracking_env):
    run_id = log_audit_run(
        action="promote",
        version=2,
        reason="motivo",
        operator="analista",
        details={},
        status="INCONSISTENT",
    )
    tags = cliente_mlflow(tracking_env).get_run(run_id).data.tags
    assert tags["status"] == "INCONSISTENT"
    sem_status = log_audit_run(
        action="rollback", version=1, reason="motivo", operator="a", details={}
    )
    assert "status" not in cliente_mlflow(tracking_env).get_run(sem_status).data.tags


def test_audit_run_recusa_status_desconhecido_sem_criar_run(tracking_env):
    with pytest.raises(ValueError, match="QUEBRADO"):
        log_audit_run(
            action="promote",
            version=2,
            reason="motivo",
            operator="a",
            details={},
            status="QUEBRADO",
        )
    experimento = configure_tracking()
    assert cliente_mlflow(tracking_env).search_runs([experimento]) == []


def test_audit_run_grava_float_nao_finito_como_texto(tracking_env):
    """AUC NaN de um challenger reprovado não pode impedir a auditoria."""
    run_id = log_audit_run(
        action="promote",
        version=2,
        reason="Override com AUC indefinida",
        operator="analista",
        details={"auc": float("nan"), "ganho": [float("inf"), 0.01]},
    )
    tags = cliente_mlflow(tracking_env).get_run(run_id).data.tags
    assert json.loads(tags["details"]) == {"auc": "nan", "ganho": ["inf", 0.01]}


def test_audit_run_recusa_operador_vazio(tracking_env):
    with pytest.raises(ValueError, match="operador"):
        log_audit_run(
            action="promote", version=2, reason="motivo", operator="  ", details={}
        )


def test_audit_run_operador_indisponivel_falha_com_mensagem(tracking_env, monkeypatch):
    def _sem_usuario():
        raise OSError("No username set in the environment")

    monkeypatch.setattr(registry.getpass, "getuser", _sem_usuario)
    with pytest.raises(ValueError, match="operador"):
        log_audit_run(action="promote", version=2, reason="motivo", details={})


def test_audit_run_falha_do_mlflow_vira_erro_redigido(tracking_env, monkeypatch):
    def _falhar(*args, **kwargs):
        raise MlflowException("API request to http://admin:segredo123@host failed")

    monkeypatch.setattr(MlflowClient, "set_terminated", _falhar)
    with pytest.raises(TrackingUnavailableError) as erro:
        log_audit_run(
            action="promote", version=2, reason="motivo", operator="a", details={}
        )
    assert "segredo123" not in str(erro.value)
    assert "Falha ao registrar o run de auditoria" in str(erro.value)
