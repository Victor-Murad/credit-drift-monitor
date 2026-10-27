"""Model Registry do MLflow: versões, aliases e o champion servido (spec da Etapa 3, §4.7).

- Cada treino vira uma versão de `REGISTERED_MODEL_NAME`, criada a partir do modelo
  logado no run de treino; o run anexa também o `bundle` joblib e o metadata em
  `bundle/` (é dali que promoção e rollback copiam o champion).
- Aliases (`champion`, `challenger`, `previous`) no lugar dos estágios fixos.
  `register_challenger` só move `challenger`; `promote_alias`/`rollback_alias` só
  movem aliases (o arquivo servido é do comando de promoção).
- O alerta que habilita o retreino é um run de lote `alert_level=CRITICO` sem a
  tag `retrain_consumed` (filtrada no cliente: tag ausente não entra no filtro do
  `search_runs`); `mark_alert_consumed` grava a versão do challenger nela.
- Cada promoção ou rollback deixa um run `run_type=audit` (`log_audit_run`): quem,
  quando (UTC) e por quê, sem segredos.
- O champion servido vive em `models/champion/` (fora dos `outs` do DVC). A cópia é
  feita por temporário no mesmo diretório + `os.replace` e conferida pelo sha256 da
  tag `model_sha256` da versão: divergência falha fechado. Quem dá o alias exporta
  ANTES (ver `src.monitoring.stage.run_train`): export falho não deixa alias.
  `staged_champion` separa as duas metades (baixar e conferir; `install`), para o
  comando de promoção conferir o arquivo antes de mover os aliases e só trocá-lo
  depois (`src.monitoring.promote`).

As constantes `src.config.REGISTERED_MODEL_NAME` e `CHAMPION_*` são lidas na chamada,
para que os testes possam redirecioná-las.
"""

from __future__ import annotations

import getpass
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH

from src import config
from src.logging_utils import RUN_ID, get_logger, log_event
from src.models.integrity import ModelIntegrityError, file_sha256
from src.models.train import MODEL_VERSION as BASELINE_MODEL_VERSION
from src.monitoring import tracking
from src.monitoring.alerts import AlertLevel
from src.monitoring.tracking import TrackingUnavailableError

logger = get_logger("monitoring.registry")

CHAMPION_ALIAS = "champion"
BUNDLE_ARTIFACT_DIR = "bundle"
MODEL_ARTIFACT_NAME = "model"
# Nomes fixos dentro de `bundle/`: os mesmos do champion servido, para reaproveitar
# a verificação de integridade (`verify_model_integrity`).
BUNDLE_MODEL_FILE = config.CHAMPION_FILE.name
BUNDLE_METADATA_FILE = config.CHAMPION_METADATA_FILE.name
# O MLflow 3.x serializa o sklearn com `skops` (sem pickle arbitrário) e recusa
# tipos fora da lista de confiança dele: estes são os do pipeline do baseline.
SKOPS_TRUSTED_TYPES = (
    "src.models.features.add_ratio_features",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
)
# Requisitos explícitos do modelo logado (evita a inferência do MLflow, que carrega
# o modelo num subprocesso a cada registro).
_MODEL_REQUIREMENTS = ("mlflow", "scikit-learn", "skops", "xgboost", "pandas", "numpy")


def _pip_requirements() -> list[str]:
    """Versões instaladas dos pacotes do modelo; pacote ausente falha com mensagem."""
    requisitos = []
    for pacote in _MODEL_REQUIREMENTS:
        try:
            requisitos.append(f"{pacote}=={installed_version(pacote)}")
        except PackageNotFoundError as exc:
            raise TrackingUnavailableError(
                f"Pacote {pacote} não instalado: é requisito do modelo registrado no "
                "MLflow. Rode `uv sync` (o pyproject.toml declara a dependência)."
            ) from exc
    return requisitos


def _log_bundle(run_id: str, bundle_path: Path, metadata_path: Path) -> None:
    """Anexa bundle e metadata em `bundle/` com os nomes fixos do champion."""
    with tempfile.TemporaryDirectory(prefix="credit-drift-monitor-bundle-") as tmp:
        pasta = Path(tmp)
        shutil.copyfile(bundle_path, pasta / BUNDLE_MODEL_FILE)
        shutil.copyfile(metadata_path, pasta / BUNDLE_METADATA_FILE)
        tracking.client().log_artifacts(
            run_id, str(pasta), artifact_path=BUNDLE_ARTIFACT_DIR
        )


def register_model_version(
    *,
    run_id: str,
    pipeline: Any,
    signature: Any,
    input_example: Any,
    bundle_path: Path,
    metadata_path: Path,
    tags: dict[str, str],
) -> int:
    """Loga o modelo no run de treino (ativo), anexa o bundle e cria a versão.

    O modelo vai com *signature*, *input example* e `pyfunc_predict_fn="predict_proba"`;
    o `threshold` e o bundle completo ficam no artefato `bundle/`. Devolve o número
    da versão criada em `REGISTERED_MODEL_NAME`, com as `tags` dadas.
    """
    ativo = mlflow.active_run()
    if ativo is None or ativo.info.run_id != run_id:
        raise ValueError(
            f"register_model_version exige o run de treino ativo ({run_id}); "
            "abra-o com mlflow.start_run antes de registrar."
        )
    bundle_path = tracking.require_path(bundle_path, "Bundle do modelo")
    metadata_path = tracking.require_path(metadata_path, "Metadata do modelo")
    requisitos = _pip_requirements()
    with tracking.tracking_errors("Falha ao registrar a versão do modelo"):
        modelo = mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            signature=signature,
            input_example=input_example,
            pyfunc_predict_fn="predict_proba",
            pip_requirements=requisitos,
            skops_trusted_types=list(SKOPS_TRUSTED_TYPES),
        )
        _log_bundle(run_id, bundle_path, metadata_path)
        versao = mlflow.register_model(
            modelo.model_uri, config.REGISTERED_MODEL_NAME, tags=tags
        )
    log_event(
        logger,
        "model_version_registered",
        model_name=config.REGISTERED_MODEL_NAME,
        version=int(versao.version),
        mlflow_run_id=run_id,
    )
    return int(versao.version)


def get_alias_version(alias: str) -> ModelVersion | None:
    """Versão que tem o `alias`, ou `None` (modelo ainda não registrado ou sem o alias)."""
    nome = config.REGISTERED_MODEL_NAME
    with tracking.tracking_errors("Falha ao consultar o Model Registry"):
        cliente = tracking.client()
        modelos = cliente.search_registered_models(filter_string=f"name = '{nome}'")
        if not modelos:
            return None
        versao = modelos[0].aliases.get(alias)
        if versao is None:
            return None
        return cliente.get_model_version(nome, str(versao))


def set_alias(alias: str, version: int) -> None:
    """Aponta o `alias` para a `version` de `REGISTERED_MODEL_NAME`."""
    with tracking.tracking_errors("Falha ao mover o alias no Model Registry"):
        tracking.client().set_registered_model_alias(
            config.REGISTERED_MODEL_NAME, alias, str(version)
        )
    log_event(
        logger,
        "model_alias_set",
        model_name=config.REGISTERED_MODEL_NAME,
        alias=alias,
        version=int(version),
    )


def bootstrap_champion(version: int) -> bool:
    """Dá o alias `champion` à `version` só se ainda não houver champion.

    Um `train` repetido nunca move o champion: daí em diante só promoção e rollback.
    Devolve se deu o alias. Não exporta: o chamador roda `export_champion` ANTES,
    para que um export falho não deixe alias apontando para um champion ausente.
    """
    if get_alias_version(CHAMPION_ALIAS) is not None:
        return False
    set_alias(CHAMPION_ALIAS, version)
    return True


def _conferir_bundle(modelo: Path, metadata: Path, esperado: str, version: int) -> None:
    """O arquivo e o metadata baixados têm de bater com a tag `model_sha256`."""
    obtido = file_sha256(modelo)
    try:
        registrado = json.loads(metadata.read_text(encoding="utf-8"))["artifact"][
            "sha256"
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelIntegrityError(
            f"Metadata da versão {version} sem artifact.sha256: champion não exportado."
        ) from exc
    if obtido != esperado or registrado != esperado:
        raise ModelIntegrityError(
            f"sha256 do bundle da versão {version} diverge da tag model_sha256 "
            f"(tag {esperado[:12]}..., arquivo {obtido[:12]}..., metadata "
            f"{str(registrado)[:12]}...). Champion não exportado."
        )


def _trocar_par(modelo: Path, metadata: Path, backup: Path) -> None:
    """Põe metadata e modelo no lugar: metadata primeiro, modelo por último.

    Se o processo morrer entre os dois `os.replace`, fica metadata novo com modelo
    antigo (ou ausente), um par que `verify_model_integrity` recusa. Se o segundo
    `os.replace` falhar com erro, o metadata anterior é restaurado (ou removido, se
    não havia) e o erro sobe.
    """
    destino_metadata = Path(config.CHAMPION_METADATA_FILE)
    havia_metadata = destino_metadata.exists()
    if havia_metadata:
        shutil.copy2(destino_metadata, backup)
    os.replace(metadata, destino_metadata)
    trocado = False
    try:
        os.replace(modelo, config.CHAMPION_FILE)
        trocado = True
    finally:
        if not trocado:
            if havia_metadata:
                os.replace(backup, destino_metadata)
            else:
                destino_metadata.unlink(missing_ok=True)


def _export_error(version: int, destino: Path, exc: OSError) -> ModelIntegrityError:
    return ModelIntegrityError(
        f"Falha ao exportar o champion (versão {version}) para {destino}: {exc}"
    )


@dataclass(frozen=True)
class StagedChampion:
    """Bundle de uma versão já baixado e conferido, ainda fora do lugar servido.

    `install` põe o par em `models/champion/` (`_trocar_par`); só vale dentro do
    `staged_champion` que o criou, que apaga o temporário ao sair.
    """

    version: int
    model_sha256: str
    model: Path
    metadata: Path
    backup: Path

    def install(self) -> None:
        """Troca o champion servido por este bundle (`OSError` vira erro de domínio)."""
        try:
            _trocar_par(self.model, self.metadata, self.backup)
        except OSError as exc:
            raise _export_error(self.version, Path(config.CHAMPION_DIR), exc) from exc


def _export_source(version: int) -> tuple[str, str]:
    """`(model_sha256, run de origem)` da `version`; sem um deles, falha fechado."""
    nome = config.REGISTERED_MODEL_NAME
    with tracking.tracking_errors("Falha ao ler a versão do Model Registry"):
        mv = tracking.client().get_model_version(nome, str(version))
    esperado = mv.tags.get("model_sha256")
    if not esperado:
        raise ModelIntegrityError(
            f"Versão {version} de {nome} sem a tag model_sha256: sem hash registrado, "
            "o champion não pode ser conferido."
        )
    if not mv.run_id:
        raise ModelIntegrityError(
            f"Versão {version} de {nome} sem run de origem: bundle indisponível."
        )
    return esperado, mv.run_id


def _baixar_e_conferir(
    run_id: str, esperado: str, version: int, tmp: Path
) -> StagedChampion:
    with tracking.tracking_errors("Falha ao baixar o bundle do run de origem"):
        baixado = Path(
            tracking.client().download_artifacts(
                run_id, BUNDLE_ARTIFACT_DIR, dst_path=str(tmp)
            )
        )
    modelo = tracking.require_path(baixado / BUNDLE_MODEL_FILE, "Bundle do champion")
    metadata = tracking.require_path(
        baixado / BUNDLE_METADATA_FILE, "Metadata do champion"
    )
    _conferir_bundle(modelo, metadata, esperado, version)
    return StagedChampion(
        version=int(version),
        model_sha256=esperado,
        model=modelo,
        metadata=metadata,
        backup=tmp / "metadata_anterior.json",
    )


@contextmanager
def staged_champion(version: int) -> Iterator[StagedChampion]:
    """Baixa o bundle da `version` e confere o sha256, sem tocar no champion servido.

    O temporário fica dentro de `CHAMPION_DIR` (mesmo sistema de arquivos, para o
    `os.replace` ser atômico) e é apagado ao sair, com ou sem `install`. Assim o
    comando de promoção confere o arquivo ANTES de mover qualquer alias (§4.7).

    Divergência de hash ou erro de arquivo (`OSError`: diretório, download local,
    leitura) levanta `ModelIntegrityError` com o caminho, sem credenciais; falha do
    MLflow levanta `TrackingUnavailableError` (mensagem redigida).
    """
    esperado, run_id = _export_source(version)
    destino = Path(config.CHAMPION_DIR)
    try:
        destino.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".export-", dir=destino))
    except OSError as exc:
        raise _export_error(version, destino, exc) from exc
    try:
        try:
            preparado = _baixar_e_conferir(run_id, esperado, version, tmp)
        except OSError as exc:
            raise _export_error(version, destino, exc) from exc
        yield preparado
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def export_champion(version: int) -> None:
    """Copia bundle + metadata do run de origem da `version` para `models/champion/`.

    `staged_champion` (baixa e confere o sha256 contra a tag `model_sha256`) e,
    só então, `install`: os arquivos trocam com `os.replace`, metadata primeiro e
    modelo por último (ver `_trocar_par` para a janela entre as duas trocas).
    Os erros são os de `staged_champion` e de `StagedChampion.install`.
    """
    with staged_champion(version) as preparado:
        preparado.install()
    nome = config.REGISTERED_MODEL_NAME
    esperado = preparado.model_sha256
    log_event(
        logger,
        "champion_exported",
        level=logging.INFO,
        model_name=nome,
        version=int(version),
        model_sha256=esperado,
        file=config.project_relative(Path(config.CHAMPION_FILE)),
    )


# MARK: erros e consultas


class RegistryStateError(RuntimeError):
    """O estado do Model Registry não permite a operação (versão ou alias ausente)."""


def _not_found(exc: MlflowException, *, malformed_id: bool = False) -> bool:
    """O MLflow disse "não existe" (e, com `malformed_id`, também "id inválido")."""
    codigos = {ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)}
    if malformed_id:
        codigos.add(ErrorCode.Name(INVALID_PARAMETER_VALUE))
    return exc.error_code in codigos


def require_version(version: int) -> ModelVersion:
    """A `version` de `REGISTERED_MODEL_NAME`; inexistente é `RegistryStateError`.

    "Não existe" é estado do Registry, não MLflow indisponível: não passa pela
    mensagem de `tracking_errors`.
    """
    nome = config.REGISTERED_MODEL_NAME
    with tracking.tracking_errors("Falha ao ler a versão do Model Registry"):
        try:
            return tracking.client().get_model_version(nome, str(version))
        except MlflowException as exc:
            if not _not_found(exc):
                raise
            ausente = exc
    raise RegistryStateError(
        f"Versão {version} de {nome} não existe no Model Registry: nenhum alias foi "
        "alterado."
    ) from ausente


# MARK: challenger

CHALLENGER_ALIAS = "challenger"
PREVIOUS_ALIAS = "previous"
ORIGIN_RETRAIN = "retrain"
# Sem `model_sha256` o challenger nunca poderia ser exportado (`export_champion`
# confere o hash). Sem `bundle_model_version` próprio, o monitor confundiria o
# baseline de AUC do retreino com o do baseline (filtro por `model_version`).
_CHALLENGER_REQUIRED_TAGS = ("model_sha256", "bundle_model_version")


def _check_challenger_tags(tags: dict[str, str]) -> None:
    for chave in _CHALLENGER_REQUIRED_TAGS:
        if not str(tags.get(chave, "")).strip():
            raise ValueError(
                f"register_challenger exige a tag {chave} não vazia: nenhuma versão "
                "foi registrada."
            )
    if tags["bundle_model_version"] == BASELINE_MODEL_VERSION:
        raise ValueError(
            f"O challenger não pode ter bundle_model_version={BASELINE_MODEL_VERSION} "
            "(a do baseline): o monitor distingue os modelos por essa versão. Use uma "
            "versão própria do retreino (ex.: retrain-<sha256[:12]>)."
        )


def register_challenger(
    *,
    run_id: str,
    pipeline: Any,
    signature: Any,
    input_example: Any,
    bundle_path: Path,
    metadata_path: Path,
    tags: dict[str, str],
) -> int:
    """Registra o modelo retreinado como nova versão e move `challenger` para ela.

    Mesmos argumentos de `register_model_version`. As tags são conferidas ANTES do
    registro (`model_sha256` e `bundle_model_version` não vazias, esta diferente da
    do baseline) e ganham `origin=retrain`, que prevalece sobre outra `origin`.
    Nunca toca em `champion` nem em `previous`. Se o alias falhar depois do registro,
    a versão fica sem alias e o erro sobe; um novo retreino cria outra versão.
    """
    _check_challenger_tags(tags)
    versao = register_model_version(
        run_id=run_id,
        pipeline=pipeline,
        signature=signature,
        input_example=input_example,
        bundle_path=bundle_path,
        metadata_path=metadata_path,
        tags={**tags, "origin": ORIGIN_RETRAIN},
    )
    set_alias(CHALLENGER_ALIAS, versao)
    return versao


# MARK: alerta crítico e consumo

BATCH_RUN_TYPE = "batch"
CRITICAL_LEVEL = AlertLevel.CRITICO.value
RETRAIN_CONSUMED_TAG = "retrain_consumed"
SEARCH_PAGE_SIZE = 1000
_FINISHED = "FINISHED"


def _search_all(cliente: Any, experiment_id: str, filtro: str) -> list[Run]:
    """Todas as páginas do `search_runs` (o padrão do MLflow devolve só a primeira)."""
    runs: list[Run] = []
    token = None
    while True:
        pagina = cliente.search_runs(
            [experiment_id],
            filter_string=filtro,
            max_results=SEARCH_PAGE_SIZE,
            page_token=token,
        )
        runs.extend(pagina)
        token = pagina.token
        if not token:
            return runs


def _critical_runs() -> list[Run]:
    """Todos os runs de lote `CRITICO` do experimento (sem experimento, lista vazia).

    O `search_runs` filtra só pelas tags que todo run de lote tem (`run_type`,
    `alert_level`); `retrain_consumed`, ausente até o primeiro retreino, e o status
    são conferidos por quem chama.
    """
    filtro = (
        f"tags.run_type = '{BATCH_RUN_TYPE}' and tags.alert_level = '{CRITICAL_LEVEL}'"
    )
    with tracking.tracking_errors("Falha ao consultar os alertas no MLflow"):
        cliente = tracking.client()
        experimento = cliente.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
        if experimento is None:
            return []
        return _search_all(cliente, experimento.experiment_id, filtro)


def _consumed_batches(runs: list[Run]) -> dict[str, str]:
    """`batch_sha256` → versão do challenger que já consumiu um alerta desse lote.

    O consumo mais recente vence (ordem de `start_time`), para a mensagem citar o
    último challenger treinado sobre o lote.
    """
    consumidos: dict[str, str] = {}
    for r in sorted(runs, key=lambda r: (r.info.start_time or 0, r.info.run_id)):
        tags = r.data.tags
        if RETRAIN_CONSUMED_TAG in tags and tags.get("batch_sha256"):
            consumidos[tags["batch_sha256"]] = tags[RETRAIN_CONSUMED_TAG]
    return consumidos


def _consumption(run: Run, consumidos: dict[str, str]) -> str | None:
    tags = run.data.tags
    if RETRAIN_CONSUMED_TAG in tags:
        return tags[RETRAIN_CONSUMED_TAG]
    sha = tags.get("batch_sha256")
    return consumidos.get(sha) if sha else None


def alert_consumed_by(run: Run) -> str | None:
    """Versão do challenger que consumiu o alerta `run` ou outro alerta do MESMO lote.

    O consumo é do lote (`batch_sha256`), não só do run: reexecuções da demo gravam
    novos alertas do mesmo t2 (o `simulate` é determinístico), e um segundo retreino
    sobre os mesmos dados é o reaproveitamento que exige o flag explícito (§4.7).
    """
    if RETRAIN_CONSUMED_TAG in run.data.tags:
        return run.data.tags[RETRAIN_CONSUMED_TAG]
    if not run.data.tags.get("batch_sha256"):
        return None
    return _consumption(run, _consumed_batches(_critical_runs()))


def find_unconsumed_critical_run(model_version: str | None = None) -> Run | None:
    """Run de lote `CRITICO` mais recente ainda não consumido por um retreino (§4.7).

    Não consumido = sem `retrain_consumed` e de um lote (`batch_sha256`) que nenhum
    retreino consumiu (`alert_consumed_by`). Só status `FINISHED` (lote cujo run
    falhou não habilita retreino) e, com `model_version`, só alertas emitidos contra
    esse modelo (a tag `model_version` do run de lote). Mais recente = maior
    `start_time`, com empate decidido pelo maior `run_id`, para a escolha ser
    determinística. Sem experimento ou alerta, `None`.
    """
    runs = _critical_runs()
    consumidos = _consumed_batches(runs)
    candidatos = [
        r
        for r in runs
        if r.info.status == _FINISHED
        and _consumption(r, consumidos) is None
        and (model_version is None or r.data.tags.get("model_version") == model_version)
    ]
    if not candidatos:
        return None
    return max(candidatos, key=lambda r: (r.info.start_time or 0, r.info.run_id))


def require_critical_run(run_id: str) -> Run:
    """O run `run_id`, desde que seja um lote `CRITICO` terminado; senão `ValueError`.

    É a conferência de um alerta escolhido à mão (ex.: `--alert-run`). Não olha
    `retrain_consumed`: reaproveitar um alerta consumido é decisão do chamador.
    """
    with tracking.tracking_errors("Falha ao consultar o run do alerta no MLflow"):
        try:
            run = tracking.client().get_run(run_id)
        except MlflowException as exc:
            # Id malformado (ex.: digitado no `--alert-run`) também é "não encontrado".
            if not _not_found(exc, malformed_id=True):
                raise
            run = None
    if run is None:
        raise ValueError(f"Run de alerta {run_id} não encontrado no MLflow.")
    tags = run.data.tags
    if tags.get("run_type") != BATCH_RUN_TYPE:
        raise ValueError(
            f"Run {run_id} não é um run de lote (run_type={tags.get('run_type')}): só "
            "um alerta de lote habilita o retreino."
        )
    nivel = tags.get("alert_level")
    if nivel != CRITICAL_LEVEL:
        raise ValueError(
            f"Run de lote {run_id} tem alert_level={nivel}, não {CRITICAL_LEVEL}: só um "
            "alerta CRÍTICO habilita o retreino."
        )
    if run.info.status != _FINISHED:
        raise ValueError(
            f"Run de lote {run_id} terminou com status {run.info.status}: um lote cujo "
            "registro falhou não habilita o retreino."
        )
    return run


def mark_alert_consumed(batch_run_id: str, version: int) -> None:
    """Grava `retrain_consumed=<versão>` no run do alerta que gerou o challenger.

    O run tem de ser um lote `CRITICO` (`require_critical_run`); um consumo anterior
    é sobrescrito (reaproveitamento já decidido pelo chamador). Os outros alertas
    `CRITICO` terminados e não consumidos do mesmo lote (`batch_sha256`) recebem a
    mesma tag: a trilha mostra em cada run que o lote já gerou um challenger.
    """
    alvo = require_critical_run(batch_run_id)
    sha = alvo.data.tags.get("batch_sha256")
    mesmo_lote = (
        [
            r.info.run_id
            for r in _critical_runs()
            if r.info.run_id != batch_run_id
            and r.info.status == _FINISHED
            and r.data.tags.get("batch_sha256") == sha
            and RETRAIN_CONSUMED_TAG not in r.data.tags
        ]
        if sha
        else []
    )
    with tracking.tracking_errors("Falha ao marcar o alerta como consumido"):
        cliente = tracking.client()
        for run_id in (batch_run_id, *mesmo_lote):
            cliente.set_tag(run_id, RETRAIN_CONSUMED_TAG, str(int(version)))
    log_event(
        logger,
        "alert_consumed",
        mlflow_run_id=batch_run_id,
        model_name=config.REGISTERED_MODEL_NAME,
        version=int(version),
        same_batch_run_ids=mesmo_lote,
    )


# MARK: promoção e rollback dos aliases


def promote_alias(version: int) -> None:
    """Move `champion` para `version` e dá `previous` ao champion anterior (§4.7).

    Só move ALIASES: exportar o arquivo e a ordem completa da promoção ficam com o
    comando de promoção. `previous` primeiro, `champion` por último: se o segundo
    falhar, o champion continua o anterior (e `previous` aponta para ele).

    Promover a versão que já é champion não faz nada (nem apaga nem move
    `previous`), para que repetir uma promoção interrompida seja seguro. Versão
    inexistente: `RegistryStateError`, sem alterar alias.
    """
    atual = get_alias_version(CHAMPION_ALIAS)
    require_version(version)
    if atual is not None and int(atual.version) == int(version):
        log_event(
            logger,
            "model_alias_promote_skipped",
            model_name=config.REGISTERED_MODEL_NAME,
            version=int(version),
            reason="already_champion",
        )
        return
    if atual is not None:
        set_alias(PREVIOUS_ALIAS, int(atual.version))
    set_alias(CHAMPION_ALIAS, version)


def rollback_alias() -> int:
    """Devolve `champion` à versão `previous`, consome o `previous` e devolve a versão.

    Sem `previous`: `RegistryStateError`, sem alterar alias. O histórico é de um
    nível: `previous` é removido depois de mover o `champion`, para que um segundo
    rollback falhe em vez de re-promover a versão revertida sem passar pela regra de
    promoção. Se a remoção falhar, os dois aliases apontam para a mesma versão e
    repetir o rollback é seguro. Não toca nos arquivos de `models/`.
    """
    nome = config.REGISTERED_MODEL_NAME
    anterior = get_alias_version(PREVIOUS_ALIAS)
    if anterior is None:
        raise RegistryStateError(
            f"{nome} não tem o alias {PREVIOUS_ALIAS}: não há versão para onde voltar "
            "(nenhuma promoção desde o último rollback). Nenhum alias foi alterado."
        )
    versao = int(anterior.version)
    set_alias(CHAMPION_ALIAS, versao)
    delete_alias(PREVIOUS_ALIAS)
    return versao


def delete_alias(alias: str) -> None:
    """Remove o `alias` de `REGISTERED_MODEL_NAME`; alias ausente não faz nada."""
    atual = get_alias_version(alias)
    if atual is None:
        return
    nome = config.REGISTERED_MODEL_NAME
    with tracking.tracking_errors(
        f"Falha ao remover o alias {alias} no Model Registry"
    ):
        tracking.client().delete_registered_model_alias(nome, alias)
    log_event(
        logger,
        "model_alias_deleted",
        model_name=nome,
        alias=alias,
        version=int(atual.version),
    )


# MARK: auditoria

AUDIT_RUN_TYPE = "audit"
ACTION_PROMOTE = "promote"
ACTION_ROLLBACK = "rollback"
AUDIT_ACTIONS = frozenset({ACTION_PROMOTE, ACTION_ROLLBACK})
# Estado reconferido depois da troca (alias `champion` x arquivo servido).
STATUS_CONSISTENT = "CONSISTENT"
STATUS_INCONSISTENT = "INCONSISTENT"
AUDIT_STATUSES = frozenset({STATUS_CONSISTENT, STATUS_INCONSISTENT})
_REDACTED = "***"
# Chaves de `details` cujo valor é segredo por natureza (redigir a mais é seguro).
_SENSITIVE_KEY = re.compile(
    r"pass(word|wd)?|secret|token|credential|api[_-]?key|auth", re.IGNORECASE
)
# URIs no meio de texto livre (motivo, detalhes) passam por `redact_uri`.
_URI_IN_TEXT = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def redact_text(texto: str) -> str:
    """Texto livre (motivo, operador) com toda URI redigida por `redact_uri`."""
    return _URI_IN_TEXT.sub(lambda m: tracking.redact_uri(m.group(0)), texto)


def _redact(valor: Any) -> Any:
    """Cópia de `details` sem segredos e sem float não finito (JSON válido)."""
    if isinstance(valor, str):
        return redact_text(valor)
    if isinstance(valor, float) and not math.isfinite(valor):
        return str(valor)
    if isinstance(valor, dict):
        return {
            str(k): _REDACTED if _SENSITIVE_KEY.search(str(k)) else _redact(v)
            for k, v in valor.items()
        }
    if isinstance(valor, list | tuple):
        return [_redact(v) for v in valor]
    return valor


def _resolve_operator(operator: str | None) -> str:
    if operator is None:
        try:
            operator = getpass.getuser()
        except (OSError, KeyError, ImportError) as exc:
            raise ValueError(
                "Não foi possível identificar o operador da auditoria "
                "(getpass.getuser falhou): informe o operador explicitamente."
            ) from exc
    if not operator.strip():
        raise ValueError("A auditoria exige um operador não vazio.")
    return operator.strip()


def _audit_time(clock: Callable[[], datetime]) -> datetime:
    agora = clock()
    if agora.tzinfo is None or agora.utcoffset() is None:
        raise ValueError(
            "O relógio da auditoria devolveu hora sem fuso: use datetime com tzinfo "
            "(UTC)."
        )
    return agora.astimezone(UTC)


def log_audit_run(
    *,
    action: str,
    version: int,
    reason: str,
    details: dict[str, Any],
    operator: str | None = None,
    clock: Callable[[], datetime] = _utc_now,
    status: str | None = None,
) -> str:
    """Run de auditoria (`run_type=audit`) de uma promoção ou rollback: quem, quando, por quê.

    Tags: `action` (`promote`/`rollback`), `model_name`, `version`, `reason`,
    `operator` (padrão: `getpass.getuser()`), `timestamp_utc` (ISO 8601 em UTC, do
    `clock`, injetável nos testes), `pipeline_run_id`, `details` (JSON) e, se dado,
    `status` (`CONSISTENT`/`INCONSISTENT`, o estado reconferido). Antes de
    gravar: motivo vazio, ação ou status desconhecidos, hora sem fuso, `details` não
    serializável ou tag acima do limite do MLflow são `ValueError`, sem run. URIs
    no motivo e nos detalhes saem redigidas e chaves sensíveis (senha, token...)
    viram `***`. Devolve o `run_id`.

    As tags vão todas no `create_run`; se só o encerramento falhar, o run fica
    `RUNNING` com o registro completo e o erro sobe (`TrackingUnavailableError`).
    """
    if action not in AUDIT_ACTIONS:
        raise ValueError(
            f"Ação de auditoria desconhecida: {action} (esperado: "
            f"{', '.join(sorted(AUDIT_ACTIONS))})."
        )
    if status is not None and status not in AUDIT_STATUSES:
        raise ValueError(
            f"Status de auditoria desconhecido: {status} (esperado: "
            f"{', '.join(sorted(AUDIT_STATUSES))})."
        )
    motivo = reason.strip() if isinstance(reason, str) else ""
    if not motivo:
        raise ValueError(
            "A auditoria exige um motivo não vazio (reason): quem promove ou reverte "
            "registra por quê."
        )
    quem = _resolve_operator(operator)
    quando = _audit_time(clock)
    try:
        detalhes = json.dumps(
            _redact(details), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"details da auditoria não é serializável em JSON: {exc}"
        ) from exc
    tags = {
        "run_type": AUDIT_RUN_TYPE,
        "action": action,
        "model_name": config.REGISTERED_MODEL_NAME,
        "version": str(int(version)),
        "reason": redact_text(motivo),
        "operator": redact_text(quem),
        "timestamp_utc": quando.isoformat(),
        "pipeline_run_id": RUN_ID,
        "details": detalhes,
    }
    if status is not None:
        tags["status"] = status
    longas = sorted(k for k, v in tags.items() if len(v) > MAX_TAG_VAL_LENGTH)
    if longas:
        raise ValueError(
            f"Tag(s) {', '.join(longas)} da auditoria passam de {MAX_TAG_VAL_LENGTH} "
            "caracteres (limite do MLflow): resuma o conteúdo."
        )

    experiment_id = tracking.configure_tracking()
    with tracking.tracking_errors("Falha ao registrar o run de auditoria"):
        cliente = tracking.client()
        run = cliente.create_run(
            experiment_id,
            start_time=int(quando.timestamp() * 1000),
            tags=tags,
            run_name=f"audit_{action}_v{int(version)}",
        )
        cliente.set_terminated(run.info.run_id, _FINISHED)
    log_event(
        logger,
        "audit_run_logged",
        level=logging.INFO,
        action=action,
        model_name=config.REGISTERED_MODEL_NAME,
        version=int(version),
        operator=tags["operator"],
        mlflow_run_id=run.info.run_id,
    )
    return run.info.run_id
