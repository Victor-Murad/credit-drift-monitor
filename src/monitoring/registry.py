"""Model Registry do MLflow: versões, aliases e o champion servido (spec da Etapa 3, §4.7).

- Cada treino vira uma versão de `REGISTERED_MODEL_NAME`, criada a partir do modelo
  logado no run de treino; o run anexa também o `bundle` joblib e o metadata em
  `bundle/` (é dali que promoção e rollback copiam o champion).
- Aliases (`champion`, `challenger`, `previous`) no lugar dos estágios fixos.
- O champion servido vive em `models/champion/` (fora dos `outs` do DVC). A cópia é
  feita por temporário no mesmo diretório + `os.replace` e conferida pelo sha256 da
  tag `model_sha256` da versão: divergência falha fechado. Quem dá o alias exporta
  ANTES (ver `src.monitoring.stage.run_train`): export falho não deixa alias.

As constantes `src.config.REGISTERED_MODEL_NAME` e `CHAMPION_*` são lidas na chamada,
para que os testes possam redirecioná-las.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion

from src import config
from src.logging_utils import get_logger, log_event
from src.models.integrity import ModelIntegrityError, file_sha256
from src.monitoring import tracking
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


def _baixar_conferir_e_trocar(
    run_id: str, esperado: str, version: int, destino: Path
) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destino) as tmp:
        with tracking.tracking_errors("Falha ao baixar o bundle do run de origem"):
            baixado = Path(
                tracking.client().download_artifacts(
                    run_id, BUNDLE_ARTIFACT_DIR, dst_path=tmp
                )
            )
        modelo = tracking.require_path(
            baixado / BUNDLE_MODEL_FILE, "Bundle do champion"
        )
        metadata = tracking.require_path(
            baixado / BUNDLE_METADATA_FILE, "Metadata do champion"
        )
        _conferir_bundle(modelo, metadata, esperado, version)
        _trocar_par(modelo, metadata, Path(tmp) / "metadata_anterior.json")


def export_champion(version: int) -> None:
    """Copia bundle + metadata do run de origem da `version` para `models/champion/`.

    Baixa o artefato `bundle/` num temporário dentro de `CHAMPION_DIR` (mesmo
    sistema de arquivos), confere o sha256 contra a tag `model_sha256` e só então
    troca os arquivos com `os.replace`, metadata primeiro e modelo por último (ver
    `_trocar_par` para a janela entre as duas trocas).

    Divergência de hash ou erro de arquivo (`OSError`: diretório, download local,
    leitura, troca) levanta `ModelIntegrityError` com o caminho, sem credenciais;
    falha do MLflow levanta `TrackingUnavailableError` (mensagem redigida).
    """
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

    destino = Path(config.CHAMPION_DIR)
    try:
        _baixar_conferir_e_trocar(mv.run_id, esperado, version, destino)
    except OSError as exc:
        raise ModelIntegrityError(
            f"Falha ao exportar o champion (versão {version}) para {destino}: {exc}"
        ) from exc
    log_event(
        logger,
        "champion_exported",
        level=logging.INFO,
        model_name=nome,
        version=int(version),
        model_sha256=esperado,
        file=config.project_relative(Path(config.CHAMPION_FILE)),
    )
