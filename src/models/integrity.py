"""Integridade do artefato do modelo antes de desserializar.

O bundle e um `.joblib` (pickle): carregar um arquivo adulterado executa
codigo arbitrario. O treino grava o sha256 em `baseline_metadata.json`
(`artifact.sha256`); aqui o hash do arquivo em disco e conferido contra ele
e a ausencia do metadata tambem bloqueia (fail-closed).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config import MODEL_METADATA_FILE


class ModelIntegrityError(RuntimeError):
    """O artefato do modelo nao bate com o hash registrado no treino."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_model_integrity(model_path: Path, metadata_path: Path | None = None) -> None:
    """Lanca `ModelIntegrityError` se o sha256 do modelo divergir do metadata.

    Por padrao o metadata e o `baseline_metadata.json` ao lado do modelo.
    """
    metadata_path = metadata_path or model_path.with_name(MODEL_METADATA_FILE.name)
    if not metadata_path.exists():
        raise ModelIntegrityError(
            f"Metadata do modelo ausente em {metadata_path}: sem hash registrado, "
            "o artefato nao pode ser verificado. Rode `make train`."
        )
    try:
        expected = json.loads(metadata_path.read_text(encoding="utf-8"))["artifact"][
            "sha256"
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelIntegrityError(
            f"Metadata invalido em {metadata_path}: sem artifact.sha256."
        ) from exc

    actual = file_sha256(model_path)
    if actual != expected:
        raise ModelIntegrityError(
            f"sha256 de {model_path.name} diverge do registrado no treino "
            f"(esperado {expected[:12]}..., obtido {actual[:12]}...). "
            "Artefato adulterado ou desatualizado: retreine com `make train`."
        )
