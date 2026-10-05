"""Logging estruturado do pipeline.

Cada evento vai para dois destinos:
- console, em formato legivel (para quem roda o script);
- `logs/pipeline.jsonl`, uma linha JSON por evento (para ser centralizado
  na Etapa 3 -- MLflow/dashboard -- sem precisar fazer parsing de texto).

Uso:
    logger = get_logger(__name__)
    log_event(logger, "contract_validation", stage="ingestion", status="BLOCKED", n_rows=508)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PIPELINE_LOG_FILE

_EXTRA_KEY = "event_fields"


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, _EXTRA_KEY, {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Retorna um logger com handler de console e de arquivo JSONL."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(console)

    log_path = log_file or PIPELINE_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(JsonLinesFormatter())
    logger.addHandler(file_handler)
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    message: str | None = None,
    **fields: Any,
) -> None:
    """Registra um evento estruturado (`event` + campos arbitrarios)."""
    text = message or " ".join([event, *(f"{k}={v}" for k, v in fields.items())])
    logger.log(level, text, extra={_EXTRA_KEY: {"event": event, **fields}})
