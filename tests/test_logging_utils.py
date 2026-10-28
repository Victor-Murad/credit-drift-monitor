"""Testes do logging estruturado (JSONL com run_id e rotacao)."""

from __future__ import annotations

import json
from logging.handlers import RotatingFileHandler

from src import logging_utils
from src.logging_utils import RUN_ID, get_logger, log_event


def test_events_carry_run_id_and_fields(tmp_path) -> None:
    log_file = tmp_path / "pipeline.jsonl"
    logger = get_logger("tests.logging.runid", log_file=log_file)

    log_event(logger, "demo_event", batch_name="b1", n_rows=3)

    line = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert line["run_id"] == RUN_ID
    assert (line["event"], line["batch_name"], line["n_rows"]) == (
        "demo_event",
        "b1",
        3,
    )


def test_file_handler_rotates(tmp_path) -> None:
    logger = get_logger("tests.logging.rotation", log_file=tmp_path / "p.jsonl")

    handler = next(h for h in logger.handlers if isinstance(h, RotatingFileHandler))

    assert handler.maxBytes == logging_utils.LOG_MAX_BYTES
    assert handler.backupCount == logging_utils.LOG_BACKUP_COUNT
