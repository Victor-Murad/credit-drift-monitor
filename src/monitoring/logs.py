"""Extração de eventos do log JSONL do pipeline por `run_id` e janela de tempo."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.logging_utils import get_logger, log_event

logger = get_logger("monitoring.logs")


def _arquivos_de_log(log_file: Path) -> list[Path]:
    """Arquivos do log em ordem cronológica: `.N` ... `.2`, `.1` e o principal.

    No `RotatingFileHandler`, quanto maior o sufixo, mais antigo o arquivo.
    """
    rotacionados = [
        p for p in log_file.parent.glob(f"{log_file.name}.*") if p.suffix[1:].isdigit()
    ]
    rotacionados.sort(key=lambda p: int(p.suffix[1:]), reverse=True)
    principal = [log_file] if log_file.exists() else []
    return [*rotacionados, *principal]


def _como_utc(momento: datetime) -> datetime:
    return momento if momento.tzinfo else momento.replace(tzinfo=UTC)


def _timestamp(evento: dict[str, Any]) -> datetime:
    return _como_utc(datetime.fromisoformat(evento["timestamp"]))


def extract_run_events(
    log_file: Path, run_id: str, *, since: datetime, until: datetime
) -> list[str]:
    """Linhas do log com `run_id` igual e timestamp em [since, until].

    Lê os rotacionados (do mais antigo ao mais novo) e depois o principal. Os
    limites são inclusivos; `since`/`until` sem timezone valem como UTC, assim
    como timestamps de log sem timezone. Linhas que não são um evento JSON
    válido são descartadas sem levantar exceção; a contagem é registrada via
    `log_event` (nível WARNING) quando maior que zero. Bytes inválidos em UTF-8
    são substituídos para não abortar a leitura.
    """
    since, until = _como_utc(since), _como_utc(until)
    linhas: list[str] = []
    descartadas = 0
    for arquivo in _arquivos_de_log(log_file):
        with arquivo.open(encoding="utf-8", errors="replace") as fh:
            for bruta in fh:
                linha = bruta.strip()
                if not linha:
                    continue
                try:
                    evento = json.loads(linha)
                    if evento.get("run_id") != run_id:
                        continue
                    momento = _timestamp(evento)
                except (ValueError, KeyError, TypeError, AttributeError):
                    descartadas += 1
                    continue
                if since <= momento <= until:
                    linhas.append(linha)
    if descartadas:
        log_event(
            logger,
            "log_lines_discarded",
            level=logging.WARNING,
            message=f"{descartadas} linha(s) de log inválida(s) descartada(s)",
            file=str(log_file),
            run_id_filtrado=run_id,
            discarded=descartadas,
        )
    return linhas
