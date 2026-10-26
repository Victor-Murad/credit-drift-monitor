"""Testes dos utilitários de infraestrutura do monitoramento (memória e logs)."""

import json
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from src.config import PIPELINE_LOG_FILE
from src.monitoring.logs import extract_run_events
from src.monitoring.resources import peak_memory_mb


def test_peak_memory_positivo_e_em_mb():
    assert 10 < peak_memory_mb() < 100_000


def test_peak_memory_converte_unidade_por_plataforma(monkeypatch):
    monkeypatch.setattr(
        resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=1_048_576)
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    assert peak_memory_mb() == 1.0
    monkeypatch.setattr(sys, "platform", "linux")
    assert peak_memory_mb() == 1024.0


def _linha(run_id: str, ts: datetime, **campos) -> str:
    payload = {
        "timestamp": ts.isoformat(),
        "run_id": run_id,
        "level": "INFO",
        "logger": "teste",
        "message": "evento",
        **campos,
    }
    return json.dumps(payload)


def _ts(minuto: int) -> datetime:
    return datetime(2026, 10, 26, 12, minuto, 0, tzinfo=UTC)


def test_extract_run_events_filtra_por_run_id_e_janela(tmp_path: Path):
    log = tmp_path / "pipeline.jsonl"
    dentro = _linha("run-a", _ts(10), event="dentro")
    linhas = [
        _linha("run-a", _ts(1), event="antes"),
        dentro,
        _linha("run-b", _ts(10), event="outro_run"),
        _linha("run-a", _ts(50), event="depois"),
    ]
    log.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    eventos = extract_run_events(log, "run-a", since=_ts(5), until=_ts(20))

    assert eventos == [dentro]


def test_extract_run_events_le_arquivo_rotacionado(tmp_path: Path):
    log = tmp_path / "pipeline.jsonl"
    antigo = _linha("run-a", _ts(10), event="antigo")
    recente = _linha("run-a", _ts(11), event="recente")
    (tmp_path / "pipeline.jsonl.1").write_text(antigo + "\n", encoding="utf-8")
    log.write_text(recente + "\n", encoding="utf-8")

    eventos = extract_run_events(log, "run-a", since=_ts(0), until=_ts(59))

    assert sorted(eventos) == sorted([antigo, recente])


def test_extract_run_events_descarta_linha_invalida_sem_levantar(tmp_path: Path):
    log = tmp_path / "pipeline.jsonl"
    valida = _linha("run-a", _ts(10), event="ok")
    log.write_text(
        "isto não é json\n" + valida + "\n" + "{truncado\n", encoding="utf-8"
    )

    eventos = extract_run_events(log, "run-a", since=_ts(0), until=_ts(59))

    assert eventos == [valida]
    registro = [
        json.loads(linha)
        for linha in PIPELINE_LOG_FILE.read_text(encoding="utf-8").splitlines()
    ]
    descarte = [e for e in registro if e.get("event") == "log_lines_discarded"]
    assert descarte[-1]["discarded"] == 2
