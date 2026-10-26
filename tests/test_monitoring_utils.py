"""Testes dos utilitários de infraestrutura do monitoramento (memória e logs)."""

import json
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

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


def test_extract_run_events_le_arquivo_rotacionado_em_ordem_cronologica(
    tmp_path: Path,
):
    # RotatingFileHandler: maior sufixo = mais antigo; o principal é o mais novo.
    # `.10` testa que a ordenação é numérica e não lexicográfica.
    log = tmp_path / "pipeline.jsonl"
    mais_antigo = _linha("run-a", _ts(1), event="mais_antigo")
    antigo = _linha("run-a", _ts(10), event="antigo")
    recente = _linha("run-a", _ts(20), event="recente")
    atual = _linha("run-a", _ts(30), event="atual")
    (tmp_path / "pipeline.jsonl.10").write_text(mais_antigo + "\n", encoding="utf-8")
    (tmp_path / "pipeline.jsonl.2").write_text(antigo + "\n", encoding="utf-8")
    (tmp_path / "pipeline.jsonl.1").write_text(recente + "\n", encoding="utf-8")
    log.write_text(atual + "\n", encoding="utf-8")

    eventos = extract_run_events(log, "run-a", since=_ts(0), until=_ts(59))

    assert eventos == [mais_antigo, antigo, recente, atual]


def test_extract_run_events_fronteiras_da_janela_sao_inclusivas(tmp_path: Path):
    log = tmp_path / "pipeline.jsonl"
    no_inicio = _linha("run-a", _ts(5), event="inicio")
    no_fim = _linha("run-a", _ts(20), event="fim")
    log.write_text(no_inicio + "\n" + no_fim + "\n", encoding="utf-8")

    eventos = extract_run_events(log, "run-a", since=_ts(5), until=_ts(20))

    assert eventos == [no_inicio, no_fim]


def test_extract_run_events_arquivo_principal_ausente(tmp_path: Path):
    log = tmp_path / "pipeline.jsonl"
    antigo = _linha("run-a", _ts(10), event="antigo")
    (tmp_path / "pipeline.jsonl.1").write_text(antigo + "\n", encoding="utf-8")

    eventos = extract_run_events(log, "run-a", since=_ts(0), until=_ts(59))

    assert eventos == [antigo]


def test_extract_run_events_janela_ingenua_vale_como_utc(tmp_path: Path):
    log = tmp_path / "pipeline.jsonl"
    dentro = _linha("run-a", _ts(10), event="dentro")
    log.write_text(dentro + "\n", encoding="utf-8")

    eventos = extract_run_events(
        log,
        "run-a",
        since=datetime(2026, 10, 26, 12, 0),
        until=datetime(2026, 10, 26, 12, 59),
    )

    assert eventos == [dentro]


def test_extract_run_events_byte_invalido_nao_aborta(tmp_path: Path):
    log = tmp_path / "pipeline.jsonl"
    valida = _linha("run-a", _ts(10), event="ok")
    log.write_bytes(b"\xff\xfe lixo\n" + valida.encode("utf-8") + b"\n")

    eventos = extract_run_events(log, "run-a", since=_ts(0), until=_ts(59))

    assert eventos == [valida]


def test_extract_run_events_descarta_linha_invalida_sem_levantar(
    tmp_path: Path, monkeypatch
):
    chamadas: list[dict] = []

    def espiao(_logger, event, **campos):
        chamadas.append({"event": event, **campos})

    monkeypatch.setattr("src.monitoring.logs.log_event", espiao)
    log = tmp_path / "pipeline.jsonl"
    valida = _linha("run-a", _ts(10), event="ok")
    log.write_text(
        "isto não é json\n" + valida + "\n" + "{truncado\n", encoding="utf-8"
    )

    eventos = extract_run_events(log, "run-a", since=_ts(0), until=_ts(59))

    assert eventos == [valida]
    assert len(chamadas) == 1
    assert chamadas[0]["event"] == "log_lines_discarded"
    assert chamadas[0]["discarded"] == 2


def test_extract_run_events_sem_descarte_nao_registra(tmp_path: Path, monkeypatch):
    chamadas: list[str] = []
    monkeypatch.setattr(
        "src.monitoring.logs.log_event", lambda *_a, **_k: chamadas.append("x")
    )
    log = tmp_path / "pipeline.jsonl"
    log.write_text(_linha("run-a", _ts(10)) + "\n", encoding="utf-8")

    extract_run_events(log, "run-a", since=_ts(0), until=_ts(59))

    assert chamadas == []
