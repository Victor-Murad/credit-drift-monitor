"""Relatorios de validacao do contrato (JSON para maquinas, HTML para pessoas)."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from src.config import project_relative
from src.validation.gate import APPROVED, ValidationResult

_MAX_FAILURE_ROWS = 100

_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
     background:#f6f7f9;color:#1d2330}
main{max-width:1100px;margin:0 auto;padding:32px 16px}
h1{font-size:1.5rem;margin:0 0 4px}
h2{font-size:1.1rem;margin:32px 0 8px}
.muted{color:#5b6475}
.badge{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:700;
       color:#fff;letter-spacing:.04em}
.ok{background:#1f8a4c}.fail{background:#c62828}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:16px}
.card{background:#fff;border:1px solid #e1e4ea;border-radius:8px;padding:12px}
.card b{display:block;font-size:1.4rem}
table{border-collapse:collapse;width:100%;background:#fff;font-size:.9rem}
th,td{border:1px solid #e1e4ea;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#eef0f4}
code{background:#eef0f4;padding:1px 4px;border-radius:4px}
.scroll{overflow-x:auto}
"""


def _table(rows: list[dict], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def render_html(result: ValidationResult, failure_cases: pd.DataFrame) -> str:
    approved = result.status == APPROVED
    badge = (
        '<span class="badge ok">APROVADO</span>'
        if approved
        else '<span class="badge fail">BLOQUEADO</span>'
    )
    verdict = (
        "O lote cumpre todas as regras do contrato e foi liberado para ingestao."
        if approved
        else "O lote viola o contrato. A ingestao foi interrompida e o arquivo "
        "foi enviado para quarentena; nenhum registro chegou ao modelo."
    )

    cards = "".join(
        f'<div class="card"><span class="muted">{label}</span><b>{value}</b></div>'
        for label, value in [
            ("Linhas", f"{result.n_rows:,}".replace(",", ".")),
            ("Colunas", result.n_columns),
            ("Regras violadas", len(result.rules_violated)),
            ("Ocorrencias", f"{result.n_failure_cases:,}".replace(",", ".")),
            ("Tempo", f"{result.duration_ms} ms"),
        ]
    )

    sections = []
    if result.rules_violated:
        rules = [
            {**r, "examples": ", ".join(r["examples"])} for r in result.rules_violated
        ]
        sections.append("<h2>Regras violadas</h2>")
        sections.append(
            _table(rules, ["rule", "column", "check", "n_failures", "examples"])
        )
        sample = failure_cases.head(_MAX_FAILURE_ROWS).to_dict("records")
        sections.append(
            f"<h2>Ocorrencias (primeiras {min(len(failure_cases), _MAX_FAILURE_ROWS)}"
            f" de {len(failure_cases)})</h2>"
        )
        sections.append(_table(sample, ["column", "check", "failure_case", "index"]))
    if result.quarantine_file:
        sections.append(
            f"<h2>Quarentena</h2><p>Arquivo original preservado em "
            f"<code>{html.escape(result.quarantine_file)}</code> para analise.</p>"
        )

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Contract - {html.escape(result.batch_name)}</title>
<style>{_CSS}</style>
</head>
<body><main>
<h1>Validacao do Data Contract {badge}</h1>
<p class="muted">Lote <code>{html.escape(result.batch_name)}</code> &middot;
etapa <code>{html.escape(result.stage)}</code> &middot;
contrato <code>{html.escape(result.contract)}</code> &middot;
{html.escape(result.validated_at)}</p>
<p>{verdict}</p>
<div class="cards">{cards}</div>
{"".join(sections)}
</main></body>
</html>
"""


def write_reports(
    result: ValidationResult, failure_cases: pd.DataFrame, report_dir: Path
) -> dict[str, str]:
    """Grava `<lote>_report.json` e `<lote>_report.html` em `report_dir`."""
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{result.batch_name}_report.json"
    html_path = report_dir / f"{result.batch_name}_report.html"

    paths = {"json": project_relative(json_path), "html": project_relative(html_path)}
    payload = {**result.to_dict(), "report_files": paths}
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    html_path.write_text(render_html(result, failure_cases), encoding="utf-8")
    return paths
