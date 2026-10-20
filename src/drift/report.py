"""Relatorios de drift (HTML do Evidently + JSON/MD nossos)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import project_relative


def write_drift_reports(
    result: Any,
    snapshot: Any,
    *,
    report_dir: Path,
) -> dict[str, str]:
    """Grava HTML (Evidently), JSON estruturado e devolve caminhos relativos."""
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = result.batch_name

    html_path = report_dir / f"{stem}_drift_report.html"
    json_path = report_dir / f"{stem}_drift_report.json"

    snapshot.save_html(str(html_path))

    payload = result.to_dict()
    payload["evidently_metrics"] = snapshot.dict().get("metrics", [])
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "html": project_relative(html_path),
        "json": project_relative(json_path),
    }
