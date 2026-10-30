"""Auditoria de vies (fairness) das decisoes de credito (Etapa 4).

`gender` nunca entra no modelo (src/config.py, AUDIT_COLUMNS): fica no lote
so para esta auditoria. A idade e feature do modelo e tambem atributo
sensivel para nao discriminacao (LGPD art. 6, IX; Estatuto da Pessoa Idosa),
entao e auditada por faixa.

Desfecho favoravel = credito aprovado (`predicted_default == 0`). Por grupo:

    approval_rate             P(aprovado | grupo)          -> paridade estatistica
    good_payer_approval_rate  P(aprovado | target=0, grupo) -> igualdade de oportunidade
    bad_payer_approval_rate   P(aprovado | target=1, grupo) -> (com a anterior) equalized odds
    default_rate_approved     inadimplencia da carteira aprovada do grupo
    roc_auc                   poder de ordenacao do score dentro do grupo

Resumo por atributo: razao de impacto disparatado (menor / maior taxa de
aprovacao) e as diferencas maximas entre grupos. O limiar de 0,80 da razao
("regra dos quatro quintos") e uma referencia de mercado dos EUA, nao um
criterio legal brasileiro: um valor abaixo dele pede investigacao humana,
nao prova discriminacao.

    uv run python -m src.governance.fairness [--batch data/production/production_base.parquet]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import (
    ID_COLUMN,
    PRODUCTION_BASE_FILE,
    REPORTS_DIR,
    TARGET_COLUMN,
    project_relative,
)
from src.logging_utils import get_logger, log_event
from src.models.predict import load_bundle, score_batch
from src.validation.gate import DataContractViolation
from src.validation.io import read_batch

logger = get_logger("governance.fairness")

FAIRNESS_REPORTS_DIR = REPORTS_DIR / "governance"
# Referencia (nao criterio legal): razao de impacto disparatado < 0,80 pede
# investigacao humana.
DISPARATE_IMPACT_MIN = 0.80
# Grupos menores que isto entram no relatorio, mas nao no resumo: taxas de
# grupos minusculos sao ruido.
MIN_GROUP_SIZE = 30
AGE_BINS = [18, 25, 35, 45, 60, np.inf]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-59", "60+"]
SENSITIVE_ATTRIBUTES = ("gender", "age_band")


def age_band(age_years: pd.Series) -> pd.Series:
    """Faixa etaria (intervalos fechados a esquerda: 25 cai em `25-34`)."""
    return pd.cut(age_years, bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)


def _rate(mask: pd.Series) -> float | None:
    return round(float(mask.mean()), 4) if len(mask) else None


def group_metrics(scored: pd.DataFrame, attribute: str) -> list[dict]:
    """Metricas por grupo de `attribute` num lote pontuado e rotulado.

    `scored` precisa de `target`, `default_probability`, `predicted_default`
    e da coluna do atributo.
    """
    rows = []
    for group, part in scored.groupby(attribute, observed=True, sort=True):
        approved = part["predicted_default"] == 0
        good = part[TARGET_COLUMN] == 0
        bad = ~good
        both_classes = part[TARGET_COLUMN].nunique() == 2
        rows.append(
            {
                "group": str(group),
                "n": int(len(part)),
                "share": round(len(part) / len(scored), 4),
                "default_rate": _rate(part[TARGET_COLUMN] == 1),
                "approval_rate": _rate(approved),
                "good_payer_approval_rate": _rate(approved[good]),
                "bad_payer_approval_rate": _rate(approved[bad]),
                "default_rate_approved": _rate(part.loc[approved, TARGET_COLUMN] == 1),
                "roc_auc": (
                    round(
                        float(
                            roc_auc_score(
                                part[TARGET_COLUMN], part["default_probability"]
                            )
                        ),
                        4,
                    )
                    if both_classes
                    else None
                ),
            }
        )
    return rows


def _gap(rows: list[dict], key: str) -> float | None:
    values = [r[key] for r in rows if r[key] is not None]
    return round(max(values) - min(values), 4) if len(values) >= 2 else None


def summarize(rows: list[dict], min_group_size: int = MIN_GROUP_SIZE) -> dict:
    """Razao de impacto disparatado e diferencas entre grupos com n suficiente."""
    eligible = [r for r in rows if r["n"] >= min_group_size]
    approvals = [r["approval_rate"] for r in eligible]
    ratio = (
        round(min(approvals) / max(approvals), 4)
        if len(approvals) >= 2 and max(approvals) > 0
        else None
    )
    return {
        "groups_compared": [r["group"] for r in eligible],
        "groups_below_min_size": [r["group"] for r in rows if r["n"] < min_group_size],
        "disparate_impact_ratio": ratio,
        "statistical_parity_difference": _gap(eligible, "approval_rate"),
        "equal_opportunity_difference": _gap(eligible, "good_payer_approval_rate"),
        "bad_payer_approval_difference": _gap(eligible, "bad_payer_approval_rate"),
        "roc_auc_gap": _gap(eligible, "roc_auc"),
        "needs_review": ratio is not None and ratio < DISPARATE_IMPACT_MIN,
    }


def audit(batch: pd.DataFrame, bundle: dict, *, batch_name: str) -> dict:
    """Pontua o lote rotulado (atras do gate) e mede o vies por atributo."""
    if TARGET_COLUMN not in batch.columns:
        raise ValueError(
            f"A auditoria exige lote rotulado (coluna '{TARGET_COLUMN}'): sem "
            "rotulo so a paridade de aprovacao seria mensuravel."
        )
    scored = score_batch(
        batch,
        bundle,
        batch_name=batch_name,
        stage="fairness_audit",
        report_dir=None,
        quarantine_dir=None,
    )
    sensitive = batch[[ID_COLUMN, "gender", "age_years"]].copy()
    sensitive["age_band"] = age_band(sensitive.pop("age_years"))
    scored = scored.merge(sensitive, on=ID_COLUMN, how="left", validate="one_to_one")

    attributes = {}
    for attribute in SENSITIVE_ATTRIBUTES:
        rows = group_metrics(scored, attribute)
        attributes[attribute] = {"groups": rows, "summary": summarize(rows)}
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "batch_name": batch_name,
        "model_version": bundle["model_version"],
        "n_rows": int(len(scored)),
        "favorable_outcome": "aprovado (predicted_default == 0)",
        "disparate_impact_min": DISPARATE_IMPACT_MIN,
        "min_group_size": MIN_GROUP_SIZE,
        "attributes": attributes,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_markdown(report: dict) -> str:
    lines = [
        "# Auditoria de viés (fairness)",
        "",
        f"> Gerado em `{report['generated_at']}` · lote `{report['batch_name']}` · "
        f"modelo `{report['model_version']}` · {report['n_rows']} linhas.",
        "> Desfecho favorável = crédito aprovado. A razão de impacto disparatado "
        f"abaixo de {report['disparate_impact_min']:.2f} é referência de "
        "investigação, não prova de discriminação.",
        "",
    ]
    for attribute, data in report["attributes"].items():
        s = data["summary"]
        lines += [
            f"## `{attribute}`",
            "",
            "| Grupo | n | Default | Aprovação | Aprovação (bons) | Aprovação (maus) "
            "| Default aprovados | ROC-AUC |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in data["groups"]:
            lines.append(
                f"| {r['group']} | {r['n']} | {_fmt(r['default_rate'])} "
                f"| {_fmt(r['approval_rate'])} | {_fmt(r['good_payer_approval_rate'])} "
                f"| {_fmt(r['bad_payer_approval_rate'])} "
                f"| {_fmt(r['default_rate_approved'])} | {_fmt(r['roc_auc'])} |"
            )
        lines += [
            "",
            f"- Razão de impacto disparatado: **{_fmt(s['disparate_impact_ratio'])}**"
            f" (revisão necessária: {'SIM' if s['needs_review'] else 'não'})",
            f"- Diferença de paridade estatística: {_fmt(s['statistical_parity_difference'])}",
            f"- Diferença de igualdade de oportunidade: {_fmt(s['equal_opportunity_difference'])}",
            f"- Diferença de aprovação entre maus pagadores: {_fmt(s['bad_payer_approval_difference'])}",
            f"- Diferença de ROC-AUC entre grupos: {_fmt(s['roc_auc_gap'])}",
        ]
        if s["groups_below_min_size"]:
            lines.append(
                f"- Fora do resumo (n < {report['min_group_size']}): "
                + ", ".join(s["groups_below_min_size"])
            )
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict, out_dir: Path = FAIRNESS_REPORTS_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['batch_name']}_fairness"
    paths = {"json": out_dir / f"{stem}.json", "md": out_dir / f"{stem}.md"}
    paths["json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audita o vies das decisoes do modelo num lote rotulado."
    )
    parser.add_argument("--batch", type=Path, default=PRODUCTION_BASE_FILE)
    parser.add_argument("--out-dir", type=Path, default=FAIRNESS_REPORTS_DIR)
    args = parser.parse_args(argv)

    if not args.batch.exists():
        print(
            f"Lote nao encontrado: {args.batch}. Rode `make data` antes.",
            file=sys.stderr,
        )
        return 1
    try:
        report = audit(
            read_batch(args.batch), load_bundle(), batch_name=args.batch.stem
        )
    except (DataContractViolation, ValueError) as exc:
        print(f"Auditoria nao executada: {exc}", file=sys.stderr)
        return 1

    paths = write_report(report, args.out_dir)
    log_event(
        logger,
        "fairness_audit",
        batch_name=report["batch_name"],
        model_version=report["model_version"],
        n_rows=report["n_rows"],
        disparate_impact={
            a: d["summary"]["disparate_impact_ratio"]
            for a, d in report["attributes"].items()
        },
        report=project_relative(paths["json"]),
    )
    print(render_markdown(report))
    print(
        f"Relatorios: {project_relative(paths['json'])}, {project_relative(paths['md'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
