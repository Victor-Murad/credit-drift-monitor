"""Demonstracao da Etapa 2: simular drift, pontuar e detectar.

Pipeline:
  1. Simula t1 (data drift) e t2 (data + concept) a partir de production_base.
  2. Garante que t0/t1/t2 passam no gate e sao pontuados pelo baseline.
  3. Compara cada fatia com a Referencia via Evidently.
  4. Grava reports/drift/drift_demo_summary.md com impacto de negocio.

    uv run python -m src.drift.demo
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import (
    DRIFT_REPORTS_DIR,
    MODEL_FILE,
    PRODUCTION_BASE_FILE,
    PRODUCTION_T1_FILE,
    PRODUCTION_T2_FILE,
    REFERENCE_FILE,
    SEED,
    TARGET_COLUMN,
    project_relative,
    set_global_seed,
)
from src.drift.detect import run_drift_detection
from src.drift.simulate import simulate_production_slices
from src.logging_utils import get_logger, log_event
from src.models.predict import load_bundle, score_batch
from src.models.train import evaluate
from src.validation.gate import DataContractViolation

logger = get_logger("drift.demo")

# Colunas monitoradas na demo (variaveis da Etapa 2 + target para concept).
DEMO_DRIFT_COLUMNS = [
    "annuity_amount",
    "ext_source_3",
    "ext_source_2",
    "income_total",
    "credit_amount",
    "years_employed",
    TARGET_COLUMN,
]

# Evidently em 200k+ linhas fica lento para a demo; amostramos a referencia
# e a corrente com a mesma seed (reprodutivel). As metricas de modelo usam
# o lote completo.
EVIDENTLY_SAMPLE_SIZE = 15_000


def _sample_for_evidently(
    reference: pd.DataFrame, current: pd.DataFrame, n: int = EVIDENTLY_SAMPLE_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = (
        reference.sample(n=min(n, len(reference)), random_state=SEED)
        if len(reference) > n
        else reference
    )
    cur = (
        current.sample(n=min(n, len(current)), random_state=SEED)
        if len(current) > n
        else current
    )
    return ref, cur


def _business_metrics(scored: pd.DataFrame, bundle: dict) -> dict:
    y_true = scored[TARGET_COLUMN].to_numpy()
    y_score = scored["default_probability"].to_numpy()
    return evaluate(y_true, y_score, bundle["threshold"])


def _score_slice(
    path: Path, bundle: dict, *, batch_name: str, report_dir: Path
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    scored = score_batch(
        df,
        bundle,
        batch_name=batch_name,
        stage="drift",
        report_dir=report_dir / "contract_gate",
    )
    metrics = _business_metrics(scored, bundle)
    return scored, metrics


def run_demo(
    *,
    report_dir: Path = DRIFT_REPORTS_DIR,
    evidently_sample: int = EVIDENTLY_SAMPLE_SIZE,
) -> dict:
    set_global_seed(SEED)

    for required in (PRODUCTION_BASE_FILE, REFERENCE_FILE, MODEL_FILE):
        if not required.exists():
            sys.exit(
                f"Artefato ausente: {required}. Rode `make etapa1` (data + train) antes."
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    gate_report_dir = report_dir / "contract_gate"
    gate_report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Simular fatias
    paths = simulate_production_slices()
    bundle = load_bundle()
    reference = pd.read_parquet(REFERENCE_FILE)

    slices = [
        ("production_t0", PRODUCTION_BASE_FILE),
        ("production_t1_data_drift", paths["t1"]),
        ("production_t2_concept_drift", paths["t2"]),
    ]

    score_results: dict[str, dict] = {}
    drift_results: dict[str, dict] = {}

    for batch_name, path in slices:
        try:
            scored, metrics = _score_slice(
                path, bundle, batch_name=batch_name, report_dir=report_dir
            )
        except DataContractViolation as exc:
            sys.exit(
                f"Lote {batch_name} violou o contrato apos o drift — "
                f"revise o clip em simulate.py. Detalhe: {exc}"
            )

        score_results[batch_name] = {
            "n_rows": len(scored),
            "contract": "APROVADO",
            "metrics": metrics,
            "path": project_relative(path),
        }

        # t0 serve de baseline de performance; ainda assim reportamos drift
        # vs referencia (deve ser estavel ou leve, por ser split da mesma origem).
        current = pd.read_parquet(path)
        ref_s, cur_s = _sample_for_evidently(reference, current, n=evidently_sample)
        drift = run_drift_detection(
            ref_s,
            cur_s,
            batch_name=batch_name,
            columns=DEMO_DRIFT_COLUMNS,
            report_dir=report_dir,
            drift_share=0.3,
        )
        drift_results[batch_name] = drift.to_dict()

    summary = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "seed": SEED,
        "evidently_sample_size": evidently_sample,
        "model_version": bundle["model_version"],
        "slices": score_results,
        "drift": drift_results,
    }

    summary_json = report_dir / "drift_demo_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    summary_md = report_dir / "drift_demo_summary.md"
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")

    log_event(
        logger,
        "drift_demo",
        status="SUCCESS",
        summary_md=project_relative(summary_md),
        t1_dataset_drift=drift_results["production_t1_data_drift"]["dataset_drift"],
        t2_concept_drift=drift_results["production_t2_concept_drift"]["concept_drift"],
        t0_auc=score_results["production_t0"]["metrics"]["roc_auc"],
        t2_auc=score_results["production_t2_concept_drift"]["metrics"]["roc_auc"],
    )
    return summary


def _render_markdown(summary: dict) -> str:
    slices = summary["slices"]
    drift = summary["drift"]
    t0 = slices["production_t0"]["metrics"]
    t1 = slices["production_t1_data_drift"]["metrics"]
    t2 = slices["production_t2_concept_drift"]["metrics"]

    def drift_row(name: str) -> str:
        d = drift[name]
        cols = (
            ", ".join(c["column"] for c in d["column_results"] if c["drifted"]) or "—"
        )
        return (
            f"| `{name}` | {'SIM' if d['dataset_drift'] else 'nao'} | "
            f"{d['n_drifted_columns']}/{len(d['columns_checked'])} | "
            f"{'SIM' if d['concept_drift'] else 'nao'} | {cols} |"
        )

    lines = [
        "# Etapa 2 — Demonstracao de Drift",
        "",
        f"> Gerado em `{summary['generated_at']}` · seed={summary['seed']} · "
        f"modelo `{summary['model_version']}` · amostra Evidently="
        f"{summary['evidently_sample_size']}.",
        "",
        "## Gate do contrato",
        "",
        "Todos os lotes (t0, t1 com data drift, t2 com data+concept) **passaram** "
        "no contrato Pandera. Drift ≠ dado inválido.",
        "",
        "## Impacto no modelo (lote completo)",
        "",
        "| Fatia | ROC-AUC | Aprovação | Default na carteira aprovada | Default base |",
        "|---|---|---|---|---|",
        f"| t0 (sem drift) | {t0['roc_auc']:.4f} | {t0['approval_rate']:.2%} | "
        f"{t0['default_rate_approved']} | {t0['default_rate_base']:.2%} |",
        f"| t1 (data drift) | {t1['roc_auc']:.4f} | {t1['approval_rate']:.2%} | "
        f"{t1['default_rate_approved']} | {t1['default_rate_base']:.2%} |",
        f"| t2 (data+concept) | {t2['roc_auc']:.4f} | {t2['approval_rate']:.2%} | "
        f"{t2['default_rate_approved']} | {t2['default_rate_base']:.2%} |",
        "",
        f"Queda de AUC t0→t2: **{t0['roc_auc'] - t2['roc_auc']:.4f}**.",
        "",
        "## Detecção Evidently (vs Dataset de Referência)",
        "",
        "| Fatia | Dataset drift | Colunas driftadas | Concept drift (`target`) | Colunas |",
        "|---|---|---|---|---|",
        drift_row("production_t0"),
        drift_row("production_t1_data_drift"),
        drift_row("production_t2_concept_drift"),
        "",
        "## Artefatos",
        "",
        f"- Sumário JSON: `{project_relative(DRIFT_REPORTS_DIR / 'drift_demo_summary.json')}`",
        f"- Relatórios HTML: `{project_relative(DRIFT_REPORTS_DIR)}/*_drift_report.html`",
        f"- t1: `{project_relative(PRODUCTION_T1_FILE)}`",
        f"- t2: `{project_relative(PRODUCTION_T2_FILE)}`",
        "",
        "## Lições para o vídeo",
        "",
        "1. **Qualidade ≠ estabilidade:** o gate aprovou t1/t2; o Evidently "
        "marcou drift nas variáveis importantes (`annuity_amount`, `ext_source_3`).",
        "2. **`income_total` pode driftar sem derrubar o modelo** (importância baixa).",
        "3. **Concept drift** (t2) altera o rótulo e degrada AUC / inadimplência "
        "da carteira aprovada mesmo com o mesmo pipeline.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    summary = run_demo()
    t0_auc = summary["slices"]["production_t0"]["metrics"]["roc_auc"]
    t2_auc = summary["slices"]["production_t2_concept_drift"]["metrics"]["roc_auc"]
    print(
        f"Demo OK. AUC t0={t0_auc:.4f} -> t2={t2_auc:.4f}. "
        f"Sumario: {DRIFT_REPORTS_DIR / 'drift_demo_summary.md'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
