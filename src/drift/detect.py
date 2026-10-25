"""Deteccao de Data Drift e Concept Drift com Evidently (ADR-0004).

Compara o Dataset de Referencia com um lote de producao (t0 / t1 / t2).
O preset `DataDriftPreset` gera `ValueDrift` por coluna (KS / chi-quadrado);
o PSI (src/drift/psi.py) acompanha cada coluna como medida de tamanho do
efeito. A coluna `target` entra no mesmo relatorio, mas o teste sobre ela
detecta apenas mudanca de PREVALENCIA do rotulo (label shift); a degradacao
de performance (queda de ROC-AUC) esta em src/drift/performance.py.

    uv run python -m src.drift.detect \\
        --reference data/reference/reference.parquet \\
        --current data/production/production_t1_data_drift.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import (
    CATEGORICAL_FEATURES,
    DRIFT_REPORTS_DIR,
    NUMERIC_FEATURES,
    PRODUCTION_T1_FILE,
    REFERENCE_FILE,
    TARGET_COLUMN,
)
from src.drift.psi import population_stability_index, psi_level
from src.drift.report import write_drift_reports
from src.logging_utils import get_logger, log_event

logger = get_logger("drift.detect")


@dataclass
class ColumnDrift:
    column: str
    method: str
    statistic: float
    threshold: float
    drifted: bool
    psi: float | None = None
    psi_level: str | None = None


@dataclass
class DriftResult:
    batch_name: str
    n_rows_reference: int
    n_rows_current: int
    columns_checked: list[str]
    n_drifted_columns: int
    drifted_share: float
    dataset_drift: bool
    column_results: list[ColumnDrift] = field(default_factory=list)
    concept_drift: bool = False
    duration_ms: float = 0.0
    report_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_columns(df: pd.DataFrame, *, include_target: bool) -> list[str]:
    preferred = [
        "annuity_amount",
        "ext_source_3",
        "ext_source_2",
        "income_total",
        "credit_amount",
        "years_employed",
        "age_years",
        "region_rating",
    ]
    cols = [c for c in preferred if c in df.columns]
    # Algumas categoricas para o relatorio HTML (sem inflar demais).
    for c in ("income_type", "contract_type", "education_type"):
        if c in df.columns and c not in cols:
            cols.append(c)
    if include_target and TARGET_COLUMN in df.columns and TARGET_COLUMN not in cols:
        cols.append(TARGET_COLUMN)
    # Fallback: qualquer feature numerica/categorica presente.
    if not cols:
        cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
        if include_target and TARGET_COLUMN in df.columns:
            cols.append(TARGET_COLUMN)
    return cols


def _metric_type(metric: dict[str, Any]) -> str:
    config = metric.get("config") or {}
    return str(config.get("type", ""))


def _parse_snapshot(
    snapshot_dict: dict[str, Any],
) -> tuple[int, float, list[ColumnDrift]]:
    """Extrai contagem de colunas driftadas e o detalhe por coluna do Snapshot."""
    column_results: list[ColumnDrift] = []
    n_drifted = 0
    share = 0.0

    for metric in snapshot_dict.get("metrics", []):
        mtype = _metric_type(metric)
        value = metric.get("value")
        config = metric.get("config") or {}

        if mtype.endswith("DriftedColumnsCount") and isinstance(value, dict):
            n_drifted = int(value.get("count", 0))
            share = float(value.get("share", 0.0))
            continue

        if not mtype.endswith("ValueDrift"):
            continue

        column = str(config.get("column", ""))
        method = str(config.get("method", ""))
        threshold = float(config.get("threshold", 0.05))
        statistic = float(value) if value is not None else float("nan")
        # Metodos "* p_value": drift quando p < threshold.
        drifted = bool(statistic == statistic and statistic < threshold)
        column_results.append(
            ColumnDrift(
                column=column,
                method=method,
                statistic=round(statistic, 6) if statistic == statistic else statistic,
                threshold=threshold,
                drifted=drifted,
            )
        )

    if not n_drifted and column_results:
        n_drifted = sum(1 for c in column_results if c.drifted)
        share = n_drifted / len(column_results) if column_results else 0.0

    return n_drifted, share, column_results


def run_drift_detection(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    batch_name: str,
    columns: list[str] | None = None,
    include_target: bool = True,
    drift_share: float = 0.3,
    threshold: float = 0.05,
    min_psi: float | None = None,
    report_dir: Path | None = DRIFT_REPORTS_DIR,
) -> DriftResult:
    """Roda Evidently DataDriftPreset e grava relatorios HTML/JSON.

    Todo `ColumnDrift` recebe o PSI. Por padrao a decisao `drifted` segue so o
    p-valor; com `min_psi` (ex.: `PSI_MODERATE`) a coluna so e marcada quando
    o teste e significativo E o efeito e relevante, o que elimina alertas de
    ruido em amostras grandes (ADR-0005).
    """
    cols = columns or _default_columns(current, include_target=include_target)
    cols = [c for c in cols if c in reference.columns and c in current.columns]
    if not cols:
        raise ValueError("Nenhuma coluna em comum entre referencia e corrente.")

    started = time.perf_counter()
    # num_method/cat_method fixos em testes de hipotese (p-value): drift quando
    # p < threshold. Sem isso o Evidently 0.7 pode cair em Wasserstein normed,
    # cuja escala inverte a regra drifted = value < threshold.
    report = Report(
        [
            DataDriftPreset(
                columns=cols,
                drift_share=drift_share,
                threshold=threshold,
                num_method="ks",
                cat_method="chisquare",
                include_tests=False,
            )
        ]
    )
    snapshot = report.run(current_data=current, reference_data=reference)
    duration_ms = (time.perf_counter() - started) * 1000

    n_drifted_preset, share_preset, column_results = _parse_snapshot(snapshot.dict())
    for col in column_results:
        col.psi = round(
            population_stability_index(reference[col.column], current[col.column]), 6
        )
        col.psi_level = psi_level(col.psi)
        if min_psi is not None and col.psi < min_psi:
            col.drifted = False
    # Preferimos a contagem a partir dos ValueDrift (p < threshold), que e
    # a mesma regra usada no sumario por coluna; o DriftedColumnsCount do
    # Evidently fica como fallback.
    if column_results:
        n_drifted = sum(1 for c in column_results if c.drifted)
        share = n_drifted / len(column_results)
    else:
        n_drifted, share = n_drifted_preset, share_preset
    dataset_drift = share >= drift_share
    concept = next((c for c in column_results if c.column == TARGET_COLUMN), None)
    concept_drift = bool(concept and concept.drifted)

    result = DriftResult(
        batch_name=batch_name,
        n_rows_reference=len(reference),
        n_rows_current=len(current),
        columns_checked=cols,
        n_drifted_columns=n_drifted,
        drifted_share=round(share, 4),
        dataset_drift=dataset_drift,
        column_results=column_results,
        concept_drift=concept_drift,
        duration_ms=round(duration_ms, 2),
    )

    if report_dir is not None:
        result.report_paths = write_drift_reports(
            result, snapshot, report_dir=report_dir
        )

    log_event(
        logger,
        "drift_detected",
        batch_name=batch_name,
        dataset_drift=dataset_drift,
        concept_drift=concept_drift,
        n_drifted_columns=n_drifted,
        drifted_share=round(share, 4),
        duration_ms=result.duration_ms,
        drifted_columns=[c.column for c in column_results if c.drifted],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detecta Data/Concept Drift (Evidently) entre referencia e corrente."
    )
    parser.add_argument("--reference", type=Path, default=REFERENCE_FILE)
    parser.add_argument("--current", type=Path, default=PRODUCTION_T1_FILE)
    parser.add_argument("--batch-name", type=str, default=None)
    parser.add_argument(
        "--min-psi",
        type=float,
        default=None,
        help="so marca drift se PSI >= valor (ex.: 0.10); padrao: so p-valor",
    )
    args = parser.parse_args(argv)

    for path in (args.reference, args.current):
        if not path.exists():
            print(f"Arquivo nao encontrado: {path}", file=sys.stderr)
            return 1

    reference = pd.read_parquet(args.reference)
    current = pd.read_parquet(args.current)
    batch_name = args.batch_name or args.current.stem

    result = run_drift_detection(
        reference, current, batch_name=batch_name, min_psi=args.min_psi
    )
    status = "DATASET_DRIFT" if result.dataset_drift else "ESTAVEL"
    print(
        f"[{status}] lote={result.batch_name} "
        f"colunas_driftadas={result.n_drifted_columns}/{len(result.columns_checked)} "
        f"share={result.drifted_share:.2%} "
        f"concept_drift={result.concept_drift}"
    )
    for col in result.column_results:
        flag = "DRIFT" if col.drifted else "ok"
        print(
            f"  [{flag}] {col.column}: {col.method}={col.statistic} "
            f"(thr={col.threshold}) psi={col.psi} ({col.psi_level})"
        )
    if result.report_paths:
        print("Relatorios:", ", ".join(result.report_paths.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
