"""Construcao do Dataset de Referencia e do lote de Producao t0 (checklist 1.3).

Fluxo:
  1. Le do CSV bruto apenas as 21 colunas usadas (minimizacao -- LGPD art. 6, III).
  2. Converte para o esquema canonico do contrato (nomes, idade em anos,
     sentinela de DAYS_EMPLOYED -> NaN).
  3. Aplica o contrato linha a linha: registros brutos que violam qualquer
     regra sao descartados (com contagem por regra no relatorio). O Dataset
     de Referencia e, por construcao, 100% aderente ao contrato ("limpo").
  4. Separa, de forma estratificada e com seed fixa:
       - data/reference/reference.parquet       (70%) -> treino do baseline
       - data/production/production_base.parquet (30%) -> nunca visto no
         treino; ponto de partida da simulacao de drift na Etapa 2.
  5. Grava a linhagem (hash do bruto, contagens, perfil estatistico) em
     reports/data_contract/reference_build_report.json.

    uv run python -m src.data.prepare
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pandera.errors as pa_errors
from sklearn.model_selection import train_test_split

from src.config import (
    CONTRACT_REPORTS_DIR,
    DAYS_EMPLOYED_SENTINEL,
    NUMERIC_FEATURES,
    PRODUCTION_BASE_FILE,
    PRODUCTION_FRACTION,
    RAW_COLUMN_MAP,
    RAW_FILE,
    REFERENCE_FILE,
    SEED,
    TARGET_COLUMN,
    project_relative,
    set_global_seed,
)
from src.logging_utils import get_logger, log_event
from src.validation.contract import CONTRACT_VERSION, contract_columns, get_schema
from src.validation.gate import run_contract

logger = get_logger("data.prepare")

DAYS_PER_YEAR = 365.25


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"Arquivo bruto nao encontrado: {path}\n"
            "Rode `uv run python -m src.data.download` (API do Kaggle) ou baixe "
            "application_train.csv manualmente para data/raw/ (ver README)."
        )
    return pd.read_csv(path, usecols=list(RAW_COLUMN_MAP))


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas brutas do Home Credit para o esquema do contrato."""
    df = raw.rename(columns=RAW_COLUMN_MAP)
    df["age_years"] = (-df.pop("days_birth") / DAYS_PER_YEAR).round(2)

    days_employed = df.pop("days_employed")
    not_employed = days_employed == DAYS_EMPLOYED_SENTINEL
    df["years_employed"] = (-days_employed.mask(not_employed) / DAYS_PER_YEAR).round(2)
    return df[contract_columns(labeled=True)]


def clean_with_contract(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove as linhas que violam o contrato e devolve (limpo, resumo).

    Usa o proprio contrato como fonte unica das regras de limpeza, assim o
    Dataset de Referencia nunca diverge do que o gate exige em producao.
    """
    _, result, failure_cases = run_contract(
        df, labeled=True, batch_name="raw_application_train", stage="reference_build"
    )
    rejected = pd.Index(failure_cases["index"].dropna().unique())
    try:
        clean = get_schema(labeled=True).validate(df.drop(index=rejected), lazy=True)
    except pa_errors.SchemaErrors as err:
        # O que sobrevive ao descarte das linhas e falha de schema (coluna
        # ausente/extra, tipo): o mapeamento de colunas esta errado.
        raise RuntimeError(
            f"Falha estrutural no contrato ao montar a referencia:\n{err.failure_cases}"
        ) from err
    summary = {
        "n_rows_raw": int(len(df)),
        "n_rows_rejected": int(len(rejected)),
        "n_rows_clean": int(len(clean)),
        "rejections_by_rule": result.rules_violated,
    }
    return clean.reset_index(drop=True), summary


def profile(df: pd.DataFrame) -> dict:
    """Perfil estatistico enxuto -- referencia numerica para a Etapa 2."""
    stats = {}
    for col in NUMERIC_FEATURES:
        s = df[col]
        stats[col] = {
            "null_rate": round(float(s.isna().mean()), 4),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "p05": round(float(s.quantile(0.05)), 4),
            "p50": round(float(s.median()), 4),
            "p95": round(float(s.quantile(0.95)), 4),
        }
    return stats


def split_reference_production(
    clean: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference, production = train_test_split(
        clean,
        test_size=PRODUCTION_FRACTION,
        stratify=clean[TARGET_COLUMN],
        random_state=seed,
    )
    return reference.reset_index(drop=True), production.reset_index(drop=True)


def main() -> None:
    set_global_seed()
    raw = load_raw()
    log_event(logger, "raw_loaded", n_rows=len(raw), n_columns=raw.shape[1])

    canonical = to_canonical(raw)
    clean, cleaning = clean_with_contract(canonical)
    log_event(
        logger,
        "reference_cleaning",
        n_rows_raw=cleaning["n_rows_raw"],
        n_rows_rejected=cleaning["n_rows_rejected"],
        n_rows_clean=cleaning["n_rows_clean"],
    )

    reference, production = split_reference_production(clean)
    for frame, path in [
        (reference, REFERENCE_FILE),
        (production, PRODUCTION_BASE_FILE),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "seed": SEED,
        "contract_version": CONTRACT_VERSION,
        "source": {
            "dataset": "Home Credit Default Risk (Kaggle) - application_train.csv",
            "file": project_relative(RAW_FILE),
            "sha256": sha256_of(RAW_FILE),
            "n_columns_raw_used": len(RAW_COLUMN_MAP),
        },
        "cleaning": cleaning,
        "splits": {
            "reference": {
                "file": project_relative(REFERENCE_FILE),
                "n_rows": int(len(reference)),
                "default_rate": round(float(reference[TARGET_COLUMN].mean()), 4),
            },
            "production_base": {
                "file": project_relative(PRODUCTION_BASE_FILE),
                "n_rows": int(len(production)),
                "default_rate": round(float(production[TARGET_COLUMN].mean()), 4),
            },
        },
        "reference_profile": profile(reference),
    }
    CONTRACT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = CONTRACT_REPORTS_DIR / "reference_build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_event(
        logger,
        "reference_built",
        reference_rows=len(reference),
        production_rows=len(production),
        report=project_relative(report_path),
    )


if __name__ == "__main__":
    main()
