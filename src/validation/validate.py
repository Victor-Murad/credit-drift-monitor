"""CLI do gate de validacao.

    uv run python -m src.validation.validate --input data/samples/invalid_batch.csv
    uv run python -m src.validation.validate --input data/reference/reference.parquet --labeled

Codigo de saida: 0 = lote aprovado, 1 = lote bloqueado pelo contrato. Assim o
mesmo comando serve como etapa bloqueante em Makefile, CI ou num DAG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.validation.gate import DataContractViolation, enforce_contract


def read_batch(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato nao suportado: {path.suffix} (use .csv ou .parquet)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="lote .csv/.parquet")
    parser.add_argument(
        "--labeled",
        action="store_true",
        help="lote rotulado (exige a coluna target) -- treino/feedback",
    )
    parser.add_argument("--batch-name", help="nome do lote (padrao: nome do arquivo)")
    parser.add_argument("--stage", default="ingestion", help="etapa do pipeline")
    args = parser.parse_args(argv)

    df = read_batch(args.input)
    try:
        enforce_contract(
            df,
            batch_name=args.batch_name or args.input.stem,
            stage=args.stage,
            labeled=args.labeled,
        )
    except DataContractViolation as exc:
        print(f"\n{exc}", file=sys.stderr)
        for rule in exc.result.rules_violated:
            print(
                f"  - {rule['rule']:<55} coluna={rule['column']:<22} "
                f"ocorrencias={rule['n_failures']}",
                file=sys.stderr,
            )
        print(f"  Relatorio: {exc.result.report_files.get('html')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
