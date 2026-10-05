"""Demonstracao do contrato bloqueando a ingestao (checklist 1.4).

Simula dois arquivos chegando de um sistema upstream:
  - valid_batch.csv   -> 500 solicitacoes limpas do lote de producao t0
  - invalid_batch.csv -> 500 solicitacoes + defeitos injetados (invalid_batch.py)

Ambos passam pelo mesmo caminho de producao (`score_batch`): gate do
contrato -> modelo. Esperado: o valido e APROVADO e pontuado; o invalido e
BLOQUEADO, vai para quarentena e nunca chega ao modelo. O script tambem
confere se CADA defeito injetado foi detectado pela regra esperada e grava
o resumo em reports/data_contract/contract_demo_summary.md.

    uv run python -m src.validation.demo
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import pandas as pd

from src.config import (
    CONTRACT_REPORTS_DIR,
    INVALID_BATCH_FILE,
    MODEL_FILE,
    PRODUCTION_BASE_FILE,
    SEED,
    TARGET_COLUMN,
    VALID_BATCH_FILE,
    project_relative,
    set_global_seed,
)
from src.logging_utils import get_logger, log_event
from src.models.predict import load_bundle, score_batch
from src.validation.gate import DataContractViolation, ValidationResult
from src.validation.invalid_batch import make_invalid_batch

logger = get_logger("data_contract.demo")

BATCH_SIZE = 500


def build_batches() -> list[dict]:
    if not PRODUCTION_BASE_FILE.exists():
        sys.exit("Lote de producao t0 nao encontrado. Rode `make data` antes.")
    production = pd.read_parquet(PRODUCTION_BASE_FILE)
    # Solicitacoes novas ainda nao tem rotulo: o gate usa o contrato de inferencia.
    incoming = production.drop(columns=[TARGET_COLUMN]).sample(
        n=2 * BATCH_SIZE, random_state=SEED
    )
    valid = incoming.iloc[:BATCH_SIZE]
    invalid, manifest = make_invalid_batch(incoming.iloc[BATCH_SIZE:], seed=SEED)

    VALID_BATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(VALID_BATCH_FILE, index=False)
    invalid.to_csv(INVALID_BATCH_FILE, index=False)
    return manifest


def was_detected(defect: dict, result: ValidationResult) -> bool:
    for rule in result.rules_violated:
        if not rule["rule"].startswith(f"{defect['expected_rule']} -"):
            continue
        if (
            defect["column"] in ("<dataframe>", rule["column"])
            or defect["column"] in rule["examples"]
        ):
            return True
    return False


def run_batch(path, bundle) -> tuple[str, ValidationResult | None, int]:
    """Executa o caminho de producao e devolve (status, resultado, n pontuados)."""
    df = pd.read_csv(path)
    try:
        scored = score_batch(df, bundle, batch_name=path.stem)
    except DataContractViolation as exc:
        return "BLOQUEADO", exc.result, 0
    return "APROVADO", None, len(scored)


def main() -> int:
    set_global_seed()
    manifest = build_batches()
    if not MODEL_FILE.exists():
        sys.exit("Modelo baseline nao encontrado. Rode `make train` antes.")
    bundle = load_bundle()

    valid_status, _, n_scored = run_batch(VALID_BATCH_FILE, bundle)
    invalid_status, invalid_result, n_scored_invalid = run_batch(
        INVALID_BATCH_FILE, bundle
    )

    checks = [
        {**d, "detected": bool(invalid_result) and was_detected(d, invalid_result)}
        for d in manifest
    ]
    success = (
        valid_status == "APROVADO"
        and invalid_status == "BLOQUEADO"
        and n_scored_invalid == 0
        and all(c["detected"] for c in checks)
    )

    defect_rows = "\n".join(
        f"| {c['expected_rule']} | {c['description']} | `{c['column']}` | {c['n_rows']} | "
        f"{'sim' if c['detected'] else '**NAO**'} |"
        for c in checks
    )
    rules_rows = "\n".join(
        f"| {r['rule']} | `{r['column']}` | `{r['check']}` | {r['n_failures']} |"
        for r in (invalid_result.rules_violated if invalid_result else [])
    )
    summary = f"""# Demonstracao do Data Contract -- bloqueio de lote invalido

Gerado por `src/validation/demo.py` em {datetime.now(tz=UTC).isoformat()} (seed {SEED}).

| Lote | Arquivo | Resultado do gate | Registros pontuados pelo modelo |
|---|---|---|---|
| valido | `{project_relative(VALID_BATCH_FILE)}` | **{valid_status}** | {n_scored} |
| invalido | `{project_relative(INVALID_BATCH_FILE)}` | **{invalid_status}** | {n_scored_invalid} |

Resultado geral da demonstracao: **{"SUCESSO" if success else "FALHA"}**.

## Defeitos injetados x deteccao

| Regra esperada | Defeito | Coluna | Linhas afetadas | Detectado? |
|---|---|---|---|---|
{defect_rows}

## Regras violadas reportadas pelo contrato (Pandera, `lazy=True`)

| Regra | Coluna | Checagem | Ocorrencias |
|---|---|---|---|
{rules_rows}

Relatorios completos: [`invalid_batch_report.html`](invalid_batch_report.html) e
[`valid_batch_report.html`](valid_batch_report.html).
"""
    CONTRACT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (CONTRACT_REPORTS_DIR / "contract_demo_summary.md").write_text(
        summary, encoding="utf-8"
    )
    (CONTRACT_REPORTS_DIR / "invalid_batch_manifest.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_event(
        logger,
        "contract_demo",
        status="SUCCESS" if success else "FAILED",
        valid_batch=valid_status,
        invalid_batch=invalid_status,
        defects_detected=sum(c["detected"] for c in checks),
        defects_injected=len(checks),
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
