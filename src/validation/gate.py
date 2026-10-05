"""Gate de validacao: executa o contrato ANTES de qualquer ingestao.

Politica fail-closed: se qualquer regra do contrato for violada, o lote
inteiro e bloqueado -- nada segue para treino ou inferencia --, o arquivo vai
para `data/quarantine/`, um relatorio JSON + HTML e gerado em
`reports/data_contract/` e um evento estruturado e registrado em
`logs/pipeline.jsonl`.

Uso tipico (treino e inferencia chamam isto antes de tocar no modelo):

    df_ok = enforce_contract(df, batch_name="lote_2026_10", stage="inference", labeled=False)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.errors as pa_errors

from src.config import CONTRACT_REPORTS_DIR, QUARANTINE_DIR, project_relative
from src.logging_utils import get_logger, log_event
from src.validation.contract import CONTRACT_VERSION, get_schema

APPROVED = "APPROVED"
BLOCKED = "BLOCKED"

# Descricao em PT-BR das checagens customizadas (as nativas sao inferidas
# pelo nome em `describe_check`).
_CUSTOM_CHECKS = {
    "annuity_not_greater_than_credit": "R8 - parcela maior que o valor total do credito",
    "employment_missing_only_if_not_employed": "R9 - tempo de emprego ausente para cliente empregado",
}

_DATAFRAME_LEVEL = "<dataframe>"


def describe_check(column: str, check: str) -> str:
    """Traduz o nome de uma checagem Pandera para a regra de negocio."""
    if check in _CUSTOM_CHECKS:
        return _CUSTOM_CHECKS[check]
    if check == "not_nullable":
        return "R2 - valor nulo em campo obrigatorio"
    if check in {"field_uniqueness", "unique"}:
        return "R1 - valor duplicado (deve ser unico)"
    if check == "column_in_schema":
        return "R10 - coluna nao prevista no contrato"
    if check == "column_in_dataframe":
        return "R10 - coluna obrigatoria ausente"
    if check.startswith(("coerce_dtype", "dtype")):
        return "R11 - tipo de dado invalido"
    if check.startswith("isin"):
        return "R6 - categoria fora do dominio permitido"
    if check.startswith(("greater_than", "less_than", "in_range")):
        rule = {
            "age_years": "R3",
            "income_total": "R4",
            "credit_amount": "R5",
            "annuity_amount": "R5",
            "ext_source_2": "R7",
            "ext_source_3": "R7",
        }.get(column, "Rx")
        return f"{rule} - valor fora do intervalo permitido"
    return check


@dataclass
class ValidationResult:
    batch_name: str
    stage: str
    contract: str
    status: str
    n_rows: int
    n_columns: int
    n_failure_cases: int
    rules_violated: list[dict[str, Any]]
    duration_ms: float
    validated_at: str
    report_files: dict[str, str] = field(default_factory=dict)
    quarantine_file: str | None = None

    @property
    def approved(self) -> bool:
        return self.status == APPROVED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataContractViolation(Exception):
    """Lancada quando um lote viola o contrato -- interrompe o pipeline."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        super().__init__(
            f"Lote '{result.batch_name}' BLOQUEADO pelo contrato {result.contract}: "
            f"{len(result.rules_violated)} regra(s) violada(s), "
            f"{result.n_failure_cases} ocorrencia(s)."
        )


def summarize_failures(
    failure_cases: pd.DataFrame, max_examples: int = 5
) -> list[dict]:
    """Agrupa as falhas do Pandera por (coluna, checagem).

    Checagens de nivel de linha (R8, R9) sao reportadas pelo Pandera uma vez
    POR COLUNA da linha que falhou; aqui elas sao colapsadas em uma entrada
    por linha, com o indice da linha como exemplo.
    """
    if failure_cases.empty:
        return []
    fc = failure_cases.copy()
    fc["column"] = fc["column"].fillna(_DATAFRAME_LEVEL).astype(str)
    fc["check"] = fc["check"].astype(str)
    summary = []

    row_level = fc["check"].isin(list(_CUSTOM_CHECKS))
    for check, group in fc[row_level].groupby("check", sort=False):
        rows = group["index"].dropna().unique()
        summary.append(
            {
                "rule": describe_check(_DATAFRAME_LEVEL, check),
                "column": _DATAFRAME_LEVEL,
                "check": check,
                "n_failures": int(len(rows)),
                "examples": [f"linha {int(i)}" for i in rows[:max_examples]],
            }
        )

    for (column, check), group in fc[~row_level].groupby(
        ["column", "check"], sort=False
    ):
        examples = group["failure_case"].dropna().astype(str).unique()[:max_examples]
        summary.append(
            {
                "rule": describe_check(column, check),
                "column": column,
                "check": check,
                "n_failures": int(len(group)),
                "examples": examples.tolist(),
            }
        )
    return sorted(summary, key=lambda item: item["rule"])


def run_contract(
    df: pd.DataFrame, *, labeled: bool, batch_name: str, stage: str
) -> tuple[pd.DataFrame | None, ValidationResult, pd.DataFrame]:
    """Executa o contrato sem efeitos colaterais.

    Retorna (df validado ou None, resultado, failure_cases do Pandera).
    `lazy=True` coleta TODAS as violacoes de uma vez em vez de parar na
    primeira (Aula 5).
    """
    schema = get_schema(labeled)
    started = time.perf_counter()
    try:
        validated = schema.validate(df, lazy=True)
        failure_cases = pd.DataFrame(
            columns=["schema_context", "column", "check", "failure_case", "index"]
        )
    except pa_errors.SchemaErrors as err:
        validated = None
        failure_cases = err.failure_cases
    duration_ms = (time.perf_counter() - started) * 1000
    rules_violated = summarize_failures(failure_cases)

    result = ValidationResult(
        batch_name=batch_name,
        stage=stage,
        contract=schema.Config.name,
        status=APPROVED if validated is not None else BLOCKED,
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        n_failure_cases=sum(r["n_failures"] for r in rules_violated),
        rules_violated=rules_violated,
        duration_ms=round(duration_ms, 2),
        validated_at=datetime.now(tz=UTC).isoformat(),
    )
    return validated, result, failure_cases


def enforce_contract(
    df: pd.DataFrame,
    *,
    batch_name: str,
    stage: str,
    labeled: bool,
    report_dir: Path | None = CONTRACT_REPORTS_DIR,
    quarantine_dir: Path | None = QUARANTINE_DIR,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Valida o lote e BLOQUEIA (exception) se o contrato for violado.

    Retorna o DataFrame validado (com tipos coeridos) quando aprovado.
    `report_dir=None` / `quarantine_dir=None` desligam os artefatos em disco.
    """
    logger = logger or get_logger("data_contract")
    validated, result, failure_cases = run_contract(
        df, labeled=labeled, batch_name=batch_name, stage=stage
    )

    if not result.approved and quarantine_dir is not None:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        quarantine_file = quarantine_dir / f"{batch_name}_{stamp}.csv"
        df.to_csv(quarantine_file, index=False)
        result.quarantine_file = project_relative(quarantine_file)

    if report_dir is not None:
        # import tardio: evita ciclo report -> gate -> report
        from src.validation.report import write_reports

        result.report_files = write_reports(result, failure_cases, report_dir)

    log_event(
        logger,
        "data_contract_validation",
        level=logging.INFO if result.approved else logging.ERROR,
        message=(
            f"[{result.status}] lote={batch_name} etapa={stage} "
            f"linhas={result.n_rows} regras_violadas={len(result.rules_violated)} "
            f"ocorrencias={result.n_failure_cases} ({result.duration_ms} ms)"
        ),
        batch_name=batch_name,
        stage=stage,
        status=result.status,
        contract=result.contract,
        contract_version=CONTRACT_VERSION,
        n_rows=result.n_rows,
        n_failure_cases=result.n_failure_cases,
        rules_violated=[r["rule"] for r in result.rules_violated],
        duration_ms=result.duration_ms,
        quarantine_file=result.quarantine_file,
    )

    if not result.approved:
        raise DataContractViolation(result)
    return validated
