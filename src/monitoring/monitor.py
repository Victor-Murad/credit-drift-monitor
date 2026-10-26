"""Avalia UM lote de produção e registra o resultado no MLflow (spec da Etapa 3, §4.8).

    uv run python -m src.monitoring.monitor --batch <arquivo.parquet|.csv> \\
        [--baseline-auc 0.75] [--labeled | --no-labeled]

Fluxo: lê o lote → carrega o champion (sha256 conferido contra a tag `model_sha256`
do alias `champion`) → contrato e pontuação (`score_batch`) → drift contra a referência
(Evidently + PSI, colunas `MONITORED_COLUMNS`) → PSI do score → AUC (se rotulado) →
nível de alerta (§4.2) → run de lote filho de um run de estágio `monitor`. A
avaliação (PSI do score, queda de AUC, motivos, piso `ATENCAO`, NaN como "não
avaliado") é a mesma do `drift_demo`: `DriftBatchObserver`, de `src.monitoring.stage`.

Códigos de saída: 0 (`OK`/`ATENCAO`), 2 (`CRITICO`), 3 (`BLOQUEADO`) e 1 (erro
operacional: MLflow indisponível, sem champion, lote ausente/ilegível/vazio, lote
rotulado sem baseline de AUC, argumento inválido).

Rótulo: inferido pela coluna `target`; `--no-labeled` ignora a coluna e `--labeled`
exige que ela exista. Baseline de AUC (lote rotulado): `--baseline-auc` ou o `roc_auc`
do run de lote mais recente com `batch_name=production_t0` e o mesmo `model_version`
(gravado pelo `make demo-drift`). Sem nenhum dos dois, falha fechado. Um lote de uma
só classe não precisa de baseline: a AUC não é avaliada (`single_class`).

`MONITORING_TRACKING=off` roda sem Registry nem MLflow (sem conferência do champion
pelo Registry e sem baseline do MLflow: lote rotulado exige `--baseline-auc`).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import NoReturn

import pandas as pd

from src import config
from src.drift.demo import (
    DATASET_DRIFT_SHARE,
    MONITORED_COLUMNS,
    BatchObservation,
    sample_for_evidently,
)
from src.drift.detect import DriftParsingError, run_drift_detection
from src.models import predict
from src.models.integrity import ModelIntegrityError
from src.models.metrics import evaluate
from src.monitoring import tracking
from src.monitoring.alerts import AlertDecision, AlertLevel
from src.monitoring.stage import (
    StagePreconditionError,
    blocked_decision,
    champion_sha256,
    contract_report_files,
    model_metadata_path,
    observer_from_metadata,
    register_batch,
)
from src.monitoring.tracking import (
    TRACKING_ENV_VAR,
    BatchRecord,
    StageRun,
    TrackingUnavailableError,
    tracked_stage,
    tracking_errors,
)
from src.validation.gate import DataContractViolation
from src.validation.io import read_batch

STAGE_NAME = "monitor"
BASELINE_BATCH_NAME = "production_t0"
# Lido na chamada (os testes o redirecionam): relatórios do Evidently e do contrato.
MONITOR_REPORTS_DIR = config.REPORTS_DIR / "monitor"

EXIT_CODES: dict[AlertLevel, int] = {
    AlertLevel.OK: 0,
    AlertLevel.ATENCAO: 0,
    AlertLevel.CRITICO: 2,
    AlertLevel.BLOQUEADO: 3,
}
EXIT_OPERATIONAL_ERROR = 1


class MonitorInputError(RuntimeError):
    """Lote ou argumento que impede a avaliação (erro operacional, código 1)."""


# MARK: leitura do lote


def _read(path: Path, labeled: bool | None) -> pd.DataFrame:
    """Lê o lote e aplica `--labeled/--no-labeled`; ausente, ilegível ou vazio é erro."""
    if not path.is_file():
        raise MonitorInputError(f"Lote não encontrado: {path}.")
    try:
        df = read_batch(path)
    except (ValueError, OSError) as exc:
        # Formato não suportado, parquet/CSV corrompido ou ilegível.
        raise MonitorInputError(f"Lote ilegível em {path}: {exc}") from exc
    tem_rotulo = config.TARGET_COLUMN in df.columns
    if labeled is True and not tem_rotulo:
        raise MonitorInputError(
            f"`--labeled` pedido, mas o lote {path} não tem a coluna "
            f"`{config.TARGET_COLUMN}`."
        )
    if labeled is False and tem_rotulo:
        df = df.drop(columns=config.TARGET_COLUMN)
    if df.empty:
        raise MonitorInputError(
            f"Lote vazio: {path} não tem linhas; nada a avaliar (nenhum run de lote "
            "é criado)."
        )
    return df


def _reference_file() -> Path:
    caminho = Path(config.REFERENCE_FILE)
    if not caminho.is_file():
        raise StagePreconditionError(
            f"Dataset de referência não encontrado: {caminho}. Rode `make pipeline` "
            "(estágio `prepare`) antes de monitorar um lote."
        )
    return caminho


# MARK: baseline de AUC


def _auc_evaluable(scored: pd.DataFrame) -> bool:
    """Lote rotulado com as duas classes: a ROC-AUC é definida e exige baseline."""
    alvo = config.TARGET_COLUMN
    return alvo in scored.columns and scored[alvo].nunique(dropna=True) >= 2


def baseline_auc_from_tracking(model_version: str) -> float | None:
    """`roc_auc` do run de lote `production_t0` mais recente deste `model_version`.

    A busca filtra no MLflow só pelas tags que todo run de lote tem (`run_type`,
    `batch_name`); o `model_version` é conferido aqui. Sem run desse modelo, ou se o
    mais recente não tem `roc_auc` finito (ex.: t0 de uma classe só), `None`.
    """
    with tracking_errors("Falha ao consultar o baseline de AUC no MLflow"):
        cliente = tracking.client()
        experimento = cliente.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
        if experimento is None:
            return None
        runs = cliente.search_runs(
            [experimento.experiment_id],
            filter_string=(
                f"tags.run_type = 'batch' and tags.batch_name = '{BASELINE_BATCH_NAME}'"
            ),
            order_by=["attributes.start_time DESC"],
        )
    do_modelo = (r for r in runs if r.data.tags.get("model_version") == model_version)
    recente = next(do_modelo, None)
    if recente is None:
        return None
    auc = recente.data.metrics.get("roc_auc")
    return float(auc) if auc is not None and math.isfinite(auc) else None


def _no_baseline_error(model_version: str, tracking_on: bool) -> MonitorInputError:
    if tracking_on:
        onde = (
            f"nenhum run de lote `{BASELINE_BATCH_NAME}` do modelo `{model_version}` "
            "com `roc_auc` no MLflow"
        )
    else:
        onde = (
            f"rastreamento desligado ({TRACKING_ENV_VAR}=off), sem consulta ao MLflow"
        )
    return MonitorInputError(
        f"Lote rotulado sem baseline de AUC: {onde} e nenhum `--baseline-auc`. Rode "
        "`make demo-drift` (com o rastreamento ligado) para registrar o baseline do "
        "t0, ou passe `--baseline-auc`. A queda de AUC não é avaliada sem baseline."
    )


# MARK: avaliação


def _business_metrics(scored: pd.DataFrame, bundle: dict) -> dict:
    """`evaluate` se a AUC for definida; senão só a taxa de aprovação (sem rótulo)."""
    if _auc_evaluable(scored):
        return evaluate(
            scored[config.TARGET_COLUMN].to_numpy(),
            scored["default_probability"].to_numpy(),
            bundle["threshold"],
        )
    return {"approval_rate": round(float((scored["predicted_default"] == 0).mean()), 4)}


def _blocked_record(
    exc: DataContractViolation, *, batch_path: Path, model_version: str
) -> BatchRecord:
    """Lote barrado pelo contrato: nenhuma pontuação, todas as linhas rejeitadas."""
    resultado = exc.result
    return BatchRecord(
        batch_name=resultado.batch_name,
        decision=blocked_decision(resultado),
        n_rows=resultado.n_rows,
        rows_rejected=resultado.n_rows,
        metrics={},
        psi_by_column={},
        n_drifted_columns=0,
        model_version=model_version,
        artifacts=contract_report_files(resultado.batch_name, resultado),
        batch_path=batch_path,
    )


def run_monitor(
    run: StageRun,
    batch_path: Path,
    *,
    baseline_auc: float | None = None,
    labeled: bool | None = None,
) -> AlertDecision:
    """Avalia o lote e registra o run de lote no `run` de estágio; devolve a decisão."""
    batch_path = Path(batch_path)
    df = _read(batch_path, labeled)
    batch_name = batch_path.stem
    reference_path = _reference_file()

    esperado = champion_sha256() if run.enabled else None
    modelo = predict.default_model_path()
    bundle = predict.load_bundle(modelo, expected_sha256=esperado)
    model_version = str(bundle["model_version"])
    if baseline_auc is None and run.enabled and config.TARGET_COLUMN in df.columns:
        baseline_auc = baseline_auc_from_tracking(model_version)
    # Metadata conferido antes de pontuar (ausente/incompatível é erro de domínio).
    observador = observer_from_metadata(
        run, model_metadata_path(modelo), baseline_auc=baseline_auc
    )

    report_dir = Path(MONITOR_REPORTS_DIR)
    try:
        scored = predict.score_batch(
            df,
            bundle,
            batch_name=batch_name,
            stage=STAGE_NAME,
            report_dir=report_dir / "contract_gate",
            quarantine_dir=config.QUARANTINE_DIR,
        )
    except DataContractViolation as exc:
        record = _blocked_record(
            exc, batch_path=batch_path, model_version=model_version
        )
        return register_batch(run, record).decision

    if baseline_auc is None and _auc_evaluable(scored):
        raise _no_baseline_error(model_version, run.enabled)

    referencia, corrente = sample_for_evidently(pd.read_parquet(reference_path), df)
    drift = run_drift_detection(
        referencia,
        corrente,
        batch_name=batch_name,
        columns=MONITORED_COLUMNS,
        report_dir=report_dir,
        drift_share=DATASET_DRIFT_SHARE,
        min_psi=config.PSI_MODERATE,
        model_version=model_version,
    )
    observacao = BatchObservation(
        batch_name=batch_name,
        scored=scored,
        metrics=_business_metrics(scored, bundle),
        drift=drift,
        report_dir=report_dir,
        model_version=model_version,
        batch_path=batch_path,
    )
    return observador(observacao).decision


# MARK: CLI


class _Parser(argparse.ArgumentParser):
    """Erro de uso sai com 1 (operacional): o 2 do argparse é o código de `CRITICO`."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_OPERATIONAL_ERROR, f"{self.prog}: erro: {message}\n")


def _auc_arg(texto: str) -> float:
    try:
        valor = float(texto)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{texto!r} não é um número") from exc
    if not math.isfinite(valor) or not 0 < valor <= 1:
        raise argparse.ArgumentTypeError(
            f"{texto!r} fora de (0, 1]: o baseline é uma ROC-AUC"
        )
    return valor


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m src.monitoring.monitor",
        description=(
            "Avalia um lote de produção (contrato, pontuação, drift, PSI do score e "
            "AUC se rotulado) e registra o nível de alerta no MLflow. Saída: 0 OK/"
            "ATENÇÃO, 2 CRÍTICO, 3 BLOQUEADO, 1 erro operacional."
        ),
    )
    parser.add_argument(
        "--batch", required=True, type=Path, help="lote .parquet ou .csv"
    )
    parser.add_argument(
        "--baseline-auc",
        type=_auc_arg,
        default=None,
        help=(
            "ROC-AUC de referência para a queda de AUC (padrão: a do run "
            f"`{BASELINE_BATCH_NAME}` do mesmo modelo no MLflow)"
        ),
    )
    parser.add_argument(
        "--labeled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"força (ou ignora) o rótulo; padrão: inferido pela coluna `{config.TARGET_COLUMN}`",
    )
    return parser


def _summary(batch_name: str, decision: AlertDecision) -> str:
    motivos = ", ".join(decision.reasons) or "nenhum"
    return (
        f"Lote {batch_name}: {decision.level.label} (motivos: {motivos}). "
        f"Ação: {decision.action}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with tracked_stage(STAGE_NAME) as run:
            decision = run_monitor(
                run, args.batch, baseline_auc=args.baseline_auc, labeled=args.labeled
            )
    except (
        TrackingUnavailableError,
        ModelIntegrityError,
        StagePreconditionError,
        MonitorInputError,
        DriftParsingError,
    ) as erro:
        # A mensagem já vem redigida (sem credenciais da URI); nada de traceback.
        print(f"ERRO no monitor: {erro}", file=sys.stderr)
        return EXIT_OPERATIONAL_ERROR
    print(_summary(args.batch.stem, decision))
    return EXIT_CODES[decision.level]


if __name__ == "__main__":
    sys.exit(main())
