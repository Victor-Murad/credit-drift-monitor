"""Treino do modelo baseline de risco de credito (checklist 1.3).

- Entrada: Dataset de Referencia (data/reference/reference.parquet), que
  passa pelo gate do contrato ANTES do treino (contrato rotulado).
- Modelo oficial: XGBoost dentro de um Pipeline scikit-learn.
- Modelo de comparacao: Regressao Logistica (mesmas features) -- serve so
  como linha de base para justificar a escolha (docs/adrs/0003).
- Particao estratificada da referencia: 60% treino / 20% validacao
  (escolha do threshold) / 20% teste (metricas reportadas).
- Saidas: models/baseline_model.joblib, models/baseline_metadata.json e
  reports/model/baseline_metrics.md.

    uv run python -m src.models.train
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    MODEL_FILE,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_FILE,
    MODEL_REPORTS_DIR,
    NUMERIC_FEATURES,
    RATIO_FEATURES,
    REFERENCE_FILE,
    SEED,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
    project_relative,
    set_global_seed,
)
from src.logging_utils import get_logger, log_event
from src.models.features import add_ratio_features, model_input
from src.models.integrity import file_sha256
from src.models.metrics import evaluate, ks_threshold, score_distribution
from src.models.portable import to_portable
from src.validation.contract import CONTRACT_VERSION
from src.validation.gate import enforce_contract

logger = get_logger("models.train")

MODEL_VERSION = "baseline-v1"

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "n_jobs": -1,
}


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def build_pipeline(
    model: str = "xgboost", seed: int = SEED, xgb_params: dict | None = None
) -> Pipeline:
    numeric = NUMERIC_FEATURES + RATIO_FEATURES
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if model == "xgboost":
        # XGBoost trata NaN nativamente: numericos passam direto.
        numeric_step: Any = "passthrough"
        classifier: Any = XGBClassifier(
            **{**XGB_PARAMS, **(xgb_params or {})}, random_state=seed
        )
    elif model == "logistic_regression":
        numeric_step = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        classifier = LogisticRegression(max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"Modelo desconhecido: {model}")

    preprocess = ColumnTransformer(
        [
            ("num", numeric_step, numeric),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(add_ratio_features)),
            ("preprocess", preprocess),
            ("model", classifier),
        ]
    )


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------
def raw_feature_importance(
    pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, seed: int = SEED
) -> list[dict]:
    """Permutation importance (queda de ROC-AUC) por coluna DO CONTRATO.

    Mede as variaveis brutas, nao as one-hot/razoes: e o que interessa para
    escolher quais variaveis sofrerao drift na Etapa 2.
    """
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=1
    )
    ranking = sorted(
        zip(X.columns, result.importances_mean, result.importances_std, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "feature": f,
            "auc_drop_mean": round(float(m), 5),
            "auc_drop_std": round(float(s), 5),
        }
        for f, m, s in ranking
    ]


# --------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------
def split_reference(
    reference: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        reference,
        test_size=TEST_SIZE,
        stratify=reference[TARGET_COLUMN],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        stratify=train_val[TARGET_COLUMN],
        random_state=seed,
    )
    return train, val, test


def fit_model(
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    seed: int = SEED,
    xgb_params: dict | None = None,
) -> tuple[Pipeline, float]:
    """Treina o pipeline `name` e escolhe o threshold (KS maximo) na VALIDACAO.

    Compartilhado com o retreino do challenger (`src.monitoring.retrain`): mesmo
    pipeline, mesmos hiperparametros e mesma regra de threshold. O threshold volta
    sem arredondar (as metricas de teste usam o valor exato).
    """
    pipe = build_pipeline(name, seed, xgb_params)
    pipe.fit(model_input(train), train[TARGET_COLUMN].to_numpy())
    threshold = ks_threshold(
        val[TARGET_COLUMN].to_numpy(), pipe.predict_proba(model_input(val))[:, 1]
    )
    return pipe, threshold


def make_bundle(
    pipeline: Pipeline, threshold: float, model_version: str, seed: int = SEED
) -> dict:
    """Bundle servido (`joblib`): pipeline, threshold arredondado e versoes."""
    return {
        "pipeline": pipeline,
        "threshold": round(threshold, 6),
        "model_version": model_version,
        "input_columns": MODEL_INPUT_COLUMNS,
        "contract_version": CONTRACT_VERSION,
        "seed": seed,
    }


def model_metadata(
    *,
    model_version: str,
    seed: int,
    splits: dict[str, int],
    xgb_params: dict | None,
    models: dict[str, dict],
    score_reference: dict,
    raw_feature_importance: list[dict] | None = None,
) -> dict:
    """Metadata do modelo oficial (xgboost), comum ao baseline e ao challenger.

    `score_reference` (scores do modelo oficial no teste da referencia) e a
    referencia do PSI do score em producao. A importancia por permutacao so entra
    quando calculada (o baseline a calcula; o retreino nao).
    """
    metadata: dict[str, Any] = {
        "model_version": model_version,
        "official_model": "xgboost",
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "seed": seed,
        "contract_version": CONTRACT_VERSION,
        "decision_rule": "negar credito se P(default) >= threshold (threshold = ponto de KS maximo na validacao)",
        "splits": splits,
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "derived_in_pipeline": RATIO_FEATURES,
            "excluded_audit_only": ["gender"],
        },
        "xgboost_params": {**XGB_PARAMS, **(xgb_params or {})},
        "models": models,
    }
    if raw_feature_importance is not None:
        metadata["raw_feature_importance"] = raw_feature_importance
    metadata["score_reference"] = score_reference
    return metadata


def train_models(
    reference: pd.DataFrame,
    seed: int = SEED,
    xgb_params: dict | None = None,
    importance_sample: int = 20_000,
) -> tuple[dict, dict]:
    """Treina XGBoost (oficial) + Regressao Logistica (comparacao).

    Retorna (bundle do modelo oficial, metadados). Nao grava nada em disco.
    """
    train, val, test = split_reference(reference, seed)
    y_test = test[TARGET_COLUMN].to_numpy()

    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}
    test_scores: dict[str, np.ndarray] = {}
    for name in ("xgboost", "logistic_regression"):
        started = time.perf_counter()
        pipe, threshold = fit_model(name, train, val, seed, xgb_params)
        test_score = pipe.predict_proba(model_input(test))[:, 1]
        results[name] = {
            "threshold": round(threshold, 6),
            "test_metrics": evaluate(y_test, test_score, threshold),
            "train_seconds": round(time.perf_counter() - started, 2),
        }
        pipelines[name] = pipe
        test_scores[name] = test_score
        log_event(
            logger,
            "model_trained",
            model=name,
            roc_auc=results[name]["test_metrics"]["roc_auc"],
            ks=results[name]["test_metrics"]["ks"],
            seconds=results[name]["train_seconds"],
        )

    official = pipelines["xgboost"]
    sample = test.sample(n=min(importance_sample, len(test)), random_state=seed)
    importance = raw_feature_importance(
        official, model_input(sample), sample[TARGET_COLUMN].to_numpy(), seed
    )

    bundle = make_bundle(official, results["xgboost"]["threshold"], MODEL_VERSION, seed)
    metadata = model_metadata(
        model_version=MODEL_VERSION,
        seed=seed,
        splits={
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        xgb_params=xgb_params,
        models=results,
        # Referencia do PSI do score em producao: scores do modelo oficial no teste.
        score_reference=score_distribution(test_scores["xgboost"]),
        raw_feature_importance=importance,
    )
    return bundle, metadata


def environment() -> dict:
    """Versoes do Python e dos pacotes do modelo (vai para o metadata)."""
    packages = ["scikit-learn", "xgboost", "pandas", "numpy", "pandera", "joblib"]
    return {"python": platform.python_version(), **{p: version(p) for p in packages}}


def write_metrics_markdown(metadata: dict) -> str:
    rows = []
    for name, res in metadata["models"].items():
        m = res["test_metrics"]
        rows.append(
            f"| {name} | {m['roc_auc']:.4f} | {m['gini']:.4f} | {m['ks']:.4f} | "
            f"{m['pr_auc']:.4f} | {m['recall']:.4f} | {m['approval_rate']:.2%} | "
            f"{m['default_rate_approved']:.2%} |"
        )
    top = "\n".join(
        f"| {i} | `{item['feature']}` | {item['auc_drop_mean']:.5f} |"
        for i, item in enumerate(metadata["raw_feature_importance"][:10], start=1)
    )
    base = next(iter(metadata["models"].values()))["test_metrics"]["default_rate_base"]
    return f"""# Metricas do modelo baseline ({metadata["model_version"]})

Gerado automaticamente por `src/models/train.py` em {metadata["trained_at"]}.
Conjunto de teste: {metadata["splits"]["test"]:,} registros do Dataset de Referencia
(taxa de default base: {base:.2%}). Threshold = ponto de KS maximo na validacao.

| Modelo | ROC-AUC | Gini | KS | PR-AUC | Recall (maus) | Taxa de aprovacao | Default na carteira aprovada |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Modelo oficial: **{metadata["official_model"]}**.

## Importancia das variaveis do contrato (permutation importance, queda de ROC-AUC)

| # | Variavel | Queda de AUC |
|---|---|---|
{top}
"""


def save_bundle(
    bundle: dict,
    metadata: dict,
    model_file: Path,
    metadata_file: Path,
    *,
    artifact_file: str | None = None,
) -> dict:
    """Grava o bundle e o metadata com `artifact` = {file, sha256} do bundle gravado.

    `artifact_file` e o nome registrado do arquivo (padrao: caminho relativo a
    raiz). Altera e devolve o proprio `metadata`.
    """
    model_file.parent.mkdir(parents=True, exist_ok=True)
    # Booster no formato de modelo do XGBoost: portavel entre sistemas operacionais.
    joblib.dump(to_portable(bundle), model_file)
    metadata["artifact"] = {
        "file": artifact_file or project_relative(model_file),
        "sha256": file_sha256(model_file),
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def main() -> tuple[dict, dict]:
    """Treina, grava os artefatos e devolve `(bundle, metadata)` (como gravados).

    O retorno serve ao estágio rastreado (`src.monitoring.stage`); o CLI ignora.
    """
    set_global_seed()
    reference = pd.read_parquet(REFERENCE_FILE)
    # Gate: o treino so acontece se a referencia cumprir o contrato rotulado.
    reference = enforce_contract(
        reference, batch_name="reference_training", stage="training", labeled=True
    )

    bundle, metadata = train_models(reference)
    metadata["data_lineage"] = {
        "reference_file": project_relative(REFERENCE_FILE),
        "reference_sha256": file_sha256(REFERENCE_FILE),
        "reference_rows": int(len(reference)),
        "build_report": "reports/data_contract/reference_build_report.json",
    }
    metadata["environment"] = environment()

    save_bundle(bundle, metadata, MODEL_FILE, MODEL_METADATA_FILE)
    MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_REPORTS_DIR / "baseline_metrics.md").write_text(
        write_metrics_markdown(metadata), encoding="utf-8"
    )
    log_event(
        logger,
        "model_saved",
        model_version=MODEL_VERSION,
        file=project_relative(MODEL_FILE),
        roc_auc=metadata["models"]["xgboost"]["test_metrics"]["roc_auc"],
    )
    return bundle, metadata


if __name__ == "__main__":
    main()
