"""Configuracao central: caminhos, seed global e constantes de dominio.

Todos os scripts importam daqui para que caminhos e listas de features nao
fiquem duplicados (e divergentes) entre as etapas do projeto.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reprodutibilidade
# --------------------------------------------------------------------------
SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fixa as fontes de aleatoriedade usadas no projeto (checklist 1.2).

    Bibliotecas que recebem `random_state` (scikit-learn, XGBoost) tambem
    recebem `SEED` explicitamente; esta funcao cobre o restante.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]


def project_relative(path: Path) -> str:
    """Caminho relativo a raiz do repo (evita gravar caminhos da maquina local
    nos relatorios versionados)."""
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PRODUCTION_DIR = DATA_DIR / "production"
QUARANTINE_DIR = DATA_DIR / "quarantine"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
CONTRACT_REPORTS_DIR = REPORTS_DIR / "data_contract"
MODEL_REPORTS_DIR = REPORTS_DIR / "model"
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"
LOGS_DIR = ROOT_DIR / "logs"

# Fonte: Home Credit Default Risk (Kaggle) -- ver docs/adrs/0001
KAGGLE_COMPETITION = "home-credit-default-risk"
RAW_FILE_NAME = "application_train.csv"
RAW_FILE = RAW_DIR / RAW_FILE_NAME

# Dataset de Referencia (treino) x lote de Producao t0 (base da Etapa 2).
# A separacao fisica em pastas diferentes e uma Boa Pratica Obrigatoria do PDF.
REFERENCE_FILE = REFERENCE_DIR / "reference.parquet"
PRODUCTION_BASE_FILE = PRODUCTION_DIR / "production_base.parquet"
# Fatias de producao com drift simulado (Etapa 2).
PRODUCTION_T1_FILE = PRODUCTION_DIR / "production_t1_data_drift.parquet"
PRODUCTION_T2_FILE = PRODUCTION_DIR / "production_t2_concept_drift.parquet"

# --------------------------------------------------------------------------
# Choques de drift (Etapa 2) -- valores reprodutiveis; clipados no contrato
# --------------------------------------------------------------------------
ANNUITY_SHOCK = 1.25
EXT_SOURCE_3_SHIFT = -0.15
INCOME_SHOCK = 1.15
CONCEPT_FLIP_RATE = 0.25
# Concept drift: flipar bons pagadores com score de bureau alto (baixo risco).
CONCEPT_SCORE_COLUMN = "ext_source_3"
CONCEPT_SCORE_MIN = 0.55

INVALID_BATCH_FILE = SAMPLES_DIR / "invalid_batch.csv"
VALID_BATCH_FILE = SAMPLES_DIR / "valid_batch.csv"

MODEL_FILE = MODELS_DIR / "baseline_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "baseline_metadata.json"

# Modelo campeao em producao (Etapa 3): copia local do que o Registry marca
# com o alias `champion`.
CHAMPION_DIR = MODELS_DIR / "champion"
CHAMPION_FILE = CHAMPION_DIR / "baseline_model.joblib"
CHAMPION_METADATA_FILE = CHAMPION_DIR / "baseline_metadata.json"

# Sobrescrevivel por variavel de ambiente (os testes usam um arquivo temporario).
PIPELINE_LOG_FILE = Path(
    os.environ.get("PIPELINE_LOG_FILE", LOGS_DIR / "pipeline.jsonl")
)

# --------------------------------------------------------------------------
# Observabilidade e Model Registry locais (Etapa 3 -- ver docs/adrs/0006)
# --------------------------------------------------------------------------
MLFLOW_EXPERIMENT = "credit-drift-monitor"
# Caminhos absolutos a partir de ROOT_DIR: o worktree e o checkout principal
# nao divergem. Sobrescreviveis por variavel de ambiente (os testes usam um
# diretorio temporario e nunca tocam o mlflow.db real).
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
)
MLFLOW_ARTIFACT_ROOT = Path(
    os.environ.get("MLFLOW_ARTIFACT_ROOT", ROOT_DIR / "mlartifacts")
)
REGISTERED_MODEL_NAME = "credit-default-xgboost"
# Politica de promocao challenger -> champion: ganho minimo de ROC-AUC exigido
# e regressao maxima tolerada nas metricas de guarda.
PROMOTION_MIN_GAIN = 0.02
PROMOTION_MAX_REGRESSION = 0.02
# Faixas (quantis do baseline) usadas no PSI do score do modelo.
SCORE_PSI_BINS = 10

# --------------------------------------------------------------------------
# Particionamento
# --------------------------------------------------------------------------
# Fracao dos registros limpos reservada como "lote de producao t0": nunca e
# vista no treino e e o ponto de partida da simulacao de drift (Etapa 2).
PRODUCTION_FRACTION = 0.30
# Dentro do Dataset de Referencia: treino / validacao (threshold) / teste.
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20

# --------------------------------------------------------------------------
# Esquema canonico (nomes apos a ingestao -- ver src/data/prepare.py)
# --------------------------------------------------------------------------
ID_COLUMN = "client_id"
TARGET_COLUMN = "target"

# Mapeamento coluna bruta (Home Credit) -> coluna canonica do contrato.
# Apenas estas colunas sao lidas do CSV bruto (minimizacao de dados, LGPD
# art. 6, III): 21 de 122 colunas.
RAW_COLUMN_MAP: dict[str, str] = {
    "SK_ID_CURR": ID_COLUMN,
    "TARGET": TARGET_COLUMN,
    "NAME_CONTRACT_TYPE": "contract_type",
    "CODE_GENDER": "gender",
    "FLAG_OWN_CAR": "own_car",
    "FLAG_OWN_REALTY": "own_realty",
    "CNT_CHILDREN": "children_count",
    "CNT_FAM_MEMBERS": "family_members",
    "AMT_INCOME_TOTAL": "income_total",
    "AMT_CREDIT": "credit_amount",
    "AMT_ANNUITY": "annuity_amount",
    "AMT_GOODS_PRICE": "goods_price",
    "NAME_INCOME_TYPE": "income_type",
    "NAME_EDUCATION_TYPE": "education_type",
    "NAME_FAMILY_STATUS": "family_status",
    "NAME_HOUSING_TYPE": "housing_type",
    "REGION_RATING_CLIENT": "region_rating",
    "EXT_SOURCE_2": "ext_source_2",
    "EXT_SOURCE_3": "ext_source_3",
    # Convertidas em anos em prepare.py
    "DAYS_BIRTH": "days_birth",
    "DAYS_EMPLOYED": "days_employed",
}

# Valor-sentinela do Home Credit em DAYS_EMPLOYED para quem nao esta
# empregado (aposentados/desempregados). Vira NaN em `years_employed`.
DAYS_EMPLOYED_SENTINEL = 365243

CATEGORICAL_FEATURES = [
    "contract_type",
    "own_car",
    "own_realty",
    "income_type",
    "education_type",
    "family_status",
    "housing_type",
]
NUMERIC_FEATURES = [
    "children_count",
    "family_members",
    "age_years",
    "years_employed",
    "income_total",
    "credit_amount",
    "annuity_amount",
    "goods_price",
    "region_rating",
    "ext_source_2",
    "ext_source_3",
]
# Razoes financeiras derivadas dentro do pipeline do modelo
# (src/models/features.py). Nao fazem parte do contrato de entrada.
RATIO_FEATURES = [
    "credit_to_income",
    "annuity_to_income",
    "credit_to_annuity",
    "goods_to_credit",
]

# Atributos mantidos APENAS para auditoria de vies (Etapa 4) -- nunca sao
# usados como feature do modelo (LGPD art. 6, IX -- nao discriminacao).
AUDIT_COLUMNS = ["gender"]

MODEL_INPUT_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# --------------------------------------------------------------------------
# Limiares de monitoramento (Etapa 2 -- ver docs/adrs/0005)
# --------------------------------------------------------------------------
# PSI: convencao de mercado (credit scoring). < 0,10 estavel; 0,10-0,25
# moderado; >= 0,25 mudanca significativa.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
# Queda de ROC-AUC (vs. baseline) a partir da qual a performance do modelo
# e considerada degradada -- sinal de concept drift que nao depende do `target`
# ter mudado de prevalencia.
AUC_DROP_ALERT = 0.05
