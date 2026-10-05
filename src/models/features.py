"""Features derivadas calculadas DENTRO do pipeline do modelo.

Ficam no pipeline (e nao no dataset) para que o contrato de entrada continue
contendo so dados brutos: um choque de renda na Etapa 2, por exemplo, se
propaga automaticamente para as razoes de comprometimento de renda.
"""

from __future__ import annotations

import pandas as pd

from src.config import MODEL_INPUT_COLUMNS


def model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona as colunas que o modelo consome (descarta id, genero, alvo)."""
    return df[MODEL_INPUT_COLUMNS]


def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """Razoes financeiras classicas de credito.

    Divisoes sao seguras: o contrato garante renda, credito e parcela > 0.
    """
    X = X.copy()
    X["credit_to_income"] = X["credit_amount"] / X["income_total"]
    X["annuity_to_income"] = X["annuity_amount"] / X["income_total"]
    X["credit_to_annuity"] = X["credit_amount"] / X["annuity_amount"]
    X["goods_to_credit"] = X["goods_price"] / X["credit_amount"]
    return X
