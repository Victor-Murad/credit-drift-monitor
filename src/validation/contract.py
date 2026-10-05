"""Contrato de dados (Data Contract) de uma solicitacao de credito -- Pandera.

Schema-as-code (Aula 5 -- Validacao de Schema e Contratos de Dados): o
contrato e uma classe Python versionada junto com o codigo e testada no CI.
A especificacao "organizacional" do mesmo contrato (dono, termos de uso,
classificacao LGPD de cada campo) fica em `contracts/credit_application.yaml`;
`tests/test_contract_spec.py` garante que os dois nao divergem.

Regras rigidas (todas bloqueiam o lote inteiro -- fail-closed):

    R1  Unicidade      client_id unico e nao nulo (sem duplicatas)
    R2  Completude     campos obrigatorios nao nulos (ex.: renda nunca nula)
    R3  Validade       idade em [18, 100] anos -- capacidade civil para contratar credito
    R4  Validade       renda anual em (0, 10.000.000]
    R5  Validade       valor do credito em (0, 5.000.000] e parcela > 0
    R6  Validade       categorias restritas a dominios conhecidos
    R7  Validade       scores externos (bureau) em [0, 1]
    R8  Consistencia   parcela nao pode ser maior que o valor total do credito
    R9  Consistencia   tempo de emprego ausente somente para aposentado/desempregado
    R10 Schema         `strict=True`: colunas fora do contrato (ex.: CPF) sao rejeitadas
    R11 Schema         tipos coeridos e verificados (int/float/str)
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

CONTRACT_NAME = "credit_application"
CONTRACT_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# Limites de negocio (documentados em docs/data_contract.md)
# --------------------------------------------------------------------------
MIN_AGE_YEARS = 18
MAX_AGE_YEARS = 100
MAX_YEARS_EMPLOYED = 60
MAX_INCOME_TOTAL = 10_000_000
MAX_CREDIT_AMOUNT = 5_000_000
MAX_CHILDREN = 20
MAX_FAMILY_MEMBERS = 25

# --------------------------------------------------------------------------
# Dominios das variaveis categoricas (valores observados no Home Credit,
# exceto codigos de "desconhecido" como 'XNA'/'Unknown', que sao rejeitados)
# --------------------------------------------------------------------------
CONTRACT_TYPES = ["Cash loans", "Revolving loans"]
GENDERS = ["F", "M"]
YES_NO = ["Y", "N"]
INCOME_TYPES = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
    "Student",
    "Businessman",
    "Maternity leave",
]
NOT_EMPLOYED_INCOME_TYPES = ["Pensioner", "Unemployed"]
EDUCATION_TYPES = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
FAMILY_STATUSES = [
    "Married",
    "Single / not married",
    "Civil marriage",
    "Separated",
    "Widow",
]
HOUSING_TYPES = [
    "House / apartment",
    "With parents",
    "Municipal apartment",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
]
REGION_RATINGS = [1, 2, 3]


class CreditApplicationSchema(pa.DataFrameModel):
    """Contrato v1 de um lote de solicitacoes de credito (sem rotulo).

    Usado no gate de inferencia: todo lote que chega para ser pontuado pelo
    modelo passa por aqui antes.
    """

    # R1 -- unicidade
    client_id: Series[int] = pa.Field(nullable=False, unique=True, gt=0)

    # R6 -- dominios categoricos
    contract_type: Series[str] = pa.Field(nullable=False, isin=CONTRACT_TYPES)
    gender: Series[str] = pa.Field(nullable=False, isin=GENDERS)
    own_car: Series[str] = pa.Field(nullable=False, isin=YES_NO)
    own_realty: Series[str] = pa.Field(nullable=False, isin=YES_NO)
    income_type: Series[str] = pa.Field(nullable=False, isin=INCOME_TYPES)
    education_type: Series[str] = pa.Field(nullable=False, isin=EDUCATION_TYPES)
    family_status: Series[str] = pa.Field(nullable=False, isin=FAMILY_STATUSES)
    housing_type: Series[str] = pa.Field(nullable=False, isin=HOUSING_TYPES)
    region_rating: Series[int] = pa.Field(nullable=False, isin=REGION_RATINGS)

    # Composicao familiar
    children_count: Series[int] = pa.Field(nullable=False, ge=0, le=MAX_CHILDREN)
    family_members: Series[int] = pa.Field(nullable=False, ge=1, le=MAX_FAMILY_MEMBERS)

    # R3 -- idade
    age_years: Series[float] = pa.Field(
        nullable=False, ge=MIN_AGE_YEARS, le=MAX_AGE_YEARS
    )
    # Ausente apenas para quem nao esta empregado (ver R9)
    years_employed: Series[float] = pa.Field(nullable=True, ge=0, le=MAX_YEARS_EMPLOYED)

    # R2 + R4 -- renda obrigatoria e limitada
    income_total: Series[float] = pa.Field(nullable=False, gt=0, le=MAX_INCOME_TOTAL)
    # R5 -- valores do emprestimo
    credit_amount: Series[float] = pa.Field(nullable=False, gt=0, le=MAX_CREDIT_AMOUNT)
    annuity_amount: Series[float] = pa.Field(nullable=False, gt=0)
    # Ausente em parte dos creditos rotativos (nao ha bem financiado)
    goods_price: Series[float] = pa.Field(nullable=True, gt=0)

    # R7 -- scores normalizados de bureaus externos (podem faltar)
    ext_source_2: Series[float] = pa.Field(nullable=True, ge=0, le=1)
    ext_source_3: Series[float] = pa.Field(nullable=True, ge=0, le=1)

    # R8 -- consistencia entre campos
    @pa.dataframe_check
    def annuity_not_greater_than_credit(cls, df: pd.DataFrame) -> Series[bool]:
        # Nulos sao responsabilidade da R2; aqui so comparamos valores presentes.
        either_null = df["annuity_amount"].isna() | df["credit_amount"].isna()
        return either_null | (df["annuity_amount"] <= df["credit_amount"])

    # R9 -- consistencia entre campos
    @pa.dataframe_check
    def employment_missing_only_if_not_employed(cls, df: pd.DataFrame) -> Series[bool]:
        return df["years_employed"].notna() | df["income_type"].isin(
            NOT_EMPLOYED_INCOME_TYPES
        )

    class Config:
        name = f"{CONTRACT_NAME}_v{CONTRACT_VERSION}"
        strict = True  # R10
        coerce = True  # R11


class LabeledCreditApplicationSchema(CreditApplicationSchema):
    """Contrato v1 de um lote rotulado (treino ou feedback de producao)."""

    target: Series[int] = pa.Field(nullable=False, isin=[0, 1])

    class Config:
        name = f"{CONTRACT_NAME}_labeled_v{CONTRACT_VERSION}"
        strict = True
        coerce = True


def get_schema(labeled: bool) -> type[CreditApplicationSchema]:
    """Retorna o contrato adequado ao lote (com ou sem a coluna `target`)."""
    return LabeledCreditApplicationSchema if labeled else CreditApplicationSchema


def contract_columns(labeled: bool) -> list[str]:
    """Colunas exigidas pelo contrato, na ordem declarada."""
    return list(get_schema(labeled).to_schema().columns)
