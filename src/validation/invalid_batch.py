"""Geracao de um lote PROPOSITALMENTE invalido (checklist 1.4).

Parte de registros limpos e injeta, de forma deterministica (seed), defeitos
tipicos de um sistema upstream com problemas. Cada defeito mira uma regra do
contrato, e o manifesto devolvido permite conferir que TODAS foram detectadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ID_COLUMN, SEED

# (codigo, descricao, regra esperada, coluna afetada)
DEFECTS = [
    ("null_income", "renda nula", "R2", "income_total"),
    ("underage", "cliente menor de 18 anos", "R3", "age_years"),
    ("age_out_of_range", "idade impossivel (150 anos)", "R3", "age_years"),
    ("duplicate_rows", "registros duplicados (mesmo client_id)", "R1", ID_COLUMN),
    ("negative_credit", "valor de credito negativo", "R5", "credit_amount"),
    (
        "income_above_limit",
        "renda acima do limite (erro de digitacao)",
        "R4",
        "income_total",
    ),
    ("unknown_category", "tipo de contrato desconhecido", "R6", "contract_type"),
    ("ext_score_out_of_range", "score externo fora de [0, 1]", "R7", "ext_source_2"),
    ("annuity_gt_credit", "parcela maior que o credito", "R8", "<dataframe>"),
    ("unexpected_pii_column", "coluna de CPF nao prevista no contrato", "R10", "cpf"),
]


def make_invalid_batch(
    clean: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, list[dict]]:
    """Devolve (lote sujo, manifesto dos defeitos injetados).

    `clean` deve estar em conformidade com o contrato e ter pelo menos ~40
    linhas (os defeitos sao aplicados em linhas distintas).
    """
    rng = np.random.default_rng(seed)
    df = clean.reset_index(drop=True).copy()
    rows = rng.permutation(len(df))
    cursor = 0

    def take(n: int) -> np.ndarray:
        nonlocal cursor
        picked = rows[cursor : cursor + n]
        cursor += n
        return picked

    # numericos viram float para aceitar NaN/valores invalidos
    df["income_total"] = df["income_total"].astype(float)
    manifest: list[dict] = []

    def record(code: str, idx) -> None:
        _, description, rule, column = next(d for d in DEFECTS if d[0] == code)
        manifest.append(
            {
                "defect": code,
                "description": description,
                "expected_rule": rule,
                "column": column,
                "n_rows": int(len(idx)),
            }
        )

    idx = take(5)
    df.loc[idx, "income_total"] = np.nan
    record("null_income", idx)

    idx = take(4)
    df.loc[idx, "age_years"] = [16.0, 17.5, 12.0, 17.9]
    record("underage", idx)

    idx = take(2)
    df.loc[idx, "age_years"] = 150.0
    record("age_out_of_range", idx)

    idx = take(3)
    df.loc[idx, "credit_amount"] = -df.loc[idx, "credit_amount"].abs()
    record("negative_credit", idx)

    idx = take(2)
    df.loc[idx, "income_total"] = 117_000_000.0
    record("income_above_limit", idx)

    idx = take(2)
    df.loc[idx, "contract_type"] = "Crypto loans"
    record("unknown_category", idx)

    idx = take(3)
    df.loc[idx, "ext_source_2"] = [1.7, -0.2, 3.0]
    record("ext_score_out_of_range", idx)

    idx = take(3)
    df.loc[idx, "annuity_amount"] = df.loc[idx, "credit_amount"].abs() * 2
    record("annuity_gt_credit", idx)

    # Coluna de PII que o sistema upstream "esqueceu" de remover. Valores
    # obviamente ficticios (nao sao CPFs validos).
    df["cpf"] = "000.000.000-00"
    record("unexpected_pii_column", df.index)

    duplicated = df.loc[take(8)]
    df = pd.concat([df, duplicated], ignore_index=True)
    record("duplicate_rows", duplicated.index)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df, manifest
