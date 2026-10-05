"""Fixtures: dados sinteticos aderentes ao contrato.

Os testes nao dependem do dataset do Kaggle (licenca nao permite versiona-lo
e o CI nao tem credenciais), entao geramos lotes sinteticos com o mesmo
esquema e um sinal aprendivel (ext_source baixo -> mais default).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Antes de qualquer import de `src`: os testes nao escrevem no log real do
# pipeline (logs/pipeline.jsonl), que sera centralizado na Etapa 3.
os.environ["PIPELINE_LOG_FILE"] = str(
    Path(tempfile.gettempdir()) / "credit-drift-monitor-tests.jsonl"
)

import numpy as np  # noqa: E402
import pandas as pd
import pytest

from src.validation import contract as c


def make_valid_df(n: int = 300, seed: int = 0, labeled: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income_type = rng.choice(c.INCOME_TYPES[:4], size=n, p=[0.5, 0.2, 0.2, 0.1])
    not_employed = np.isin(income_type, c.NOT_EMPLOYED_INCOME_TYPES)
    age = rng.uniform(21, 68, size=n).round(2)
    credit = rng.uniform(50_000, 1_500_000, size=n).round(1)
    children = rng.integers(0, 4, size=n)
    ext2 = rng.uniform(0, 1, size=n)
    ext3 = rng.uniform(0, 1, size=n)
    ext3[rng.random(n) < 0.15] = np.nan

    df = pd.DataFrame(
        {
            "client_id": np.arange(100_001, 100_001 + n),
            "contract_type": rng.choice(c.CONTRACT_TYPES, size=n, p=[0.9, 0.1]),
            "gender": rng.choice(c.GENDERS, size=n),
            "own_car": rng.choice(c.YES_NO, size=n),
            "own_realty": rng.choice(c.YES_NO, size=n),
            "income_type": income_type,
            "education_type": rng.choice(c.EDUCATION_TYPES[:3], size=n),
            "family_status": rng.choice(c.FAMILY_STATUSES[:3], size=n),
            "housing_type": rng.choice(c.HOUSING_TYPES[:3], size=n),
            "region_rating": rng.choice(c.REGION_RATINGS, size=n),
            "children_count": children,
            "family_members": children + rng.integers(1, 3, size=n),
            "age_years": age,
            "years_employed": np.where(
                not_employed, np.nan, rng.uniform(0, 15, size=n).round(2)
            ),
            "income_total": rng.uniform(30_000, 500_000, size=n).round(1),
            "credit_amount": credit,
            "annuity_amount": (credit / rng.uniform(10, 40, size=n)).round(1),
            "goods_price": (credit * 0.9).round(1),
            "ext_source_2": ext2,
            "ext_source_3": ext3,
        }
    )
    if labeled:
        logit = -1.0 - 3.0 * (ext2 - 0.5) - 0.02 * (age - 40)
        df["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return make_valid_df()


@pytest.fixture
def valid_unlabeled_df() -> pd.DataFrame:
    return make_valid_df(labeled=False)
