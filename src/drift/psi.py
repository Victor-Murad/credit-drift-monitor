"""Population Stability Index (PSI) -- tamanho do efeito do data drift.

O teste KS do Evidently responde "a distribuicao mudou?" (p-valor), mas com
dezenas de milhares de linhas qualquer diferenca minima e "significativa".
O PSI mede QUANTO mudou e permite filtrar alertas de ruido (ADR-0005).

    PSI = soma_i (atual_i - ref_i) * ln(atual_i / ref_i)

Numericas: faixas por quantis da Referencia. Categoricas: uma faixa por
categoria. Ausentes formam uma faixa propria (mudanca na taxa de nulos conta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PSI_MODERATE, PSI_SIGNIFICANT

# Evita ln(0) / divisao por zero em faixas vazias.
EPSILON = 1e-4
DEFAULT_BINS = 10


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1 / len(counts))
    return np.clip(counts / total, EPSILON, None)


def _numeric_counts(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    ref_values = reference.dropna().to_numpy(dtype=float)
    cur_values = current.dropna().to_numpy(dtype=float)

    inner = np.unique(np.quantile(ref_values, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    ref_counts = np.histogram(ref_values, bins=edges)[0]
    cur_counts = np.histogram(cur_values, bins=edges)[0]
    return (
        np.append(ref_counts, reference.isna().sum()),
        np.append(cur_counts, current.isna().sum()),
    )


def _categorical_counts(
    reference: pd.Series, current: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    ref = reference.astype(object).where(reference.notna(), "__missing__")
    cur = current.astype(object).where(current.notna(), "__missing__")
    categories = pd.Index(ref.unique()).union(pd.Index(cur.unique()))
    return (
        ref.value_counts().reindex(categories, fill_value=0).to_numpy(),
        cur.value_counts().reindex(categories, fill_value=0).to_numpy(),
    )


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = DEFAULT_BINS
) -> float:
    """PSI de uma coluna: Referencia (esperado) vs. lote corrente (observado)."""
    if reference.dropna().empty:
        raise ValueError("Referencia sem valores nao nulos: PSI indefinido.")
    numeric = pd.api.types.is_numeric_dtype(
        reference
    ) and not pd.api.types.is_bool_dtype(reference)
    # Rotulo binario (`target`) e numerico mas tem 2 valores: trata como categoria.
    if numeric and reference.nunique(dropna=True) > 2:
        ref_counts, cur_counts = _numeric_counts(reference, current, bins)
    else:
        ref_counts, cur_counts = _categorical_counts(reference, current)

    ref_share, cur_share = _shares(ref_counts), _shares(cur_counts)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def score_psi(reference: dict, scores: np.ndarray) -> float:
    """PSI do score do lote contra a distribuicao de referencia do treino.

    `reference` e o `score_reference` gravado no metadata do treino
    (`{"edges": [...], "shares": [...]}`, ver `score_distribution`).
    """
    edges = np.asarray(reference["edges"], dtype=float)
    ref_share = _shares(np.asarray(reference["shares"], dtype=float))
    if len(ref_share) != len(edges) + 1:
        raise ValueError(
            "score_reference invalido: 'shares' deve ter len('edges') + 1 itens."
        )
    values = np.asarray(scores, dtype=float)
    bins = np.searchsorted(edges, values, side="right")
    cur_share = _shares(np.bincount(bins, minlength=len(ref_share)))
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_level(psi: float) -> str:
    """Classifica o PSI: `stable`, `moderate` ou `significant`."""
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"
