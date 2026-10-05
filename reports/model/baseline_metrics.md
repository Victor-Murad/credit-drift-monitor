# Metricas do modelo baseline (baseline-v1)

Gerado automaticamente por `src/models/train.py` em 2026-10-05T21:15:31.322878+00:00.
Conjunto de teste: 43,049 registros do Dataset de Referencia
(taxa de default base: 8.07%). Threshold = ponto de KS maximo na validacao.

| Modelo | ROC-AUC | Gini | KS | PR-AUC | Recall (maus) | Taxa de aprovacao | Default na carteira aprovada |
|---|---|---|---|---|---|---|---|
| xgboost | 0.7494 | 0.4989 | 0.3695 | 0.2434 | 0.6820 | 65.45% | 3.92% |
| logistic_regression | 0.7307 | 0.4613 | 0.3429 | 0.2180 | 0.6403 | 67.41% | 4.31% |

Modelo oficial: **xgboost**.

## Importancia das variaveis do contrato (permutation importance, queda de ROC-AUC)

| # | Variavel | Queda de AUC |
|---|---|---|
| 1 | `ext_source_3` | 0.05831 |
| 2 | `annuity_amount` | 0.04842 |
| 3 | `ext_source_2` | 0.04837 |
| 4 | `credit_amount` | 0.04146 |
| 5 | `goods_price` | 0.02538 |
| 6 | `education_type` | 0.00858 |
| 7 | `years_employed` | 0.00606 |
| 8 | `contract_type` | 0.00541 |
| 9 | `age_years` | 0.00290 |
| 10 | `income_total` | 0.00194 |
