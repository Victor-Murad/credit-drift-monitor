# Metricas do modelo baseline (baseline-v1)

Gerado automaticamente por `src/models/train.py` em 2026-10-25T16:52:30.447030+00:00.
Conjunto de teste: 43,049 registros do Dataset de Referencia
(taxa de default base: 8.07%). Threshold = ponto de KS maximo na validacao.

| Modelo | ROC-AUC | Gini | KS | PR-AUC | Recall (maus) | Taxa de aprovacao | Default na carteira aprovada |
|---|---|---|---|---|---|---|---|
| xgboost | 0.7491 | 0.4982 | 0.3706 | 0.2435 | 0.6872 | 65.02% | 3.88% |
| logistic_regression | 0.7307 | 0.4613 | 0.3429 | 0.2180 | 0.6403 | 67.41% | 4.31% |

Modelo oficial: **xgboost**.

## Importancia das variaveis do contrato (permutation importance, queda de ROC-AUC)

| # | Variavel | Queda de AUC |
|---|---|---|
| 1 | `ext_source_3` | 0.05700 |
| 2 | `annuity_amount` | 0.05041 |
| 3 | `ext_source_2` | 0.04826 |
| 4 | `credit_amount` | 0.04025 |
| 5 | `goods_price` | 0.02624 |
| 6 | `education_type` | 0.00893 |
| 7 | `years_employed` | 0.00578 |
| 8 | `contract_type` | 0.00413 |
| 9 | `age_years` | 0.00257 |
| 10 | `own_car` | 0.00182 |
