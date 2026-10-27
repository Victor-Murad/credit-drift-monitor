# Demonstracao do Data Contract -- bloqueio de lote invalido

Gerado por `src/validation/demo.py` em 2026-10-27T10:38:22.131947+00:00 (seed 42).

| Lote | Arquivo | Resultado do gate | Registros pontuados pelo modelo |
|---|---|---|---|
| valido | `data/samples/valid_batch.csv` | **APROVADO** | 500 |
| invalido | `data/samples/invalid_batch.csv` | **BLOQUEADO** | 0 |

Resultado geral da demonstracao: **SUCESSO**.

## Defeitos injetados x deteccao

| Regra esperada | Defeito | Coluna | Linhas afetadas | Detectado? |
|---|---|---|---|---|
| R2 | renda nula | `income_total` | 5 | sim |
| R3 | cliente menor de 18 anos | `age_years` | 4 | sim |
| R3 | idade impossivel (150 anos) | `age_years` | 2 | sim |
| R5 | valor de credito negativo | `credit_amount` | 3 | sim |
| R4 | renda acima do limite (erro de digitacao) | `income_total` | 2 | sim |
| R6 | tipo de contrato desconhecido | `contract_type` | 2 | sim |
| R7 | score externo fora de [0, 1] | `ext_source_2` | 3 | sim |
| R8 | parcela maior que o credito | `<dataframe>` | 3 | sim |
| R10 | coluna de CPF nao prevista no contrato | `cpf` | 500 | sim |
| R1 | registros duplicados (mesmo client_id) | `client_id` | 8 | sim |

## Regras violadas reportadas pelo contrato (Pandera, `lazy=True`)

| Regra | Coluna | Checagem | Ocorrencias |
|---|---|---|---|
| R1 - valor duplicado (deve ser unico) | `client_id` | `field_uniqueness` | 16 |
| R10 - coluna nao prevista no contrato | `credit_application_v1.0.0` | `column_in_schema` | 1 |
| R2 - valor nulo em campo obrigatorio | `income_total` | `not_nullable` | 5 |
| R3 - valor fora do intervalo permitido | `age_years` | `greater_than_or_equal_to(18)` | 4 |
| R3 - valor fora do intervalo permitido | `age_years` | `less_than_or_equal_to(100)` | 2 |
| R4 - valor fora do intervalo permitido | `income_total` | `less_than_or_equal_to(10000000)` | 2 |
| R5 - valor fora do intervalo permitido | `credit_amount` | `greater_than(0)` | 3 |
| R6 - categoria fora do dominio permitido | `contract_type` | `isin(['Cash loans', 'Revolving loans'])` | 2 |
| R7 - valor fora do intervalo permitido | `ext_source_2` | `greater_than_or_equal_to(0)` | 1 |
| R7 - valor fora do intervalo permitido | `ext_source_2` | `less_than_or_equal_to(1)` | 2 |
| R8 - parcela maior que o valor total do credito | `<dataframe>` | `annuity_not_greater_than_credit` | 6 |

Relatorios completos: [`invalid_batch_report.html`](invalid_batch_report.html) e
[`valid_batch_report.html`](valid_batch_report.html).
