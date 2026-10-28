<!--
================================================================================
LOG DE MANUTENÇÃO DE DOCUMENTAÇÃO
--------------------------------------------------------------------------------
Data       | Autor          | Descrição
--------------------------------------------------------------------------------
2026-10-20 | Eduardo Batista | Adequação ao padrão canônico de ADR (título, log e seção 2).
================================================================================
-->

# ADR 0001: Dataset: Home Credit Default Risk

| Metadado | Detalhe |
| :--- | :--- |
| **Status** | **Implementada** (2026-10-20) |
| **Decisores** | Eduardo Batista |
| **Tags** | `[dataset]`, `[etapa-1]` |
| **Etapa** | 1 (checklist 1.1) |

## Contexto

O enunciado pede um dataset **público, tabular, de classificação binária financeira com pelo menos 5.000 amostras**, e sugere três: German Credit Data, Home Credit Default Risk e Lending Club Loan Data. Ele também precisa comportar um cenário plausível de **degradação econômica** (inflação, novos perfis de cliente) para a Etapa 2.

| Opção | Linhas | Atende ≥ 5.000? | Observações |
|---|---|---|---|
| German Credit (UCI/Kaggle) | 1.000 | **Não** | Descartado pelo próprio critério do PDF. |
| Lending Club | ~2,2 milhões | Sim | Mais de 2 GB e mais de 150 colunas; pesado demais para o escopo e para rodar nas máquinas do grupo. |
| **Home Credit Default Risk** | **307.511** | **Sim** | Uma única tabela (`application_train.csv`, ~166 MB) já basta. Tem renda, idade, emprego, valor do crédito e parcela, scores de bureau e alvo binário (`TARGET`). |

## 2. Decisão de Arquitetura

Usar o **`application_train.csv` do Home Credit Default Risk** (Kaggle), lendo **apenas 21 das 122 colunas** (minimização de dados, LGPD art. 6º, III).

## Por que ele comporta o cenário de degradação econômica

- **Inflação/juros:** `income_total` (renda), `credit_amount` e `annuity_amount` (parcela) permitem simular o encarecimento do crédito e a perda de renda real. As razões de comprometimento de renda (`annuity_to_income`, `credit_to_income`) são calculadas dentro do pipeline do modelo, então um choque nessas variáveis se propaga para as features.
- **Desemprego / novo perfil de cliente:** `income_type` (inclui `Unemployed`/`Pensioner`), `years_employed`, `age_years` e os scores de bureau `ext_source_2/3` permitem simular a chegada de clientes mais jovens, com menos tempo de emprego e score pior.
- **Concept drift:** como o alvo é real, dá para simular que a mesma renda e o mesmo score passam a gerar mais inadimplência (relação X → y alterada).

## Consequências

- O Kaggle exige login e aceite das regras da competição para baixar. O download é automatizado em `src/data/download.py` (API) e também há instrução para baixar manualmente.
- **As regras da competição não permitem redistribuir os dados.** Por isso nenhum dataset entra no Git (`.gitignore`): tudo é regenerado com `make data`. A linhagem (SHA-256 do arquivo bruto e contagens) fica versionada em `reports/data_contract/reference_build_report.json`.
- O dataset não tem data de solicitação. A "passagem do tempo" é simulada na Etapa 2 a partir do lote de produção t0 (`data/production/production_base.parquet`).
