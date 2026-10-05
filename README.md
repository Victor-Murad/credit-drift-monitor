# credit-drift-monitor

**Camada de sustentação e confiabilidade de um modelo de Credit Scoring em produção:** data contracts, detecção de Data/Concept Drift, observabilidade e governança (LGPD).

> Tech Challenge — Fase 4 (Model Monitoring and Security) · Pós-Tech Machine Learning Engineering · FIAP

## Contexto

Uma fintech de concessão de crédito tem um modelo de Credit Scoring em produção. Com as mudanças recentes na economia (inflação, novos perfis de clientes), a diretoria suspeita que o modelo está tomando decisões erradas e sofrendo **degradação silenciosa**.

Este projeto constrói a camada que impede essa degradação de passar despercebida:

1. **Contratos de dados** que barram dados ruins antes de chegarem ao modelo.
2. **Simulação e detecção de drift** ao longo do tempo (Data Drift e Concept Drift).
3. **Observabilidade** do pipeline e do modelo (logs, métricas, alertas).
4. **Governança e privacidade** (LGPD), com análise causal da degradação.

## Sumário

- [Contexto](#contexto)
- [Arquitetura e fluxo de dados](#arquitetura-e-fluxo-de-dados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Etapa 1 — Validação de Dados e Contratos](#etapa-1--validação-de-dados-e-contratos)
- [Etapa 2 — Simulação e Detecção de Drift](#etapa-2--simulação-e-detecção-de-drift)
- [Etapa 3 — Observabilidade de Pipelines e Modelos](#etapa-3--observabilidade-de-pipelines-e-modelos)
- [Etapa 4 — Governança, LGPD e Análise Causal](#etapa-4--governança-lgpd-e-análise-causal)
- [Decisões de arquitetura (ADRs)](#decisões-de-arquitetura-adrs)
- [Vídeo STAR](#vídeo-star)
- [Equipe](#equipe)

## Arquitetura e fluxo de dados

```
 Kaggle: application_train.csv (307.511 linhas, 122 colunas)
              │  lê só 21 colunas (minimização, LGPD art. 6º, III)
              ▼
 src/data/prepare.py ── esquema canônico ── contrato aplicado linha a linha
              │
      ┌───────┴──────────────────────────┐
      ▼ 70% (estratificado, seed 42)     ▼ 30%
 data/reference/reference.parquet   data/production/production_base.parquet
 (Dataset de Referência: treino)    (lote de Produção t0, nunca visto no treino;
      │                              a Etapa 2 aplica o drift a partir dele)
      ▼
 ┌───────────────────────── GATE DO CONTRATO (Pandera, fail-closed) ─────────┐
 │ lote viola o contrato? ── sim ──► BLOQUEADO: quarentena + relatório HTML/JSON│
 │                                   + log estruturado (logs/pipeline.jsonl)  │
 └──────────── não ──────────────────────────────────────────────────────────┘
      ▼
 src/models/train.py ──► models/baseline_model.joblib (XGBoost)
 src/models/predict.py (inferência: sempre atrás do mesmo gate)
```

## Estrutura do repositório

```
├── contracts/credit_application.yaml   # Data Contract (formato DCS): dono, termos, classificação LGPD
├── data/                               # NÃO versionado (licença Kaggle); gerado por `make data`
│   ├── raw/                            #   application_train.csv
│   ├── reference/                      #   Dataset de Referência (treino)
│   ├── production/                     #   lotes de produção/análise (drift na Etapa 2)
│   ├── quarantine/                     #   lotes bloqueados pelo contrato
│   └── samples/                        #   lotes de demonstração (válido / inválido)
├── docs/adr/                           # decisões de arquitetura
├── models/                             # baseline treinado + metadados (versionados)
├── reports/
│   ├── data_contract/                  # evidências do contrato (HTML/JSON/MD)
│   └── model/                          # métricas do baseline
├── src/
│   ├── config.py                       # caminhos, seed, features
│   ├── logging_utils.py                # logs estruturados (JSON lines)
│   ├── data/                           # download + construção dos datasets
│   ├── validation/                     # contrato Pandera, gate, relatórios, demo
│   └── models/                         # features, treino, predição
└── tests/                              # pytest com dados sintéticos (sem Kaggle)
```

## Como executar

### Pré-requisitos

- [**uv**](https://docs.astral.sh/uv/). Ele instala o Python 3.12 sozinho. No Windows: `winget install --id=astral-sh.uv -e`; no macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Conta no Kaggle com as [regras da competição](https://www.kaggle.com/competitions/home-credit-default-risk/rules) aceitas.
- `make` é opcional: todos os comandos têm um equivalente com `uv run` (tabela abaixo).

### Passo a passo

```bash
# 1. Ambiente (cria .venv a partir do uv.lock)
uv sync

# 2. Dados: escolha UMA das opções
#    a) API do Kaggle: salve o token em ~/.kaggle/access_token
#       (ou na variavel KAGGLE_API_TOKEN) e rode:
uv run python -m src.data.download
#    b) Manual: baixe application_train.csv na aba "Data" da competição
#       e coloque em data/raw/application_train.csv

# 3. Etapa 1 completa
uv run python -m src.data.prepare        # Referência + Produção t0
uv run python -m src.models.train        # baseline (passa pelo gate antes)
uv run python -m src.validation.demo     # contrato bloqueando lote inválido

# 4. Testes
uv run pytest
```

| `make` | Equivalente sem make |
|---|---|
| `make setup` | `uv sync` |
| `make download-data` | `uv run python -m src.data.download` |
| `make data` | `uv run python -m src.data.prepare` |
| `make train` | `uv run python -m src.models.train` |
| `make contract-demo` | `uv run python -m src.validation.demo` |
| `make validate INPUT=arq.csv` | `uv run python -m src.validation.validate --input arq.csv` |
| `make predict INPUT=arq.csv` | `uv run python -m src.models.predict --input arq.csv` |
| `make etapa1` | os três comandos do passo 3 |
| `make test` | `uv run pytest` |

Reprodutibilidade: `SEED = 42` é fixada em todos os passos (`src/config.py`) e as dependências estão travadas no `uv.lock`.

## Etapa 1 — Validação de Dados e Contratos

> Foco: garantir que lixo não entre no modelo. Detalhes, evidências e o checklist 1.1–1.4 em [`docs/etapa-1-validacao-de-dados.md`](docs/etapa-1-validacao-de-dados.md).

**Entregável do PDF: "Script de validação de dados operante e modelo baseline treinado".**

- **Data Contract (Pandera)**: [`src/validation/contract.py`](src/validation/contract.py), com 11 regras rígidas: unicidade, completude, idade 18–100, limites de renda e de crédito, domínios categóricos, scores em [0, 1], parcela ≤ crédito, consistência de emprego, `strict` (bloqueia colunas não previstas, como CPF) e tipos.
- **Pipeline de validação (gate)**: [`src/validation/gate.py`](src/validation/gate.py). É executado antes do treino e antes de toda inferência. É fail-closed: o lote vai para quarentena e o gate gera relatório e log.
- **Demonstração de bloqueio**: [`reports/data_contract/contract_demo_summary.md`](reports/data_contract/contract_demo_summary.md) e [`invalid_batch_report.html`](reports/data_contract/invalid_batch_report.html).
- **Modelo baseline**: [`models/baseline_model.joblib`](models/) e as métricas em [`reports/model/baseline_metrics.md`](reports/model/baseline_metrics.md).


## Etapa 2 — Simulação e Detecção de Drift

> _Em construção._ Ponto de partida: `data/production/production_base.parquet` (lote t0 limpo, com rótulo, nunca visto no treino) e `src/models/predict.py` (gera predições passando pelo gate). Variáveis candidatas ao drift estão em [`docs/etapa-1-validacao-de-dados.md`](docs/etapa-1-validacao-de-dados.md#variáveis-definidas-para-o-drift-da-etapa-2).

## Etapa 3 — Observabilidade de Pipelines e Modelos

> _Em construção._ Os eventos de validação, treino e predição já são gravados em formato estruturado em `logs/pipeline.jsonl` (`src/logging_utils.py`), prontos para centralização.

## Etapa 4 — Governança, LGPD e Análise Causal

> _Em construção._ Insumos já existentes: minimização de dados (21 de 122 colunas), classificação LGPD por campo em [`contracts/credit_application.yaml`](contracts/credit_application.yaml), `gender` excluído do modelo e mantido só para auditoria de viés, e nenhum dado pessoal versionado.

## Decisões de arquitetura (ADRs)

| ADR | Decisão |
|---|---|
| [0001](docs/adr/0001-dataset-home-credit.md) | Dataset Home Credit Default Risk (German Credit tem < 5.000 linhas) |
| [0002](docs/adr/0002-pandera-data-contracts.md) | Data Contracts com Pandera (e não Great Expectations) |
| [0003](docs/adr/0003-modelo-baseline-xgboost.md) | Baseline XGBoost em Pipeline scikit-learn, threshold por KS |

## Vídeo STAR

> _Link a ser adicionado na Etapa 4._

## Equipe

| Integrante | RM | Responsabilidade |
|---|---|---|
| Victor Murad | _preencher_ | Etapa 1 — Validação de dados, contratos e baseline |
| _a definir_ | | Etapa 2 — Drift |
| _a definir_ | | Etapa 3 — Observabilidade |
| _a definir_ | | Etapa 4 — Governança e vídeo |

Convenção de commits e fluxo de trabalho: [`CONTRIBUTING.md`](CONTRIBUTING.md).
