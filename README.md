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

### Objetivo do modelo de crédito

O modelo estima a probabilidade de default (dificuldade de pagamento, `target = 1`) de uma solicitação de crédito pessoal. Um threshold, escolhido pelo ponto de KS máximo na validação, converte a probabilidade em decisão: `aprovado` ou `negado` ([`src/models/predict.py`](src/models/predict.py)). O objetivo de negócio é reduzir a inadimplência da carteira aprovada sem recusar bons pagadores em excesso. Este repositório não busca o melhor modelo possível: ele sustenta um modelo razoável (ROC-AUC 0,749) em produção, com dados confiáveis, degradação visível e decisões auditáveis.

## Sumário

- [Contexto](#contexto)
- [Requisitos obrigatórios do repositório](#requisitos-obrigatórios-do-repositório)
- [Arquitetura e fluxo de dados](#arquitetura-e-fluxo-de-dados)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Tecnologias e bibliotecas](#tecnologias-e-bibliotecas)
- [Como executar](#como-executar)
- [Etapa 1 — Validação de Dados e Contratos](#etapa-1--validação-de-dados-e-contratos)
- [Etapa 2 — Simulação e Detecção de Drift](#etapa-2--simulação-e-detecção-de-drift)
- [Etapa 3 — Observabilidade de Pipelines e Modelos](#etapa-3--observabilidade-de-pipelines-e-modelos)
- [Etapa 4 — Governança, LGPD e Análise Causal](#etapa-4--governança-lgpd-e-análise-causal)
  - [Plano de proteção de dados (LGPD)](#41-plano-de-proteção-de-dados-lgpd)
  - [Mitigação de vieses algorítmicos](#42-mitigação-de-vieses-algorítmicos)
  - [Análise causal da degradação do modelo](#43-análise-causal-da-degradação-do-modelo-ambiguidade-2)
  - [Justificativa técnica das ferramentas](#44-justificativa-técnica-das-ferramentas)
- [Validação de reprodutibilidade](#validação-de-reprodutibilidade)
- [Limitações conhecidas](#limitações-conhecidas)
- [Decisões de arquitetura (ADRs)](#decisões-de-arquitetura-adrs)
- [Vídeo STAR](#vídeo-star)
- [Equipe](#equipe)

## Requisitos obrigatórios do repositório

| # | Requisito | Onde está | Evidência de execução |
| :-: | :--- | :--- | :--- |
| 1 | Pipeline de validação de dados (Data Contracts) configurado | [`src/validation/contract.py`](src/validation/contract.py) (11 regras Pandera), [`src/validation/gate.py`](src/validation/gate.py) (gate fail-closed), [`contracts/credit_application.yaml`](contracts/credit_application.yaml) (camada organizacional) | [`contract_demo_summary.md`](reports/data_contract/contract_demo_summary.md): lote inválido bloqueado, 10 de 10 defeitos detectados; [`invalid_batch_report.html`](reports/data_contract/invalid_batch_report.html) |
| 2 | Scripts de simulação e detecção estatística de Drift | [`src/drift/simulate.py`](src/drift/simulate.py), [`src/drift/detect.py`](src/drift/detect.py) (KS e qui-quadrado via Evidently), [`src/drift/psi.py`](src/drift/psi.py), [`src/drift/performance.py`](src/drift/performance.py) | [`drift_demo_summary.md`](reports/drift/drift_demo_summary.md) e os JSON por fatia em [`reports/drift/`](reports/drift/) |
| 3 | Dashboards ou relatórios HTML comprovando o monitoramento | Relatórios HTML do Evidently e do contrato; MLflow UI local (`make tracking-ui`) | [`production_t1_data_drift_drift_report.html`](reports/drift/production_t1_data_drift_drift_report.html), [`production_t2_concept_drift_drift_report.html`](reports/drift/production_t2_concept_drift_drift_report.html), [`reports/drift/contract_gate/`](reports/drift/contract_gate/), [aceite da Etapa 3](docs/ai/reports/2026-10-27-etapa3-aceite.md) |
| 4 | Documentação consolidada de Governança e LGPD | [Etapa 4](#etapa-4--governança-lgpd-e-análise-causal) deste README, [ADR-0008](docs/adrs/0008-auditoria-de-vies-sem-dependencia-nova.md), [`src/governance/fairness.py`](src/governance/fairness.py) | Testes em [`tests/test_fairness.py`](tests/test_fairness.py) e [`tests/test_repo_hygiene.py`](tests/test_repo_hygiene.py) |
| 5 | Histórico de commits semântico e organizado | [`CONTRIBUTING.md`](CONTRIBUTING.md) (Conventional Commits em PT-BR), [`CHANGELOG.md`](CHANGELOG.md) | `git log`: 77 dos 79 commits até `8368e91` seguem o padrão; os 2 restantes são merges de PR gerados pelo GitHub |

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
              │
              ▼  Etapa 2
 src/drift/simulate.py ──► production_t1 (data) / t2 (data+concept)
 src/drift/detect.py   ──► Evidently vs reference ──► reports/drift/
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
├── docs/adrs/                           # decisões de arquitetura
├── docs/video-star-roteiro.md          # roteiro do vídeo STAR (Etapa 4)
├── models/                             # baseline treinado + metadados (versionados)
│   └── champion/                       #   modelo servido (gerado; ignorado pelo git)
├── reports/
│   ├── data_contract/                  # evidências do contrato (HTML/JSON/MD)
│   ├── model/                          # métricas do baseline
│   ├── drift/                          # evidências de Data/Concept Drift (Etapa 2)
│   └── monitor/, retrain/              # saídas por lote dos comandos (geradas; ignoradas pelo git)
├── src/
│   ├── config.py                       # caminhos, seed, features
│   ├── logging_utils.py                # logs estruturados (JSON lines)
│   ├── data/                           # download + construção dos datasets
│   ├── validation/                     # contrato Pandera, gate, relatórios, demo
│   ├── models/                         # features, treino, predição
│   ├── drift/                          # simulação + detecção de drift (Evidently)
│   ├── monitoring/                     # Etapa 3: estágios rastreados, alerta, MLflow, Registry, monitor
│   └── governance/                     # Etapa 4: auditoria de viés (fairness); saída em reports/governance/
├── dvc.yaml / dvc.lock                 # Etapa 3: pipeline reprodutível (só hashes no git)
├── mlflow.db / mlartifacts/            # NÃO versionados: MLflow local (gerados por `make pipeline`)
└── tests/                              # pytest com dados sintéticos (sem Kaggle)
```

## Tecnologias e bibliotecas

Python 3.12, ambiente travado no [`uv.lock`](uv.lock). A coluna "Uso no código" foi conferida por busca de imports e chamadas em `src/` (2026-10-29); estar declarada no [`pyproject.toml`](pyproject.toml) não basta.

| Biblioteca | Papel | Uso no código |
| :--- | :--- | :--- |
| Pandera | Data Contract executável | `src/validation/contract.py`, `gate.py`, `src/data/prepare.py` |
| Evidently (`<0.8`) | Detecção de drift (KS, qui-quadrado) e relatórios HTML | `src/drift/detect.py` |
| scikit-learn, XGBoost, joblib | Pipeline do modelo, métricas e persistência | `src/models/`, `src/governance/fairness.py` |
| pandas, NumPy, PyArrow | Dados; PyArrow é o motor do Parquet | todo `src/`; `to_parquet`/`read_parquet` |
| MLflow (`>=3,<4`) | Tracking, métricas de saúde, Model Registry | só em `src/monitoring/` (contrato do import-linter) |
| skops | Serialização segura do modelo logado no MLflow | indireto: `mlflow.sklearn.log_model(..., skops_trusted_types=...)` em `src/monitoring/registry.py` |
| DVC (grupo `dev`) | Pipeline reprodutível e hashes de dados | CLI: [`dvc.yaml`](dvc.yaml), `make repro` |
| kaggle (grupo `data`) | Download do dataset | CLI chamada por `subprocess` em `src/data/download.py` |
| pytest, pytest-cov, pytest-xdist, Ruff, import-linter, PyYAML (grupo `dev`) | Qualidade | `tests/`, `make lint`; PyYAML só em `tests/test_contract_spec.py` |

Não há dependência declarada sem uso. SciPy chega como dependência transitiva e não é importada em `src/`. **Great Expectations, Datadog, Prometheus-client e Alibi Detect não estão instalados nem são usados**; a comparação com eles está em [Justificativa técnica das ferramentas](#44-justificativa-técnica-das-ferramentas).

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
#    Num git worktree, não copie o CSV: ligue o arquivo do checkout principal
#    por link simbólico (data/raw/ é ignorado pelo git; o Kaggle proíbe
#    redistribuir o dataset, então ele nunca é versionado):
#       ln -s <checkout-principal>/data/raw/application_train.csv data/raw/application_train.csv

# 3. Pipeline completo das Etapas 1 a 3, rastreado no MLflow local
make pipeline                            # prepare -> train -> contract_demo -> drift_demo
make tracking-ui                         # MLflow UI em http://127.0.0.1:5001

# 4. Testes
uv run pytest
```

| `make`                              | Equivalente sem make                                                 |
| ------------------------------------- | -------------------------------------------------------------------- |
| `make setup`                        | `uv sync`                                                          |
| `make download-data`                | `uv run python -m src.data.download`                               |
| `make pipeline`                     | os quatro estágios abaixo, em ordem, com o mesmo `PIPELINE_RUN_ID` exportado |
| `make data`                         | `uv run python -m src.monitoring.stage prepare`                    |
| `make train`                        | `uv run python -m src.monitoring.stage train`                      |
| `make contract-demo`                | `uv run python -m src.monitoring.stage contract_demo`              |
| `make demo-drift` / `make drift-demo` / `make etapa2` | `uv run python -m src.monitoring.stage drift_demo` |
| `make etapa1`                       | `make data train contract-demo`                                    |
| `make monitor BATCH=arq.parquet`    | `uv run python -m src.monitoring.monitor --batch arq.parquet [--baseline-auc 0.75]` |
| `make retrain [ARGS=...]`           | `uv run python -m src.monitoring.retrain [--alert-run <id>]`       |
| `make promote [ARGS=...]`           | `uv run python -m src.monitoring.promote [--version N] [--override --reason "..."]` |
| `make rollback [ARGS=...]`          | `uv run python -m src.monitoring.promote rollback [--reason "..."]` |
| `make tracking-ui`                  | `uv run mlflow ui --backend-store-uri sqlite:///<caminho absoluto>/mlflow.db --port 5001` |
| `make repro`                        | `uv run dvc repro`                                                 |
| `make validate INPUT=arq.csv`       | `uv run python -m src.validation.validate --input arq.csv`         |
| `make predict INPUT=arq.csv`        | `uv run python -m src.models.predict --input arq.csv`              |
| `make simulate-drift`               | `uv run python -m src.drift.simulate`                              |
| `make detect-drift CURRENT=...`     | `uv run python -m src.drift.detect --current ... [--min-psi 0.10]` |
| `make fairness [BATCH=...]`         | `uv run python -m src.governance.fairness [--batch ...]`           |
| `make test`                         | `uv run pytest`                                                    |

Os módulos das Etapas 1 e 2 continuam executáveis sem rastreamento (`uv run python -m src.data.prepare`, `src.models.train`, `src.validation.demo`, `src.drift.demo`). Os estágios de `src.monitoring.stage` envolvem esses mesmos módulos e registram tudo no MLflow; `MONITORING_TRACKING=off` desliga o registro.

Qualidade: `make lint` (Ruff + import-linter) e `make test-cov` (cobertura ≥ 80%, hoje ~97%). Os testes usam dados sintéticos e rodam o pipeline inteiro em sandbox, sem Kaggle e sem tocar no `mlflow.db` real. A exceção é o teste de aceite real (`tests/integration/test_acceptance_real.py`). Ele só lê o `mlflow.db` da raiz e pula quando falta o banco ou o CSV, ou quando roda em CI.

Reprodutibilidade: `SEED = 42` é fixada em todos os passos (`src/config.py`) e as dependências estão travadas no `uv.lock`.

### Problemas comuns no Windows

| Sintoma                                                                                                 | Causa                                                                        | Solução                                                                                                |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `Uma política de Controle de Aplicativo bloqueou este arquivo` (os error 4551 / `DLL load failed`) | O **Smart App Control** do Windows 11 bloqueia as DLLs do pandas/numpy | Desligar em *Segurança do Windows → Controle de aplicativos e do navegador → Smart App Control*     |
| `ModuleNotFoundError: sklearn...` mesmo com o arquivo existindo                                       | O caminho passa de 260 caracteres (limite do Windows)                        | Clonar o repositório numa pasta de caminho curto (ex.: `C:\dev\`) ou ativar *Long Paths* no Windows |
| `403 Forbidden` no download do Kaggle                                                                 | As regras da competição não foram aceitas                                 | Clicar em *Late Submission → I Understand and Accept* na página da competição                      |
| `XGBoostError: input stream corrupted` ao carregar um `.joblib` antigo                                  | Bundle gravado antes da correção de portabilidade: o pickle do booster do XGBoost guarda o estado interno, que não é portável entre sistemas operacionais. Desde então o booster é gravado no formato de modelo do XGBoost ([`src/models/portable.py`](src/models/portable.py), nota na [ADR-0003](docs/adrs/0003-modelo-baseline-xgboost.md)) | Retreinar com `make pipeline` (ou `uv run python -m src.monitoring.stage train`) |
| `make: command not found`                                                                               | O Windows não traz o GNU make                                                | Usar os equivalentes `uv run ...` da tabela acima                                                         |

## Etapa 1 — Validação de Dados e Contratos

> Foco: garantir que lixo não entre no modelo. Detalhes, evidências e o checklist 1.1–1.4 em [`docs/etapa-1-validacao-de-dados.md`](docs/etapa-1-validacao-de-dados.md).

**Entregável do PDF: "Script de validação de dados operante e modelo baseline treinado".**

- **Data Contract (Pandera)**: [`src/validation/contract.py`](src/validation/contract.py), com 11 regras rígidas: unicidade, completude, idade 18–100, limites de renda e de crédito, domínios categóricos, scores em [0, 1], parcela ≤ crédito, consistência de emprego, `strict` (bloqueia colunas não previstas, como CPF) e tipos.
- **Pipeline de validação (gate)**: [`src/validation/gate.py`](src/validation/gate.py). É executado antes do treino e antes de toda inferência. É fail-closed: o lote vai para quarentena e o gate gera relatório e log.
- **Demonstração de bloqueio**: [`reports/data_contract/contract_demo_summary.md`](reports/data_contract/contract_demo_summary.md) e [`invalid_batch_report.html`](reports/data_contract/invalid_batch_report.html).
- **Modelo baseline**: [`models/baseline_model.joblib`](models/) e as métricas em [`reports/model/baseline_metrics.md`](reports/model/baseline_metrics.md).

**Resultados:**

|                                           |                                                                                                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Registros brutos rejeitados pelo contrato | 21 de 307.511 (renda de 117 mi, gênero `XNA`, parcela nula…)                                                                     |
| Dataset de Referência / Produção t0    | 215.243 / 92.247 linhas (default 8,07% em ambos)                                                                                     |
| Baseline XGBoost (teste)                  | ROC-AUC **0,749** · Gini **0,499** · KS **0,370** (Regressão Logística: AUC 0,731)                             |
| Impacto no negócio                       | aprova 65% dos pedidos, e a inadimplência da carteira cai de 8,07% para **3,88%**                                             |
| Demonstração do contrato                | lote válido **APROVADO** (500 pontuados); lote sujo **BLOQUEADO** (0 pontuados); **10 de 10** defeitos detectados |

## Etapa 2 — Simulação e Detecção de Drift

> Foco: tornar a degradação silenciosa visível. Detalhes e checklist 2.1–2.4 em [`docs/etapa-2-simulacao-e-deteccao-de-drift.md`](docs/etapa-2-simulacao-e-deteccao-de-drift.md).

**Entregável: simulação de drift operante + relatório de detecção (Data Drift e Concept Drift).**

- **Simulação:** [`src/drift/simulate.py`](src/drift/simulate.py) — a partir de `production_base.parquet`, gera t1 (data drift em `annuity_amount`, `ext_source_3`, `income_total`) e t2 (data + concept drift no `target`), com clip para continuar passando no contrato.
- **Detecção (Evidently):** [`src/drift/detect.py`](src/drift/detect.py) — compara cada fatia com a Referência (`DataDriftPreset`, KS / chi-square). [ADR-0004](docs/adrs/0004-evidently-drift-detection.md). Nos JSON, o campo `statistic` de cada coluna é o **p-valor** do teste (drift quando p < 0,05), conferido contra `scipy.stats.ks_2samp` na [validação de reprodutibilidade](#validação-de-reprodutibilidade).
- **Tamanho do efeito (PSI):** [`src/drift/psi.py`](src/drift/psi.py) — PSI em toda coluna (< 0,10 estável · 0,10–0,25 moderado · ≥ 0,25 significativo). `--min-psi 0.10` só marca drift quando o KS é significativo **e** o efeito é relevante, evitando falso alerta em amostra grande. [ADR-0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md).
- **Concept drift por performance:** [`src/drift/performance.py`](src/drift/performance.py) — queda de ROC-AUC vs. baseline (alerta a partir de 0,05).
- **Demo ponta a ponta:** [`src/drift/demo.py`](src/drift/demo.py) / `make etapa2` — gate → score → Evidently → [`reports/drift/drift_demo_summary.md`](reports/drift/drift_demo_summary.md).

**Resultados (Home Credit, seed 42, 92.247 linhas):**

|                      |                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------ |
| Gate em t0 / t1 / t2 | todos **APROVADOS** (drift ≠ dado inválido)                                        |
| Evidently t0         | sem dataset drift (AUC **0,753**, alinhado ao baseline)                              |
| Evidently t1         | **Data Drift** em `annuity_amount` (PSI 0,17), `ext_source_3` (0,97), `income_total` (0,76); AUC 0,722 |
| Evidently t2         | mesmo Data Drift; label shift em `target` com efeito desprezível (PSI 0,02); **queda de AUC 0,180 → modelo degradado** (AUC **0,573**) |
| Impacto no negócio  | default na carteira aprovada: 3,76% (t0) → **9,82%** (t2)                           |
| Lição              | t1 detecta drift com queda leve de AUC; t2 (concept) degrada de fato                       |

> **Limitação conhecida (concept drift):** o teste do Evidently sobre `target` mede mudança de *prevalência* do rótulo (label/prior shift), não mudança em P(y|X); por isso o campo é `label_shift` em `DriftResult` e nos relatórios JSON (o antigo `concept_drift` segue como alias obsoleto; a remoção foi adiada para a próxima regeneração dos relatórios com o dataset, ver a 2ª emenda da [ADR-0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md)). O sinal confiável de concept drift é a **queda de ROC-AUC** com rótulos de feedback (`performance` no sumário da demo). Sem rótulos (inferência pura) só é possível monitorar data drift e PSI. Em t0, o KS sozinho marca `annuity_amount` por ruído de amostra grande (PSI 0,00); a demo usa `min_psi=0,10` e, com isso, t0 fica sem nenhuma coluna marcada. Pelo mesmo critério, o teste sobre `target` em t2 deixa de alertar (PSI 0,02) enquanto o AUC cai 0,181: é a prova de que só a performance revela o concept drift.

## Etapa 3 — Observabilidade de Pipelines e Modelos

> Foco: um lugar único para ver a saúde do pipeline, a estabilidade da execução e os alertas de degradação do modelo. Desenho em [`docs/ai/specs/2026-10-25-etapa3-observabilidade-design.md`](docs/ai/specs/2026-10-25-etapa3-observabilidade-design.md); decisão em [ADR-0006](docs/adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md).

**Entregável: ambiente de rastreamento (MLflow) com a saúde do pipeline, a estabilidade da execução e os alertas de degradação do modelo.** A extensão de retreino, promoção e rollback (§4.7 da spec, [ADR-0007](docs/adrs/0007-retreino-registry-e-promocao-humana.md)) está implementada e foi aceita com os dados reais (relatório em [`docs/ai/reports/2026-10-27-etapa3-aceite.md`](docs/ai/reports/2026-10-27-etapa3-aceite.md)): o challenger real ficou elegível, foi promovido e revertido com auditoria. A repromoção da v3 exige `--override` e fica com o usuário.

- **MLflow local:** o banco SQLite fica em `mlflow.db` e os artefatos em `mlartifacts/`, os dois na raiz do repositório e fora do git. O experimento se chama `credit-drift-monitor`. Não há servidor remoto.
- **Runs:** cada estágio (`prepare`, `train`, `contract_demo`, `drift_demo`) gera um run de estágio com a tag `pipeline_run_id` e o trecho do `logs/pipeline.jsonl` daquele `run_id` anexado como artefato. Cada lote avaliado gera um run filho com nível, motivos, ação, métricas e os relatórios do Evidently e do contrato.
- **Métricas de saúde (checklist 3.3):**
  - status: `stage_success` e a tag `stage_status`;
  - latência: `stage_duration_s`, mais o `duration_ms` dos eventos de contrato e de detecção no log anexado;
  - volume: `rows_processed` e `rows_rejected`;
  - alertas: `alerts_failure_count` e `alerts_drift_count`;
  - modelo: `roc_auc`, `auc_drop`, `approval_rate`, `default_rate_approved` e `score_psi`;
  - estabilidade: `peak_memory_mb` e `stage_duration_s`. "Infraestrutura" aqui é o processo local: não há cluster, e essa limitação é assumida.
- **Model Registry:** o modelo `credit-default-xgboost` recebe o alias `champion` no primeiro `train`. O modelo servido fica em `models/champion/`, conferido pelo sha256 da versão `champion`. Um `train` repetido registra uma versão nova, mas não move o alias. O retreino acrescenta o alias `challenger`, e a promoção dá o alias `previous` ao champion anterior (ver [Retreino, promoção e rollback](#retreino-promoção-e-rollback-extensão-47)).

### Nível de alerta por lote

A regra é uma função pura, em [`src/monitoring/alerts.py`](src/monitoring/alerts.py). A precedência é `BLOQUEADO` > `CRÍTICO` > `ATENÇÃO` > `OK`. Nas tags do MLflow os códigos são ASCII: `OK`, `ATENCAO`, `CRITICO` e `BLOQUEADO`.

| Nível | Regra | Ação |
| :--- | :--- | :--- |
| `BLOQUEADO` | o contrato barrou o lote | inspecionar o relatório do contrato e a quarentena; corrigir a fonte; não pontuar |
| `CRÍTICO` | lote rotulado com queda de ROC-AUC ≥ 0,05 frente ao t0 | tratar como concept drift: investigar e avaliar retreino com validação humana |
| `ATENÇÃO` | alguma coluna com drift (p-valor **e** PSI ≥ 0,10, sem contar `target`) **ou** PSI do score ≥ 0,10 | investigar a coluna; um data drift sozinho não prova piora do modelo |
| `OK` | nenhuma das anteriores | nenhuma |

**Lote sem rótulo, ou com uma só classe (`single_class`):** a AUC não é avaliada (`performance_evaluated=false`), e o nível máximo é `ATENÇÃO`.

**Piso de `ATENÇÃO` para anomalias:** um lote vazio (`empty_batch`), uma AUC não finita (`auc_undefined`) ou um PSI do score não finito (`score_psi_undefined`) nunca terminam `OK`. Eles sobem no mínimo a `ATENÇÃO`, com o motivo nas `alert_reasons`. É um máximo: o piso nunca rebaixa um nível maior. `no_baseline_auc` (lote rotulado sem AUC de referência) e `single_class` entram nos motivos sem piso.

### Resultado real (Home Credit, seed 42, `pipeline_run_id` `20261030T024043Z`)

| Lote | Nível | Motivos | ROC-AUC | Queda de AUC | PSI do score |
| :--- | :--- | :--- | ---: | ---: | ---: |
| t0 (`production_t0`) | `OK` | nenhum | 0,7534 | 0,0000 | 0,0003 |
| t1 (`production_t1_data_drift`) | `ATENÇÃO` | `drift:annuity_amount`, `drift:ext_source_3`, `drift:income_total`, `score_psi>=0.10` | 0,7223 | 0,0311 | 0,1455 |
| t2 (`production_t2_concept_drift`) | `CRÍTICO` | `auc_drop>=0.05` e os mesmos de t1 | 0,5733 | 0,1801 | 0,1455 |
| lote inválido da demo do contrato | `BLOQUEADO` | as regras violadas (R1 a R8 e R10); `intentional_demo=true` | não avaliada | | |

No estágio `contract_demo`, `alerts_failure_count` = 1 e `rows_rejected` = 508. No `drift_demo`, `alerts_drift_count` = 2.

### Comandos

```bash
make pipeline                  # os 4 estágios, sempre executados, com um único PIPELINE_RUN_ID
make tracking-ui               # MLflow UI em http://127.0.0.1:5001 (a 5000 colide com o AirPlay no macOS)
make monitor BATCH=data/production/production_t2_concept_drift.parquet [BASELINE_AUC=0.7534]
make demo-drift                # só o estágio drift_demo (apelidos: drift-demo, etapa2)
make retrain                   # treina o challenger a partir do alerta CRITICO (nunca promove)
make promote                   # comando humano: promove o challenger, se elegível
make rollback                  # comando humano: devolve o champion à versão previous
make repro                     # dvc repro: roda só os estágios com mudança
```

O `monitor` avalia **um** lote: contrato, pontuação, drift, PSI do score, AUC (se o lote for rotulado), nível de alerta e registro no MLflow. A AUC de referência vem do run `production_t0` do mesmo modelo no MLflow, ou de `--baseline-auc`. Um lote rotulado sem nenhuma das duas falha fechado.

> **Código de saída do monitor:** o próprio monitor sai com 0 (`OK`/`ATENÇÃO`), 2 (`CRÍTICO`), 3 (`BLOQUEADO`) ou 1 (erro operacional). O `make monitor` **não repassa** esse código: o GNU make sai com 2 em qualquer falha de receita. O código real aparece só no texto (`make: *** [monitor] Error 3` e a linha `monitor: codigo de saida 3 ...`). O mesmo vale para `make retrain`, `make promote` e `make rollback`. O código real do módulo aparece em `Error N` e na linha `codigo de saida`; o do `make` é sempre 2, inclusive no erro de `BATCH` ausente. Para usar num script, chame o módulo direto:
>
> ```bash
> uv run python -m src.monitoring.monitor --batch data/production/production_t2_concept_drift.parquet
> echo $?   # 2 (CRÍTICO)
> ```

> **Variáveis de ambiente:** `MLFLOW_TRACKING_URI` e `MLFLOW_ARTIFACT_ROOT` trocam o banco e o diretório de artefatos (o padrão é `mlflow.db` e `mlartifacts/` na raiz, com caminho absoluto). Se uma delas estiver exportada no shell, o pipeline grava em outro lugar. O teste de aceite real (`tests/integration/test_acceptance_real.py`) sempre lê o `mlflow.db` da raiz, então ele deixa de enxergar a execução ou enxerga uma antiga. Antes de rodar `make pipeline` e o aceite, confira que as duas não estão definidas. `MONITORING_TRACKING=off` desliga o rastreamento; sem ele, um MLflow indisponível derruba o estágio com mensagem clara.

### Retreino, promoção e rollback (extensão §4.7)

Decisão completa, com os limiares pré-registrados, na [ADR-0007](docs/adrs/0007-retreino-registry-e-promocao-humana.md). Os três comandos exigem o rastreamento ligado: com `MONITORING_TRACKING=off` eles falham fechado, com código 1.

**Fluxo:**

1. **Alerta.** `make pipeline` (ou `make demo-drift`, ou `make monitor BATCH=<lote rotulado>`) grava um run de lote `CRITICO` quando a queda de AUC passa de 0,05. Isso é automático.
2. **Retreino.** `make retrain` treina o challenger com o treino da referência mais metade estratificada do lote `CRITICO`. A outra metade é o holdout de comparação e nunca entra no treino. O challenger vira uma versão nova no Registry, com o alias `challenger`. O comando imprime o veredito da regra com os números, mas **nunca move o `champion`**.
3. **Promoção.** `make promote` é uma decisão humana. O comando reavalia a regra, confere o sha256 do artefato, move `previous` e `champion`, troca `models/champion/` de forma atômica, reconfere e grava um run de auditoria (quem, quando, por quê).
4. **Rollback.** `make rollback`, também humano, devolve o `champion` à versão `previous`, reexporta o arquivo dela e grava outro run de auditoria.

| Etapa | Quem age | O que exige |
| :--- | :--- | :--- |
| Alerta `CRITICO` | automático (monitor e demo) | lote rotulado com queda de AUC ≥ 0,05 |
| Retreino | comando explícito (`make retrain`) | um alerta `CRITICO` não consumido, emitido contra o champion atual |
| Promoção | **pessoa** (`make promote`) | challenger elegível pela regra, ou `--override` com `--reason` |
| Rollback | **pessoa** (`make rollback`) | uma versão `previous` |

**Regra de promoção** (`src/monitoring/promotion.py`): o challenger é elegível se passa no contrato e na integridade, se a AUC dele no holdout supera a do champion em pelo menos `PROMOTION_MIN_GAIN` = 0,02 e se, no teste da referência, a AUC dele não fica abaixo da do champion em mais de `PROMOTION_MAX_REGRESSION` = 0,02. Os dois limiares foram fixados antes do retreino real e não se ajustam depois de ver o resultado.

**Argumentos (`ARGS=`):** os alvos repassam `ARGS` ao módulo. Aspas dentro dele valem.

```bash
make retrain ARGS="--alert-run <run_id>"                          # um alerta específico
make retrain ARGS="--force-no-alert --reason 'motivo'"            # sem alerta (exceção registrada)
make retrain ARGS="--alert-run <run_id> --allow-reuse --reason 'motivo'"  # alerta já consumido
make promote ARGS="--version 3"                                   # outra versão que não o challenger
make promote ARGS="--override --reason 'motivo'"                  # promove inelegível (auditado)
make rollback ARGS="--reason 'motivo'"
```

| Comando | 0 | 2 | 3 | 1 |
| :--- | :--- | :--- | :--- | :--- |
| `retrain` | challenger registrado (elegível ou não) | recusa: sem alerta `CRITICO` válido | o contrato barrou a referência ou o lote | erro operacional |
| `promote` | promovido e estado `CONSISTENT` | recusa: challenger inelegível sem `--override` | | erro operacional (inclui troca interrompida, `INCONSISTENT`) |
| `rollback` | revertido e estado `CONSISTENT` | | | erro operacional (inclui sem `previous`) |

- `models/baseline_model.joblib` é só o artefato do baseline: promoção e rollback **nunca** o tocam. O modelo servido é o de `models/champion/`, e um `train` repetido não move o `champion`.
- Depois de uma promoção ou de um rollback, rode `make demo-drift` antes de usar o `make monitor` com um lote rotulado. O baseline de AUC do monitor é o run `production_t0` do mesmo `model_version`, e cada challenger tem um `model_version` próprio (`retrain-<sha256[:12]>`). Sem esse baseline, o monitor falha fechado; a alternativa é passar `BASELINE_AUC=`.

### DVC: reprodução local

- [`dvc.yaml`](dvc.yaml) tem os estágios `prepare → train → {contract_demo, drift_demo}`. Cada estágio chama `uv run python -m src.monitoring.stage <estágio>` e declara os `outs` arquivo por arquivo.
- `models/` e `reports/` são `outs` com `cache: false`, então continuam versionados no git (exceto `models/champion/`, `reports/monitor/` e `reports/retrain/`, que são estado local ignorado). Os outs de `data/` vão para o cache do DVC, que fica fora do git.
- O CSV bruto é `deps` do `prepare`, e o hash vai para o [`dvc.lock`](dvc.lock), que é versionado (só hashes e tamanhos). Não se usa `dvc add`.
- `make repro` (`uv run dvc repro`) roda só os estágios com mudança. Cada estágio recebe um `PIPELINE_RUN_ID` próprio. Sem mudança, não gera runs novos. Para a demonstração completa no MLflow, use `make pipeline`. Os relatórios têm timestamp, então o `dvc.lock` e `reports/` mudam a cada rodada. Isso é esperado.
- **Remoto local (opcional), só em `.dvc/config.local`:**

  ```bash
  uv run dvc remote add --local -d local ~/dvc-remotes/credit-drift-monitor
  ```

  A flag `--local` grava em `.dvc/config.local`, que o `.gitignore` deixa fora do repositório. O versionamento de dados fica só na máquina do usuário, para a demo: o Kaggle proíbe redistribuir o dataset, e um remoto externo (S3, Drive, SSH) seria redistribuição e exigiria credenciais (ADR-0006). Nunca faça `dvc push` para um serviço externo.

> **Champion e `dvc repro`:** o champion servido fica em `models/champion/`, fora dos `outs` do DVC, porque promoção e rollback o reescrevem. Os dois arquivos dele (`baseline_model.joblib` e `baseline_metadata.json`) são `deps` de `contract_demo` e `drift_demo`. Assim, depois de uma promoção ou de um rollback, o `dvc repro` marca as duas demos como desatualizadas e as reexecuta com o champion novo. Limitação: o champion só nasce no bootstrap do `train` com o rastreamento ligado. Com `MONITORING_TRACKING=off`, o `dvc repro` para nessas demos por dependência ausente. Nesse caso, use `make pipeline`.

## Etapa 4 — Governança, LGPD e Análise Causal

> Foco: documentar como o projeto trata dados pessoais, em que base legal a decisão de crédito se apoia, como o viés é medido e por que o modelo degradou. **Convenção desta seção:** **[implementado]** marca o que existe no código ou no repositório e foi verificado; **[recomendação]** marca uma proposta que ainda não está operacional; **[validação jurídica]** marca uma premissa que depende de parecer jurídico. Este texto não substitui o parecer do encarregado (DPO) nem do jurídico.

**Escopo.** Este é um projeto acadêmico sobre um dataset público (Home Credit, Kaggle). Para fins exclusivamente acadêmicos, a LGPD se aplica de forma restrita (art. 4º, II, "b": ainda valem os arts. 7º e 11). O plano abaixo foi escrito para o cenário do enunciado, uma fintech que opera este modelo em produção, e aponta onde o repositório já atende a esse cenário.

### 4.1 Plano de proteção de dados (LGPD)

#### A) Dados pessoais tratados (PII)

O projeto lê **21 das 122 colunas** do CSV bruto (`usecols` em [`src/data/prepare.py`](src/data/prepare.py); lista em `RAW_COLUMN_MAP`, [`src/config.py`](src/config.py)). Todas as linhas se referem a pessoas naturais, então todo campo abaixo é dado pessoal (art. 5º, I). A coluna `pii` do contrato marca só os campos de maior risco.

| Campo | Natureza | Classificação no contrato | Uso |
| :--- | :--- | :--- | :--- |
| `client_id` | Identificador pseudônimo (`SK_ID_CURR`, já pseudonimizado pela origem) | `pii`, `restricted` | Unicidade (R1) e junção na auditoria; **não é feature** |
| `gender` | Dado pessoal. Não é sensível pelo art. 5º, II, mas é protegido contra discriminação (art. 6º, IX) | `pii`, `restricted` | **Só auditoria de viés**; excluído do modelo em `model_input` ([`src/models/features.py`](src/models/features.py)) |
| `age_years` | Derivado de `DAYS_BIRTH`: a data de nascimento não é guardada, só a idade em anos | `pii`, `confidential` | Feature do modelo e atributo auditado |
| `income_total`, `credit_amount`, `annuity_amount`, `goods_price` | Dados financeiros | `confidential` (renda também `pii`) | Features |
| `ext_source_2`, `ext_source_3` | Scores de bureaus externos (dados de terceiros) | `confidential` | Features (as duas mais importantes do modelo) |
| `income_type`, `education_type`, `family_status`, `housing_type`, `children_count`, `family_members`, `own_car`, `own_realty`, `region_rating`, `contract_type` | Perfil socioeconômico | sem marcação | Features |
| `target` | Histórico de adimplência | sem marcação | Rótulo (treino e feedback) |

**Ponto de atenção [validação jurídica]:** `income_type = "Maternity leave"` revela licença-maternidade, informação próxima de dado referente à saúde (gestação). Se o jurídico a classificar como dado sensível, o art. 11 passa a se aplicar a essa categoria.

#### B) Tratamento em cada fase

| Fase | O que acontece com os dados pessoais | Status |
| :--- | :--- | :--- |
| Ingestão | Só as 21 colunas necessárias são lidas (minimização, art. 6º, III). O token do Kaggle fica fora do repositório (`.gitignore`: `access_token`, `kaggle.json`, `.env`). | [implementado] |
| Processamento | A data de nascimento relativa vira idade em anos; o sentinela de emprego vira nulo. O contrato é `strict` (R10) e **barra colunas não previstas**: a demo injeta uma coluna `cpf` e o lote é bloqueado. | [implementado] |
| Treinamento | `model_input` descarta `client_id`, `gender` e `target`. O artefato do modelo guarda só parâmetros e estatísticas agregadas (medianas do imputador, categorias, distribuição do score), nunca linhas. | [implementado] |
| Validação | Um lote rejeitado é copiado inteiro para `data/quarantine/` (fora do git). O relatório do contrato traz até 5 exemplos de valores por regra violada, incluindo `client_id`. | [implementado]; risco abaixo |
| Monitoramento | O log JSONL (`logs/pipeline.jsonl`, fora do git) guarda contagens, regras e duração, sem valores de registros (conferido no `log_event` do gate). Os relatórios do Evidently cobrem 7 colunas monitoradas, sem `client_id`. O MLflow local recebe esses relatórios e o trecho do log. | [implementado] |
| Versionamento | `data/` nunca entra no git (teste `test_nada_de_data_no_indice_do_git`). O DVC usa só remoto local, sem `dvc push` externo ([ADR-0006](docs/adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md)). Os relatórios versionados não podem trazer caminho absoluto com nome de usuário (teste `test_relatorios_versionados_sem_caminho_absoluto_de_usuario`). | [implementado] |

**Anonimização, pseudonimização e mascaramento.** O projeto **não anonimiza** dados: idade, renda e perfil continuam associados a um registro. `client_id` chega pseudonimizado da origem, mas o projeto não aplica uma segunda pseudonimização. Pela LGPD, dado pseudonimizado continua sendo dado pessoal. Também não há mascaramento nos relatórios: os exemplos de falha aparecem em claro.

**Riscos identificados**

| Risco | Situação | Medida |
| :--- | :--- | :--- |
| Quarentena com o lote bruto completo e sem prazo de expurgo | aberto | [recomendação] expurgo automático (ver retenção) |
| Exemplos de `client_id` e de valores nos relatórios do contrato versionados (`invalid_batch_report.*`) | aberto | [recomendação] mascarar exemplos de campos `pii: true` em `src/validation/report.py` e regenerar as evidências com o dataset |
| Caminho local com nome de usuário em `reference_build_report.json` (vinha de symlink de worktree) | **corrigido nesta etapa** | `project_relative` não segue o symlink; campo corrigido; teste de regressão |
| Reidentificação por quase-identificadores (idade, gênero, família, região) | baixo (dataset público já pseudonimizado) | [recomendação] não publicar linhas; agregar relatórios |
| Desserialização de pickle malicioso | mitigado | sha256 conferido antes de carregar ([`src/models/integrity.py`](src/models/integrity.py)); skops no MLflow |
| Credenciais da URI do MLflow em logs | mitigado | `redact_uri` em [`src/monitoring/tracking.py`](src/monitoring/tracking.py) |
| `mlflow.db` e `mlartifacts/` locais sem criptografia | aceito na demo | [recomendação] disco criptografado e controle de acesso em produção |

**Medidas organizacionais [recomendação]:** encarregado (DPO) nomeado; registro das operações de tratamento (art. 37); relatório de impacto (RIPD, art. 38) antes de produção, dado o alto impacto da decisão de crédito; acesso a `data/`, quarentena e MLflow restrito por papel; treinamento da equipe; plano de resposta a incidentes (art. 48).

#### C) Base legal para a decisão de crédito

A **base legal do tratamento** responde "posso tratar estes dados?" (arts. 7º e 11). A **justificativa da decisão automatizada** responde "como devo decidir, explicar e revisar?" (art. 20 e princípios do art. 6º). São obrigações diferentes, e uma base legal válida não dispensa os deveres do art. 20.

**Hipóteses do art. 7º aplicáveis** (a escolha final é do controlador, com o jurídico):

| Hipótese | Aplicação | Premissa / ponto em aberto |
| :--- | :--- | :--- |
| **Art. 7º, X — proteção do crédito** | Hipótese principal para pontuar o risco do solicitante. É a base declarada em `terms.usage` do [contrato](contracts/credit_application.yaml). | [validação jurídica] alinhamento com a Lei 12.414/2011 (Cadastro Positivo) e com o CDC, art. 43, sobre o uso de dados de bureau (`ext_source_*`) |
| Art. 7º, V — procedimentos preliminares de contrato, a pedido do titular | Tratamento da própria solicitação que o cliente enviou | Complementar à X; não cobre usos secundários |
| Art. 7º, IX — legítimo interesse | Possível para usos secundários: monitoramento de drift, retreino e auditoria de viés | [validação jurídica] exige teste de balanceamento (art. 10) e transparência. Alternativa: tratar o monitoramento como parte compatível da finalidade de proteção ao crédito |
| Art. 7º, II — obrigação legal ou regulatória | Gestão de risco de crédito exigida de instituições reguladas pelo Banco Central | [validação jurídica] depende do tipo de licença da fintech e da norma aplicável |
| Art. 7º, I — consentimento | **Não recomendado** como base principal: é revogável a qualquer momento, e a assimetria entre fintech e solicitante enfraquece a liberdade do consentimento | |

**Art. 11 (dados sensíveis):** nenhum campo atual está entre as categorias do art. 5º, II (origem racial ou étnica, religião, opinião política, filiação sindical, saúde, vida sexual, dado genético ou biométrico). Por isso o art. 11 **não se aplica hoje**, com a ressalva sobre "Maternity leave" acima. Qualquer feature nova dessas categorias exige reavaliar a base legal antes de entrar no contrato. O `strict=True` (R10) impede que ela entre sem uma mudança explícita e versionada do contrato.

**Art. 20 (decisão unicamente automatizada).** A decisão em [`src/models/predict.py`](src/models/predict.py) é unicamente automatizada (score ≥ threshold → `negado`), então o art. 20 se aplica.

| Dever | Situação |
| :--- | :--- |
| Direito de solicitar revisão (caput) | [recomendação] canal de revisão, com análise humana das negativas contestadas. Hoje a revisão humana existe só no nível do **modelo** (promoção e rollback, [ADR-0007](docs/adrs/0007-retreino-registry-e-promocao-humana.md)), não da **decisão individual** |
| Informações claras sobre critérios e procedimentos (§1º), respeitado o segredo comercial | Parcial. [implementado] importância global das variáveis em [`baseline_metrics.md`](reports/model/baseline_metrics.md), regras do contrato, threshold e versão do modelo no metadata. [recomendação] motivos individuais (*reason codes*) por decisão |
| Possibilidade de auditoria pela ANPD (§2º) | [implementado] rastreabilidade: sha256 do modelo, versão do contrato, runs e auditoria de promoção no MLflow, `dvc.lock` e histórico do git |
| Não discriminação (art. 6º, IX) | [implementado] `gender` fora do modelo e auditoria executável ([4.2](#42-mitigação-de-vieses-algorítmicos)) |

**Princípios do art. 6º mapeados**

| Princípio | Evidência no projeto |
| :--- | :--- |
| Finalidade e adequação (I, II) | `terms.usage` e `terms.limitations` no contrato (proíbem marketing, perfilamento fora do crédito e reidentificação) |
| Necessidade (III) | 21 de 122 colunas; `client_id` e `gender` fora do modelo |
| Qualidade dos dados (V) | Data Contract com 11 regras, fail-closed, executado antes do treino e de toda inferência |
| Transparência (VI) | README, ADRs, relatórios HTML; *reason codes* individuais pendentes |
| Segurança (VII) | Segredos fora do git, sha256 antes de desserializar, redação de URIs, sem remoto externo |
| Prevenção (VIII) | Gate bloqueia dado inválido; alertas `ATENÇÃO` e `CRÍTICO` antes que a degradação chegue à carteira |
| Não discriminação (IX) | Auditoria de viés por gênero e faixa etária |
| Responsabilização (X) | Runs de auditoria no MLflow, promoção humana, histórico de commits semântico |

#### D) Política de retenção de dados

**Não existe política de retenção implementada.** O único controle automático é a rotação do log por **volume** (10 MB × 5 arquivos, [`src/logging_utils.py`](src/logging_utils.py)). A tabela separa o estado atual da proposta; os prazos propostos são **[recomendação]** e não estão operacionais.

| Artefato | Contém dado pessoal? | Finalidade | Hoje | Proposta [recomendação] |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/application_train.csv` | sim (122 colunas) | fonte | fora do git, sem prazo | apagar depois de gerar referência e produção; nunca redistribuir (regra do Kaggle) |
| `data/reference/`, `data/production/` | sim (21 colunas) | treino, simulação e monitoramento | fora do git; cache local do DVC | manter só as versões que treinaram o `champion` e o `previous`; apagar as demais na troca de champion, depois da janela de auditoria |
| `data/quarantine/` | sim (lote bruto rejeitado) | investigar incidente de qualidade | sem prazo, sem expurgo | apagar 30 dias depois de fechado o incidente |
| `data/samples/` | sim (derivado de registros reais) | demonstração | sem prazo | regenerável; apagar ao fim da demonstração |
| `logs/pipeline.jsonl` | não traz valores de registros | observabilidade | rotação por volume | somar um prazo de 90 dias |
| `mlflow.db`, `mlartifacts/` | em parte (exemplos nos relatórios do contrato) | rastreabilidade e auditoria | sem prazo | manter runs de modelos promovidos e de auditoria; expurgar runs de lote antigos (`mlflow gc`) |
| `models/` | não (parâmetros e agregados) | inferência | versionado no git | manter enquanto em uso, mais a janela de auditoria |
| `reports/` versionados | em parte (exemplos de falha) | evidência | git (histórico permanente) | mascarar exemplos antes de versionar: apagar do git exige reescrever o histórico |

**Prazos legais em produção [validação jurídica]:** o CDC (art. 43, §1º) limita informações negativas a 5 anos, e a Lei 12.414/2011 (art. 14) limita as de adimplemento a 15 anos. A guarda para defesa em processos e as normas do regulador também definem prazos mínimos.

**Descarte seguro [recomendação]:** apagar os arquivos de `data/` e limpar o cache do DVC (`dvc gc`); remover os runs no MLflow e rodar `mlflow gc`; em disco não criptografado, usar sobrescrita segura; registrar o descarte (data, artefato, responsável).

### 4.2 Mitigação de vieses algorítmicos

**O que os dados permitem auditar.** O contrato traz `gender` (F/M) e `age_years` (na referência: mediana de 43,1 anos, p05 de 25,8 e p95 de 63,5), e os lotes rotulados trazem o `target`. Isso permite medir viés por gênero e por faixa etária. **Não há** raça ou etnia, deficiência ou localização fina, então viés nesses atributos não é mensurável aqui.

**Fontes de viés possíveis**

- **Viés de seleção (reject inference):** o `target` do Home Credit só existe para créditos concedidos. O modelo aprende com quem já foi aprovado e herda as recusas históricas.
- **Viés histórico:** discriminação passada nas concessões ou na cobrança fica codificada nos rótulos.
- **Scores de terceiros:** `ext_source_2` e `ext_source_3` são as variáveis mais importantes e não são auditáveis por dentro. Podem carregar viés do bureau.
- **Proxies de atributos protegidos:** retirar `gender` não basta ("fairness by unawareness"). `income_type = "Maternity leave"` é proxy direto de gênero; `"Pensioner"`, `family_status = "Widow"` e `housing_type = "With parents"` se associam à idade; `region_rating` pode refletir desigualdade regional e socioeconômica. Essas associações são **hipóteses** e não foram medidas.
- **Idade como feature:** `age_years` entra no modelo (9ª em importância, queda de AUC de 0,0029 na permutação). O uso pode ser estatisticamente legítimo, mas pode ter efeito discriminatório contra pessoas idosas [validação jurídica: Estatuto da Pessoa Idosa, Lei 10.741/2003].

**Identificação: auditoria executável [implementado].** [`src/governance/fairness.py`](src/governance/fairness.py) ([ADR-0008](docs/adrs/0008-auditoria-de-vies-sem-dependencia-nova.md)) pontua um lote rotulado atrás do mesmo gate e mede, por gênero e por faixa etária (18-24, 25-34, 35-44, 45-59, 60+):

| Métrica | Pergunta |
| :--- | :--- |
| Taxa de aprovação e diferença de paridade estatística | Os grupos são aprovados na mesma proporção? |
| Razão de impacto disparatado (menor ÷ maior aprovação) | Algum grupo fica abaixo de 0,80 da taxa do mais aprovado? (referência dos EUA, não critério legal brasileiro) |
| Aprovação entre bons pagadores (igualdade de oportunidade) | Bons pagadores de todos os grupos têm a mesma chance? |
| Aprovação entre maus pagadores (com a anterior: equalized odds) | O erro de aprovar mau pagador é igual entre grupos? |
| Inadimplência da carteira aprovada e ROC-AUC por grupo | O risco aceito e a qualidade do score são iguais entre grupos? |

```bash
make fairness                         # padrão: production_base (t0), nunca visto no treino
make fairness BATCH=<lote rotulado>   # saída em reports/governance/<lote>_fairness.{json,md}
```

**Resultado com dados reais [implementado].** Lote `production_base` (t0, 92.247 linhas, nunca visto no treino), modelo `baseline-v1`. Relatório completo em [`reports/governance/production_base_fairness.md`](reports/governance/production_base_fairness.md) (só agregados por grupo, sem registros individuais).

| Atributo | Grupo | n | Default real | Aprovação | Aprovação entre bons pagadores | Default na carteira aprovada | ROC-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gênero | F | 60.688 | 6,98% | 67,35% | 69,93% | 3,42% | 0,750 |
| Gênero | M | 31.559 | 10,17% | 60,33% | 64,13% | 4,51% | 0,751 |
| Idade | 18-24 | 3.695 | 12,48% | **37,86%** | **41,16%** | 4,86% | 0,704 |
| Idade | 25-34 | 21.629 | 10,69% | 50,91% | 54,58% | 4,26% | 0,743 |
| Idade | 35-44 | 25.395 | 8,58% | 63,90% | 67,09% | 4,01% | 0,749 |
| Idade | 45-59 | 30.754 | 6,43% | 72,65% | 74,97% | 3,45% | 0,752 |
| Idade | 60+ | 10.774 | 4,79% | **82,91%** | **84,19%** | 3,32% | 0,710 |

| Atributo | Razão de impacto disparatado | Diferença de paridade | Diferença de igualdade de oportunidade | Revisão necessária |
| :--- | ---: | ---: | ---: | :--- |
| Gênero | 0,896 | 7,0 p.p. | 5,8 p.p. | não |
| Faixa etária | **0,457** | **45,1 p.p.** | **43,0 p.p.** | **sim** |

**Leitura.**

- **Gênero:** a razão de impacto (0,896) fica acima da referência de 0,80. As mulheres são mais aprovadas, coerente com a inadimplência real menor do grupo (6,98% contra 10,17%), e o score ordena igualmente bem os dois grupos (ROC-AUC 0,750 e 0,751). Excluir `gender` do modelo não eliminou a diferença de 5,8 p.p. entre bons pagadores, que deve continuar monitorada.
- **Idade: achado que exige revisão humana.** A razão de impacto é 0,457, muito abaixo de 0,80: quem tem de 18 a 24 anos é aprovado em 37,9% dos casos, contra 82,9% de quem tem 60 anos ou mais. Parte da diferença reflete risco real (a inadimplência cai de 12,5% para 4,8% com a idade), mas **a diferença persiste entre bons pagadores**: um jovem que paga em dia tem 41,2% de chance de aprovação, contra 84,2% de um idoso que também paga (igualdade de oportunidade de 43,0 p.p.). O score também ordena pior nas pontas (ROC-AUC 0,704 entre 18-24 e 0,710 entre 60+, contra 0,75 nas faixas centrais).
- **Origem provável [hipótese]:** `age_years` é feature direta (9ª em importância), e `years_employed`, `ext_source_2/3` e `income_type = "Pensioner"` carregam informação de idade. Retirar só `age_years` provavelmente reduz, mas não elimina, a diferença.
- **Enquadramento [validação jurídica]:** idade não é dado sensível pelo art. 5º, II, mas o princípio da não discriminação (art. 6º, IX) veda tratamento discriminatório ilícito ou abusivo. Usar a idade como fator de risco pode ser legítimo se for justificado e proporcional; uma diferença deste tamanho entre bons pagadores precisa dessa justificativa documentada antes de o modelo ir para produção.

**Próximos passos propostos [recomendação]:** (1) treinar uma variante sem `age_years` e comparar AUC e razão de impacto (troca entre desempenho e equidade); (2) medir a associação das demais features com a idade; (3) se a diferença entre bons pagadores persistir, avaliar reponderação ou restrições de fairness; (4) incluir a auditoria no fluxo de promoção do challenger.

**Estratégias de mitigação**

| Momento | Estratégia | Status |
| :--- | :--- | :--- |
| Antes do treino | Excluir o atributo protegido do modelo | [implementado] (`gender`) |
| Antes do treino | Medir associação entre features e atributos protegidos; agrupar ou retirar proxies diretos (ex.: "Maternity leave") | [recomendação] |
| Antes do treino | Reponderar amostras para equilibrar grupos e rótulos | [recomendação] |
| No treino | Restrições de fairness (ex.: Fairlearn), que exige nova ADR e dependência | [recomendação] |
| Depois do treino | Thresholds por grupo: corrige a paridade, mas usa o atributo protegido na decisão | [recomendação, validação jurídica] |
| Processo | Revisão humana de negativas contestadas e de casos perto do threshold (art. 20) | [recomendação] |
| Governança | Toda promoção de challenger roda a auditoria e bloqueia se `needs_review` | [recomendação] |

**Monitoramento contínuo [recomendação].** O viés também deriva: um data drift numa proxy (renda, tipo de renda) pode mudar as taxas por grupo sem mudar a AUC global. Proposta: rodar `make fairness` em cada lote rotulado de feedback e registrar a razão de impacto no MLflow ao lado da AUC e do PSI.

### 4.3 Análise causal da degradação do modelo (ambiguidade #2)

**Natureza da evidência.** O Home Credit não tem dimensão temporal. A degradação deste projeto é **simulada** por intervenções controladas sobre o lote t0 ([`src/drift/simulate.py`](src/drift/simulate.py)), não observada no tempo:

- **t0 → t1:** só as features mudam (`annuity_amount` × 1,25; `ext_source_3` − 0,15; `income_total` × 1,15, com clip nos limites do contrato). O `target` não muda (default de 8,07% nos dois).
- **t1 → t2:** as features são **idênticas** (mesmos PSIs e mesmo PSI do score, 0,1455). Só o rótulo muda: 25% dos bons pagadores com `ext_source_3` ≥ 0,55 viram default.

Como cada passo muda uma única coisa e o modelo é o mesmo, a diferença de desempenho entre fatias vizinhas é **atribuível à intervenção**. É um experimento, não uma correlação. Isso vale **dentro da simulação**: a causa da degradação de um modelo real (inflação, novos perfis) não pode ser provada com este dataset.

**Evidências** ([`drift_demo_summary.md`](reports/drift/drift_demo_summary.md); alertas do [aceite da Etapa 3](docs/ai/reports/2026-10-27-etapa3-aceite.md)):

| Fatia | Contrato | Colunas com drift (PSI) | ROC-AUC | Aprovação | Default na carteira aprovada | Alerta |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| t0 | aprovado | nenhuma | 0,7534 | 64,95% | 3,76% | `OK` |
| t1 | aprovado | `annuity_amount` (0,17), `ext_source_3` (0,97), `income_total` (0,76) | 0,7223 | 52,04% | 3,67% | `ATENÇÃO` |
| t2 | aprovado | as mesmas de t1; `target`: qui-quadrado com p ≈ 0, mas PSI 0,018 | 0,5733 | 52,04% | 9,82% | `CRÍTICO` |

A queda total de AUC (0,1801) se decompõe em **0,0311** (t0 → t1, choque nas features) e **0,1490** (t1 → t2, mudança de P(y|X)), ou seja, 83% da queda vem da mudança de conceito.

| # | Fator | Classificação | Evidência | Impacto no modelo | Mitigação |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | Mudança na relação X → y (concept drift simulado) | **Causa comprovada** (na simulação) | t1 e t2 têm features idênticas e o mesmo modelo; só o rótulo difere; AUC 0,7223 → 0,5733 | Score perde poder de ordenação; inadimplência aprovada de 3,67% → 9,82% | Alerta `CRÍTICO` por queda de AUC com rótulos de feedback; retreino guardado pelo alerta. No aceite, o challenger ganhou +0,1316 de AUC no holdout do t2 ([ADR-0007](docs/adrs/0007-retreino-registry-e-promocao-humana.md)) |
| 2 | Choque conjunto nas três features (data drift) | **Causa comprovada** (na simulação), do conjunto | t0 → t1 com rótulos inalterados: AUC −0,0311, aprovação de 64,95% → 52,04% | Sobretudo na **política**: o modelo fica mais conservador e recusa mais, com inadimplência aprovada estável (3,76% → 3,67%) | Alerta `ATENÇÃO` (KS e PSI ≥ 0,10, PSI do score); investigar antes de retreinar |
| 3 | Contribuição de cada feature no item 2 | **Hipótese** | Os três choques foram aplicados juntos. Indício: `ext_source_3` é a variável mais importante (0,058 na permutação) e `income_total` fica fora do top 10 | Provável predominância de `ext_source_3` | [recomendação] ablação: simular um choque por vez |
| 4 | Mecanismo da queda de AUC em t1 sem mudança de rótulo | **Hipótese** | Não medido. (a) O clip de `ext_source_3` em 0 cria empates (p05 da referência = 0,1547, então perto de 5% dos valores não nulos colapsam em 0). (b) Choques de tamanhos diferentes alteram as razões derivadas (`annuity_to_income` × 1,087; `credit_to_annuity` × 0,8) | Pequeno (−0,0311) | Mesma ablação, com medição de empates |
| 5 | Tamanho do PSI de `income_total` (0,76 para um choque de 15%) | **Hipótese** técnica | A renda é concentrada em valores redondos (p05, p50 e p95 da referência: 67.500, 146.250 e 337.500). O choque multiplicativo desloca blocos inteiros de massa entre faixas de quantil | O PSI superestima o efeito em variáveis discretizadas; não afeta o modelo (baixa importância) | Ler o PSI junto com a importância; [recomendação] faixas fixas para variáveis com valores repetidos |
| 6 | Mudança na distribuição do alvo (label shift) | **Evidência, não causa principal** | Default de 8,07% → 11,75% em t2. O qui-quadrado é significativo, mas o PSI de 0,018 é "estável" | A prevalência sozinha não explica a queda: o que muda é o rótulo na região de `ext_source_3` alto, onde o modelo vê baixo risco | Monitorar `label_shift`, sem tratá-lo como alerta de concept drift ([ADR-0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md)) |
| 7 | Qualidade dos dados e falha de pipeline | **Descartada** para t0–t2 | As três fatias passaram no contrato; o modelo é o mesmo (sha256 conferido); só 21 de 307.511 linhas brutas foram rejeitadas (0,007%) | nenhum | Gate fail-closed: dado inválido é `BLOQUEADO` e nunca chega a ser pontuado |
| 8 | Diferença entre referência e produção sem drift | **Descartada** | t0: nenhuma coluna marcada; AUC 0,7534 contra 0,7494 no teste do treino; PSI do score 0,0003 | nenhum | Separação física e estratificada referência/produção |
| 9 | Causas econômicas reais (inflação, novos perfis, mudança de comportamento) | **Hipótese não testável** com este dataset | Os choques **representam** essas causas (inflação → parcela e renda nominais maiores; piora dos bureaus → `ext_source_3` menor; mudança de comportamento → concept drift), mas não as observam | — | [recomendação] lotes reais datados com rótulos de feedback; correlacionar alertas com indicadores macro, sem tratar correlação como causa |

**Limitação.** Sem rótulos de feedback, só o data drift e o PSI do score são observáveis: o concept drift (item 1) só aparece quando os rótulos chegam, com atraso. Nesse intervalo, um alerta `ATENÇÃO` é o único sinal, e ele não prova que o modelo piorou.

### 4.4 Justificativa técnica das ferramentas

As escolhas foram conferidas contra o `uv.lock` e os imports reais (ver [Tecnologias e bibliotecas](#tecnologias-e-bibliotecas)).

**Validação de dados: Pandera (instalado e usado) × Great Expectations (não instalado).** Decisão completa na [ADR-0002](docs/adrs/0002-pandera-data-contracts.md).

| Critério | Pandera (eleita) | Great Expectations |
| :--- | :--- | :--- |
| Contrato | Classe Python tipada (`DataFrameModel`), revisada em PR | Suites e contexto (data source, asset, batch definition) |
| Regras de qualidade | `Field` e `@dataframe_check`; as 11 regras cabem num arquivo | Expectations prontas, catálogo maior |
| Integração com o pipeline | Uma chamada `validate(lazy=True)` dentro do gate | Requer o contexto do GE em cada ponto de chamada |
| Manutenção | Leve; testado com pytest direto | API mudou muito da 0.x para a 1.x |
| Bloqueio de coluna inesperada (CPF) | `strict=True` | Possível, com mais configuração |
| Relatório HTML | Não nativo; o projeto gera o seu (`src/validation/report.py`) | Data Docs nativos (a única vantagem relevante aqui) |
| Adequação à arquitetura | Biblioteca de domínio, sem infraestrutura, coerente com o import-linter | Traz estado e configuração próprios |

**Observabilidade: MLflow (instalado e usado) × Datadog (não instalado).** Decisão completa na [ADR-0006](docs/adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md).

| Critério | MLflow local (eleita) | Datadog |
| :--- | :--- | :--- |
| Rastreamento de experimentos | Runs de estágio e de lote, tag `pipeline_run_id` | Não é o foco (APM e métricas de infraestrutura) |
| Versionamento de modelos | Model Registry com aliases `champion`, `challenger` e `previous` | Não tem registry de modelos |
| Parâmetros e métricas | Métricas de saúde, PSI, AUC, nível de alerta por lote | Métricas e dashboards em tempo real |
| Monitoramento e alerta | Nível de alerta nas tags do run e códigos de saída do `monitor`; **sem push de alerta** | Monitores e notificações nativos (o ponto forte) |
| Rastreabilidade | Liga dado (hash), modelo (sha256), contrato e relatório; runs de auditoria de promoção | Logs e traces, sem linhagem de modelo |
| Privacidade e licença | Nada sai da máquina (LGPD e licença do Kaggle) e não há credencial | SaaS externo: exige enviar dados ou métricas e guardar chave de API |
| Integração | Só a camada `src/monitoring` importa MLflow (contrato do import-linter) | Exigiria agente e SDK |

A ausência de notificação ativa (e-mail, Slack) é uma lacuna assumida da demonstração local. Em produção, os dois se complementam: o MLflow como registro do ciclo de vida do modelo e uma ferramenta como o Datadog para alertas operacionais.

**Demais escolhas:** Evidently para os testes estatísticos e os relatórios HTML de drift ([ADR-0004](docs/adrs/0004-evidently-drift-detection.md)), com PSI próprio e queda de AUC como sinais complementares ([ADR-0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md)); DVC para a reprodução local ([ADR-0006](docs/adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md)); métricas de fairness próprias em vez de Fairlearn ([ADR-0008](docs/adrs/0008-auditoria-de-vies-sem-dependencia-nova.md)).

## Validação de reprodutibilidade

Execução de 2026-10-29, Windows 11, `uv` 0.11.7, a partir de um **clone novo** (sem `.venv`):

| Passo | Comando | Resultado |
| :--- | :--- | :--- |
| Ambiente | `uv sync --locked` | ok, 2 min 28 s |
| Lint | `uv run ruff check .`, `uv run ruff format --check .`, `uv run lint-imports` | ok; 3 contratos de camadas mantidos |
| Testes (antes da correção) | `uv run pytest -n auto` | **11 erros de coleta**: `src/monitoring/resources.py` importava `resource`, que só existe em Unix. Toda a camada de monitoramento quebrava no Windows |
| Testes (depois da correção) | `uv run pytest -n auto --cov=src` | **436 passaram, 37 pulados, 0 falhas; cobertura 97,34%** |
| Gate do contrato via API, lote sintético | `enforce_contract` | lote válido aprovado; lote com os defeitos de `make_invalid_batch` **bloqueado**, com R1–R8 e R10 detectadas |
| Drift via API, lote sintético | `run_drift_detection(min_psi=0.10)` | sem choque: nenhuma coluna marcada; com `apply_data_drift`: as 3 colunas com choque marcadas |
| Conferência independente dos cálculos | PSI reimplementado do zero; KS conferido com `scipy.stats.ks_2samp` | PSI idêntico em 7 colunas; p-valor do Evidently idêntico ao do SciPy |
| Pipeline com dados reais (`make pipeline`) | — | **bloqueado**: sem `application_train.csv` e sem token do Kaggle nesta máquina. A última execução real é de 2026-10-27 (`pipeline_run_id` `20261027T103457Z`, [aceite](docs/ai/reports/2026-10-27-etapa3-aceite.md)) |

Os testes pulados precisam de `make` (ausente no Windows), de privilégio para criar symlink, do módulo Unix `resource` ou do dataset e do `mlflow.db` reais (aceite real). O CI roda em Linux e cobre os ramos Unix.

**Correções feitas nesta validação:** medição de memória de pico no Windows via `psapi` (só biblioteca padrão); testes que usavam `/proc/inexistente` como caminho impossível agora usam um caminho sob um arquivo, impossível em qualquer sistema operacional (no Windows, o `/proc/...` era criável e os testes falhavam).

### Execução com dados reais no Windows (2026-10-30)

Execução completa com o dataset, Windows 11, `pipeline_run_id` `20261030T024043Z`. **Os artefatos versionados (modelo, relatórios e `dvc.lock`) e os números deste README vêm desta execução.**

| Passo | Comando | Resultado |
| :--- | :--- | :--- |
| Pipeline | `uv run python -m src.monitoring.stage <prepare, train, contract_demo, drift_demo>` | os 4 estágios com sucesso; alertas `OK` (t0), `ATENÇÃO` (t1), `CRÍTICO` (t2) e `BLOQUEADO` (lote inválido) |
| Auditoria de viés | `uv run python -m src.governance.fairness` | relatório em `reports/governance/` ([4.2](#42-mitigação-de-vieses-algorítmicos)) |
| Testes | `uv run pytest -n auto` | **445 passaram, 32 pulados, 0 falhas** (o aceite real, que exige o dataset e o `mlflow.db`, passa a rodar) |
| DVC | `uv run dvc status` | `Data and pipelines are up to date` |

**Correção feita nesta execução: modelo portável entre sistemas operacionais.** O bundle treinado no macOS falhava no Windows com `XGBoostError: input stream corrupted`, porque o pickle do booster guarda o estado interno do XGBoost, que não é portável entre plataformas. O booster passou a ser gravado no formato de modelo do XGBoost (`save_raw("json")`, [`src/models/portable.py`](src/models/portable.py)) e é reconstruído no `load_bundle`; as predições da ida e volta são idênticas (`tests/test_portable.py`). Em relação à execução no macOS, os números mudam só na 3ª ou 4ª casa decimal (ex.: AUC do t0 0,7533 → 0,7534).

**Dica para Windows:** se o `dvc` falhar com `WinError 3`, o caminho do cache passou de 260 caracteres. Aponte o cache para uma pasta curta, só na sua máquina: `uv run dvc cache dir --local C:/dvc-cache` (grava em `.dvc/config.local`, fora do git).

## Limitações conhecidas

- **Dataset e credenciais:** o pipeline real exige o CSV do Kaggle (licença proíbe redistribuir) e, para o download automático, um token. Sem eles, só os testes sintéticos rodam.
- **Degradação simulada:** a análise causal prova causas dentro da simulação, não no mundo real ([4.3](#43-análise-causal-da-degradação-do-modelo-ambiguidade-2)).
- **Concept drift exige rótulos:** sem feedback, só o data drift e o PSI do score são observáveis.
- **Campo `statistic` dos relatórios de drift:** guarda o **p-valor** do teste (KS ou qui-quadrado), não a estatística D. A decisão (p < 0,05) está correta; o nome engana e foi mantido para não quebrar o esquema dos JSON.
- **Viés por idade sem mitigação:** a auditoria encontrou razão de impacto de 0,457 entre faixas etárias ([4.2](#42-mitigação-de-vieses-algorítmicos)). O achado está medido e documentado, mas a mitigação (variante sem `age_years`, reponderação) é recomendação, não implementação.
- **LGPD:** retenção, mascaramento de exemplos, revisão humana de decisões individuais e *reason codes* são recomendações, não implementações.
- **Alertas sem notificação ativa:** o nível de alerta fica no MLflow e no código de saída; nada envia e-mail ou mensagem.
- **Alias `concept_drift`:** ainda presente nos JSON de drift; sai na próxima regeneração dos relatórios (2ª emenda da ADR-0005).

## Decisões de arquitetura (ADRs)

| ADR                                                              | Decisão                                                            |
| ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| [0001](docs/adrs/0001-dataset-home-credit.md)                     | Dataset Home Credit Default Risk (German Credit tem < 5.000 linhas) |
| [0002](docs/adrs/0002-pandera-data-contracts.md)                  | Data Contracts com Pandera (e não Great Expectations)              |
| [0003](docs/adrs/0003-modelo-baseline-xgboost.md)                 | Baseline XGBoost em Pipeline scikit-learn, threshold por KS         |
| [0004](docs/adrs/0004-evidently-drift-detection.md)               | Detecção de Data/Concept Drift com Evidently                      |
| [0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md) | PSI e queda de ROC-AUC como sinais complementares ao KS             |
| [0006](docs/adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md) | MLflow (SQLite) e DVC locais, sem remoto externo (licença Kaggle)   |
| [0007](docs/adrs/0007-retreino-registry-e-promocao-humana.md)      | Retreino só com alerta `CRITICO`, Registry com aliases e promoção humana |
| [0008](docs/adrs/0008-auditoria-de-vies-sem-dependencia-nova.md)    | Auditoria de viés com métricas próprias, sem Fairlearn/AIF360       |

## Vídeo STAR

> **Link do vídeo: _pendente (a gravar e publicar pelo grupo)_.**

Roteiro de até 5 minutos, baseado só nas evidências deste repositório, com as telas a mostrar em cada trecho: [`docs/video-star-roteiro.md`](docs/video-star-roteiro.md).

## Equipe

| Integrante    | RM            | Responsabilidade                                      |
| ------------- | ------------- | ----------------------------------------------------- |
| Victor Murad  | _preencher_ | Etapa 1 — Validação de dados, contratos e baseline |
| VonLuqi       | _preencher_ | Etapa 2 — Drift                                      |
| _a definir_ |               | Etapa 3 — Observabilidade                            |
| _a definir_ |               | Etapa 4 — Governança e vídeo                       |

Convenção de commits e fluxo de trabalho: [`CONTRIBUTING.md`](CONTRIBUTING.md).
