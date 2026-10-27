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
├── models/                             # baseline treinado + metadados (versionados)
├── reports/
│   ├── data_contract/                  # evidências do contrato (HTML/JSON/MD)
│   ├── model/                          # métricas do baseline
│   └── drift/                          # evidências de Data/Concept Drift (Etapa 2)
├── src/
│   ├── config.py                       # caminhos, seed, features
│   ├── logging_utils.py                # logs estruturados (JSON lines)
│   ├── data/                           # download + construção dos datasets
│   ├── validation/                     # contrato Pandera, gate, relatórios, demo
│   ├── models/                         # features, treino, predição
│   ├── drift/                          # simulação + detecção de drift (Evidently)
│   └── monitoring/                     # Etapa 3: estágios rastreados, alerta, MLflow, Registry, monitor
├── dvc.yaml / dvc.lock                 # Etapa 3: pipeline reprodutível (só hashes no git)
├── mlflow.db / mlartifacts/            # NÃO versionados: MLflow local (gerados por `make pipeline`)
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
| `make tracking-ui`                  | `uv run mlflow ui --backend-store-uri sqlite:///<caminho absoluto>/mlflow.db --port 5001` |
| `make repro`                        | `uv run dvc repro`                                                 |
| `make validate INPUT=arq.csv`       | `uv run python -m src.validation.validate --input arq.csv`         |
| `make predict INPUT=arq.csv`        | `uv run python -m src.models.predict --input arq.csv`              |
| `make simulate-drift`               | `uv run python -m src.drift.simulate`                              |
| `make detect-drift CURRENT=...`     | `uv run python -m src.drift.detect --current ... [--min-psi 0.10]` |
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
| Baseline XGBoost (teste)                  | ROC-AUC **0,749** · Gini **0,498** · KS **0,371** (Regressão Logística: AUC 0,731)                             |
| Impacto no negócio                       | aprova 65% dos pedidos, e a inadimplência da carteira cai de 8,07% para **3,88%**                                             |
| Demonstração do contrato                | lote válido **APROVADO** (500 pontuados); lote sujo **BLOQUEADO** (0 pontuados); **10 de 10** defeitos detectados |

## Etapa 2 — Simulação e Detecção de Drift

> Foco: tornar a degradação silenciosa visível. Detalhes e checklist 2.1–2.4 em [`docs/etapa-2-simulacao-e-deteccao-de-drift.md`](docs/etapa-2-simulacao-e-deteccao-de-drift.md).

**Entregável: simulação de drift operante + relatório de detecção (Data Drift e Concept Drift).**

- **Simulação:** [`src/drift/simulate.py`](src/drift/simulate.py) — a partir de `production_base.parquet`, gera t1 (data drift em `annuity_amount`, `ext_source_3`, `income_total`) e t2 (data + concept drift no `target`), com clip para continuar passando no contrato.
- **Detecção (Evidently):** [`src/drift/detect.py`](src/drift/detect.py) — compara cada fatia com a Referência (`DataDriftPreset`, KS / chi-square). [ADR-0004](docs/adrs/0004-evidently-drift-detection.md).
- **Tamanho do efeito (PSI):** [`src/drift/psi.py`](src/drift/psi.py) — PSI em toda coluna (< 0,10 estável · 0,10–0,25 moderado · ≥ 0,25 significativo). `--min-psi 0.10` só marca drift quando o KS é significativo **e** o efeito é relevante, evitando falso alerta em amostra grande. [ADR-0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md).
- **Concept drift por performance:** [`src/drift/performance.py`](src/drift/performance.py) — queda de ROC-AUC vs. baseline (alerta a partir de 0,05).
- **Demo ponta a ponta:** [`src/drift/demo.py`](src/drift/demo.py) / `make etapa2` — gate → score → Evidently → [`reports/drift/drift_demo_summary.md`](reports/drift/drift_demo_summary.md).

**Resultados (Home Credit, seed 42, 92.247 linhas):**

|                      |                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------ |
| Gate em t0 / t1 / t2 | todos **APROVADOS** (drift ≠ dado inválido)                                        |
| Evidently t0         | sem dataset drift (AUC **0,753**, alinhado ao baseline)                              |
| Evidently t1         | **Data Drift** em `annuity_amount` (PSI 0,17), `ext_source_3` (0,97), `income_total` (0,76); AUC 0,722 |
| Evidently t2         | mesmo Data Drift; label shift em `target` com efeito desprezível (PSI 0,02); **queda de AUC 0,181 → modelo degradado** (AUC **0,572**) |
| Impacto no negócio  | default na carteira aprovada: 3,74% (t0) → **9,84%** (t2)                           |
| Lição              | t1 detecta drift com queda leve de AUC; t2 (concept) degrada de fato                       |

> **Limitação conhecida (concept drift):** o teste do Evidently sobre `target` mede mudança de *prevalência* do rótulo (label/prior shift), não mudança em P(y|X); por isso o campo é `label_shift` em `DriftResult` e nos relatórios JSON (o antigo `concept_drift` segue como alias obsoleto, removido na Etapa 4). O sinal confiável de concept drift é a **queda de ROC-AUC** com rótulos de feedback (`performance` no sumário da demo). Sem rótulos (inferência pura) só é possível monitorar data drift e PSI. Em t0, o KS sozinho marca `annuity_amount` por ruído de amostra grande (PSI 0,00); a demo usa `min_psi=0,10` e, com isso, t0 fica sem nenhuma coluna marcada. Pelo mesmo critério, o teste sobre `target` em t2 deixa de alertar (PSI 0,02) enquanto o AUC cai 0,181: é a prova de que só a performance revela o concept drift.

## Etapa 3 — Observabilidade de Pipelines e Modelos

> Foco: um lugar único para ver a saúde do pipeline, a estabilidade da execução e os alertas de degradação do modelo. Desenho em [`docs/ai/specs/2026-10-25-etapa3-observabilidade-design.md`](docs/ai/specs/2026-10-25-etapa3-observabilidade-design.md); decisão em [ADR-0006](docs/adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md).

**Entregável: ambiente de rastreamento (MLflow) com a saúde do pipeline, a estabilidade da execução e os alertas de degradação do modelo.** A parte de retreino, promoção e rollback (§4.7 da spec) ainda não foi feita.

- **MLflow local:** o banco SQLite fica em `mlflow.db` e os artefatos em `mlartifacts/`, os dois na raiz do repositório e fora do git. O experimento se chama `credit-drift-monitor`. Não há servidor remoto.
- **Runs:** cada estágio (`prepare`, `train`, `contract_demo`, `drift_demo`) gera um run de estágio com a tag `pipeline_run_id` e o trecho do `logs/pipeline.jsonl` daquele `run_id` anexado como artefato. Cada lote avaliado gera um run filho com nível, motivos, ação, métricas e os relatórios do Evidently e do contrato.
- **Métricas de saúde (checklist 3.3):**
  - status: `stage_success` e a tag `stage_status`;
  - latência: `stage_duration_s`, mais o `duration_ms` dos eventos de contrato e de detecção no log anexado;
  - volume: `rows_processed` e `rows_rejected`;
  - alertas: `alerts_failure_count` e `alerts_drift_count`;
  - modelo: `roc_auc`, `auc_drop`, `approval_rate`, `default_rate_approved` e `score_psi`;
  - estabilidade: `peak_memory_mb` e `stage_duration_s`. "Infraestrutura" aqui é o processo local: não há cluster, e essa limitação é assumida.
- **Model Registry:** o modelo `credit-default-xgboost` recebe o alias `champion` no primeiro `train`. O modelo servido fica em `models/champion/`, conferido pelo sha256 da versão `champion`. Um `train` repetido registra uma versão nova, mas não move o alias.

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

### Resultado real (Home Credit, seed 42, `pipeline_run_id` `20261027T103457Z`)

| Lote | Nível | Motivos | ROC-AUC | Queda de AUC | PSI do score |
| :--- | :--- | :--- | ---: | ---: | ---: |
| t0 (`production_t0`) | `OK` | nenhum | 0,7533 | 0,0000 | 0,0003 |
| t1 (`production_t1_data_drift`) | `ATENÇÃO` | `drift:annuity_amount`, `drift:ext_source_3`, `drift:income_total`, `score_psi>=0.10` | 0,7224 | 0,0309 | 0,1431 |
| t2 (`production_t2_concept_drift`) | `CRÍTICO` | `auc_drop>=0.05` e os mesmos de t1 | 0,5720 | 0,1813 | 0,1431 |
| lote inválido da demo do contrato | `BLOQUEADO` | as regras violadas (R1 a R8 e R10); `intentional_demo=true` | não avaliada | | |

No estágio `contract_demo`, `alerts_failure_count` = 1 e `rows_rejected` = 508. No `drift_demo`, `alerts_drift_count` = 2.

### Comandos

```bash
make pipeline                  # os 4 estágios, sempre executados, com um único PIPELINE_RUN_ID
make tracking-ui               # MLflow UI em http://127.0.0.1:5001 (a 5000 colide com o AirPlay no macOS)
make monitor BATCH=data/production/production_t2_concept_drift.parquet [BASELINE_AUC=0.7533]
make demo-drift                # só o estágio drift_demo (apelidos: drift-demo, etapa2)
make repro                     # dvc repro: roda só os estágios com mudança
```

O `monitor` avalia **um** lote: contrato, pontuação, drift, PSI do score, AUC (se o lote for rotulado), nível de alerta e registro no MLflow. A AUC de referência vem do run `production_t0` do mesmo modelo no MLflow, ou de `--baseline-auc`. Um lote rotulado sem nenhuma das duas falha fechado.

> **Código de saída do monitor:** o próprio monitor sai com 0 (`OK`/`ATENÇÃO`), 2 (`CRÍTICO`), 3 (`BLOQUEADO`) ou 1 (erro operacional). O `make monitor` **não repassa** esse código: o GNU make sai com 2 em qualquer falha de receita. O código real aparece só no texto (`make: *** [monitor] Error 3` e a linha `monitor: codigo de saida 3 ...`). Para usar num script, chame o módulo direto:
>
> ```bash
> uv run python -m src.monitoring.monitor --batch data/production/production_t2_concept_drift.parquet
> echo $?   # 2 (CRÍTICO)
> ```

> **Variáveis de ambiente:** `MLFLOW_TRACKING_URI` e `MLFLOW_ARTIFACT_ROOT` trocam o banco e o diretório de artefatos (o padrão é `mlflow.db` e `mlartifacts/` na raiz, com caminho absoluto). Se uma delas estiver exportada no shell, o pipeline grava em outro lugar. O teste de aceite real (`tests/integration/test_acceptance_real.py`) sempre lê o `mlflow.db` da raiz, então ele deixa de enxergar a execução ou enxerga uma antiga. Antes de rodar `make pipeline` e o aceite, confira que as duas não estão definidas. `MONITORING_TRACKING=off` desliga o rastreamento; sem ele, um MLflow indisponível derruba o estágio com mensagem clara.

### DVC: reprodução local

- [`dvc.yaml`](dvc.yaml) tem os estágios `prepare → train → {contract_demo, drift_demo}`. Cada estágio chama `uv run python -m src.monitoring.stage <estágio>` e declara os `outs` arquivo por arquivo.
- `models/` e `reports/` são `outs` com `cache: false`, então continuam versionados no git. Os outs de `data/` vão para o cache do DVC, que fica fora do git.
- O CSV bruto é `deps` do `prepare`, e o hash vai para o [`dvc.lock`](dvc.lock), que é versionado (só hashes e tamanhos). Não se usa `dvc add`.
- `make repro` (`uv run dvc repro`) roda só os estágios com mudança. Cada estágio recebe um `PIPELINE_RUN_ID` próprio. Sem mudança, não gera runs novos. Para a demonstração completa no MLflow, use `make pipeline`. Os relatórios têm timestamp, então o `dvc.lock` e `reports/` mudam a cada rodada. Isso é esperado.
- **Remoto local (opcional), só em `.dvc/config.local`:**

  ```bash
  uv run dvc remote add --local -d local ~/dvc-remotes/credit-drift-monitor
  ```

  A flag `--local` grava em `.dvc/config.local`, que o `.gitignore` deixa fora do repositório. O versionamento de dados fica só na máquina do usuário, para a demo: o Kaggle proíbe redistribuir o dataset, e um remoto externo (S3, Drive, SSH) seria redistribuição e exigiria credenciais (ADR-0006). Nunca faça `dvc push` para um serviço externo.

> **Champion e `dvc repro`:** o champion servido fica em `models/champion/`, fora dos `outs` do DVC, porque promoção e rollback o reescrevem. Os dois arquivos dele (`baseline_model.joblib` e `baseline_metadata.json`) são `deps` de `contract_demo` e `drift_demo`. Assim, depois de uma promoção ou de um rollback, o `dvc repro` marca as duas demos como desatualizadas e as reexecuta com o champion novo. Limitação: o champion só nasce no bootstrap do `train` com o rastreamento ligado. Com `MONITORING_TRACKING=off`, o `dvc repro` para nessas demos por dependência ausente. Nesse caso, use `make pipeline`.

## Etapa 4 — Governança, LGPD e Análise Causal

> _Em construção._ Insumos já existentes: minimização de dados (21 de 122 colunas), classificação LGPD por campo em [`contracts/credit_application.yaml`](contracts/credit_application.yaml), `gender` excluído do modelo e mantido só para auditoria de viés, e nenhum dado pessoal versionado.

## Decisões de arquitetura (ADRs)

| ADR                                                              | Decisão                                                            |
| ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| [0001](docs/adrs/0001-dataset-home-credit.md)                     | Dataset Home Credit Default Risk (German Credit tem < 5.000 linhas) |
| [0002](docs/adrs/0002-pandera-data-contracts.md)                  | Data Contracts com Pandera (e não Great Expectations)              |
| [0003](docs/adrs/0003-modelo-baseline-xgboost.md)                 | Baseline XGBoost em Pipeline scikit-learn, threshold por KS         |
| [0004](docs/adrs/0004-evidently-drift-detection.md)               | Detecção de Data/Concept Drift com Evidently                      |
| [0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md) | PSI e queda de ROC-AUC como sinais complementares ao KS             |
| [0006](docs/adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md) | MLflow (SQLite) e DVC locais, sem remoto externo (licença Kaggle)   |

## Vídeo STAR

> _Link a ser adicionado na Etapa 4._

## Equipe

| Integrante    | RM            | Responsabilidade                                      |
| ------------- | ------------- | ----------------------------------------------------- |
| Victor Murad  | _preencher_ | Etapa 1 — Validação de dados, contratos e baseline |
| VonLuqi       | _preencher_ | Etapa 2 — Drift                                      |
| _a definir_ |               | Etapa 3 — Observabilidade                            |
| _a definir_ |               | Etapa 4 — Governança e vídeo                       |

Convenção de commits e fluxo de trabalho: [`CONTRIBUTING.md`](CONTRIBUTING.md).
