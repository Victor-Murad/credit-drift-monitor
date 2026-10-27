# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o versionamento segue [SemVer 2.0.0](https://semver.org/lang/pt-BR/).

## [Unreleased]
### Adicionado
- Data contract Pandera com 11 regras rígidas, gate fail-closed com quarentena e relatórios HTML/JSON (Etapa 1).
- Dataset de Referência e lote de Produção t0 separados, modelo baseline XGBoost persistido e demonstração do bloqueio de lote inválido (Etapa 1).
- Simulação de Data/Concept Drift e detecção com Evidently, com relatórios HTML/JSON e sumário de impacto no modelo (Etapa 2).
- PSI por coluna (`src/drift/psi.py`) e filtro opcional de efeito `min_psi` / `--min-psi` na detecção de drift ([ADR-0005](docs/adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md)).
- Queda de ROC-AUC vs. baseline como sinal de concept drift (`src/drift/performance.py`), exposta no sumário da demo (`performance`).
- Demo de drift usa `min_psi=0,10` (validado com dados reais): t0 sem falso alerta; sumário passa a incluir PSI por coluna e a degradação de AUC.
- Testes de integração do pipeline ponta a ponta em sandbox (`tests/integration/`) e testes de download, PSI e CLIs.
- Mitigações do architecture review ([relatório](docs/ai/reports/2026-10-25-architecture-review-src.md)):
  - detecção de drift falha fechada (`DriftParsingError`) quando o Snapshot do Evidently não traz `ValueDrift`; Evidently limitado a `<0.8`;
  - `load_bundle` confere o sha256 do modelo contra `baseline_metadata.json` antes de desserializar (`src/models/integrity.py`);
  - métricas em `src/models/metrics.py` e `read_batch` em `src/validation/io.py`, desfazendo dois acoplamentos entre camadas; contrato `layers` (drift → models → validation) no import-linter;
  - todo evento de log carrega `run_id` (ou `PIPELINE_RUN_ID`); `drift_detected` registra `model_version` e `min_psi`; o log JSONL rotaciona (10 MB × 5);
  - workflow de CI (`.github/workflows/ci.yml`: `uv sync --locked`, `make lint`, `make test-cov`) e Dependabot para `github-actions`.
- Núcleo da Etapa 3, observabilidade com MLflow e DVC locais ([ADR-0006](docs/adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md), [spec](docs/ai/specs/2026-10-25-etapa3-observabilidade-design.md)):
  - camada `src/monitoring` (contrato `monitoring → drift → models → validation` no import-linter; `mlflow` e `dvc` só nela) com MLflow em SQLite (`mlflow.db`) e artefatos em `mlartifacts/`, ambos locais e fora do git;
  - estágios rastreados `prepare`, `train`, `contract_demo` e `drift_demo` (`python -m src.monitoring.stage <estágio>`), cada um com um run de estágio (`stage_success`, `stage_duration_s`, `rows_processed`, `rows_rejected`, `alerts_failure_count`, `alerts_drift_count`, `peak_memory_mb`, tag `pipeline_run_id`) e o log JSONL do `run_id` anexado;
  - um run de lote filho por lote avaliado, com nível de alerta (`OK`, `ATENCAO`, `CRITICO`, `BLOQUEADO`), motivos, ação do runbook, PSI por coluna, PSI do score, ROC-AUC, queda de AUC, `label_shift` e os relatórios do Evidently e do contrato;
  - regra pura de nível de alerta (`src/monitoring/alerts.py`) e PSI do score contra a distribuição do teste do treino (`score_reference` no `baseline_metadata.json`);
  - Model Registry com o alias `champion` (bootstrap no primeiro `train`) e o champion servido em `models/champion/`, conferido por sha256;
  - comando `monitor` para avaliar um lote (códigos 0, 2, 3 e 1);
  - alvos `make pipeline`, `make monitor BATCH=`, `make demo-drift`, `make tracking-ui` (porta 5001) e `make repro`;
  - `dvc.yaml` com os 4 estágios, `outs` arquivo por arquivo (`cache: false` em `models/` e `reports/`), o CSV bruto como `deps` do `prepare` e o `dvc.lock` versionado;
  - teste de aceite real (`tests/integration/test_acceptance_real.py`), que só roda na máquina com o dataset e o `mlflow.db`. Na execução real de 2026-10-27 (`pipeline_run_id` `20261027T103457Z`), t0 deu `OK`, t1 `ATENCAO` e t2 `CRITICO`.
- Extensão §4.7 da Etapa 3: retreino, Model Registry e promoção humana ([ADR-0007](docs/adrs/0007-retreino-registry-e-promocao-humana.md)). Implementada e aceita com os dados reais em 2026-10-27 (challenger v3: ganho +0,1316 no holdout, regressão +0,0089, elegível; promoção, rollback e auditoria `CONSISTENT`; ver `docs/ai/reports/2026-10-27-etapa3-aceite.md`):
  - regra pura de promoção (`src/monitoring/promotion.py`) com os limiares pré-registrados `PROMOTION_MIN_GAIN` = 0,02 (ganho no holdout do lote `CRITICO`) e `PROMOTION_MAX_REGRESSION` = 0,02 (regressão no teste da referência);
  - `retrain` (`python -m src.monitoring.retrain`): só roda com um alerta `CRITICO` não consumido do champion atual, ou com `--force-no-alert`/`--allow-reuse` e `--reason`. Treina com o treino da referência mais metade do lote, avalia no holdout que ficou de fora, registra o challenger (alias `challenger`, `model_version` próprio `retrain-<sha256[:12]>`) e nunca move o `champion`. Códigos 0, 2, 3 e 1;
  - `promote` e `promote rollback` (`python -m src.monitoring.promote`): comandos humanos que reavaliam a regra, conferem o sha256, movem `champion` e `previous`, trocam `models/champion/` de forma atômica, reconferem e gravam um run de auditoria (`CONSISTENT`/`INCONSISTENT`). `--override` exige `--reason`. Códigos 0, 2 e 1;
  - alvos `make retrain`, `make promote` e `make rollback`, que repassam `ARGS=` e mostram o código de saída exato;
  - os arquivos de `models/champion/` passam a ser `deps` de `contract_demo` e `drift_demo` no `dvc.yaml`: o `dvc repro` reexecuta as demos depois de uma promoção ou de um rollback.

### Alterado
- `make data`, `make train`, `make contract-demo` e `make demo-drift` (e os apelidos `drift-demo`, `etapa1` e `etapa2`) passam a chamar `src.monitoring.stage`, com rastreamento no MLflow. `MONITORING_TRACKING=off` desliga o rastreamento.
- `DriftResult.concept_drift` renomeado para `label_shift`. O JSON mantém `concept_drift` como alias obsoleto até a Etapa 4, e a ADR-0005 (item 4) ganhou uma emenda.
- ADR-0006 passa a `Implementada` (2026-10-27) e ganha uma nota: a limitação do `dvc repro` com promoção e rollback foi tratada na ADR-0007.
- Com o rastreamento desligado (`MONITORING_TRACKING=off`), o `dvc repro` passa a parar em `contract_demo` e `drift_demo`, porque não há champion; nesse caso, use `make pipeline`.
- `models/` e `reports/` regenerados pela execução real da Etapa 3 em 2026-10-27, agora com `score_reference` no metadata e `label_shift` nos relatórios de drift. Os números do baseline e da demo não mudaram.
- Piso do Evidently em `pyproject.toml` corrigido para `>=0.7` (a API usada, `from evidently import Report`, não existe antes).
- Meta de cobertura do repositório (`fail_under`) elevada de 68 para 80; cobertura atual ~95%.
- Sumário de drift passa a rotular o teste sobre `target` como *label shift* e a explicar a limitação (não mede mudança em P(y|X)).
- Links para ADRs na documentação das Etapas 1 e 2 corrigidos para `docs/adrs/`.

- Artefatos de `models/` e `reports/` regenerados com o dataset real (reprodução ponta a ponta em 2026-10-25). O retreino nesta máquina (xgboost 3.4.1) deu ROC-AUC 0,7491 (antes 0,7494): pequenas diferenças numéricas entre ambientes, sem mudar as conclusões.
