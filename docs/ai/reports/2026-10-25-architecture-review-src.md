# Architecture Review: `src/` (2026-10-25)

> **Alvo:** `src/` (pipeline batch em Python: CLI + biblioteca; sem API, sem banco, sem rede em runtime).
> **Base:** commit `efd27fa` (branch `etapa3`).
> **Normas:** `guia-arquitetura-software-design-clean-code.md` e, no que se aplica, `guia-system-design-sistemas-distribuidos.md`.
> **Limitação de contexto:** não há SLO, volume nem requisito de consistência definidos. As lentes 3 e 4 foram avaliadas só no que se aplica a um pipeline batch local.

## Scorecard (50 pts)

| Lente | Nota | Status | Foco |
| :--- | :---: | :---: | :--- |
| 1. Fronteiras, domínio e dependências | 8/10 | OK | Sem ciclos, domínio puro; contrato do import-linter fraco |
| 2. Design interno, SOLID e Clean Code | 7/10 | WARN | Sem `except` silencioso; `demo.py` e `train.py` misturam responsabilidades |
| 3. Dados, consistência e persistência | 7/10 | WARN | Reprodutível (seed, sha256); `joblib.load` não verificava o hash |
| 4. Escala, resiliência e comunicação | 6/10 | WARN | Parser do Evidently falhava aberto |
| 5. Governança, observabilidade e evolução | 6/10 | WARN | 5 ADRs e logs JSONL; sem `run_id`, sem CI, sem rotação de log |

**Total: 34/50, maturidade intermediária.** Adequada às Etapas 1 e 2; três pontos a fechar antes da Etapa 3.

## Riscos e violações

| Sev. | Prioridade | Risco | Evidência | Ação |
| :---: | :---: | :--- | :--- | :--- |
| Médio | Antes da Etapa 3 | **Falha aberta** na detecção: se o Evidently mudar os nomes internos (`DriftedColumnsCount`, `ValueDrift`), `column_results` fica vazio e o lote sai "estável" | `src/drift/detect.py` (`_parse_snapshot`); `evidently>=0.7` sem teto | Lançar erro se não houver `ValueDrift`; limitar a versão |
| Médio | Antes da Etapa 3 | **Logs sem correlação**: sem `run_id`; `model_version` ausente em `drift_detected`; `FileHandler` sem rotação | `src/logging_utils.py`, `src/drift/detect.py` | `run_id` e `model_version` nos eventos; rotação |
| Médio | Sprint | **`joblib.load` sem verificar integridade** (o `baseline_metadata.json` já grava o sha256) | `src/models/predict.py:34`, `src/models/train.py:353` | Comparar o sha256 antes de carregar; falhar se divergir |
| Médio | Sprint | **Sem CI** (`.github/` só tem `dependabot.yml` e o template de PR) | `.github/workflows/` ausente | Workflow com lint, import-linter e cobertura |
| Médio | Sprint | **Acoplamento sem contrato**: `drift/demo.py` importa `evaluate` de `models/train.py`; `models/predict.py` importa `read_batch` de `validation/validate.py` (CLI) | Grafo de imports | Extrair `metrics` e `read_batch`; contrato `layers` |
| Baixo | Trimestre | `DriftResult.concept_drift` mede label shift; o sinal real é `performance.degraded` (ADR-0005) | `src/drift/detect.py` | Renomear para `label_shift` com alias legado |
| Baixo | Trimestre | `drift/demo.py` (318 linhas) e `models/train.py` (372 linhas) misturam orquestração, regra e renderização | — | Extrair a renderização quando a Etapa 3 tocar nesses arquivos |

## Plano de ação

**Quick wins:** parser fail-closed e teto do Evidently; verificação do sha256 do modelo; contrato `layers` no import-linter.
**Estruturais:** `src/models/metrics.py` e módulo de leitura de lote; `run_id` e `model_version` nos logs; workflow de CI.
**Decisões que exigem ADR (Tipo 1):** destino e esquema da telemetria da Etapa 3 (MLflow, JSONL ou ambos); política de alerta (`min_psi=PSI_MODERATE` e `performance.degraded` como gatilhos; rótulo atrasado).

## Status da mitigação (branch `etapa3-mitigacoes`)

| Item | Estado | Como foi fechado |
| :--- | :---: | :--- |
| Parser fail-closed + teto do Evidently | feito | `DriftParsingError` em `detect.py`; `evidently>=0.7,<0.8` |
| Verificação do sha256 no `load_bundle` | feito | `src/models/integrity.py`; recusa também se o metadata faltar |
| `metrics.py` + `read_batch` fora do CLI + contrato `layers` | feito | `src/models/metrics.py`, `src/validation/io.py` (e não `src/data/io.py`, que criaria ciclo `data` ↔ `validation`); contrato com 1 import ignorado (`validation.demo` → `models.predict`) |
| `run_id` / `model_version` / rotação de log | feito | `RUN_ID` em `logging_utils.py`; `RotatingFileHandler` 10 MB × 5 |
| Workflow de CI | feito, **não testado no GitHub** | `.github/workflows/ci.yml`; actions fixadas por tag (`v4`, `v5`), não por SHA |
| Renomear `concept_drift` → `label_shift` | adiado | Fica para a Etapa 3, junto com a política de alerta |
| ADR: telemetria e política de alerta | decidido em parte | Usuário escolheu **MLflow + DVC**; escopo e política de alerta seguem no brainstorming |

Verificação local: 84 testes passando, cobertura 95,6%, `make lint` (ruff e 2 contratos do import-linter) limpo.

## Atualização (2026-10-27): núcleo da Etapa 3

| Item | Estado | Evidência |
| :--- | :---: | :--- |
| Renomear `concept_drift` → `label_shift` | feito | `54ab4a3` (alias legado no JSON até a Etapa 4) e emenda da ADR-0005 em `f2f1b1b`. Os relatórios de `reports/drift/` foram regenerados com `label_shift` em `9f7e4bd` |
| ADR: telemetria e política de alerta | feito | ADR-0006 `Implementada` (2026-10-27). A política de alerta é a regra pura em `src/monitoring/alerts.py` (spec §4.2), com o piso de `ATENÇÃO` para anomalias documentado na spec e no README |
| Correlação log ↔ execução | feito | tag `pipeline_run_id` em cada run de estágio e trecho do `logs/pipeline.jsonl` do `run_id` anexado. Conferido pelo aceite real |
| Contrato de camadas | ampliado | `monitoring → drift → models → validation` e um contrato `forbidden` para `mlflow`/`dvc` fora de `src/monitoring/`. Os 3 contratos estão KEPT |
| `drift/demo.py` e `models/train.py` misturando responsabilidades | não revisto aqui | a Etapa 3 só adicionou ganchos (`on_batch`, retorno de `train.main`). A extração da renderização não foi avaliada nesta atualização |
| Workflow de CI | feito, **ainda não testado no GitHub** | sem mudança; o aceite real pula em CI (`CI` definido) |

Execução real com o dataset do Kaggle, `pipeline_run_id` `20261027T103457Z`, no MLflow local:
- t0 `OK`: AUC 0,7533;
- t1 `ATENCAO`: `drift:annuity_amount`, `drift:ext_source_3`, `drift:income_total` e `score_psi>=0.10` (0,143); AUC 0,7224, queda de 0,031;
- t2 `CRITICO`: `auc_drop>=0.05`, queda de 0,181; AUC 0,5720;
- lote inválido `BLOQUEADO`, com `intentional_demo=true` e `alerts_failure_count`=1.

`tests/integration/test_acceptance_real.py` deu 5 passed, 0 skipped. `make test-cov` deu 287 passed, com 97,17% de cobertura. O primeiro `dvc.lock` saiu do `make repro`, e a segunda rodada pulou os 4 estágios.

Pendente (Fase B, spec §4.7): retreino, challenger, promoção, rollback e ADR-0007.
