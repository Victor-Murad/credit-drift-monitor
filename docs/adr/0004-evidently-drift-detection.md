# ADR-0004 — Detecção de drift com Evidently

- **Status:** aceita
- **Etapa:** 2 (simulação e detecção de Data Drift e Concept Drift)

## Contexto

A Etapa 2 pede simular degradação econômica (Data Drift em variáveis importantes e Concept Drift na relação X → y) e **detectar** essa mudança com relatório versionado. O PDF não obriga uma biblioteca específica; o material da disciplina e o CONTRIBUTING do repositório apontam para **Evidently**.

Alternativas consideradas: PSI “na mão”, `alibi-detect`, Deepchecks. PSI isolado cobre só distribuição univariada e não gera o relatório HTML didático do vídeo. Alibi-detect e Deepchecks são mais pesados e menos alinhados ao stack já usado (pandas + scikit-learn).

## Decisão

Usar **Evidently** ≥ 0.7 (`Report` + `DataDriftPreset`) em `src/drift/detect.py`, comparando `data/reference/reference.parquet` com as fatias de produção (`production_base`, `production_t1_*`, `production_t2_*`). A coluna `target` entra no mesmo preset (`ValueDrift`) para evidenciar Concept Drift; não há `TargetDriftPreset` separado nesta API.

## Justificativa

| Critério | Evidently | PSI manual | Alibi-detect |
|---|---|---|---|
| Relatório HTML pronto para o vídeo | Sim | Não | Limitado |
| Data Drift + Target/Concept Drift | Presets nativos | Só PSI | Sim, mais config |
| Integração com DataFrames pandas | Direta | Direta | Direta |
| Já previsto no CONTRIBUTING | `chore(deps): adicionar evidently` | — | — |

## Consequências

- Dependência `evidently` no `pyproject.toml` / `uv.lock`.
- Relatórios em `reports/drift/` (HTML + JSON + sumário Markdown).
- A simulação (`src/drift/simulate.py`) continua independente do Evidently: gera Parquets que **passam no contrato Pandera** (clip). Drift ≠ dado inválido.
