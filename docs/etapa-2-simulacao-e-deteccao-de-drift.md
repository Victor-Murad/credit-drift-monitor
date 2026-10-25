# Etapa 2 — Simulação e Detecção de Drift

> **Disciplina:** Drift / Model Monitoring · **Peso:** 25%
> **Foco:** tornar a degradação silenciosa **visível** (Data Drift e Concept Drift).
> **Entregável:** script de simulação de drift operante + relatório de detecção versionado, com impacto no modelo.

Todos os números abaixo vêm dos artefatos gerados por `make etapa2` sobre o **Home Credit** (seed 42): Referência 215.243 / Produção t0 92.247, default 8,07%. Pré-requisito: Etapa 1 (`make data` + `make train`).

---

## 2.1 Alinhamento e definição do problema

| Item do checklist | Status | Onde / observação |
|---|---|---|
| Confirmar variáveis importantes (≥ 2) para o drift | ✅ | Definidas na Etapa 1 via permutation importance. |
| Separar Referência e Produção em pastas distintas | ✅ | Já feito em `data/reference/` e `data/production/` (Boa Prática do PDF). |
| Simular Data Drift em variáveis importantes | ✅ | `annuity_amount`, `ext_source_3` (+ `income_total` complementar). |
| Simular Concept Drift (relação X → y) | ✅ | Flip de `target` em faixas de baixo risco (`ext_source_3` alto). |
| Garantir que lotes driftados passam no contrato | ✅ | Clip aos limites Pandera; gate APROVADO em t0/t1/t2. |
| Detectar drift com ferramenta (Evidently) | ✅ | [ADR-0004](adrs/0004-evidently-drift-detection.md). |
| Medir impacto no modelo / negócio | ✅ | AUC, aprovação e default na carteira aprovada (t0 vs t1 vs t2). |

### Variáveis e choques (contrato de simulação)

| Tipo | Variável | Choque | Narrativa |
|---|---|---|---|
| Data drift (obrigatório) | `annuity_amount` | ×1,25 (clip: parcela ≤ crédito) | Choque de juros / Selic. |
| Data drift (obrigatório) | `ext_source_3` | −0,15 (clip [0, 1]) | Deterioração do score de bureau. |
| Data drift (complementar) | `income_total` | ×1,15 (clip ≤ 10M) | Exemplo do PDF; importância baixa. |
| Concept drift | `target` | flip 25% dos bons com `ext_source_3 ≥ 0,55` | Mesmo perfil passa a inadimplir mais. |

> **Drift ≠ dado inválido.** Uma parcela maior é plausível; uma negativa não. A simulação clipa nos limites do contrato para o gate continuar liberando o lote — a estabilização fica a cargo do Evidently.

---

## 2.2 Setup e módulo de drift

| Item | Status | Onde |
|---|---|---|
| Dependência Evidently | ✅ | [`pyproject.toml`](../pyproject.toml) + [`uv.lock`](../uv.lock) |
| ADR da escolha | ✅ | [ADR-0004](adrs/0004-evidently-drift-detection.md) |
| Pacote `src/drift/` | ✅ | `simulate.py`, `detect.py`, `report.py`, `demo.py` |
| Caminhos e constantes | ✅ | [`src/config.py`](../src/config.py): `DRIFT_REPORTS_DIR`, `PRODUCTION_T1_FILE`, `PRODUCTION_T2_FILE`, choques |
| Makefile | ✅ | `make simulate-drift`, `make detect-drift`, `make drift-demo`, `make etapa2` |
| Testes sintéticos (sem Kaggle) | ✅ | `tests/test_simulate.py`, `tests/test_detect.py` |

Estrutura espelhando a Etapa 1 (`src/validation/`):

```
src/drift/
  simulate.py   # aplica choques + clip + grava t1/t2
  detect.py     # Evidently DataDriftPreset (KS / chi-square) + PSI por coluna
  psi.py        # Population Stability Index (tamanho do efeito)
  performance.py# queda de ROC-AUC vs baseline (sinal de concept drift)
  report.py     # HTML (Evidently) + JSON estruturado
  demo.py       # orquestra: simular → gate → score → detectar → sumário
```

---

## 2.3 Simulação das fatias de produção

Ponto de partida: [`data/production/production_base.parquet`](../data/production/) (lote t0 da Etapa 1, nunca visto no treino, com rótulo).

```bash
uv run python -m src.drift.simulate
# ou: make simulate-drift
```

| Fatia | Arquivo | O que muda |
|---|---|---|
| **t0** | `production_base.parquet` | Sem drift (controle). |
| **t1** | `production_t1_data_drift.parquet` | Só Data Drift (`apply_data_drift`). |
| **t2** | `production_t2_concept_drift.parquet` | Data Drift + Concept Drift (`apply_concept_drift`). |

Na execução documentada abaixo (Home Credit, seed 42): taxa de default t0/t1 = **8,07%** (igual à Etapa 1); t2 sobe para **11,75%** após o flip de rótulos. Lote: **92.247** linhas.

---

## 2.4 Detecção (Evidently) e impacto no modelo

```bash
uv run python -m src.drift.demo
# ou: make etapa2
```

Fluxo da demo:

1. Simula t1 e t2.
2. Pontua t0/t1/t2 com `score_batch` (sempre atrás do gate do contrato).
3. Compara cada fatia com `reference.parquet` via Evidently (`num_method=ks`, `cat_method=chisquare`).
4. Grava `reports/drift/drift_demo_summary.md` + HTML/JSON por fatia.

### Gate do contrato

| Lote | Status do gate | Registros pontuados |
|---|---|---|
| t0 | **APROVADO** | todos |
| t1 (data drift) | **APROVADO** | todos |
| t2 (data+concept) | **APROVADO** | todos |

### Detecção vs Dataset de Referência

Amostra Evidently: 15.000 linhas da Referência e da corrente (seed 42). Limiar de *dataset drift*: share ≥ 30% das colunas monitoradas. Coluna marcada = KS/chi² significativo **e** PSI ≥ 0,10 (`min_psi`, [ADR-0005](adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md)).

| Fatia | Dataset drift | Colunas marcadas (PSI) | Label shift (`target`) |
|---|---|---|---|
| t0 | **não** (0/7) | nenhuma (sem `min_psi`, o KS marcava `annuity_amount` com PSI 0,00: ruído) | não |
| t1 | **SIM** (3/7) | `annuity_amount` (0,17), `ext_source_3` (0,97), `income_total` (0,76) | não |
| t2 | **SIM** (3/7) | as mesmas três | não (KS significativo, mas PSI 0,02) |

> **Atenção ao "Concept drift (`target`)":** o teste sobre `target` detecta só mudança de *prevalência* do rótulo (label/prior shift). Não mede mudança em P(y|X). O sinal confiável é a queda de ROC-AUC (seção abaixo e `performance` no sumário). Com dados reais, o KS sozinho marcava `annuity_amount` em t0 (PSI 0,00) e `target` em t2 (PSI 0,02); `min_psi=0,10`, usado na demo, filtra esses alertas ([ADR-0005](adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md)).

Leitura: t1 prova Data Drift sem Concept Drift; t2 adiciona a mudança de rótulo. t0 permanece **sem** dataset drift frente à Referência (split estratificado da Etapa 1).

### Impacto no modelo (baseline-v1, Home Credit, 92.247 linhas)

| Fatia | ROC-AUC | Aprovação | Default na carteira aprovada | Default base |
|---|---|---|---|---|
| t0 | **0,7533** | 64,57% | **3,74%** | 8,07% |
| t1 (data) | 0,7224 | 51,84% | 3,67% | 8,07% |
| t2 (data+concept) | **0,5720** | 51,84% | **9,84%** | 11,75% |

- AUC em t0 (**0,753**) alinhado ao teste da Etapa 1 (**0,749**).
- Queda de AUC t0→t2: **0,181** (limiar de alerta 0,05: t2 degradado, t1 não).
- Em t1 o Evidently detecta drift nas três variáveis; o AUC cai pouco (−0,031) — lição do vídeo: *drift detectado ≠ modelo degradado* (e `income_total` tem importância baixa).
- Em t2 o concept drift derruba o AUC e sobe a inadimplência da carteira aprovada (3,74% → 9,84%).

### Artefatos versionados

- [`reports/drift/drift_demo_summary.md`](../reports/drift/drift_demo_summary.md) — sumário da demonstração
- [`reports/drift/drift_demo_summary.json`](../reports/drift/drift_demo_summary.json) — mesmo conteúdo estruturado
- `reports/drift/production_t*_drift_report.html` — HTML do Evidently (~4 MB cada; úteis no vídeo)
- `reports/drift/contract_gate/` — relatórios do gate confirmando APROVADO

Para regenerar: `make data && make etapa2` (exige `data/raw/application_train.csv`).

---

## Checklist de saída da Etapa 2

- [x] Script de simulação de drift operante: `src/drift/simulate.py`
- [x] Detecção Data/Concept Drift com Evidently: `src/drift/detect.py`
- [x] Demonstração ponta a ponta: `src/drift/demo.py` + `make etapa2`
- [x] Evidências em `reports/drift/` (MD/JSON/HTML)
- [x] Lotes driftados passam no contrato (qualidade ≠ estabilidade)
- [x] ADR-0004 documentando a escolha do Evidently
- [x] PSI por coluna + filtro de efeito opcional e queda de AUC como sinal de concept drift ([ADR-0005](adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md))
- [x] `reports/drift/` regenerado com `make etapa2` (dataset real, 2026-10-25): artefatos versionados trazem `psi` e `performance`

## O que a Etapa 3 recebe pronto

| Recurso | Uso na Etapa 3 |
|---|---|
| Eventos `drift_simulated`, `drift_detected`, `drift_demo`, `batch_scored` em `logs/pipeline.jsonl` | Centralizar métricas e alertas |
| `reports/drift/*.json` | Painéis / histórico de drift por lote |
| Comparativo t0/t1/t2 (AUC, aprovação, default) | Alertas de degradação de performance |
| `performance.<fatia>.{drop,degraded}` no sumário JSON | Métrica de saúde do modelo (queda de AUC) e alerta |
| `column_results[].{psi,psi_level}` + `min_psi` | Contar alertas de drift sem falso positivo de amostra grande |
| Separação gate (contrato) vs Evidently (estabilidade) | Dois tipos de alerta distintos |
