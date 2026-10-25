<!--
================================================================================
LOG DE MANUTENÇÃO DE DOCUMENTAÇÃO
--------------------------------------------------------------------------------
Data       | Autor          | Descrição
--------------------------------------------------------------------------------
2026-10-25 | Eduardo Batista | Criação da ADR 0005 detalhando PSI e queda de ROC-AUC como sinais complementares ao KS.
================================================================================
-->

# ADR 0005: PSI e queda de ROC-AUC como sinais complementares de drift

| Metadado | Detalhe |
| :--- | :--- |
| **Status** | **Implementada** (2026-10-25) |
| **Decisores** | Eduardo Batista |
| **Tags** | `[drift]`, `[observability]`, `[model-monitoring]` |
| **Impacto Sistêmico** | Médio (`src/drift/`, relatórios de `reports/drift/`, insumo da Etapa 3) |
| **ADRs Relacionadas** | Extends ADR-0004 |

## Contexto

A ADR-0004 adotou o Evidently com teste KS (numéricas) e qui-quadrado (categóricas), decidindo por p-valor. A validação da Etapa 2 expôs duas limitações que afetam diretamente a Etapa 3, que contará alertas de drift:

### Riscos Identificados e Problemas a Resolver:
1. **Falso alerta por tamanho de amostra**: com 15 mil linhas, o KS acusa deslocamentos desprezíveis. Em t0 (sem drift induzido), `annuity_amount` aparece como driftada. O p-valor mede *se* a distribuição mudou, não *quanto*.
2. **"Concept drift" que não é concept drift**: o teste sobre a coluna `target` detecta apenas mudança de prevalência do rótulo (label/prior shift), não mudança em P(y|X). O sinal confiável (ROC-AUC 0,753 → 0,573 em t2) existia só em `demo.py`, fora dos relatórios.
3. **PSI ausente**: o PDF aceita "PSI e/ou KS", mas a métrica padrão de credit scoring (PSI) não era calculada.

---

## 2. Decisão de Arquitetura

1. **PSI por coluna** (`src/drift/psi.py`), calculado em todo `ColumnDrift`: faixas por decis da Referência (numéricas) ou por categoria; ausentes formam faixa própria. Classificação: `< 0,10` estável · `0,10–0,25` moderado · `≥ 0,25` significativo (convenção de mercado; constantes em `src/config.py`).
2. **Filtro de efeito opcional**: `run_drift_detection(min_psi=...)` / `--min-psi` só marca a coluna quando KS é significativo **e** PSI ≥ limiar. O padrão da função continua sendo só p-valor (comportamento retrocompatível). A demo (`src/drift/demo.py`) usa `min_psi=PSI_MODERATE`, decisão validada com os dados reais em 2026-10-25 (ver Consequências); a Etapa 3 deve fazer o mesmo ao contar alertas.
3. **Queda de ROC-AUC como sinal de concept drift** (`src/drift/performance.py`): `assess_auc_drop` compara o AUC de cada fatia com o baseline t0; degradado se a queda ≥ `AUC_DROP_ALERT` (0,05). Entra em `drift_demo_summary.json/.md` (chave `performance`) e no evento `drift_demo`.
4. O campo `concept_drift` de `DriftResult` permanece por compatibilidade, mas é documentado como **label shift**; o sumário o rotula assim.

### Matriz de Avaliação de Alternativas (Trade-off Matrix)

| Critério de Avaliação | A: Só KS (status quo) | B: PSI informativo + filtro opcional + AUC (eleita) | C: Trocar a regra de decisão para KS+PSI já |
| :--- | :--- | :--- | :--- |
| **Custo / Esforço** | Baixo | Baixo | Baixo |
| **Falsos alertas** | Alto | Baixo (com `min_psi`) | Baixo |
| **Risco de regressão nos relatórios versionados** | Nenhum | Nenhum | Alto (sem dataset para regenerar) |
| **Detecta concept drift real** | Não | Sim (via AUC, exige rótulos) | Não |
| **Complexidade operacional** | Baixa | Baixa | Baixa |

---

## 6. Matriz de Conformidade e Mitigação de Riscos

| Risco Mapeado no Contexto | Impacto Potencial | Mitigação Arquitetural e Técnica Adotada |
| :--- | :--- | :--- |
| Falso alerta por amostra grande | Médio | PSI reportado em toda coluna + `min_psi` opcional |
| Concept drift não detectado por falta de rótulos | Alto | Limitação documentada: a queda de AUC exige feedback rotulado (atrasado); sem rótulo só há data drift e PSI |
| Relatórios versionados defasados do código | Baixo | Regenerar com `make etapa2` quando o dataset estiver disponível |

---

## 7. Consequências e Resultados

### Positivas
- **Alertas auditáveis**: cada coluna driftada traz p-valor *e* tamanho do efeito.
- **Insumo pronto para a Etapa 3**: `performance.*.degraded` e `psi_level` viram métricas de saúde do modelo e alertas sem novo cálculo.

### Mitigações e Desafios Gerenciados
- **Validação com dados reais (Home Credit, seed 42):** sem `min_psi`, o KS marcava `annuity_amount` em t0 (PSI 0,00, falso alerta) e `target` em t2 (PSI 0,02). Com `min_psi=0,10`, t0 fica sem alertas; t1/t2 mantêm `annuity_amount` (0,17), `ext_source_3` (0,97) e `income_total` (0,76). O `income_total` continua marcado: o PSI alto vem de valores repetidos (rendas redondas) que cruzam as faixas com o choque de +15%.
- **O teste sobre `target` deixa de alertar em t2** com o filtro, enquanto a queda de AUC (0,181) acusa a degradação. Isso confirma que apenas a performance revela o concept drift simulado.
- **Os campos `psi`/`performance`** já estão nos artefatos versionados em `reports/drift/`.
