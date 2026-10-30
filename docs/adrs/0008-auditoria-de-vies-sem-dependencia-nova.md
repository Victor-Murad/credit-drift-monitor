<!--
================================================================================
LOG DE MANUTENÇÃO DE DOCUMENTAÇÃO
--------------------------------------------------------------------------------
Data       | Autor   | Descrição
--------------------------------------------------------------------------------
2026-10-29 | Etapa 4 | Criação da ADR 0008: auditoria de viés executável, sem dependência nova.
================================================================================
-->

# ADR 0008: Auditoria de viés executável com métricas próprias (sem Fairlearn/AIF360)

| Metadado | Detalhe |
| :--- | :--- |
| **Status** | **Implementada** (2026-10-29); execução com dados reais pendente |
| **Tags** | `[governance]`, `[fairness]`, `[lgpd]`, `[etapa-4]` |
| **Etapa** | 4 (checklist 4.1, mitigação de vieses) |
| **ADRs Relacionadas** | Extends ADR-0003 (`gender` fora do modelo) |

## Contexto

A ADR-0003 deixou `gender` no contrato só para auditoria, mas nenhum código media viés. O critério de Governança (15%) pede a mitigação de vieses documentada. Sem medir, a documentação ficaria só em intenção. Os dados permitem a medição: o contrato traz `gender` (F/M) e `age_years`, e os lotes rotulados (`production_base`) trazem o `target`.

## 2. Decisão de Arquitetura

- Módulo `src/governance/fairness.py`, camada irmã de `src/monitoring` no import-linter: pode usar `models` e `validation`, nunca o inverso, e não importa MLflow.
- Métricas calculadas com pandas e `roc_auc_score` do scikit-learn, já dependências do projeto. Por grupo de `gender` e de faixa etária: taxa de aprovação, aprovação entre bons e entre maus pagadores, inadimplência da carteira aprovada e ROC-AUC. Por atributo: razão de impacto disparatado e diferenças máximas entre grupos.
- O lote passa pelo mesmo gate do contrato antes de ser pontuado (`score_batch`).
- O limiar 0,80 (regra dos quatro quintos) só marca `needs_review`. É uma referência de investigação, não um critério legal brasileiro.
- `make fairness [BATCH=...]` grava `reports/governance/<lote>_fairness.{json,md}`.

### Alternativas descartadas

| Critério | Métricas próprias (eleita) | Fairlearn / AIF360 |
| :--- | :--- | :--- |
| Dependência nova no `uv.lock` | Nenhuma | Sim (AIF360 traz dependências pesadas) |
| Métricas necessárias aqui | Cobertas (paridade, igualdade de oportunidade, equalized odds, AUC por grupo) | Cobertas e muitas outras |
| Mitigação in-processing | Não tem | Fairlearn tem (`ExponentiatedGradient`) |
| Testabilidade | Valores conferidos à mão em `tests/test_fairness.py` | Idem, com mais superfície |

Se a auditoria com dados reais indicar necessidade de mitigação in-processing, a troca para o Fairlearn deve virar uma ADR nova.

## Consequências

- A auditoria é reproduzível e testada (`tests/test_fairness.py`, dados sintéticos). Nenhum resultado com dados reais foi produzido na Etapa 4, porque o dataset não estava disponível na máquina. O README não cita números de fairness.
- Auditar `gender` exige mantê-lo no lote, o que é um tratamento de dado pessoal com finalidade própria (não discriminação, LGPD art. 6º, IX). Essa finalidade está declarada no contrato (`terms.limitations`).
- A auditoria não roda dentro do `make pipeline` nem do `dvc.yaml`: é um comando explícito, para não mudar os estágios aceitos na Etapa 3.
