# Auditoria de viés (fairness)

> Gerado em `2026-10-30T02:41:37.221287+00:00` · lote `production_base` · modelo `baseline-v1` · 92247 linhas.
> Desfecho favorável = crédito aprovado. A razão de impacto disparatado abaixo de 0.80 é referência de investigação, não prova de discriminação.

## `gender`

| Grupo | n | Default | Aprovação | Aprovação (bons) | Aprovação (maus) | Default aprovados | ROC-AUC |
|---|---|---|---|---|---|---|---|
| F | 60688 | 0.0698 | 0.6735 | 0.6993 | 0.3296 | 0.0342 | 0.7499 |
| M | 31559 | 0.1017 | 0.6033 | 0.6413 | 0.2675 | 0.0451 | 0.7511 |

- Razão de impacto disparatado: **0.8958** (revisão necessária: não)
- Diferença de paridade estatística: 0.0702
- Diferença de igualdade de oportunidade: 0.0580
- Diferença de aprovação entre maus pagadores: 0.0621
- Diferença de ROC-AUC entre grupos: 0.0012

## `age_band`

| Grupo | n | Default | Aprovação | Aprovação (bons) | Aprovação (maus) | Default aprovados | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 18-24 | 3695 | 0.1248 | 0.3786 | 0.4116 | 0.1475 | 0.0486 | 0.7035 |
| 25-34 | 21629 | 0.1069 | 0.5091 | 0.5458 | 0.2028 | 0.0426 | 0.7427 |
| 35-44 | 25395 | 0.0858 | 0.6390 | 0.6709 | 0.2989 | 0.0401 | 0.7493 |
| 45-59 | 30754 | 0.0643 | 0.7265 | 0.7497 | 0.3891 | 0.0345 | 0.7516 |
| 60+ | 10774 | 0.0479 | 0.8291 | 0.8419 | 0.5756 | 0.0332 | 0.7101 |

- Razão de impacto disparatado: **0.4566** (revisão necessária: SIM)
- Diferença de paridade estatística: 0.4505
- Diferença de igualdade de oportunidade: 0.4303
- Diferença de aprovação entre maus pagadores: 0.4281
- Diferença de ROC-AUC entre grupos: 0.0481
