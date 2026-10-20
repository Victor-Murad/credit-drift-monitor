# Etapa 2 — Demonstracao de Drift

> Gerado em `2026-10-20T01:43:35.988602+00:00` · seed=42 · modelo `baseline-v1` · amostra Evidently=15000.

## Gate do contrato

Todos os lotes (t0, t1 com data drift, t2 com data+concept) **passaram** no contrato Pandera. Drift ≠ dado inválido.

## Impacto no modelo (lote completo)

| Fatia | ROC-AUC | Aprovação | Default na carteira aprovada | Default base |
|---|---|---|---|---|
| t0 (sem drift) | 0.7534 | 64.95% | 0.0376 | 8.07% |
| t1 (data drift) | 0.7223 | 52.04% | 0.0367 | 8.07% |
| t2 (data+concept) | 0.5733 | 52.04% | 0.0982 | 11.75% |

Queda de AUC t0→t2: **0.1801**.

## Detecção Evidently (vs Dataset de Referência)

| Fatia | Dataset drift | Colunas driftadas | Concept drift (`target`) | Colunas |
|---|---|---|---|---|
| `production_t0` | nao (abaixo do limiar 30%) | 1/7 | nao | annuity_amount (ruido KS; sem dataset drift) |
| `production_t1_data_drift` | SIM | 3/7 | nao | annuity_amount, ext_source_3, income_total |
| `production_t2_concept_drift` | SIM | 4/7 | SIM | annuity_amount, ext_source_3, income_total, target |

## Artefatos

- Sumário JSON: `reports/drift/drift_demo_summary.json`
- Relatórios HTML: `reports/drift/*_drift_report.html`
- t1: `data/production/production_t1_data_drift.parquet`
- t2: `data/production/production_t2_concept_drift.parquet`

## Lições para o vídeo

1. **Qualidade ≠ estabilidade:** o gate aprovou t1/t2; o Evidently marcou drift nas variáveis importantes (`annuity_amount`, `ext_source_3`).
2. **`income_total` pode driftar sem derrubar o modelo** (importância baixa).
3. **Concept drift** (t2) altera o rótulo e degrada AUC / inadimplência da carteira aprovada mesmo com o mesmo pipeline.
