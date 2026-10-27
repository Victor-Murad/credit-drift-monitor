# Etapa 2 — Demonstracao de Drift

> Gerado em `2026-10-27T10:38:26.727947+00:00` · seed=42 · modelo `baseline-v1` · amostra Evidently=15000.

## Gate do contrato

Todos os lotes (t0, t1 com data drift, t2 com data+concept) **passaram** no contrato Pandera. Drift ≠ dado inválido.

## Impacto no modelo (lote completo)

| Fatia | ROC-AUC | Aprovação | Default na carteira aprovada | Default base |
|---|---|---|---|---|
| t0 (sem drift) | 0.7533 | 64.57% | 0.0374 | 8.07% |
| t1 (data drift) | 0.7224 | 51.84% | 0.0367 | 8.07% |
| t2 (data+concept) | 0.5720 | 51.84% | 0.0984 | 11.75% |

Queda de AUC t0→t2: **0.1813**.

## Detecção Evidently (vs Dataset de Referência)

| Fatia | Dataset drift | Colunas driftadas | Label shift (`target`) | Colunas |
|---|---|---|---|---|
| `production_t0` | nao | 0/7 | nao | — |
| `production_t1_data_drift` | SIM | 3/7 | nao | annuity_amount (PSI 0.17), ext_source_3 (PSI 0.97), income_total (PSI 0.76) |
| `production_t2_concept_drift` | SIM | 3/7 | nao | annuity_amount (PSI 0.17), ext_source_3 (PSI 0.97), income_total (PSI 0.76) |

> Coluna marcada = KS/chi² significativo **e** PSI ≥ 0.10 (filtra ruído de amostra grande; ADR-0005).

> **Como ler:** o teste sobre `target` detecta só mudança na *prevalência* do rótulo (label/prior shift), não mudança em P(y|X). O PSI é o tamanho do efeito (< 0,10 estável · 0,10–0,25 moderado · ≥ 0,25 significativo): KS marcado com PSI baixo é provável ruído de amostra grande.

## Degradação de performance (sinal de Concept Drift)

| Fatia | ROC-AUC | Queda vs t0 | Limiar | Modelo degradado |
|---|---|---|---|---|
| `production_t0` | 0.7533 | 0.0000 | 0.05 | nao |
| `production_t1_data_drift` | 0.7224 | 0.0309 | 0.05 | nao |
| `production_t2_concept_drift` | 0.5720 | 0.1813 | 0.05 | SIM |

A queda de ROC-AUC com rótulos de feedback é o sinal confiável de concept drift: mede se o modelo passou a errar mais, independentemente de o rótulo ter mudado de prevalência.

## Artefatos

- Sumário JSON: `reports/drift/drift_demo_summary.json`
- Relatórios HTML: `reports/drift/*_drift_report.html`
- t1: `data/production/production_t1_data_drift.parquet`
- t2: `data/production/production_t2_concept_drift.parquet`

## Lições para o vídeo

1. **Qualidade ≠ estabilidade:** o gate aprovou t1/t2; o Evidently marcou drift nas variáveis importantes (`annuity_amount`, `ext_source_3`).
2. **`income_total` pode driftar sem derrubar o modelo** (importância baixa).
3. **Concept drift** (t2) altera o rótulo e degrada AUC / inadimplência da carteira aprovada mesmo com o mesmo pipeline; o alerta confiável é a queda de AUC, não o teste sobre `target`.
