# Roteiro do vídeo STAR (até 5 minutos)

> Todos os números vêm de artefatos versionados no repositório. Ao lado de cada fala está o arquivo de onde ela sai. Não acrescente resultados que não estejam nesses arquivos. Os números de fairness vêm de `reports/governance/production_base_fairness.md`.

## Preparação (antes de gravar)

1. Numa máquina com `data/raw/application_train.csv`, rode `make pipeline`. No Windows, sem `make`, rode os quatro estágios com `uv run python -m src.monitoring.stage <estágio>`. O pipeline regenera os relatórios e o `mlflow.db`.
2. Rode `make fairness` (no Windows: `uv run python -m src.governance.fairness`) para regenerar `reports/governance/production_base_fairness.md`.
3. Abra em abas: o README no GitHub, `reports/data_contract/invalid_batch_report.html`, `reports/drift/production_t1_data_drift_drift_report.html`, `reports/drift/production_t2_concept_drift_drift_report.html`, `reports/drift/drift_demo_summary.md` e o MLflow UI (`make tracking-ui`, em http://127.0.0.1:5001).
4. Deixe um terminal pronto na raiz do repositório.

Se os números da nova execução diferirem dos abaixo (por exemplo, diferença numérica entre sistemas operacionais), **use os da nova execução** e atualize este roteiro.

## S — Situação (0:00–0:40)

**Fala:** "Uma fintech tem um modelo de Credit Scoring em produção. A economia mudou, com inflação e novos perfis de cliente, e a diretoria suspeita que o modelo está errando sem que ninguém perceba. É a degradação silenciosa: o modelo continua respondendo, mas as respostas pioram. Em crédito, isso é inadimplência na carteira ou bons clientes recusados."

**Tela:** topo do README (Contexto e Objetivo do modelo).

## T — Tarefa (0:40–1:10)

**Fala:** "A tarefa é construir a camada de sustentação: barrar dado ruim antes do modelo, detectar drift com estatística, observar o pipeline e documentar governança e LGPD. O modelo base é um XGBoost com ROC-AUC de 0,749 no teste e KS de 0,370."

**Tela:** tabela "Requisitos obrigatórios do repositório" no README; depois `reports/model/baseline_metrics.md`.

## A — Ação (1:10–3:45)

### 1. Data Contract (1:10–1:50)

**Fala:** "O contrato é escrito em Pandera, com 11 regras: unicidade, idade de 18 a 100 anos, limites de renda e crédito, domínios categóricos, scores em 0 a 1, parcela menor ou igual ao crédito, e `strict`, que bloqueia colunas não previstas, como um CPF. O gate é fail-closed: se o lote viola uma regra, ele inteiro vai para a quarentena. Na demonstração, injetamos 10 defeitos num lote e o contrato pegou os 10."

**Telas:** `src/validation/contract.py` (lista R1–R11 no docstring); `reports/data_contract/invalid_batch_report.html` (status BLOCKED, regras violadas, coluna `cpf` barrada). Fonte: `reports/data_contract/contract_demo_summary.md`.

**Decisão a citar:** Pandera em vez de Great Expectations, porque o contrato fica como código testável e há `strict` para PII inesperada (ADR-0002).

### 2. Simulação e detecção de drift (1:50–2:50)

**Fala:** "Separamos fisicamente referência e produção: 215.243 linhas de treino e 92.247 de produção, nunca vistas no treino. A partir da produção simulamos t1, com choque de parcela, renda e score de bureau, e t2, com o mesmo choque mais mudança de conceito: bons pagadores de score alto passam a dar default. Drift não é dado inválido: as três fatias passam no contrato. O Evidently aplica KS por coluna, e o PSI mede o tamanho do efeito. Só marcamos drift quando o teste é significativo e o PSI é pelo menos 0,10. Isso evita falso alarme em amostra grande."

**Telas:** `reports/drift/production_t1_data_drift_drift_report.html` (colunas com drift); tabela "Detecção Evidently" em `reports/drift/drift_demo_summary.md`: `annuity_amount` com PSI 0,17, `ext_source_3` com 0,97 e `income_total` com 0,76.

**Ponto técnico a citar:** "O teste sobre o `target` em t2 mede só a prevalência do rótulo. O PSI dele é 0,018, estável, mas a AUC cai 0,181. Concept drift se vê pela performance, não pelo teste de distribuição."

### 3. Observabilidade (2:50–3:10)

**Fala:** "Cada estágio e cada lote vira um run no MLflow local, com saúde, latência, volume, PSI, AUC e um nível de alerta. O retreino só roda a partir de um alerta crítico, e a promoção do novo modelo é uma decisão humana, auditada."

**Tela:** MLflow UI, experimento `credit-drift-monitor`, runs de lote com a tag do nível de alerta. Fonte dos números: `docs/ai/reports/2026-10-27-etapa3-aceite.md`.

### 4. Governança e LGPD (3:10–3:45)

**Fala:** "Na governança: lemos só 21 das 122 colunas, e gênero fica fora do modelo. A base legal é a proteção do crédito, no art. 7º, X, e, como a decisão é automatizada, o art. 20 garante ao titular o direito de pedir revisão. A auditoria de viés achou um problema real: por gênero, a razão de impacto é 0,90; por idade, é 0,46. Entre bons pagadores, um jovem de 18 a 24 anos é aprovado em 41% dos casos, e um idoso, em 84%. Isso ficou documentado como achado que exige revisão humana antes de produção."

**Tela:** README, seção 4.1 (tabelas de base legal e de retenção, com as marcações [implementado] e [recomendação]); depois `reports/governance/production_base_fairness.md` (tabela por faixa etária).

## R — Resultado (3:45–4:45)

**Fala:** "O resultado é que a degradação deixou de ser silenciosa. Em t0 o alerta é OK, com AUC de 0,753. Em t1 é ATENÇÃO: há drift em três colunas e a aprovação cai de 65,0% para 52,0%. Em t2 é CRÍTICO: a AUC cai para 0,573 e a inadimplência na carteira aprovada sobe de 3,8% para 9,8%. Como cada fatia muda uma coisa só, conseguimos separar as causas: 0,031 da queda de AUC vem do choque nas features e 0,149 da mudança de conceito, ou seja, 83%. No retreino, o challenger ganhou 0,13 de AUC no holdout. E o lote sujo foi bloqueado antes de ser pontuado."

**Telas:** tabela de impacto em `reports/drift/drift_demo_summary.md`; tabela da análise causal no README (seção 4.3); `reports/drift/production_t2_concept_drift_drift_report.html`.

**Lições (fala curta):**
1. "Qualidade de dado não é estabilidade: o gate aprovou t1 e t2, e o drift apareceu mesmo assim."
2. "Concept drift exige rótulos: sem feedback, só se vê o data drift."
3. "Nossa causa é comprovada na simulação. No mundo real, ela é uma hipótese que precisa de lotes datados."

## Encerramento (4:45–5:00)

**Fala:** "Contrato, drift, observabilidade e governança, tudo reproduzível com `uv sync` e `make pipeline`. O link do repositório está na descrição."

**Tela:** README, seção "Validação de reprodutibilidade".

## Checklist antes de publicar

- [ ] Duração ≤ 5 minutos.
- [ ] As quatro letras do STAR aparecem explicitamente (título na tela ou na fala).
- [ ] Os números falados conferem com os arquivos citados.
- [ ] Nenhum dado pessoal na tela: não abra `data/` nem a quarentena e não role até os exemplos de `client_id` no relatório do contrato.
- [ ] Nenhum token ou caminho pessoal visível no terminal.
- [ ] Link publicado com permissão de visualização para os avaliadores e adicionado na seção "Vídeo STAR" do README.
