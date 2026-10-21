<!--
================================================================================
LOG DE MANUTENÇÃO DE DOCUMENTAÇÃO
--------------------------------------------------------------------------------
Data       | Autor          | Descrição
--------------------------------------------------------------------------------
2026-10-20 | Eduardo Batista | Adequação ao padrão canônico de ADR (título, log e seção 2).
================================================================================
-->

# ADR 0003: Modelo baseline: XGBoost em Pipeline scikit-learn

- **Status:** aceita
- **Etapa:** 1 (checklist 1.1 e 1.3)

## Contexto

O PDF lista **"Scikit-Learn / XGBoost — para treinar o modelo base de risco de crédito"** e pede "um modelo de classificação simples". O foco da fase não é a performance do modelo, e sim a camada de sustentação (contratos, drift, observabilidade, governança). O baseline só precisa ser bom o suficiente para que a degradação seja mensurável.

## 2. Decisão de Arquitetura

- **Modelo oficial:** `XGBClassifier` com hiperparâmetros fixos e conservadores (500 árvores, profundidade 4, `learning_rate` 0,05), dentro de um `Pipeline` do scikit-learn: features derivadas → one-hot → XGBoost. As duas bibliotecas citadas no PDF ficam em uso.
- **Modelo de comparação:** Regressão Logística com as mesmas features. Ela não vai para produção; serve só para mostrar que a escolha do XGBoost tem ganho mensurável (`reports/model/baseline_metrics.md`).
- **Sem tuning de hiperparâmetros:** hiperparâmetros fixos tornam o baseline estável e fácil de reproduzir, o que é o que interessa para comparar antes e depois do drift.
- **Ponto de corte:** o threshold que maximiza o **KS** na validação (estatística J de Youden). Com isso a decisão "aprovado/negado" existe e pode ser monitorada na Etapa 2 (taxa de aprovação, inadimplência da carteira aprovada).
- **Partição:** 60% treino / 20% validação (threshold) / 20% teste (métricas), estratificada e com `SEED = 42`.

## Métricas reportadas

ROC-AUC, **Gini** e **KS** (padrão de mercado em crédito), PR-AUC (o alvo é desbalanceado, cerca de 8% de default), Brier (calibração), precisão, recall e F1 no threshold. Do lado do negócio: **taxa de aprovação** e **inadimplência da carteira aprovada**.

## Equidade (preparação para a Etapa 4)

`gender` faz parte do contrato, mas **não entra no modelo**: fica disponível só para auditoria de viés. O pipeline descarta explicitamente qualquer coluna fora de `MODEL_INPUT_COLUMNS`.

## Consequências

- O artefato `models/baseline_model.joblib` guarda o pipeline completo junto com o threshold, a versão do modelo e a versão do contrato. Ele é pequeno e fica versionado no Git para que as Etapas 2 e 3 usem exatamente o mesmo modelo.
- Para carregar o pickle é preciso usar o mesmo ambiente (`uv.lock`).
