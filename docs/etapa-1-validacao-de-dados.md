# Etapa 1 — Validação de Dados e Contratos

> **Disciplina:** Validação de Dados e Bibliotecas de Qualidade · **Peso:** 25%
> **Foco:** garantir que lixo não entre no modelo.
> **Entregável do PDF:** *"Script de validação de dados operante e modelo baseline treinado."*

Todos os números abaixo vêm dos artefatos gerados pelos scripts (seed 42). Para reproduzir: `make etapa1`, ou os três comandos `uv run` do [README](../README.md#como-executar).

---

## 1.1 Alinhamento e definição do problema

| Item do checklist | Status | Onde / observação |
|---|---|---|
| Ler o enunciado e revisar os pesos (Validação 25%, Drift 25%, Observabilidade 20%, Governança 15%, Vídeo 15%) | ✅ | Os itens de maior peso (Validação + Drift = 50%) guiaram as prioridades. |
| Confirmar com a coordenação: prazo, formato e se é individual ou em grupo | ⏳ **grupo** | Ver [dúvidas para o Discord](#dúvidas-para-a-coordenação-discord). |
| Combinar a divisão de responsabilidades | ⏳ **grupo** | Tabela "Equipe" no README. |
| Selecionar o dataset (classificação binária financeira, ≥ 5.000 amostras) | ✅ | **Home Credit Default Risk**, com 307.511 linhas. O German Credit foi descartado por ter só 1.000. [ADR-0001](adr/0001-dataset-home-credit.md) |
| Validar que o dataset comporta degradação econômica | ✅ | Tem renda, parcela, crédito, emprego, idade e scores de bureau. [ADR-0001](adr/0001-dataset-home-credit.md) |
| Definir o algoritmo do modelo base | ✅ | XGBoost em Pipeline scikit-learn, com Regressão Logística para comparação. [ADR-0003](adr/0003-modelo-baseline-xgboost.md) |
| Definir pelo menos 2 variáveis para o drift da Etapa 2 | ✅ | Ver [seção abaixo](#variáveis-definidas-para-o-drift-da-etapa-2). |
| Registrar dúvidas do enunciado no Discord | ⏳ **grupo** | Mensagem pronta [abaixo](#dúvidas-para-a-coordenação-discord). |

### Variáveis definidas para o drift da Etapa 2

O PDF pede que o drift seja aplicado a *"pelo menos duas variáveis **importantes**"*. Para escolher, medimos a importância de cada coluna do contrato no baseline com *permutation importance*, ou seja, quanto o ROC-AUC cai quando a coluna é embaralhada no conjunto de teste ([`reports/model/baseline_metrics.md`](../reports/model/baseline_metrics.md)):

| # | Variável | Queda de AUC | Narrativa econômica |
|---|---|---|---|
| 1 | `ext_source_3` | 0,0583 | Score de bureau: endividamento das famílias e novos perfis com histórico pior. |
| 2 | `annuity_amount` | 0,0484 | Parcela: alta de juros/Selic aumenta a parcela para o mesmo valor de crédito. |
| 3 | `ext_source_2` | 0,0484 | Score de bureau (segunda fonte). |
| 4 | `credit_amount` | 0,0415 | Valor do crédito: a inflação aumenta o ticket pedido. |
| 7 | `years_employed` | 0,0061 | Desemprego: clientes com menos tempo de casa. |
| 10 | `income_total` | 0,0019 | Renda: é o exemplo citado no PDF ("aumentar artificialmente a renda"). |

**Definição para a Etapa 2:**

- **Drift obrigatório, em variáveis importantes:**
  1. **`annuity_amount`**: choque de juros, com parcelas maiores para o mesmo crédito. Afeta também as razões `annuity_to_income` e `credit_to_annuity`, calculadas dentro do pipeline.
  2. **`ext_source_3`** (e/ou `ext_source_2`): deterioração do score de bureau causada pelo novo perfil de cliente.
- **Drift complementar, recomendado para enriquecer a análise:**
  - **`income_total`**: inflação nominal da renda, que é o exemplo do PDF. Como a importância dela é baixa, esse drift deve ser **detectado** pelo Evidently sem degradar muito o modelo. É uma boa lição para o vídeo: *drift detectado ≠ modelo degradado*.
  - **`years_employed` / `income_type`**: aumento do desemprego.
- **Concept drift:** alterar a relação entre X e y (ex.: a mesma faixa de score passa a ter mais default) usando o rótulo de `production_base.parquet`.

> **Atenção para a Etapa 2:** o lote de produção com drift **precisa continuar passando no contrato**. Drift não é dado inválido: uma renda maior é um valor plausível, enquanto uma renda negativa é dado sujo. Se a simulação gerar valores fora das regras (ex.: `ext_source_3 > 1`, idade < 18), o gate vai bloquear o lote. Faça o *clip* para os limites do contrato. Essa separação entre os papéis de **qualidade** (contrato) e **estabilidade** (drift) também vale explicar no vídeo.

### Dúvidas para a coordenação (Discord)

Mensagem pronta, baseada nas ambiguidades listadas no checklist do grupo:

> Olá! Algumas dúvidas sobre o Tech Challenge da Fase 4 (MLET):
> 1. Qual é a data de entrega e o formato de submissão (link do repositório + link do vídeo na plataforma)?
> 2. O trabalho é em grupo? Existe um número máximo de integrantes?
> 3. O German Credit, sugerido no enunciado, tem 1.000 linhas, mas o mínimo pedido é de 5.000 amostras. Podemos usar o Home Credit Default Risk (307 mil linhas), que também está na lista sugerida?
> 4. A "análise causal do problema" citada no contexto deve ficar no README, junto da governança (Etapa 4)? Existe algum formato esperado?
> 5. O vídeo STAR pode ser publicado como não listado no YouTube?

---

## 1.2 Setup do projeto e do ambiente

| Item do checklist | Status | Onde |
|---|---|---|
| Criar o repositório GitHub | ✅ | `Victor-Murad/credit-drift-monitor` |
| Estrutura de pastas (`data/reference`, `data/production`, `src`, `notebooks`, `reports`, `docs`) | ✅ | [README → Estrutura](../README.md#estrutura-do-repositório) |
| `.gitignore` (dados sensíveis, artefatos, venv, caches) | ✅ | [`.gitignore`](../.gitignore). Dados do Kaggle e credenciais nunca são versionados. |
| Convenção de commits semânticos | ✅ | [`CONTRIBUTING.md`](../CONTRIBUTING.md): Conventional Commits em PT-BR |
| README inicial (título, contexto, sumário) | ✅ | [`README.md`](../README.md) |
| Gerenciador de dependências | ✅ | **uv** + [`pyproject.toml`](../pyproject.toml), Python 3.12 (mesmo padrão da Fase 3) |
| Scikit-Learn e XGBoost | ✅ | `scikit-learn`, `xgboost` |
| Great Expectations **ou** Pandera, com a escolha documentada | ✅ | **Pandera**. [ADR-0002](adr/0002-pandera-data-contracts.md) |
| Lock file commitado | ✅ | [`uv.lock`](../uv.lock) |
| Instalação limpa em ambiente novo | ✅ | Clone limpo → `uv sync --frozen` → `pytest` (41 testes passando) |
| Seeds globais | ✅ | `SEED = 42` e `set_global_seed()` em [`src/config.py`](../src/config.py), além de `random_state` em todos os splits e modelos. Há um teste de reprodutibilidade em `tests/test_train.py`. |

**Observação para quem usa Windows 11:** se o **Smart App Control** estiver ligado, o Windows bloqueia as DLLs do pandas/numpy (`Uma política de Controle de Aplicativo bloqueou este arquivo`). É preciso desligá-lo em *Segurança do Windows → Controle de aplicativos e do navegador*.

---

## 1.3 Dataset de Referência e modelo baseline

### Construção dos datasets ([`src/data/prepare.py`](../src/data/prepare.py))

1. Lê do CSV bruto **somente 21 das 122 colunas** (minimização de dados, LGPD art. 6º, III).
2. Converte para o esquema canônico: idade e tempo de emprego em anos, e o sentinela `DAYS_EMPLOYED = 365243` (aposentado/desempregado) vira nulo.
3. Aplica **o próprio contrato** linha a linha, de modo que a referência é limpa por construção e nunca diverge do que o gate exige em produção.
4. Faz uma divisão estratificada e com seed entre Referência e Produção t0, gravadas em **pastas separadas** (Boa Prática Obrigatória do PDF).

**Sujeira real encontrada no Home Credit** ([`reference_build_report.json`](../reports/data_contract/reference_build_report.json)): 21 de 307.511 registros rejeitados.

| Regra | Problema encontrado | Registros |
|---|---|---|
| R2 | `annuity_amount` (parcela) nula | 12 |
| R6 | `gender = "XNA"` (código de desconhecido) | 4 |
| R4 | renda acima de 10 milhões (117.000.000; 18.000.090; 13.500.000) | 3 |
| R2/R11 | `family_members` nulo (falha também na conversão para inteiro) | 2 |
| R6 | `family_status = "Unknown"` | 2 |

A R9 (tempo de emprego ausente só para quem não está empregado) teve **zero** violações. Isso confirma nos dados reais que o sentinela aparece apenas para aposentados e desempregados.

| Conjunto | Arquivo | Linhas | Taxa de default |
|---|---|---|---|
| **Referência** (treino) | `data/reference/reference.parquet` | 215.243 (70%) | 8,07% |
| **Produção t0** (nunca visto no treino; base da Etapa 2) | `data/production/production_base.parquet` | 92.247 (30%) | 8,07% |

### Modelo baseline ([`src/models/train.py`](../src/models/train.py))

- Antes de treinar, a referência passa pelo **gate do contrato rotulado** ([`reference_training_report.html`](../reports/data_contract/reference_training_report.html)): **APROVADO**, 215.243 linhas, em cerca de 90 ms.
- Partição da referência: 60% treino / 20% validação (threshold) / 20% teste (43.049 registros).
- Features: 7 categóricas + 11 numéricas do contrato + 4 razões financeiras derivadas no pipeline. `gender` fica **fora** do modelo e é mantido só para auditoria de viés.

| Modelo | ROC-AUC | Gini | KS | PR-AUC | Recall (maus) | Aprovação | Default na carteira aprovada |
|---|---|---|---|---|---|---|---|
| **XGBoost (oficial)** | **0,7494** | **0,4989** | **0,3695** | **0,2434** | 0,6820 | 65,45% | **3,92%** |
| Regressão Logística | 0,7307 | 0,4613 | 0,3429 | 0,2180 | 0,6403 | 67,41% | 4,31% |

Na leitura de negócio: aprovando cerca de 65% dos pedidos, a inadimplência da carteira cai de 8,07% (taxa base) para 3,92%.

Artefatos versionados:
- [`models/baseline_model.joblib`](../models/): pipeline completo, threshold, versão do modelo e versão do contrato.
- [`models/baseline_metadata.json`](../models/baseline_metadata.json): métricas, hiperparâmetros, importância das variáveis, linhagem (SHA-256 da referência e do modelo) e versões das bibliotecas.

---

## 1.4 Data Contract e demonstração de bloqueio

### O contrato ([`src/validation/contract.py`](../src/validation/contract.py) + [`contracts/credit_application.yaml`](../contracts/credit_application.yaml))

O PDF exige pelo menos 3 regras rígidas. O contrato tem **11**, e todas bloqueiam o lote inteiro (fail-closed):

| Regra | Dimensão (Aula 1) | Regra | Justificativa de negócio |
|---|---|---|---|
| R1 | Unicidade | `client_id` único e não nulo | Duplicatas inflam métricas e causam data leakage (Aula 3). |
| R2 | Completude | campos obrigatórios não nulos (**renda nunca nula**) | Sem renda não há como medir comprometimento de renda. |
| R3 | Validade | **idade entre 18 e 100 anos** | Capacidade civil para contratar crédito (Código Civil, art. 5º). |
| R4 | Validade | **renda em (0; 10.000.000]** | Acima disso é erro de digitação ou caso para análise manual. |
| R5 | Validade | crédito em (0; 5.000.000] e parcela > 0 | Valores negativos ou absurdos são erro de sistema. |
| R6 | Validade | categorias restritas a domínios conhecidos | Rejeita códigos de "desconhecido" (`XNA`, `Unknown`) e categorias novas não tratadas. |
| R7 | Validade | scores de bureau entre 0 e 1 | São scores normalizados. |
| R8 | Consistência | parcela ≤ valor total do crédito | Uma parcela maior que a dívida é impossível. |
| R9 | Consistência | tempo de emprego nulo só para aposentado ou desempregado | Ausência de dado precisa ter explicação de negócio. |
| R10 | Schema | `strict=True`: nenhuma coluna fora do contrato | **Barra PII não prevista** (ex.: CPF vazando do sistema de origem). É uma regra de LGPD aplicada no código. |
| R11 | Schema | tipos coeridos e verificados | Uma renda `"abc"` é rejeitada. |

O contrato tem duas camadas: o schema Pandera, que executa, e a especificação no formato *Data Contract Specification* (YAML), que registra dono, termos de uso e classificação LGPD de cada campo. O teste `tests/test_contract_spec.py` impede que as duas divirjam.

### Pipeline de validação ([`src/validation/gate.py`](../src/validation/gate.py))

`enforce_contract()` é chamado **antes do treino** (`train.py`) e **antes de toda inferência** (`predict.py`). Quando o lote viola o contrato:

1. o Pandera valida com `lazy=True`, coletando **todas** as violações de uma vez (Aula 5);
2. o lote é **bloqueado** inteiro com `DataContractViolation`, e o modelo nunca é chamado;
3. o arquivo original vai para `data/quarantine/` para análise;
4. são gerados os relatórios `reports/data_contract/<lote>_report.{html,json}`;
5. um evento estruturado é gravado em `logs/pipeline.jsonl`, pronto para a Etapa 3.

Pela linha de comando (exit code 1 quando bloqueia, para usar como etapa bloqueante em Makefile, CI ou DAG):

```bash
uv run python -m src.validation.validate --input data/samples/invalid_batch.csv
```

### Demonstração ([`src/validation/demo.py`](../src/validation/demo.py))

Dois arquivos chegam "do sistema de origem" e seguem o **mesmo caminho de produção** (gate → modelo):

| Lote | Resultado do gate | Registros pontuados pelo modelo |
|---|---|---|
| `valid_batch.csv` (500 solicitações limpas) | **APROVADO** | 500 |
| `invalid_batch.csv` (500 solicitações + 10 tipos de defeito, 508 linhas) | **BLOQUEADO** | **0** |

**Os 10 defeitos injetados foram detectados**, cada um pela regra esperada: renda nula, menor de idade, idade de 150 anos, duplicatas, crédito negativo, renda de 117 milhões, contrato "Crypto loans", score fora de [0, 1], parcela maior que o crédito e **coluna de CPF não prevista**. A R8 aparece com 6 ocorrências porque as 3 linhas com crédito negativo também ficam com a parcela maior que o crédito.

Evidências versionadas:
- [`reports/data_contract/contract_demo_summary.md`](../reports/data_contract/contract_demo_summary.md): resumo dos defeitos injetados e da detecção.
- [`reports/data_contract/invalid_batch_report.html`](../reports/data_contract/invalid_batch_report.html): relatório visual do bloqueio (bom para mostrar no vídeo).
- [`reports/data_contract/valid_batch_report.html`](../reports/data_contract/valid_batch_report.html): relatório do lote aprovado.

Saída no terminal (trecho):

```
| INFO  | data_contract  | [APPROVED] lote=valid_batch etapa=inference linhas=500 regras_violadas=0 ocorrencias=0
| INFO  | models.predict | batch_scored batch_name=valid_batch model_version=baseline-v1 n_rows=500 approval_rate=0.622
| ERROR | data_contract  | [BLOCKED] lote=invalid_batch etapa=inference linhas=508 regras_violadas=11 ocorrencias=44
| INFO  | data_contract.demo | contract_demo status=SUCCESS ... defects_detected=10 defects_injected=10
```

---

## Checklist de saída da Etapa 1

- [x] Script de validação de dados operante e versionado no repositório: `src/validation/` (contrato, gate, CLI e demonstração)
- [x] Modelo baseline treinado e persistido: `models/baseline_model.joblib` (XGBoost, ROC-AUC 0,749)
- [x] Contrato de dados com pelo menos 3 regras (são 11), com evidência de bloqueio de lote inválido: `reports/data_contract/`

**Pendências do grupo, não técnicas:** confirmar prazo e formato com a coordenação, fechar a divisão das Etapas 2 a 4 e postar as dúvidas no Discord.

## O que a Etapa 2 recebe pronto

| Recurso | Uso na Etapa 2 |
|---|---|
| `data/reference/reference.parquet` | Dataset de Referência para o Evidently (`reference_data`) |
| `data/production/production_base.parquet` | Ponto de partida do Dataset de Produção (aplicar o drift aqui), com rótulo para concept drift |
| `src.models.predict.score_batch(df, bundle, batch_name=...)` | Gera predições já passando pelo gate do contrato |
| `reports/data_contract/reference_build_report.json → reference_profile` | Estatísticas de referência (média, p5, p50, p95) das variáveis numéricas |
| `logs/pipeline.jsonl` | Eventos estruturados para centralizar na Etapa 3 |

A Etapa 2 está implementada em [`docs/etapa-2-simulacao-e-deteccao-de-drift.md`](etapa-2-simulacao-e-deteccao-de-drift.md).
