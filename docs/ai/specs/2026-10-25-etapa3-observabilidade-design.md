# Etapa 3: Observabilidade de pipelines e modelos (design)

> **Status:** aprovada em 2026-10-26. Núcleo (§4.1 a §4.6, §4.8) implementado e comprovado em 2026-10-27. A extensão 4.7 (retreino, Registry, promoção) foi implementada e testada com dados sintéticos em 2026-10-27 ([ADR-0007](../../adrs/0007-retreino-registry-e-promocao-humana.md), limiares pré-registrados); a demonstração com os dados reais (critério 10) segue pendente.
> **Convenção:** o que o usuário decidiu está marcado como **[decidido]**. O que é palpite meu está marcado como **[minha decisão]** e pode ser corrigido.
> **Origem:** brainstorming de 2026-10-25, `docs/tech-challenge-fase4-checklist.md` (Etapa 3), o PDF do Tech Challenge Fase 4 e as aulas do MOD3 (Monitoramento de Pipelines e Infraestrutura).
> **Decisões já registradas:** [ADR-0006](../../adrs/0006-mlflow-e-dvc-locais-na-maquina-do-usuario.md) (MLflow + DVC locais), [ADR-0005](../../adrs/0005-psi-e-queda-de-auc-como-sinais-de-drift.md) (PSI e queda de AUC), [relatório de architecture review](../reports/2026-10-25-architecture-review-src.md).

## 1. Intenção

O projeto já valida dados (Etapa 1) e detecta drift (Etapa 2), mas só deixa o resultado em arquivos soltos e em `logs/pipeline.jsonl`. A Etapa 3 entrega o **ambiente de rastreamento** que o enunciado pede: um lugar único onde se vê a saúde do pipeline, a estabilidade da execução e os **alertas de degradação do modelo**.

O entregável do PDF: *"ambiente de rastreamento (ex.: MLflow) evidenciando a saúde completa do pipeline, a estabilidade da infraestrutura e os alertas de degradação do modelo"*. O critério de avaliação relacionado é Observabilidade e Logs (20%).

O motivo de fazer isso bem: a aula de alertas (Aula 3) ensina que a falha mais perigosa é a silenciosa, quando o pipeline termina com sucesso mas o dado ou o modelo está errado. O ambiente de rastreamento precisa tornar essa falha visível.

## 2. Objetivos, não-objetivos e anti-objetivos

**Objetivos**
1. **[decidido]** MLflow como ferramenta de rastreamento, rodando só na máquina do usuário.
2. **[decidido]** O alerta fica registrado no MLflow. Não há notificação externa.
3. Logs do pipeline e resultados de drift **no mesmo lugar**: no run do MLflow.
4. As métricas de saúde exigidas pelo checklist 3.3, documentadas e visíveis no MLflow.
5. Um nível de alerta por lote (`OK`, `ATENÇÃO`, `CRÍTICO`, `BLOQUEADO`), com a regra documentada e uma ação associada a cada nível.
6. **[decidido]** DVC mínimo: `dvc.yaml` reproduzindo o pipeline, com remoto local (ADR-0006).
7. **[decidido em 2026-10-26]** Ciclo de retreino, Model Registry e promoção de modelo versionado, ligado ao alerta `CRÍTICO` (seção 4.7). Isso vai além do que o PDF da Fase 4 exige para a Etapa 3; entra por decisão do usuário e é tratado como extensão, entregue depois do núcleo (seção 4.1 a 4.6). O núcleo já nasce no backend SQLite, para não precisar migrar depois.
8. Aderência rastreável: cada exigência do checklist e da matéria aponta para o artefato que a cumpre (seção 9).

**Não-objetivos**
- Datadog, Prometheus, Grafana e dashboard próprio. O PDF aceita MLflow *ou* dashboard, e o checklist recomenda fazer uma só bem feita.
- Notificação externa (e-mail, Slack). O alerta é um registro, não um disparo.
- Retreino agendado ou disparado sem comando (por *schedule*, por evento ou por orquestrador). O retreino desta spec é sempre um comando explícito, guardado por um alerta registrado (seção 4.7).
- Deploy canário, *blue-green* e *shadow* em tráfego real: não há tráfego real, só lotes.
- Docker/compose, servidor MLflow remoto e `dvc push` para serviço externo.
- Documentação final e vídeo (Etapa 4).
- Mudar a lógica de detecção de drift (KS, qui-quadrado, PSI, `min_psi`).

**Anti-objetivos (seria falha mesmo que "funcione")**
- Qualquer dado do Kaggle sair da máquina do usuário (ADR-0001, ADR-0006).
- Preencher o MLflow com runs sintéticos ou inventados para "mostrar" alertas. Os runs vêm do pipeline real.
- Alerta sem ação associada (Aula 3: *"se não há nada que alguém possa fazer, o alerta não deveria existir"*) ou que dispare em ruído, como o falso alerta de `annuity_amount` em t0 que o `min_psi` já eliminou.
- Credenciais em texto puro (AGENTS.md).

## 3. Restrições

- Python 3.12, `uv sync --locked`, import-linter e ruff limpos, cobertura ≥ 80% (hoje 95,6%), testes sem depender do dataset do Kaggle.
- Nova camada `src/monitoring` acima de `drift` no contrato `layers`: `monitoring → drift → models → validation`. As camadas inferiores **não** importam `mlflow`.
- Regra de dependência: a decisão de nível de alerta é função pura (sem MLflow); o adaptador do MLflow só registra.
- **Pontos de entrada (revisão de 2026-10-26):** o Makefile e o `dvc.yaml` chamam `python -m src.monitoring.stage <estágio>`, que envolve o `main()` do estágio inferior. As camadas de baixo não importam `src.monitoring`; elas oferecem *ganchos* opcionais: `run_demo(on_batch=...)`, `validation.demo.main(on_batch=...)` e `train.main()` devolvendo `(bundle, metadata)`. Um contrato `forbidden` do import-linter barra `mlflow` e `dvc` fora de `src/monitoring/`.
- Falhar fechado: se o MLflow não estiver disponível ou o registro falhar, o pipeline **não** pode seguir como se tivesse registrado (silent failure). **[minha decisão]**: o estágio sai com código diferente de zero e mensagem clara, e o evento de falha vai ao `logs/pipeline.jsonl`, que é o único registro possível nesse caso. Rodar sem MLflow exige pedido explícito, pela variável `MONITORING_TRACKING=off` (que atravessa `make` e `dvc` sem mudar a CLI de cada estágio).
- Proibido `except` silencioso (regra do repositório).
- O modelo (`models/`) e os relatórios (`reports/`) continuam no git, porque o checklist 3.4 exige consolidá-los como evidência. Os datasets (`data/`), o cache do DVC e o `mlruns/` ficam fora.

## 4. Desenho

### 4.1 Estrutura de runs no MLflow

Um experimento `credit-drift-monitor`. **[minha decisão]**

| Run | Quando | Conteúdo |
| :--- | :--- | :--- |
| **Run de estágio** (pai) | Cada execução de `prepare`, `train`, `contract-demo` e `drift-demo` | Status, latência, volume, pico de memória, contagem de alertas, `pipeline_run_id`, log JSONL do `run_id` como artefato |
| **Run de lote** (filho aninhado) | Cada lote avaliado no `drift-demo` (t0, t1, t2) | Nível de alerta e motivos, PSI por coluna, `n_drifted_columns`, AUC, queda de AUC, PSI do score, relatórios HTML/JSON do Evidently e do contrato |
| **Run de treino** (dentro do estágio `train`) | Cada treino do baseline | Hiperparâmetros, métricas de teste, modelo com *signature* e *input example* (Aula 6), `reference_sha256` e `model_sha256` |

O `run_id` do `logs/pipeline.jsonl` (mitigação do review) entra como tag `pipeline_run_id`, ligando o run do MLflow aos eventos do log.

O `contract-demo` também emite um run de lote filho por lote (o válido e o inválido). O lote válido (500 linhas, sem rótulo) registra só o resultado do contrato (`OK`); não roda Evidently nem PSI do score. O bloqueio **intencional** do lote inválido gera `BLOQUEADO` e conta em `alerts_failure_count`, porque o alerta é real; o estágio mantém `stage_success=1` e o run recebe a tag `intentional_demo=true`, para que não se confunda com falha de pipeline. **[minha decisão]**

**Contrato de nomes** (fixo; os testes e a consulta de alertas dependem dele) **[minha decisão, da revisão de 2026-10-26]**:

| Item | Nome | Observação |
| :--- | :--- | :--- |
| Tag | `run_type` | `stage`, `batch`, `training`, `retrain`, `audit` |
| Tag | `pipeline_run_id`, `stage`, `stage_status` (`SUCCESS`/`FAILED`) | nos runs de estágio |
| Tag | `alert_level` | código ASCII, para filtros do `search_runs`: `OK`, `ATENCAO`, `CRITICO`, `BLOQUEADO` |
| Tag | `alert_label`, `alert_reasons` | rótulo com acento (`ATENÇÃO`, `CRÍTICO`) e motivos separados por `;` (`drift:<coluna>`, `score_psi>=0.10`, `auc_drop>=0.05`, `single_class`; no `BLOQUEADO`, as regras violadas do contrato, ex.: `R1`; `empty_batch`, `auc_undefined`, `score_psi_undefined` e `no_baseline_auc` estão na nota do piso `ATENCAO`, §4.2) |
| Tag | `batch_name`, `batch_path`, `batch_sha256`, `model_version` | o `simulate` regera t1/t2, então o hash do lote identifica o que foi avaliado |
| Tag | `performance_evaluated` (`true`/`false`), `intentional_demo`, `retrain_consumed` | |
| Métricas | as da §4.3 | `rows_rejected` = linhas dos lotes `BLOQUEADO` (o gate é tudo ou nada; `n_failure_cases` conta ocorrências, não linhas) |

Nos textos desta spec os níveis aparecem com acento (`ATENÇÃO`, `CRÍTICO`); no código e nas tags valem os códigos ASCII acima.

A coluna `target` entra no teste de *label shift*, mas **não** conta como coluna com drift para o nível `ATENÇÃO` nem para `n_drifted_columns` do alerta; o resultado sobre `target` é registrado à parte (`label_shift`).

**Identificador da execução** **[minha decisão]**: o `Makefile` exporta um único `PIPELINE_RUN_ID` por execução de `make pipeline` (o `logging_utils` já lê essa variável), e cada run de estágio o recebe como tag. Não há run-raiz extra. Reexecuções acumulam runs, distinguidos por essa tag; os testes usam um *tracking URI* temporário.

### 4.2 Nível de alerta (função pura)

Precedência: `BLOQUEADO` > `CRÍTICO` > `ATENÇÃO` > `OK`. Limiares reaproveitam `src/config.py` (`PSI_MODERATE=0,10`, `PSI_SIGNIFICANT=0,25`, `AUC_DROP_ALERT=0,05`).

| Nível | Regra | Ação associada (runbook) |
| :--- | :--- | :--- |
| `BLOQUEADO` | O contrato barrou o lote; ele nem chegou ao modelo | Inspecionar o relatório do contrato e a quarentena; corrigir a fonte; não pontuar |
| `CRÍTICO` | O lote tem rótulo e a queda de ROC-AUC frente ao baseline é ≥ `AUC_DROP_ALERT` | Tratar como concept drift: investigar e avaliar retreino com validação humana |
| `ATENÇÃO` | Alguma coluna marcada com drift (p-valor e `PSI ≥ min_psi`), **ou** PSI da distribuição de scores ≥ 0,10 | Investigar a coluna; um data drift sozinho não prova piora do modelo |
| `OK` | Nenhuma das anteriores | Nenhuma |

**Baseline da queda de AUC** **[minha decisão]**: o AUC do lote t0, como já faz `assess_auc_drop` (ADR-0005). Em produção, t0 corresponde à primeira janela com rótulo. O run de lote registra também `auc_drop_vs_training` (contra o AUC de teste do treino, 0,7491), só informativo e fora da regra do alerta. Com os números reais, as duas medidas dão as mesmas classificações: t1 cai 0,027 e t2 cai 0,177 contra o treino.

**Referência do PSI do score** **[minha decisão]**: as probabilidades do split de teste do treino, em decis. O treino persiste essas bordas e proporções em `score_reference` no `baseline_metadata.json`; o PSI é calculado por uma função nova em `drift/psi.py` que aceita bordas e proporções de referência (a atual só aceita séries brutas). O critério de aceite (t0 = `OK`) valida a escolha; se o t0 passar de 0,10, revisa-se a referência, não o limiar.

Lote rotulado com uma só classe (ROC-AUC indefinida) recebe o mesmo tratamento do lote sem rótulo, com o motivo `single_class`.

> **Nota (2026-10-27, implementação):** há um piso de `ATENÇÃO` para anomalias. Um lote vazio (`empty_batch`), uma AUC não finita (`auc_undefined`) ou um PSI do score não finito (`score_psi_undefined`) nunca terminam `OK`: sobem no mínimo a `ATENÇÃO`, com o motivo nas `alert_reasons`. É um máximo: o piso nunca rebaixa um nível maior. `no_baseline_auc` e `single_class` entram nos motivos sem piso. Sem rótulo, ou com `single_class`, o teto continua `ATENÇÃO`. A regra está em `src/monitoring/stage.py` (`_CAUTIOUS_REASONS`, `_at_least_attention`).

Sem rótulo, a AUC não é calculada: o run registra `performance_evaluated=false` e o nível máximo possível é `ATENÇÃO`, sustentado pelo PSI do score (proxy ensinado na Aula 4 de Monitoramento, p. 5: *distribuição de predições quando não há ground truth*).

Resultado esperado com os dados reais da demo (seed 42): t0 `OK`; t1 `ATENÇÃO` (3 colunas, AUC 0,7224, queda < 0,05); t2 `CRÍTICO` (queda de 0,18). Este resultado é **critério de aceite** (seção 8).

### 4.3 Métricas de saúde (checklist 3.3)

| Exigência do checklist | Métrica no MLflow | Fonte |
| :--- | :--- | :--- |
| Status de execução das tarefas | `stage_success` (1/0) e tag `stage_status` | Resultado do estágio, inclusive falha |
| Latência | `stage_duration_s`; `duration_ms` do contrato, da detecção e da pontuação | Eventos já logados |
| Volume de dados processados | `rows_processed`, `rows_rejected` | `n_rows` dos eventos |
| Quantidade de alertas de falha ou drift | `alerts_failure_count` (lotes `BLOQUEADO`, estágios com falha) e `alerts_drift_count` (`ATENÇÃO` ou `CRÍTICO`) | Níveis de alerta |
| Saúde do modelo (ROC-AUC, queda de performance) | `roc_auc`, `auc_drop`, `approval_rate`, `default_rate_approved`, `score_psi` | `evaluate`, `assess_auc_drop` |
| Estabilidade da infraestrutura | `peak_memory_mb` do processo e `stage_duration_s` | `resource` da biblioteca padrão **[minha decisão]** |

"Infraestrutura" aqui é o processo local; não há cluster. Isso é uma limitação assumida e será dita na documentação.

### 4.4 Logs centralizados (checklist 3.2)

Cada run de estágio anexa como artefato as linhas de `logs/pipeline.jsonl` (e dos arquivos rotacionados) com o seu `run_id` **dentro da janela de tempo do estágio**, porque o `PIPELINE_RUN_ID` é comum a todos os estágios (sucesso e falha de ingestão e transformação) e, no estágio de drift, os relatórios de drift. O arquivo local continua existindo.

### 4.5 DVC mínimo (ADR-0006)

- `dvc init` e um `dvc.yaml` com os estágios `prepare → train → contract_demo → drift_demo`, com `deps` e `outs`.
- Remoto **local**, em diretório fora do repositório, configurado em `.dvc/config.local` (não versionado).
- `reports/` e `models/` ficam no git: no `dvc.yaml` entram como `outs` com `cache: false`, para o DVC reproduzir sem tirar os arquivos do git. **Os `outs` são listados arquivo por arquivo em cada estágio, sem diretório compartilhado**: o DVC apaga os `outs` antes de rodar o estágio e rejeita `outs` sobrepostos, e três estágios escrevem em `reports/data_contract/`. Um teste confere que nenhum `out` é prefixo de outro entre os estágios. **[minha decisão, da revisão de 2026-10-26]**
- `data/raw/application_train.csv` entra como `deps` do estágio `prepare`; o hash fica no `dvc.lock`. **Não se usa `dvc add`**: o arquivo `.dvc` nasceria em `data/raw/`, que o `.gitignore` ignora (e o DVC recusa um `.dvc` ignorado pelo git). O download do Kaggle continua manual (`make download-data` fica fora do `dvc.yaml`). O CSV já está na máquina do usuário, em `data/raw/` do checkout principal; um worktree novo precisa ter o arquivo ligado por link simbólico para a execução real.
- `dvc.lock` é versionado. Os relatórios têm timestamp, então o `dvc.lock` e os relatórios mudam a cada rodada de `dvc repro`; isso é esperado, e não se remove o timestamp, porque os relatórios são a evidência.
- O orquestrador do `make pipeline` é o próprio Makefile (encadeia os estágios, sempre executando); o `dvc repro` (`make repro`) pula estágios sem mudança e por isso não gera runs novos. Os dois caminhos são documentados.
- Um teste garante que nada de `data/` (exceto `.gitkeep`) entra no índice do git.

### 4.6 Ajuste de nomenclatura (adiado do review)

`DriftResult.concept_drift` mede label shift (ADR-0005). Passa a se chamar `label_shift`, com o alerta lendo `performance.degraded` para concept drift. Os relatórios de `reports/drift/` são regenerados. A ADR-0005 (item 4, "permanece por compatibilidade") recebe uma emenda; o JSON mantém `concept_drift` como alias obsoleto até a Etapa 4.

### 4.7 Retreino, Model Registry e promoção (extensão decidida em 2026-10-26)

Base na matéria: gatilho de retreino por performance e por drift (Aula 4 de Monitoramento, p. 8), *champion/challenger* (mesma aula), aliases do Registry no lugar dos estágios fixos (MOD4 Aula 8, p. 8), critérios de promoção codificados como "governança como código" (MOD1 Aula 7, p. 24) e o ciclo de CT: *ingestão, validação, retreino, avaliação, registro de nova versão* (MOD1 Aula 8, p. 12).

Tudo abaixo é proposta minha, **a confirmar**, exceto o que está marcado como decidido.

**Gatilho** **[minha decisão]**. O retreino só roda se existir um alerta `CRÍTICO` registrado no MLflow (queda de AUC, sinal de concept drift). Um `ATENÇÃO` não dispara retreino, porque *"nem todo data drift significa que o modelo piorou"* (MOD1 Aula 5, p. 23). Um comando sem alerta prévio exige um flag explícito.

**Dados do retreino** **[minha decisão]**. Treino = **treino da referência** (o `split_reference` de sempre, com a seed do projeto) mais metade estratificada do lote `CRÍTICO` rotulado. A validação (escolha do *threshold*) e o teste da referência são **idênticos aos do champion**, para a regra 3 não ter vazamento. A outra metade é o *holdout* de comparação e nunca entra no treino (sem vazamento).  O contrato (`enforce_contract`) valida os dados antes do treino, como já acontece hoje.

**Challenger** **[minha decisão]**. Mesmo pipeline XGBoost e mesmos hiperparâmetros do baseline; só os dados mudam. É registrado como nova versão do modelo `credit-default-xgboost` no Model Registry, com o alias `challenger` e as tags: run de origem, `reference_sha256`, `model_sha256`, lote e alerta que motivaram o retreino.

**Aliases em vez de estágios** **[minha decisão]**. Usa-se `champion`, `challenger` e `previous` (MLflow 3.x, versão fixada pelo `uv.lock`). Os estágios `Staging/Production` estão em desuso nas versões recentes; o PDF da Fase 2 os cita, mas esta é a Fase 4. O Registry costuma exigir *backend* em banco (SQLite local); o construtor confirma na versão instalada.

**Regra de promoção (função pura, em `src/monitoring`)** **[minha decisão sobre os números]**. O challenger só é elegível se:
1. passa no contrato e no teste de integridade do artefato;
2. no holdout do lote `CRÍTICO`, sua ROC-AUC supera a do champion em pelo menos `PROMOTION_MIN_GAIN` (proposta: 0,02);
3. no split de teste da referência, sua ROC-AUC não fica abaixo da do champion em mais de `PROMOTION_MAX_REGRESSION` (proposta: 0,02), para não "esquecer" o comportamento anterior.

Os dois limiares ficam em `src/config.py` e são **pré-registrados** na ADR-0007 **antes** do retreino real; não são ajustados depois de ver o resultado. Se o challenger real for reprovado, quem decide é o usuário, e a decisão fica registrada. A decisão (elegível ou não, com os números) vai para o MLflow e para o `logs/pipeline.jsonl`.

**Promoção e rollback.** Promover move o alias `champion` para a nova versão e dá `previous` à anterior. O rollback move os aliases de volta. Ambos geram um run de auditoria no MLflow com quem, quando e por quê (MOD1 Aula 7: versionamento com trilha de auditoria e política de rollback).

**Quem promove** **[decidido em 2026-10-26]**. A promoção é um **comando humano separado** (`make promote`). A rotina de retreino treina o challenger, avalia e imprime o veredito da regra (elegível ou não, com os números), mas **nunca move o alias `champion`**. O comando de promoção reavalia a regra antes de agir e recusa um challenger que não seja elegível, a menos que o flag explícito de exceção seja usado; nesse caso o motivo é obrigatório e fica no run de auditoria. Razão: crédito é um domínio com governança forte, e a matéria ensina que o alerta aciona uma pessoa para decidir (MOD1 Aula 5, p. 23). A promoção automática (CT com *guardrails*, Aula 8 de Retreino) fica como evolução possível, trocando só esse passo.

**Efeito no restante do sistema.** O `load_bundle` continua lendo um arquivo e verificando o sha256 (ver "Champion e `models/`" abaixo). Uma ADR nova (ADR-0007) registra o ciclo de retreino e promoção.

**Demonstração com os dados reais.** O lote t2 (AUC 0,5720, `CRÍTICO`) alimenta o retreino. O resultado esperado é um challenger que recupera a AUC no holdout de t2, mantém a do teste da referência dentro da tolerância e é promovido. Isto é critério de aceite (seção 8) e precisa ser **medido**, não presumido: se o challenger não passar, a spec não muda o dado, e o resultado é registrado como está.

**Detalhes fechados após a revisão do desenho** (todos **[minha decisão]**):

- **Como o retreino acha o alerta.** O comando aceita `--alert-run <id>`; sem ele, consulta o MLflow pelo run de lote mais recente com `alert_level=CRÍTICO` ainda não consumido. Ao treinar o challenger, o run de lote recebe a tag `retrain_consumed=<versão>`. Um segundo retreino sobre o mesmo alerta exige o flag explícito. O retreino exige o rastreamento ligado, porque depende do Registry: com `MONITORING_TRACKING=off` ele falha fechado.
- **Divisão do lote t2.** Estratificada pelo `target`, com a seed do projeto. O t2 tem 92.247 linhas; o holdout tem cerca de 46 mil e algo como 5,4 mil inadimplentes (taxa base de 11,75%), o suficiente para um ganho de 0,02 de AUC não ser ruído. O AUC do champion é **recalculado no mesmo holdout**, e não reaproveitado do 0,5720 do lote inteiro.
- **Contrato.** O t2 já passa no contrato rotulado (o `drift-demo` exige isso), então `enforce_contract` aceita a referência mais a metade do t2. O teste de integridade reaproveita `src/models/integrity.py`.
- **Primeira versão do Registry.** O `train` registra cada reconstrução do baseline como nova versão do Registry (tag `origin=baseline`). Só **na primeira vez**, quando ainda não existe `champion`, ele recebe o alias (*bootstrap*). Daí em diante só o comando de promoção ou de rollback move esse alias; um `train` repetido nunca o move.
- **Champion e `models/`.** `models/baseline_model.joblib` (`out` do estágio `train`) é só o artefato do baseline e nunca é tocado por promoção. O champion servido vive em `models/champion/` (`baseline_model.joblib` + `baseline_metadata.json`, mesmos nomes para reaproveitar a verificação de integridade), **fora dos `outs` do DVC**. O `train` cria essa cópia no bootstrap; promoção e rollback a reescrevem, **copiando o artefato do run de origem** (cada run de treino anexa o `bundle` e o metadata), de forma atômica (arquivo temporário + `os.replace`). `load_bundle` usa `models/champion/` se existir e `models/` caso contrário, verifica o sha256 como hoje e, com o rastreamento ligado, `src.monitoring` passa `expected_sha256` (a tag `model_sha256` da versão `champion`); divergência falha fechado. A ordem da promoção é: copiar para temporário e conferir o hash, mover `previous`, mover `champion`, `os.replace`, reconferir o estado e gravar esse estado no run de auditoria; qualquer divergência encerra com erro e o `load_bundle` recusa até a correção.
- **Evidência de recuperação.** Os números de recuperação do challenger usam **somente o holdout**, nunca o t2 inteiro, que contém as linhas de treino.
- **Se o challenger real for reprovado.** O critério 10 não exige que ele seja promovido na demonstração real; exige que o veredito seja correto, medido e registrado. O ramo de promoção e o de rollback (e, portanto, o alias `previous`) são exercitados por testes com scores controlados. Não se usa o flag de exceção para fabricar uma promoção que feche o critério: se reprovar, a decisão sobre calibrar os limiares ou aceitar o resultado volta para o usuário.
- **Backend do MLflow.** SQLite desde o início (`mlflow.db`), com o diretório de artefatos (`mlartifacts/`), ambos fora do git, com URI absoluto a partir da raiz do repositório; os testes usam um URI temporário e nunca o banco real.

### 4.8 Interface de comandos **[minha proposta, a confirmar]**

Os nomes seguem o que a operação faz, e não o número da etapa do curso.

| Comando | O que faz |
| :--- | :--- |
| `make pipeline` | Ponto de entrada: `prepare → train → contract-demo → demo-drift`, com um único `PIPELINE_RUN_ID` e o rastreamento ligado. Encadeia os estágios diretamente (sempre executa), sem o `dvc repro` |
| `make monitor BATCH=<arquivo>` | Avalia **um** lote de produção: contrato, pontuação, drift, PSI do score, AUC (se rotulado), nível de alerta e registro no MLflow. Código de saída: 0 (`OK`/`ATENÇÃO`), 2 (`CRÍTICO`), 3 (`BLOQUEADO`), 1 (erro operacional). O baseline de AUC é o do run `production_t0` do mesmo `model_version` no MLflow, ou `--baseline-auc`; lote rotulado sem nenhum dos dois falha fechado e manda rodar `make demo-drift` |
| `make demo-drift` | Simula t1 e t2 e avalia t0, t1 e t2 (hoje `drift-demo`). Violação de contrato num lote simulado continua sendo falha do estágio (`sys.exit`) |
| `make retrain` | Treina o challenger a partir de um alerta `CRÍTICO` e registra no Registry (nunca promove) |
| `make promote` | Promove o challenger a `champion` (comando humano; recusa challenger inelegível) |
| `make rollback` | Devolve o `champion` à versão `previous` |
| `make tracking-ui` | Sobe o `mlflow ui` na porta 5001 (a 5000 colide com o AirPlay no macOS), com URI absoluto do banco local |
| `make repro` | `dvc repro` direto: pula estágios sem mudança (não gera runs novos) e não define o `PIPELINE_RUN_ID` |

`etapa1`, `etapa2` e `drift-demo` continuam como apelidos, porque a documentação e o AGENTS.md os citam; a Etapa 3 não ganha um `etapa3`.

## 5. O que fica para o construtor

- Versão do MLflow (o backend é SQLite, decidido na seção 4.7) e o detalhe da API de aliases na versão instalada.
- Nome exato dos módulos dentro de `src/monitoring/`; os alvos do Makefile estão na seção 4.8.
- Como medir o pico de memória de forma portátil entre macOS e Linux (`ru_maxrss` tem unidades diferentes).
- Como o teste evita gravar no `mlflow.db` real e custa pouco com `pytest-xdist` (sugestão: banco-modelo migrado uma vez por sessão e copiado por teste).
- Detalhes da *signature* e do *input example* do modelo.

## 6. Riscos

| Risco | Mitigação |
| :--- | :--- |
| Duas ferramentas novas (MLflow e DVC) consumirem o tempo da Etapa 3 | DVC mínimo; MLflow primeiro, DVC depois; o checklist avisa contra duplicar integrações |
| A extensão 4.7 (retreino, Registry, promoção) não é exigida pelo PDF e aumenta bastante o escopo | Entregue depois do núcleo 4.1 a 4.6, em fases separadas no plano; o núcleo precisa estar completo e aceito antes |
| Promover um challenger pior sem perceber (por exemplo, treinado em dado já contaminado) | Holdout nunca usado no treino, guarda de regressão na referência, contrato antes do treino e rollback com `previous` |
| Retreino sobre dado sintético de drift não provar nada sobre produção real | Registrar na documentação que t1 e t2 são simulados (ADR-0004); o ciclo é demonstrado, não validado em produção |
| Dados do Kaggle vazarem pelo cache do DVC ou pelo `mlruns/` | `.gitignore` e o teste da seção 4.5 |
| Alerta de ruído (fadiga de alerta) | O `min_psi` já filtra; nenhuma regra nova usa só p-valor |
| MLflow indisponível mascarar falha | Falhar fechado (seção 3) |
| Resultado real divergir do esperado em 4.2 | Aceite exige a execução real; se divergir, a regra é revista, não o dado |

## 7. Fora desta spec

A documentação final do repositório, o relatório técnico e o vídeo STAR (Etapa 4).

## 8. Critérios de aceite

1. `make pipeline` roda o pipeline com os dados reais e registra os runs no MLflow local.
2. `make tracking-ui` (o `mlflow ui`) mostra: t0 `OK`, t1 `ATENÇÃO`, t2 `CRÍTICO`, cada um com os motivos.
3. Cada métrica da tabela 4.3 aparece em pelo menos um run.
4. O log JSONL do `run_id` está anexado a cada run de estágio, e o `pipeline_run_id` liga os dois.
5. Um lote inválido gera run `BLOQUEADO` e incrementa `alerts_failure_count`.
6. Com o MLflow indisponível, o estágio falha com mensagem clara (e só roda sem ele com `MONITORING_TRACKING=off`).
7. `dvc repro` reproduz o pipeline; `git status` não mostra nada de `data/` nem de `mlruns/`. Mudanças em `dvc.lock` e nos relatórios após a rodada são esperadas (timestamps).
8. Testes novos para a regra de alerta (todas as linhas da tabela 4.2, inclusive o caso sem rótulo), o adaptador e o teste do índice do git; cobertura ≥ 80%; `make lint` limpo; import-linter com o contrato `monitoring → drift → models → validation`.
9. O CHANGELOG, a ADR-0006, a emenda da ADR-0005 e a nova ADR-0007 (retreino e promoção, com os limiares de promoção pré-registrados), com status `Implementada`, atualizados.
10. Extensão 4.7: o alerta `CRÍTICO` de t2 habilita o retreino; o challenger é treinado sem tocar no holdout; a regra de promoção é testada em todos os ramos (ganho insuficiente, regressão na referência, contrato falhando, elegível); o Registry mostra `champion` e `challenger` com as tags (e `previous` após uma promoção, exercitada nos testes se o challenger real for reprovado); um rollback devolve o alias anterior; e cada promoção ou rollback deixa um run de auditoria. Nenhum retreino roda sem alerta `CRÍTICO` ou o flag explícito. A rotina de retreino nunca move o alias `champion`; só o comando de promoção o faz, e recusa challenger inelegível sem o flag de exceção com motivo.

## 9. Rastreabilidade: checklist e matéria

| Exigência | Origem | Onde é cumprida |
| :--- | :--- | :--- |
| Escolher MLflow ou dashboard e adicionar às dependências | Checklist 3.1 | Objetivo 1; ADR-0006 |
| Prometheus só se houver integração real | Checklist 3.1 | Não-objetivo |
| Centralizar logs de execução (sucesso e falha) | Checklist 3.2 | Seção 4.4 |
| Resultados de drift no mesmo local | Checklist 3.2 | Run de lote (4.1) |
| Métricas de saúde operacionais e do modelo | Checklist 3.3 | Seção 4.3 |
| Dashboard/ambiente de rastreamento final | Checklist 3.3 e entregável do PDF | Critério de aceite 2 |
| Consolidar artefatos e histórico semântico | Checklist 3.4 | Restrição de `models/` e `reports/` no git |
| `mlflow ui` funcional | Checklist 3.4 | Critério de aceite 2 |
| Limiares de PSI (0,10 e 0,25) | Aula 4 de Monitoramento; Aula 2 de Data Drift | Seção 4.2 |
| Data drift não implica piora do modelo; humano decide | MOD1 Aula 5, p. 23 | `ATENÇÃO` vs `CRÍTICO` |
| Métrica de performance só com ground truth; proxy sem ele | Aula 4 de Monitoramento, p. 5; MOD1 Aula 5, p. 12 | Seção 4.2 (caso sem rótulo, PSI do score) |
| Registrar resultado no MLflow e alertar por limiar | MOD4 Aula 4, p. 11 | Seção 4.1 |
| Alerta com ação associada; evitar fadiga | Aula 3 de Alertas, p. 8 | Coluna "Ação associada"; anti-objetivos |
| Falhas silenciosas vs ruidosas | Aula 3 de Alertas, p. 5 | `BLOQUEADO` e falhar fechado |
| Logs estruturados com IDs de correlação | Aula 5 de Observabilidade, p. 10 | `run_id` / `pipeline_run_id` |
| Params, métricas, artefatos, *signature*, *input example*, versão de dados e código | Aula 6 de MLflow, p. 6, 8 e 9 | Run de treino (4.1) |
| Versionar dados com DVC; rastrear qual snapshot gerou o modelo | MOD1 Aula 8, p. 5; Aula 6 de MLflow, p. 9 | Seção 4.5; `reference_sha256` |
| DVC com remoto local ou S3 | PDF da Fase 2 (Etapa 3) | ADR-0006 |
| Métricas de infraestrutura (CPU, memória) | Aula 7 de Infraestrutura | `peak_memory_mb` (4.3), com limitação assumida |
| Gatilhos de retreino (performance, drift) | Aula 4 de Monitoramento, p. 8 | 4.7 (gatilho só por `CRÍTICO`) |
| Champion/challenger e promoção quando supera | Aula 4 de Monitoramento, p. 8 | 4.7 (regra de promoção) |
| Registry com aliases (`champion`/`challenger`; o `baseline` da aula vira `previous`) | MOD4 Aula 8, p. 8 | 4.7 |
| Governança como código: critérios de promoção em pipeline | MOD1 Aula 7, p. 24 | 4.7 (regra pura + limiares em `config.py`) |
| Ciclo CT: validar, retreinar, avaliar, registrar versão | MOD1 Aula 8, p. 12 | 4.7 |
| Versão anotada (origem, motivo), rollback | MOD1 Aula 5, p. 20 | Tags do Registry; alias `previous` |
