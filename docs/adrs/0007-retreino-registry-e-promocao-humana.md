<!--
================================================================================
LOG DE MANUTENÇÃO DE DOCUMENTAÇÃO
--------------------------------------------------------------------------------
Data       | Autor          | Descrição
--------------------------------------------------------------------------------
2026-10-27 | Eduardo Batista | Criação da ADR 0007: retreino guardado por alerta CRITICO, Model Registry com aliases e promoção humana, com os limiares de promoção pré-registrados antes do retreino real.
================================================================================
-->

# ADR 0007: Retreino guardado por alerta crítico, Model Registry com aliases e promoção humana

| Metadado | Detalhe |
| :--- | :--- |
| **Status** | **Aprovada** (2026-10-27) |
| **Decisores** | Eduardo Batista |
| **Tags** | `[mlops]`, `[model-registry]`, `[retraining]`, `[governance]` |
| **Impacto Sistêmico** | Alto (`src/monitoring/retrain.py`, `promote.py`, `promotion.py`, `registry.py`; `models/champion/`; `dvc.yaml`) |
| **ADRs Relacionadas** | Extends ADR-0003, ADR-0005 e ADR-0006 |

## Contexto

O núcleo da Etapa 3 (ADR-0006) registra no MLflow um alerta `CRITICO` quando um lote rotulado perde 0,05 ou mais de ROC-AUC frente ao t0 (ADR-0005). Na execução real de 2026-10-27, o t2 caiu 0,1813 (AUC 0,5720). O runbook desse nível diz "avaliar retreino com validação humana", mas até aqui não havia como fazer isso de forma rastreável.

O usuário decidiu em 2026-10-26 acrescentar a extensão §4.7 da [spec da Etapa 3](../ai/specs/2026-10-25-etapa3-observabilidade-design.md): retreino, Model Registry e promoção de um modelo versionado, ligados ao alerta `CRITICO`. Essa parte vai além do que o PDF da Fase 4 exige.

### Riscos Identificados e Problemas a Resolver:
1. **Vazamento na avaliação**: se o challenger for avaliado em linhas que viu no treino, o ganho de AUC é ilusório e a promoção escolhe o modelo errado.
2. **Limiar ajustado depois do resultado**: escolher o ganho mínimo depois de ver o número do challenger real transforma a regra em justificativa.
3. **Promoção sem dono**: crédito é um domínio de governança forte. Trocar o modelo servido sem uma pessoa responsável e sem trilha de auditoria contraria a matéria (MOD1 Aula 5, p. 23; MOD1 Aula 7).
4. **Modelo servido divergente do Registry**: uma troca interrompida no meio deixa o arquivo servido e o alias `champion` apontando para versões diferentes.
5. **Retreino disparado por ruído**: um data drift sem queda de desempenho não prova que o modelo piorou e não justifica treinar de novo.

---

## 2. Decisão de Arquitetura

### 2.1 Gatilho: só um alerta `CRITICO`
- `make retrain` só treina se houver um run de lote `alert_level=CRITICO` ainda não consumido, emitido contra o champion atual (mesmo `model_version`). Sem `--alert-run`, o comando usa o mais recente.
- `ATENCAO` nunca dispara retreino: *"nem todo data drift significa que o modelo piorou"* (MOD1 Aula 5, p. 23).
- Retreinar sem alerta exige `--force-no-alert --reason`. Reaproveitar um alerta já consumido, ou emitido contra outro champion, exige `--alert-run <id> --allow-reuse --reason`. O motivo fica na tag `retrain_reason` do run.
- O consumo vale para o lote inteiro (`batch_sha256`): depois do retreino, os alertas `CRITICO` do mesmo lote recebem `retrain_consumed=<versão>`, e um alerta do mesmo lote gravado depois também não habilita outro retreino.
- O retreino exige o rastreamento ligado. Com `MONITORING_TRACKING=off`, falha fechado (código 1).

### 2.2 Dados do retreino sem vazamento
- O `split_reference` de sempre (seed 42) divide a referência primeiro. Treino do challenger = treino da referência + metade estratificada (por `target`, seed 42) do lote `CRITICO` rotulado.
- A validação (escolha do threshold por KS) e o teste da referência são **os mesmos do champion**.
- A outra metade do lote é o **holdout de comparação** e nunca entra no treino. Um teste confere que nenhum `client_id` aparece em mais de uma parte (treino, validação, teste e holdout). Se aparecer, o comando falha fechado antes de treinar.
- A AUC do champion é **recalculada no mesmo holdout**, e não reaproveitada da AUC do lote inteiro.
- O contrato rotulado (`enforce_contract`) valida a referência e o lote antes do treino (código 3 se barrar). Também se conferem o sha256 do champion, a referência igual à do champion e o lote igual ao avaliado no alerta.
- A evidência de recuperação usa **somente o holdout**, nunca o t2 inteiro, que contém as linhas de treino.

### 2.3 Challenger e aliases do Registry
- Mesmo pipeline XGBoost e mesmos hiperparâmetros do baseline (ADR-0003). Só os dados mudam.
- Cada challenger vira uma nova versão de `credit-default-xgboost` com o alias `challenger`. A versão leva as tags de linhagem (run de origem, `reference_sha256`, `model_sha256`, lote, alerta, champion comparado) e o veredito com as quatro AUCs.
- O challenger carrega um `model_version` próprio (`retrain-<sha256[:12]>` do pipeline serializado) no bundle, no metadata e nas tags. Assim o monitor não confunde o baseline de AUC de um modelo com o de outro.
- Aliases `champion`, `challenger` e `previous` (MLflow 3.x), no lugar dos estágios `Staging/Production`, que estão em desuso (MOD4 Aula 8, p. 8). O `previous` é a versão que era champion antes da última promoção.

### 2.4 Regra de promoção e limiares pré-registrados
A regra é uma função pura (`src/monitoring/promotion.py`, `decide_promotion`). O challenger só é elegível se:
1. passa no contrato e na integridade do artefato (motivos `contract_failed` e `integrity_failed`);
2. no holdout do lote `CRITICO`, sua ROC-AUC supera a do champion em pelo menos `PROMOTION_MIN_GAIN` (motivo `gain_below_min`);
3. no teste da referência, sua ROC-AUC não fica abaixo da do champion em mais de `PROMOTION_MAX_REGRESSION` (motivo `regression_above_max`).

Os limiares são inclusivos, e as diferenças são arredondadas a 9 casas antes de comparar, só para absorver o erro de ponto flutuante. AUC não finita reprova o eixo correspondente.

**Limiares pré-registrados** (`src/config.py`):

| Constante | Valor | Justificativa |
| :--- | :---: | :--- |
| `PROMOTION_MIN_GAIN` | **0,02** | Fica bem acima do ruído de AUC do holdout. O holdout do t2 tem cerca de 46 mil linhas e 5,4 mil inadimplentes, e o erro-padrão de uma AUC nessa amostra fica perto de 0,004 (Hanley e McNeil). Mesmo tratando as duas AUCs como independentes, o erro-padrão da diferença fica perto de 0,006, e 0,02 passa de três vezes esse valor. Avaliar no lote inteiro daria mais linhas, mas vazaria o treino. |
| `PROMOTION_MAX_REGRESSION` | **0,02** | É uma tolerância simétrica ao ganho mínimo: o challenger pode ceder na referência no máximo o que precisa ganhar no lote novo. O teste da referência tem cerca de 43 mil linhas, com ruído da mesma ordem do holdout. A guarda impede que o modelo "esqueça" o comportamento anterior. |

**Os limiares foram fixados antes do retreino real.** Eles entraram em `src/config.py` em 2026-10-26 (commit `986e19e`), e nenhum retreino nem run de auditoria existia no `mlflow.db` real na data desta ADR (2026-10-27). Eles não serão ajustados depois de ver o resultado. Se o challenger real for reprovado, quem decide é o usuário (calibrar os limiares numa nova ADR ou aceitar o resultado), e a decisão fica registrada. Não se usa o `--override` para fabricar uma promoção que feche o aceite.

### 2.5 Promoção e rollback: comandos humanos
- O retreino treina, avalia e imprime o veredito com os números, mas **nunca move o alias `champion`**. A promoção automática (CT com guardrails) fica como evolução possível, trocando só esse passo.
- `make promote` **reavalia a regra** a partir das métricas do run de origem, e não das tags da versão, que são editáveis. Também recusa um challenger comparado com outro champion (`champion_mismatch`) ou já revertido por rollback (`rolled_back`). Inelegível sai com código 2 sem tocar em nada.
- `--override --reason` promove mesmo inelegível. O motivo é obrigatório e fica no run de auditoria, com `eligible=false` e os motivos. O override **não** dispensa a integridade do arquivo.
- A ordem da promoção é: baixar o artefato do run de origem para um temporário e conferir o sha256, mover `previous`, mover `champion`, `os.replace`, reconferir o estado servido e gravar o run de auditoria (`run_type=audit`, com quem, quando, por quê e o status `CONSISTENT` ou `INCONSISTENT`). Uma falha no meio desfaz o que for possível, grava `INCONSISTENT` e sai com 1. O `load_bundle` recusa um estado partido até a correção.
- `make rollback` devolve o `champion` à versão `previous`, reexporta o arquivo dela, reconfere e audita. Sem `previous`, falha com código 1 e não altera nada.
- Códigos de saída: `retrain` 0 (challenger registrado, elegível ou não), 2 (recusa de política), 3 (contrato barrou) e 1 (erro operacional). `promote` 0, 2 e 1. `rollback` 0 e 1.

### 2.6 Champion servido, `train` repetido e DVC
- O champion servido vive em `models/champion/` (`baseline_model.joblib` + `baseline_metadata.json`), **fora dos `outs` do DVC**, conferido pelo sha256 da tag `model_sha256` da versão `champion`. O diretório é estado local, ignorado pelo git.
- `models/baseline_model.joblib` (`out` do estágio `train`) é só o artefato do baseline e **nunca é tocado** por promoção nem por rollback.
- O `train` registra cada reconstrução do baseline como versão nova (`origin=baseline`). Só no bootstrap, quando ainda não existe `champion`, ele recebe o alias. **Um `train` repetido nunca move o champion.**
- **Dívida do `dvc repro` com promoção e rollback (decisão da Tarefa 15):** os dois arquivos de `models/champion/` são `deps` de `contract_demo` e `drift_demo` no `dvc.yaml`. Depois de uma promoção ou de um rollback, o `dvc repro` marca as duas demos como desatualizadas e as reexecuta com o champion novo. Limitação aceita: com `MONITORING_TRACKING=off` não há champion, e o `dvc repro` para nessas demos por dependência ausente; nesse caso, usa-se `make pipeline`.

### Matriz de Avaliação de Alternativas (Trade-off Matrix)

| Critério de Avaliação | A: Promoção automática pela regra | B: Retreino guardado + promoção humana (eleita) | C: Retreino manual, sem Registry |
| :--- | :--- | :--- | :--- |
| **Custo / Esforço** | Médio | Médio | Baixo |
| **Governança e auditoria** | Baixo (ninguém responde pela troca) | Alto (run de auditoria com quem, quando e por quê) | Baixo (sem trilha) |
| **Risco de promover modelo pior** | Médio | Baixo (regra + pessoa + rollback) | Alto |
| **Complexidade Operacional** | Média | Média | Baixa |
| **Aderência à matéria** | Média (CT com guardrails) | Alta (alerta aciona uma pessoa; champion/challenger) | Baixa |

```text
 monitor / demo-drift ──► run de lote CRITICO (MLflow)
                                │  make retrain (só com CRITICO não consumido, ou --force-no-alert)
                                ▼
   treino ref + 1/2 do lote ──► challenger (Registry, alias challenger) ──► veredito impresso
                                │  make promote (humano; reavalia a regra; --override exige --reason)
                                ▼
   sha256 conferido ─► previous ─► champion ─► os.replace(models/champion/) ─► reconferência ─► run de auditoria
                                │  make rollback (humano)
                                ▼
   champion volta para previous ─► arquivo reexportado ─► reconferência ─► run de auditoria
```

---

## 6. Matriz de Conformidade e Mitigação de Riscos

| Risco Mapeado no Contexto | Impacto Potencial | Mitigação Arquitetural e Técnica Adotada |
| :--- | :--- | :--- |
| Vazamento na avaliação | Alto | Holdout fora do treino; validação e teste da referência iguais aos do champion; conferência de `client_id` disjunto; AUC do champion recalculada no holdout |
| Limiar ajustado depois do resultado | Alto | Limiares em `src/config.py` desde 2026-10-26 e registrados nesta ADR antes do retreino real |
| Promoção sem dono | Alto | Promoção e rollback só por comando humano; run de auditoria; `--override` com motivo obrigatório |
| Arquivo servido divergente do Registry | Alto | Troca atômica (`os.replace`), reconferência, `INCONSISTENT` auditado e `load_bundle(expected_sha256)` recusando o estado partido |
| Retreino disparado por ruído | Médio | Gatilho só por `CRITICO`; consumo por lote; flags explícitos com motivo |

---

## 7. Consequências e Resultados

### Positivas
- **Ciclo de CT rastreável**: alerta, retreino, veredito, promoção e rollback ficam no MLflow, ligados por tags (MOD1 Aula 8, p. 12).
- **Governança como código**: a regra de promoção é uma função pura com limiares versionados (MOD1 Aula 7, p. 24).
- **Rollback sempre possível**: o alias `previous` e o artefato de cada run permitem voltar à versão anterior.

### Mitigações e Desafios Gerenciados
- **Demonstração com dado simulado**: t1 e t2 são simulados (ADR-0004). O ciclo é demonstrado, não validado em produção.
- **Challenger real pode ser reprovado**: o aceite exige um veredito correto, medido e registrado, não uma promoção. Os ramos de promoção e de rollback (e o alias `previous`) são exercitados por testes com scores controlados.
- **Vazamento entre gerações de modelo**: a conferência de `client_id` cobre as partes de um retreino, não o treino de um champion que já seja um challenger promovido. Se o `simulate` regerar o t2, linhas do treino anterior podem cair no novo holdout. Reaproveitar o mesmo t2 é seguro, porque a divisão é determinística na seed.
- **Baseline do monitor depois de uma promoção**: o monitor usa o run `production_t0` do mesmo `model_version`. Depois de promover, é preciso rodar `make demo-drift` (ou passar `--baseline-auc`) para o champion novo ter baseline. Sem isso, um lote rotulado falha fechado.
- **Auditoria depois da troca**: se o MLflow cair entre a troca reconferida e a gravação do run de auditoria, o comando sai com 1 e avisa que a troca foi feita sem auditoria. Não há como auditar antes sem registrar algo que ainda não aconteceu.
