# Relatório de aceite da Etapa 3: extensão §4.7 com os dados reais

> **Data:** 2026-10-27 · **Branch:** `etapa3-mitigacoes` · **Spec:** [§4.7 e §8, critério 10](../specs/2026-10-25-etapa3-observabilidade-design.md) · **ADR:** [0007](../../adrs/0007-retreino-registry-e-promocao-humana.md)
>
> Execução real, na máquina do usuário, com o `application_train.csv` do Kaggle e o `mlflow.db` local. Nenhum dado nem run sintético entrou nesta evidência. O núcleo (critérios 1 a 9) foi aceito antes, no `pipeline_run_id` `20261027T103457Z` (t0 `OK`, t1 `ATENCAO`, t2 `CRITICO`).

## Resumo

| Item | Resultado |
| :--- | :--- |
| Veredito do retreino | **ELEGÍVEL**, sem motivos de recusa |
| Ganho no holdout do t2 (46.124 linhas) | **+0,1316** (champion 0,5738 → challenger 0,7054; mínimo 0,02) |
| Regressão no teste da referência (43.049 linhas) | **+0,0089** (champion 0,7491 → challenger 0,7402; máximo 0,02) |
| Promoção v1 → v3 | feita, `CONSISTENT`, auditada |
| `make demo-drift` com o champion v3 | pontuou os três lotes com `retrain-987906221151` |
| Rollback v3 → v1 | feito, `CONSISTENT`, auditado; arquivo servido de volta ao sha256 da v1 |
| Repromoção da v3 | **pendente, decisão do usuário** (ver [Repromoção](#repromoção-pendente-decisão-do-usuário)) |
| Estado final | `champion` = v1 (baseline), `challenger` = v3 (`rolled_back=true`), sem `previous` |
| Aceite automatizado | `tests/integration/test_acceptance_real.py`: **8 passed, 0 skipped** |

Os limiares `PROMOTION_MIN_GAIN` e `PROMOTION_MAX_REGRESSION` (0,02 cada) estão em `src/config.py` desde o commit `986e19e` (2026-10-26) e foram registrados na ADR-0007 antes deste retreino. Nada foi ajustado depois do resultado.

## Estado de partida

Lido do `mlflow.db` real pelo `MlflowClient` (o mesmo backend que o `make tracking-ui` serve), antes de qualquer comando:

- `credit-default-xgboost` v1 e v2, ambas `origin=baseline`, `bundle_model_version=baseline-v1` e `model_sha256` `be8bd7bca128a1f932302048fccfa211d05b03aad672324000072ea28f47910a`. A v2 veio de um `train` repetido, que não move o alias.
- `champion` → v1. Sem `challenger` e sem `previous`. Nenhum run `run_type=retrain` nem `run_type=audit`.
- `models/champion/baseline_model.joblib` e `models/baseline_model.joblib` com o sha256 `be8bd7bc…` da v1.
- Alertas `CRITICO` do t2, nenhum consumido: `f76a9172…` (do `drift_demo` do pipeline `20261027T103457Z`), `bb321c99…` (de um `drift_demo` avulso) e `297661f4…` (de um `make monitor`). Os três têm o mesmo lote (`batch_sha256` `10dda6e4…`).

## Passo 1: retreino

```bash
make retrain ARGS="--alert-run f76a9172350740e3b1cb5f9c6232a6bb"
```

O `--alert-run` amarra o retreino ao alerta do pipeline aceito. Sem ele, o comando pegaria o alerta mais recente (`297661f4…`, do `make monitor`), que é do mesmo lote. Saída: código 0, em 2,6 s.

```text
Challenger v3 (retrain-987906221151) registrado com o alias challenger. Veredito: ELEGÍVEL à promoção (motivos: nenhum).
  Holdout do lote (46124 linhas): champion 0.5738 x challenger 0.7054 (ganho +0.1316; mínimo 0.02).
  Teste da referência: champion 0.7491 x challenger 0.7402 (regressão +0.0089; máximo 0.02).
O champion (v1) não foi alterado: promover é decisão humana (`make promote`).
```

| Métrica (run `d0172e9e…`) | Valor |
| :--- | ---: |
| `champion_holdout_auc` | 0,573783 |
| `challenger_holdout_auc` | 0,705383 |
| `gain_on_holdout` | 0,131600 |
| `champion_reference_auc` | 0,749123 |
| `challenger_reference_auc` | 0,740204 |
| `regression_on_reference` | 0,008919 |
| `threshold` (KS na validação) | 0,083836 |
| `train_rows` (treino da referência + metade do t2) | 175.268 |
| `critical_train_rows` / `holdout_rows` | 46.123 / 46.124 |
| `validation_rows` / `test_rows` (iguais às do champion) | 43.049 / 43.049 |

- O contrato rotulado aprovou a referência (215.243 linhas) e o t2 (92.247 linhas), sem violações.
- Run de estágio `retrain`: `ea494d0d6b54424fb6788416ff31f477` (`stage_status=SUCCESS`, `pipeline_run_id` `d50e8b20e3fb`). Run filho `retrain_challenger`: `d0172e9ed5ef41ed805ea7ba90264945`, com os params `promotion_min_gain=0.02` e `promotion_max_regression=0.02`.
- A versão 3 tem as tags `origin=retrain`, `bundle_model_version=retrain-987906221151`, `model_sha256` `81955014392fe05ae7ed2016107d7e41064d2cde10b793e6db71dab476fff5c2`, `reference_sha256` `d83ed8a5…` (igual à do champion), `batch_name=production_t2_concept_drift`, `batch_sha256` `10dda6e4…`, `trigger=alert`, `alert_run_id=f76a9172…`, `alert_model_version=baseline-v1`, `champion_version=1`, `retrain_run_id=d0172e9e…`, `promotion_eligible=true` e as quatro AUCs.
- Os três alertas `CRITICO` do mesmo lote receberam `retrain_consumed=3`.
- O AUC do champion foi recalculado no mesmo holdout (0,5738), e não reaproveitado do lote inteiro (0,5720).

## Passo 2: promoção

```bash
make promote
```

Código 0. `previous` → v1, `champion` → v3, `models/champion/baseline_model.joblib` trocado e conferido (sha256 `81955014…`). Run de auditoria `audit_promote_v3` `37d1e454ac9143429e992497ba491320`: `action=promote`, `operator=eduardobatista`, `timestamp_utc=2026-10-27T17:21:28Z`, `status=CONSISTENT`, `reason` padrão ("Challenger elegível pela regra de promoção (§4.7).") e `details` com `from_version=1`, `to_version=3`, `previous_version=1`, `eligible=true`, `override=false` e as quatro AUCs reavaliadas.

## Passo 3: `make demo-drift` com o champion novo

```bash
make demo-drift
```

Código 0. Run de estágio `drift_demo` `e78d7b2bd474450cbffee140ece0c2a5` (`pipeline_run_id` `a1064d5b9a72`). Os três runs de lote levam `model_version=retrain-987906221151`, o `bundle_model_version` exclusivo da v3. O `load_bundle` só carrega o arquivo servido se o sha256 dele for o da tag `model_sha256` da versão `champion` (`81955014…`), então o `drift_demo` usou o champion novo.

| Lote | Run | Nível | ROC-AUC | Queda vs t0 |
| :--- | :--- | :--- | ---: | ---: |
| t0 | `11813edc…` | `OK` | 0,7457 | 0,0000 |
| t1 | `34c2e4d2…` | `ATENCAO` (3 colunas e `score_psi` ≥ 0,10) | 0,7259 | 0,0198 |
| t2 | `04332552…` | `ATENCAO` (mesmos motivos) | 0,7239 | 0,0218 |

> **Leitura correta:** o t2 deixa de ser `CRITICO` com a v3, mas a AUC de 0,7239 é do t2 **inteiro**, que contém as 46.123 linhas do treino do challenger. Ela não serve de evidência de recuperação. A evidência é só o holdout do Passo 1 (0,7054 contra 0,5738). Esses runs também servem de baseline `production_t0` do `model_version` `retrain-987906221151` para o `make monitor`.

Os relatórios versionados em `reports/drift/` gerados por essa rodada foram restaurados para a versão comitada, que corresponde ao pipeline aceito e ao champion final (v1). Os números da rodada com a v3 ficam neste relatório e nos runs do MLflow.

## Passo 4: rollback

```bash
make rollback ARGS="--reason 'Aceite real da Etapa 3: demonstração do rollback (será repromovido em seguida).'"
```

Código 0. `champion` → v1, `previous` consumido (apagado), v3 marcada `rolled_back=true`, arquivo servido reexportado da v1 e conferido (sha256 `be8bd7bc…`). Run de auditoria `audit_rollback_v1` `c6574c294093461e95e15f30c32d05ee`: `action=rollback`, `status=CONSISTENT`, `timestamp_utc=2026-10-27T17:21:57Z`, `details` com `from_version=3`, `to_version=1`, `previous_version=null`. `models/baseline_model.joblib` ficou com o sha256 `be8bd7bc…` o tempo todo.

## Repromoção (pendente, decisão do usuário)

O plano era repromover a v3, para deixar o champion final igual ao challenger.

1. `make promote` **recusou** a v3, com código 2 e sem tocar em nada:

   ```text
   RECUSADO na promoção: Versão 3 inelegível à promoção (motivos: rolled_back): ganho no holdout +0.1316 (mínimo 0.02), regressão na referência +0.0089 (máximo 0.02). Nada foi alterado.
   ```

   É o comportamento projetado: uma versão revertida só volta com `--override --reason` (ADR-0007, §2.5). A regra numérica continua aprovando, e o único motivo é `rolled_back`.
2. `make promote ARGS="--override --reason '…'"` **não foi executado**. O classificador de permissões da sessão bloqueou o comando por usar um flag de exceção, e o bloqueio não foi contornado. O override é uma decisão humana, e fica com o usuário.

**Estado final, portanto:** `champion` = v1 (baseline), servido em `models/champion/` com o sha256 `be8bd7bc…`; `challenger` = v3 (`rolled_back=true`); sem `previous`.

Para deixar a v3 como champion da demo, o usuário roda, por conta própria:

```bash
make promote ARGS="--override --reason 'Repromoção da v3 após o rollback de demonstração do aceite; a regra numérica aprova.'"
make demo-drift   # baseline do monitor e relatórios com o champion novo
```

O run de auditoria vai gravar `override=true`, `eligible=false` e `reasons=["rolled_back"]`, ao lado das quatro AUCs que aprovam.

## Estado final do Registry e dos runs

Lido pelo `MlflowClient` no `mlflow.db` real:

| Versão | Origem | `bundle_model_version` | `model_sha256` | Aliases |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `baseline` | `baseline-v1` | `be8bd7bc…` | `champion` |
| 2 | `baseline` | `baseline-v1` | `be8bd7bc…` | |
| 3 | `retrain` (`rolled_back=true`) | `retrain-987906221151` | `81955014…` | `challenger` |

- Runs `run_type=retrain`: 1 (`d0172e9e…`). Runs `run_type=audit`: 2 (`37d1e454…` promote, `c6574c29…` rollback), ambos `CONSISTENT`.
- O ambiente de rastreamento não ganhou smoke tests nem runs extras: só os runs de estágio `retrain` e `drift_demo`, os 3 runs de lote do `drift_demo`, o run de retreino e os 2 de auditoria.

## Aceite automatizado

O teste `tests/integration/test_acceptance_real.py` ganhou três checagens do critério 10, com a lógica em `tests/integration/acceptance_checks.py` e testada com um Registry sintético em `tests/test_acceptance_checks.py`:

- `test_versoes_de_retreino_com_tags_e_run_de_origem`: cada versão `origin=retrain` tem as tags do challenger, `bundle_model_version` com o prefixo `retrain-`, o run de retreino de origem com o mesmo `model_sha256` e `challenger_version`, os params iguais aos limiares pré-registrados, o veredito igual ao da regra recalculada a partir das métricas do run e o alerta `CRITICO` com `retrain_consumed`;
- `test_aliases_coerentes_com_os_arquivos`: `champion` existe e o `model_sha256` dele é o de `models/champion/baseline_model.joblib`; `challenger` aponta para uma versão `origin=retrain`; `previous` é diferente do `champion`; `models/baseline_model.joblib` tem o sha256 de uma versão `origin=baseline`;
- `test_runs_de_auditoria_de_promocao_e_rollback`: cada run de auditoria tem ação, status, operador, motivo, hora e `details`; promoção sempre com `previous_version`; o mais recente é `CONSISTENT` e descreve os aliases de agora.

Sem versão `origin=retrain`, as duas primeiras PULAM; sem run de auditoria, a terceira PULA. Havendo retreino ou auditoria e alguma divergência, a falha é dura.

| Comando | Resultado |
| :--- | :--- |
| `uv run pytest tests/integration/test_acceptance_real.py` | 8 passed, 0 skipped |
| `make lint` | ruff limpo; import-linter com 3 contratos KEPT |
| `make test-cov` | 457 passed; cobertura 97,63%; os mesmos 16 warnings de terceiros da base (evidently, litestar, scipy) |

## Critério 10 da spec

| Exigência | Evidência |
| :--- | :--- |
| O `CRÍTICO` de t2 habilita o retreino | alerta `f76a9172…` consumido (`retrain_consumed=3`) |
| Challenger treinado sem tocar no holdout | partes disjuntas conferidas antes do treino; recuperação medida só no holdout |
| Regra testada em todos os ramos | testes das Tarefas 12 a 15; no real, o ramo elegível e a recusa por `rolled_back` |
| Registry com `champion`, `challenger` e tags; `previous` após uma promoção | v3 com as tags; `previous=1` depois da promoção (auditoria `37d1e454…`) |
| Rollback devolve o alias anterior | auditoria `c6574c29…`; alias e arquivo de volta à v1 |
| Run de auditoria por promoção e rollback | 2 runs `run_type=audit`, ambos `CONSISTENT` |
| O retreino nunca move o `champion` | `champion` continuou em v1 depois do `make retrain` |
| A promoção recusa inelegível sem o flag de exceção | `make promote` com código 2 para a v3 revertida |

## Pendências

1. **Repromoção da v3:** decisão do usuário (comando acima). Sem ela, a demo fica com o baseline v1 servido e a v3 como challenger revertido.
2. **Conferência visual da UI:** `make tracking-ui` e navegador, com o usuário. Este relatório leu o mesmo backend pela API do cliente, sem subir o servidor.
