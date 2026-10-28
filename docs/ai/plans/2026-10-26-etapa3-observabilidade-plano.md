# Etapa 3: Observabilidade, Registry e promoção — Plano de implementação

> **Para quem executar:** SUB-SKILL OBRIGATÓRIA: `superpowers:subagent-driven-development` (recomendado) ou `superpowers:executing-plans`. Os passos usam `- [ ]` para acompanhamento.
> **Revisão:** revisado por um agente Opus em 2026-10-26 (21 achados, incorporados aqui e na spec).
> **Status:** aprovado em 2026-10-26 e executado (Tasks 1 a 17). Os passos `- [ ]` abaixo são o roteiro original e não foram marcados; a evidência de execução está em `docs/ai/reports/2026-10-27-etapa3-aceite.md`.

**Goal:** Entregar o ambiente de rastreamento (MLflow + DVC locais) com níveis de alerta por lote e, depois do núcleo aceito, o ciclo de retreino, Model Registry e promoção humana.

**Architecture:** Nova camada `src/monitoring` no topo (`monitoring → drift → models → validation`). Só ela importa `mlflow`. As camadas de baixo ganham *ganchos* opcionais (observador de lote, `expected_sha256`) para a camada de cima observá-las sem que elas conheçam o MLflow. A regra de alerta e a regra de promoção são funções puras.

**Tech Stack:** Python 3.12, MLflow (backend SQLite local, Registry com aliases), DVC (remoto local), pytest, ruff, import-linter, `uv --locked`.

**Spec:** `docs/ai/specs/2026-10-25-etapa3-observabilidade-design.md` (seções citadas como §4.x). ADRs: 0005, 0006, e a 0007 (nova, Tarefa 16).

## Global Constraints

- Respostas, docs e mensagens de erro em PT-BR; identificadores de código em inglês ou como já existem.
- Testes não dependem do dataset do Kaggle; `make lint` (ruff + import-linter) limpo; cobertura `fail_under = 80` (hoje 95,6%, não reduzir).
- `mlflow` só pode ser importado em `src/monitoring/` (contrato `forbidden` novo, Tarefa 1).
- `except` silencioso proibido; `BLE001` proíbe `except Exception`. Para registrar falha de estágio, usar `try/finally` com flag de sucesso, nunca `except Exception`.
- Falhar fechado: MLflow indisponível ⇒ estágio sai com código ≠ 0 e mensagem clara; só `MONITORING_TRACKING=off` desliga o rastreamento (§3).
- Nenhum dado do Kaggle sai da máquina; nenhum run sintético para "mostrar" alerta (§2, anti-objetivos); sem credenciais em texto puro.
- Limiares vêm de `src/config.py`: `PSI_MODERATE=0.10`, `PSI_SIGNIFICANT=0.25`, `AUC_DROP_ALERT=0.05`; novos: `PROMOTION_MIN_GAIN=0.02`, `PROMOTION_MAX_REGRESSION=0.02` (palpites, calibrar na Tarefa 17).
- Códigos de nível de alerta em ASCII no código e nas tags: `OK`, `ATENCAO`, `CRITICO`, `BLOQUEADO` (rótulo com acento só para exibição). Nomes de tags e métricas: tabela "Contrato de nomes" da spec §4.1.
- Experimento MLflow `credit-drift-monitor`; modelo registrado `credit-default-xgboost`; aliases `champion`, `challenger`, `previous` (sem `Staging/Production`).
- Commits em Conventional Commits, atômicos, terminando com `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` e `Claude-Session: https://claude.ai/code/session_0155tkMstfoZc5LjuEWpYXXK`. Nada de `git push` nem merge na `main` sem pedido do usuário. O registro em `docs/commits/` é gerado no fim, com `/ai-guides-suite:commit-e-documentar`.
- Depois de cada tarefa que muda código: `graphify update .`.

## Decisões deste plano (revisar; as da spec seguem valendo)

1. **`make pipeline` encadeia os estágios** (`src.monitoring.stage`), sem passar pelo `dvc repro`: `dvc repro` pula estágios sem mudança e não registraria runs. O `make repro` é o caminho DVC (§4.8 permite).
2. **Ganchos em vez de acoplamento:** `drift.demo.run_demo(on_batch=...)`, `validation.demo.main(on_batch=...)` e `models.predict.load_bundle(expected_sha256=...)`. É o que mantém o MLflow fora das camadas inferiores.
3. **`make monitor` sem baseline de AUC falha fechado** para lote rotulado: o baseline é o AUC do t0 do mesmo `model_version`, lido do MLflow (último run de lote `production_t0` desse modelo) ou de `--baseline-auc`. Sem nenhum dos dois, sai com erro orientando a rodar `make demo-drift`. Lote sem rótulo nunca precisa de baseline.
4. **`train` repetido não move o `champion`.** Cada reconstrução do baseline vira nova versão do Registry (tag `origin=baseline`); só na primeira vez, sem `champion`, ela recebe o alias (bootstrap). Evita que `make pipeline` desfaça uma promoção.
5. **O champion servido vive em `models/champion/`**, fora dos `outs` do DVC (o `train` reescreve `models/baseline_*` a cada `dvc repro`). Cada run de treino anexa o `bundle` joblib e o metadata; promoção e rollback **copiam** esse artefato para `models/champion/` (arquivo temporário + `os.replace`), preservando o sha256. `load_bundle(path=None, *, expected_sha256=None)` usa `models/champion/` se existir, senão `models/`; `src.monitoring` passa o `expected_sha256` (tag `model_sha256` da versão `champion`). A comparação é por hash, e não pela string `bundle["model_version"]`, porque `MODEL_VERSION="baseline-v1"` é constante.
6. **Testes de MLflow nunca tocam o `mlflow.db` do repositório:** `tests/conftest.py` aponta `MLFLOW_TRACKING_URI`/`MLFLOW_ARTIFACT_ROOT` para um diretório temporário de sessão antes de importar `src` (como já faz com o log), e a fixture `tracking_env` entrega um banco novo por teste, copiado de um banco-modelo já migrado, criado uma vez por sessão (o SQLite dá "database is locked" com workers do xdist compartilhando um arquivo, e migrar custa segundos).
7. **Falha de MLflow é provocada** com URI de SQLite em diretório inexistente/sem permissão; `tracked_stage` faz uma checagem de saúde (criar/obter experimento e `search_runs(max_results=1)`) **antes** de rodar o estágio, e `MONITORING_TRACKING=off` usa um `NullStageRun` explícito.

## Review Focus

Entradas que a spec não exercita e que mais provavelmente quebram o uso real. Cada linha tem o teste na tarefa que a implementa.

1. **Lote rotulado com uma só classe** (ROC-AUC indefinida): esperado `performance_evaluated=false`, motivo `single_class` e nível máximo `ATENÇÃO`, nunca exceção (Tarefa 2 e 8).
2. **Lote vazio ou com 0 linhas aprovadas**: esperado erro claro do contrato (já barra) e, em `monitor`, nenhum run de lote com métricas inventadas (Tarefa 8).
3. **Scores constantes** (bordas de decil repetidas) no PSI do score: esperado PSI definido, sem divisão por zero nem `NaN` (Tarefa 3).
4. **Reexecução com o mesmo `PIPELINE_RUN_ID`** e **`mlflow.db` bloqueado/inacessível**: esperado runs acumulados distinguíveis e, no segundo caso, falha fechada com mensagem, sem criar `mlflow.db` em outro diretório (Tarefas 4 e 5).
5. **Retreino sobre alerta já consumido, sem alerta `CRÍTICO` ou com `--alert-run` de um lote não crítico**: esperado recusa com mensagem; o flag explícito é a única saída (Tarefa 14).

## Estrutura de arquivos

| Arquivo | Responsabilidade | Tarefa |
| :--- | :--- | :--- |
| `src/config.py` (mod.) | Constantes MLflow/promoção | 1 |
| `src/monitoring/alerts.py` | `AlertLevel`, `AlertDecision`, `decide_alert_level` (puro) | 2 |
| `src/models/metrics.py` (mod.) | `score_distribution` | 3 |
| `src/drift/psi.py` (mod.) | `score_psi` | 3 |
| `src/monitoring/resources.py` | `peak_memory_mb` | 4 |
| `src/monitoring/logs.py` | `extract_run_events` | 4 |
| `src/monitoring/tracking.py` | Adaptador MLflow: `tracked_stage`, `StageRun`, `log_batch` | 5 |
| `src/monitoring/stage.py` | CLI `python -m src.monitoring.stage <estágio>` | 6, 7, 8 |
| `src/monitoring/registry.py` | Registry (versões, aliases, tags, auditoria) | 6, 13 |
| `src/drift/demo.py`, `src/validation/demo.py` (mod.) | Ganchos `on_batch` | 7, 8 |
| `src/monitoring/monitor.py` | CLI de um lote | 10 |
| `src/monitoring/promotion.py` | `decide_promotion` (puro) | 12 |
| `src/monitoring/retrain.py` | `make retrain` | 14 |
| `src/monitoring/promote.py` | `make promote` e `make rollback` | 15 |
| `dvc.yaml`, `.dvc/`, `tests/test_dvc_yaml.py` | DVC mínimo | 11 |
| `tests/test_monitoring_*.py`, `tests/test_repo_hygiene.py` | Testes novos | todas |

---

# FASE A — Núcleo (§4.1 a §4.6)

### Task 1: Dependências, configuração, higiene do git e contrato de camadas

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `src/config.py`, `.gitignore`, `tests/conftest.py`
- Create: `src/monitoring/__init__.py`, `tests/test_repo_hygiene.py`

**Interfaces:**
- Produces (`src/config.py`): `MLFLOW_EXPERIMENT: str`, `MLFLOW_TRACKING_URI: str`, `MLFLOW_ARTIFACT_ROOT: Path`, `REGISTERED_MODEL_NAME: str`, `PROMOTION_MIN_GAIN: float`, `PROMOTION_MAX_REGRESSION: float`, `SCORE_PSI_BINS: int`, `CHAMPION_DIR: Path` (`models/champion`), `CHAMPION_FILE: Path`, `CHAMPION_METADATA_FILE: Path`. `MLFLOW_TRACKING_URI` e `MLFLOW_ARTIFACT_ROOT` leem as variáveis de ambiente homônimas; o padrão é `sqlite:///<ROOT_DIR>/mlflow.db` e `<ROOT_DIR>/mlartifacts` (caminho absoluto, para o worktree e o checkout principal não divergirem).
- Produces (`tests/conftest.py`): fixture `tracking_env(tmp_path, monkeypatch)` que aponta `MLFLOW_TRACKING_URI`/`MLFLOW_ARTIFACT_ROOT` para o `tmp_path` e retorna o URI. Como `src.config` lê o ambiente na importação, a fixture também reatribui `src.config.MLFLOW_*` com `monkeypatch.setattr`. O `conftest.py` define os mesmos padrões de sessão antes de qualquer `import src`, e `tracking_env` copia um banco-modelo migrado (criado uma vez por sessão) para cada teste.

- [ ] **Step 1: Teste de higiene (falha primeiro)** em `tests/test_repo_hygiene.py`:
  - `test_nada_de_data_no_indice_do_git`: `git ls-files data/` só devolve `.gitkeep` (execute com `subprocess`, `check=True`; `pytest.skip` se não for repositório git).
  - `test_artefatos_do_mlflow_ignorados`: `git check-ignore mlflow.db mlartifacts mlruns` retorna 0 para os três.
  - `test_mlflow_so_na_camada_de_monitoring`: nenhum `.py` fora de `src/monitoring/` e `tests/` contém `import mlflow` ou `from mlflow` (varredura de texto).
- [ ] **Step 2:** `uv run pytest tests/test_repo_hygiene.py -v` → FALHA nos dois últimos (ainda não ignorados; o terceiro passa e fica como guarda).
- [ ] **Step 3: Dependências.** `uv add "mlflow>=3.0,<4"` (grupo principal) e `uv add --dev "dvc>=3.50,<4"`. O construtor confirma na versão instalada: (a) `mlflow.set_registry_uri`/aliases (`MlflowClient.set_registered_model_alias`) e o backend SQLite funcionam; (b) `uv sync --locked` ainda é limpo. Se o teto de versão mudar, ajustar o comentário em `pyproject.toml` explicando o porquê (padrão do `evidently`).
- [ ] **Step 4: `.gitignore`** com `mlflow.db`, `mlartifacts/`, `mlruns/`, `.dvc/config.local`, `.dvc/cache`, `.dvc/tmp`. **`src/config.py`** com as constantes da seção Interfaces (`PROMOTION_*` = 0.02, `SCORE_PSI_BINS` = 10).
- [ ] **Step 5: Contrato de camadas** em `pyproject.toml`: `layers = ["src.monitoring", "src.drift", "src.models", "src.validation"]` (mantendo o `ignore_imports` existente) e um contrato `forbidden` com `source_modules = ["src.drift", "src.models", "src.validation", "src.data", "src.config"]` e `forbidden_modules = ["mlflow", "dvc"]`.
- [ ] **Step 6:** `make lint && make test` → verde; `uv sync --locked` limpo.
- [ ] **Step 7: Commit** `chore(deps): adicionar mlflow e dvc, config e camada monitoring`.

---

### Task 2: Regra de alerta (função pura)

**Files:** Create `src/monitoring/alerts.py`, `tests/test_monitoring_alerts.py`.

**Interfaces — Produces:**
```python
class AlertLevel(StrEnum):  # valores exatos
    OK = "OK"; ATENCAO = "ATENCAO"; CRITICO = "CRITICO"; BLOQUEADO = "BLOQUEADO"
    @property
    def label(self) -> str: ...      # "ATENÇÃO", "CRÍTICO" (exibição)

@dataclass(frozen=True)
class AlertDecision:
    level: AlertLevel
    reasons: tuple[str, ...]        # estáveis: "contract_blocked", "auc_drop>=0.05", "drift:<coluna>", "score_psi>=0.10", "single_class"
    performance_evaluated: bool
    action: str                     # texto do runbook da §4.2 para o nível

def decide_alert_level(
    *, blocked: bool, drifted_columns: Sequence[str],
    score_psi: float | None, auc_drop: float | None,
    performance_skip_reason: str | None = None,
) -> AlertDecision
```
`drifted_columns` **não** inclui `target` (o chamador o remove; o resultado sobre `target` vai à parte como `label_shift`). `auc_drop=None` significa "não avaliado" (lote sem rótulo ou `single_class`); nesse caso `performance_evaluated=False`. Limiares vêm de `src.config`.

- [ ] **Step 1: Testes** (um por linha da tabela §4.2 e por regra de precedência):
  - `test_bloqueado_vence_tudo`: `blocked=True`, `auc_drop=0.30`, colunas driftadas ⇒ `BLOQUEADO`, motivos `("contract_blocked",)`.
  - `test_critico_por_queda_de_auc`: `auc_drop=0.177` ⇒ `CRÍTICO`, `performance_evaluated=True`; `auc_drop=0.05` (igual ao limiar) ⇒ `CRÍTICO`; `0.0499` ⇒ não crítico.
  - `test_atencao_por_coluna`: `drifted_columns=["annuity_amount"]`, `auc_drop=0.027` ⇒ `ATENÇÃO` com motivo `drift:annuity_amount` (caso t1 da §4.2).
  - `test_atencao_por_psi_do_score`: `score_psi=0.10` ⇒ `ATENÇÃO`; `0.0999` ⇒ `OK`.
  - `test_ok`: sem colunas, `score_psi=0.02`, `auc_drop=0.0` ⇒ `OK`, `action` não vazio.
  - `test_sem_rotulo_nunca_passa_de_atencao`: `auc_drop=None` com `score_psi=0.40` ⇒ `ATENÇÃO`, `performance_evaluated=False`.
  - `test_classe_unica_registra_motivo`: `auc_drop=None, performance_skip_reason="single_class"` ⇒ motivo `single_class` presente, nível máximo `ATENÇÃO` (Review Focus 1).
  - `test_toda_decisao_tem_acao`: percorre os quatro níveis e confirma `action` não vazia (anti-objetivo "alerta sem ação").
- [ ] **Step 2:** rodar → FALHA (`ModuleNotFoundError`).
- [ ] **Step 3: Implementar** `AlertLevel`, `AlertDecision`, `decide_alert_level` e `ACTIONS: dict[AlertLevel, str]` com os textos da coluna "Ação associada" da §4.2. Precedência `BLOQUEADO > CRÍTICO > ATENÇÃO > OK`; módulo sem `import mlflow`.
- [ ] **Step 4:** rodar → PASSA. **Step 5: Commit** `feat(monitoring): regra pura de nível de alerta`.

---

### Task 3: Distribuição de referência do score e PSI do score

**Files:** Modify `src/models/metrics.py`, `src/models/train.py`, `src/drift/psi.py`; Test `tests/test_psi.py`, `tests/test_train.py`.

**Interfaces — Produces:**
```python
# src/models/metrics.py
def score_distribution(scores: np.ndarray, n_bins: int = SCORE_PSI_BINS) -> dict:
    """{"edges": [n_bins-1 bordas internas, estritamente crescentes], "shares": [n_bins proporções somando 1]}"""
# src/drift/psi.py
def score_psi(reference: dict, scores: np.ndarray) -> float
```
`train_models` passa a incluir `metadata["score_reference"] = score_distribution(<scores do split de teste, modelo xgboost>)` (só metadados; o `sha256` do modelo não muda).

- [ ] **Step 1: Testes.**
  - `test_score_distribution_formato`: scores uniformes → `len(edges)==9`, `len(shares)==10`, `sum(shares)==pytest.approx(1)`, bordas crescentes.
  - `test_score_distribution_scores_constantes` (Review Focus 3): `np.full(100, 0.3)` → bordas deduplicadas (≥ 1 faixa, `shares` soma 1), sem exceção.
  - `test_score_psi_identico_e_zero`: `score_psi(dist, mesmos_scores) < 1e-9`.
  - `test_score_psi_deslocado_alto`: scores deslocados (`+0.3`, clip em 1) ⇒ `> PSI_SIGNIFICANT`.
  - `test_score_psi_constante_e_finito`: referência constante vs. scores constantes diferentes ⇒ `math.isfinite`.
  - `test_train_grava_score_reference` em `tests/test_train.py`: `metadata["score_reference"]["shares"]` soma 1.
- [ ] **Step 2:** rodar → FALHA. **Step 3: Implementar.** `score_distribution` usa quantis do score (bordas internas únicas, via `np.unique`); `score_psi` agrupa `scores` com `np.searchsorted(edges, ..., side="right")`, reaproveita `_shares`/`EPSILON` de `psi.py` e calcula o PSI por soma `(c - r) * ln(c / r)`.
- [ ] **Step 4:** rodar → PASSA, `make lint`. **Step 5: Commit** `feat(drift): PSI do score contra a distribuição do teste do treino`.

---

### Task 4: Utilitários de infraestrutura (memória e extração de log)

**Files:** Create `src/monitoring/resources.py`, `src/monitoring/logs.py`; Test `tests/test_monitoring_utils.py`.

**Interfaces — Produces:**
```python
def peak_memory_mb() -> float                                   # resources.py; portátil macOS (bytes) e Linux (KiB)
def extract_run_events(log_file: Path, run_id: str, *, since: datetime, until: datetime) -> list[str]
# logs.py; linhas JSONL com "run_id" == run_id e timestamp em [since, until] (o PIPELINE_RUN_ID é comum a todos os estágios)
```
`extract_run_events` também lê os arquivos rotacionados (`pipeline.jsonl.1` …) e ignora linhas que não são JSON válido **contando-as** (devolve só as válidas; quem chama registra a contagem de descartadas via `log_event`, nada silencioso).

- [ ] **Step 1: Testes.** `test_peak_memory_positivo_e_em_mb` (entre 10 e 100 000); `test_peak_memory_converte_unidade_por_plataforma` (monkeypatch de `sys.platform` e de `resource.getrusage` para `ru_maxrss=1_048_576`: darwin ⇒ 1.0, linux ⇒ 1024.0); `test_extract_run_events_filtra_por_run_id_e_janela` (eventos do mesmo `run_id` fora da janela ficam de fora); `test_extract_run_events_le_arquivo_rotacionado`; `test_extract_run_events_descarta_linha_invalida_sem_levantar`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar.** **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): memória de pico e extração de logs por run_id`.

---

### Task 5: Adaptador do MLflow (`tracked_stage`, run de lote)

**Files:** Create `src/monitoring/tracking.py`; Test `tests/test_monitoring_tracking.py` (usa `tracking_env`).

**Interfaces — Produces:**
```python
class TrackingUnavailableError(RuntimeError): ...
def tracking_enabled() -> bool                 # False só se MONITORING_TRACKING=off (lido na chamada)
def configure_tracking() -> str                # define URI, cria o experimento com artifact_location=MLFLOW_ARTIFACT_ROOT, devolve experiment_id; levanta TrackingUnavailableError

@dataclass
class BatchRecord:
    batch_name: str; decision: AlertDecision; n_rows: int; rows_rejected: int
    metrics: dict[str, float]            # roc_auc, auc_drop, auc_drop_vs_training, approval_rate, default_rate_approved, score_psi
    psi_by_column: dict[str, float]; n_drifted_columns: int
    model_version: str; artifacts: list[Path] = field(default_factory=list)
    intentional_demo: bool = False

class StageRun:                                   # real (MLflow) ou nulo (rastreamento desligado), mesma interface
    run_id: str | None
    def add_rows(self, processed: int, rejected: int = 0) -> None
    def log_batch(self, record: BatchRecord) -> str | None      # run filho aninhado; conta alertas
    def set_metric(self, key: str, value: float) -> None
    def attach(self, path: Path) -> None

@contextmanager
def tracked_stage(stage: str, *, pipeline_run_id: str = RUN_ID) -> Iterator[StageRun]
```
`tracked_stage` primeiro faz a checagem de saúde do MLflow (decisão 7) e abre o run pai (tag `run_type=stage`); os runs de lote levam `run_type=batch`, `alert_level` (código ASCII), `alert_label`, `alert_reasons`, `batch_name`, `batch_path`, `batch_sha256`, `model_version`, `performance_evaluated` (contrato de nomes da spec §4.1). **Em `finally`** grava `stage_success` (1/0), `stage_duration_s`, `peak_memory_mb`, `rows_processed`, `rows_rejected`, `alerts_failure_count` (lotes `BLOQUEADO` + 1 se o estágio falhou), `alerts_drift_count` (lotes `ATENÇÃO`/`CRÍTICO`), as tags `pipeline_run_id`, `stage`, `stage_status` (`SUCCESS`/`FAILED`) e anexa o JSONL do `run_id` na janela do estágio (Tarefa 4) em `logs/`. `rows_rejected` = soma de `n_rows` dos lotes `BLOQUEADO`. Um lote `BLOQUEADO` conta em `alerts_failure_count` mesmo quando `intentional_demo=True` (spec §4.1: o alerta é real); a tag `intentional_demo=true` só distingue o bloqueio de demonstração de uma falha de pipeline, e `stage_success` fica 1.

- [ ] **Step 1: Testes** (SQLite temporário; consulta com `MlflowClient`):
  - `test_stage_sucesso_grava_metricas_e_tags`: após `with tracked_stage("train")`, o run tem `stage_success=1`, `stage_duration_s>0`, `peak_memory_mb>0`, tags `pipeline_run_id`, `stage`, `stage_status=SUCCESS`.
  - `test_stage_falha_grava_e_repropaga`: bloco que levanta `ValueError` ⇒ exceção sai intacta, run com `stage_success=0`, `stage_status=FAILED`, `alerts_failure_count>=1`. Idem com `SystemExit`.
  - `test_log_batch_cria_run_filho_com_nivel_e_motivos`: run filho com `parentRunId` = pai, tags `run_type=batch`, `alert_level=CRITICO`, `alert_label=CRÍTICO`, `alert_reasons`, `batch_sha256`, métricas do `record`, `performance_evaluated`; `search_runs(filter_string="tags.alert_level = 'CRITICO'")` o encontra.
  - `test_contadores_de_alerta`: um lote `ATENÇÃO`, um `CRÍTICO`, um `BLOQUEADO` ⇒ `alerts_drift_count==2`, `alerts_failure_count==1`.
  - `test_log_jsonl_anexado_ao_run`: artefato `logs/pipeline_<run_id>.jsonl` existe e só tem eventos do `run_id` dentro da janela do estágio (um evento do mesmo `run_id` de um estágio anterior não entra).
  - `test_reexecucao_com_mesmo_pipeline_run_id_acumula_runs` (Review Focus 4): duas execuções ⇒ dois runs de estágio com a mesma tag.
  - `test_tracking_off_nao_cria_banco`: `MONITORING_TRACKING=off` ⇒ `NullStageRun` (`run_id is None`), métodos no-op, nenhum banco criado.
  - `test_mlflow_inacessivel_falha_fechado` (Review Focus 4): `MLFLOW_TRACKING_URI` apontando para um diretório inexistente/sem permissão (`sqlite:////proc/inexistente/mlflow.db`) ⇒ `TrackingUnavailableError` com mensagem que cita o URI e `MONITORING_TRACKING=off`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar** com `mlflow.start_run` aninhado (`nested=True`) para filhos. Converter qualquer erro de conexão/permissão do MLflow em `TrackingUnavailableError` (capturando as exceções específicas do MLflow/SQLAlchemy/OSError, não `Exception`). **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): adaptador MLflow com runs de estágio e de lote`.

---

### Task 6: Registry e treino rastreado (estágios `prepare` e `train`)

**Files:** Create `src/monitoring/registry.py`, `src/monitoring/stage.py`; Modify `src/models/train.py` (retorno de `main()`), `src/config.py` já traz `CHAMPION_*` (Tarefa 1); Test `tests/test_monitoring_stage.py`, `tests/test_monitoring_registry.py`.

**Interfaces — Produces:**
```python
# registry.py
def register_model_version(*, run_id: str, pipeline, signature, input_example, bundle_path: Path,
                           metadata_path: Path, tags: dict[str, str]) -> int   # versão criada em REGISTERED_MODEL_NAME
def get_alias_version(alias: str) -> ModelVersion | None
def set_alias(alias: str, version: int) -> None
def bootstrap_champion(version: int) -> bool      # dá o alias champion só se ainda não existir champion; devolve se deu
def export_champion(version: int) -> None         # copia bundle+metadata do run de origem para models/champion/ (atômico), conferindo o sha256 (usado no bootstrap e na Tarefa 15)
# stage.py
def main(argv: list[str] | None = None) -> int     # python -m src.monitoring.stage {prepare,train,contract_demo,drift_demo}
```
`src/models/train.main()` passa a devolver `(bundle, metadata)` (hoje devolve `None`); o CLI antigo `python -m src.models.train` continua funcionando.

- [ ] **Step 1: Testes.**
  - `test_stage_train_registra_run_de_treino`: com um `reference` sintético pequeno (fixtures `make_valid_df`) e caminhos redirecionados para `tmp_path`, o estágio `train` gera: run filho de treino com `xgboost_params` como parâmetros, `roc_auc`/`ks` de teste, `reference_sha256` e `model_sha256` como tags, modelo com *signature* e *input example*, artefatos `bundle/baseline_model.joblib` e `bundle/baseline_metadata.json`.
  - `test_train_bootstrap_registra_v1_como_champion`: `get_alias_version("champion").version == "1"` e `models/champion/baseline_model.joblib` existe com o sha256 da tag `model_sha256`.
  - `test_train_repetido_nao_move_champion`: a segunda execução cria a v2 (tag `origin=baseline`), mas `champion` continua na v1 e `models/champion/` não muda.
  - `test_run_de_treino_anexa_bundle_e_metadata`: artefatos `bundle/baseline_model.joblib` e `bundle/baseline_metadata.json`; o modelo logado tem *signature*, *input example* e é carregável por `mlflow.pyfunc` com `predict_proba`.
  - `test_stage_prepare_registra_volume`: `rows_processed` = linhas da referência + produção.
  - `test_stage_falha_fechada_sem_mlflow`: URI inválido ⇒ código de saída ≠ 0 e mensagem; `MONITORING_TRACKING=off` ⇒ executa o estágio e retorna 0 sem criar `mlflow.db`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar.** `stage.main` resolve o estágio por nome, abre `tracked_stage`, chama a função do módulo de baixo (`src.data.prepare.main`, `src.models.train.main`, …) e, para `train`, loga parâmetros/métricas/tags, `mlflow.sklearn.log_model(..., registered_model_name=None)` + `register_model_version` (a versão é criada a partir do modelo logado), anexa o `bundle` e o metadata, e chama `bootstrap_champion` e, se ele deu o alias, `export_champion`. O modelo é logado com `mlflow.sklearn.log_model(..., pyfunc_predict_fn="predict_proba")`; o `threshold` e o `bundle` completos ficam no artefato `bundle/`. Tags da versão: `origin=baseline`, `reference_sha256`, `model_sha256`, `bundle_model_version`. **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): estágios prepare/train rastreados e bootstrap do Registry`.

---

### Task 7: Gancho de lote na demo de drift e estágio `drift_demo`

**Files:** Modify `src/drift/demo.py`, `src/models/predict.py`, `src/monitoring/stage.py`; Test `tests/test_monitoring_drift_stage.py`, ajustar `tests/test_detect.py` se necessário.

**Interfaces — Produces:**
```python
# src/drift/demo.py
@dataclass(frozen=True)
class BatchObservation:
    batch_name: str; scored: pd.DataFrame; metrics: dict; drift: DriftResult
    report_dir: Path; model_version: str
def run_demo(*, report_dir=..., evidently_sample=..., on_batch: Callable[[BatchObservation], None] | None = None) -> dict
# src/models/predict.py
def load_bundle(path: Path | None = None, *, expected_sha256: str | None = None) -> dict
# path=None ⇒ CHAMPION_FILE se existir, senão MODEL_FILE; se expected_sha256 e o sha256 do arquivo divergirem ⇒ sys.exit("Modelo recusado: ...")
```
`on_batch` é chamado uma vez por lote, **na ordem t0, t1, t2**, depois de calculadas as métricas e a detecção.

- [ ] **Step 1: Testes.**
  - `test_run_demo_chama_on_batch_na_ordem` (com artefatos sintéticos e `evidently_sample` pequeno): `[t0, t1, t2]` e cada observação traz `scored["default_probability"]`.
  - `test_load_bundle_recusa_hash_divergente`: `expected_sha256="0"*64` ⇒ `SystemExit` com "Modelo recusado"; `None` e hash igual passam. `test_load_bundle_prefere_champion_dir`: com `models/champion/` presente, carrega dele.
  - `test_stage_drift_demo_classifica_lotes`: com um observador montado pelo estágio e dados sintéticos controlados (t0 estável; t1 com colunas deslocadas; t2 com AUC derrubado por inversão de rótulos), os runs filhos ficam `OK`, `ATENÇÃO`, `CRÍTICO` nessa ordem. **Os dados são sintéticos de teste; este teste não substitui o aceite real da Tarefa 11.**
  - `test_stage_drift_demo_baseline_e_o_auc_do_t0`: `auc_drop` do t0 é 0 e do t2 é `auc_t0 - auc_t2`.
  - `test_run_filho_traz_relatorios_e_psi_do_score`: artefatos HTML/JSON do Evidently anexados; métrica `score_psi` presente; `auc_drop_vs_training` calculado contra o `roc_auc` do `baseline_metadata.json` (só informativo, fora da regra).
  - `test_target_nao_conta_como_coluna_driftada`: `drifted_columns` entregue a `decide_alert_level` não contém `target`, mesmo com `label_shift=True`.
  - `test_lote_sem_rotulo_nao_avalia_performance`: observação sem coluna `target` ⇒ `performance_evaluated=false`, nível ≤ `ATENÇÃO`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar.** O estágio `drift_demo` lê a tag `model_sha256` do alias `champion` (Registry) e passa como `expected_sha256` a `load_bundle` **dentro de `run_demo`** via parâmetro novo `bundle_loader: Callable[[], dict] | None` (padrão `load_bundle`); o observador do estágio calcula `score_psi` com `metadata["score_reference"]`, monta `AlertDecision` e `BatchRecord` e chama `StageRun.log_batch`. **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): estágio drift_demo com nível de alerta por lote`.

---

### Task 8: Estágio `contract_demo` (BLOQUEADO)

**Files:** Modify `src/validation/demo.py`, `src/monitoring/stage.py`; Test `tests/test_monitoring_contract_stage.py`.

**Interfaces — Produces:**
```python
# src/validation/demo.py
def main(on_batch: Callable[[str, str, ValidationResult | None, int], None] | None = None) -> int
# callback(batch_name, status "APROVADO"|"BLOQUEADO", result, n_pontuados)
```

- [ ] **Step 1: Testes.**
  - `test_contract_demo_gera_run_bloqueado`: roda com os lotes da demo (sintéticos via fixtures) ⇒ um run filho `APROVADO→OK` e um `BLOQUEADO` com tag `intentional_demo=true`.
  - `test_contract_demo_estagio_continua_com_sucesso`: `stage_success=1` mesmo com o lote inválido bloqueado.
  - `test_bloqueado_incrementa_alerts_failure_count_no_run_do_lote` conforme a spec §4.1 (o bloqueio intencional conta).
  - `test_lote_vazio_nao_gera_run_com_metricas` (Review Focus 2): `on_batch` com 0 linhas pontuadas e sem `result` ⇒ erro explícito, nenhum run de lote com `roc_auc`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar** o gancho e o observador (motivos do `BLOQUEADO` = `rules_violated` do `ValidationResult`; relatório do contrato anexado). **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): estágio contract_demo registra o bloqueio do lote inválido`.

---

### Task 9: Renomear `concept_drift` → `label_shift` (§4.6)

**Files:** Modify `src/drift/detect.py`, `src/drift/demo.py`, `tests/test_detect.py`, docs que citam o campo (`docs/etapa2*`, `README.md`); emendar a ADR-0005 item 4 (nota de emenda datada, sem reescrever o histórico); regenerar `reports/drift/` na Tarefa 11.

**Interfaces:** `DriftResult.label_shift: bool` substitui `concept_drift`; `to_dict()` emite `label_shift` e, **por compatibilidade com os relatórios antigos**, também `concept_drift` com o mesmo valor, marcado como obsoleto no docstring (removido na Etapa 4). O alerta de concept drift passa a ler só `performance.degraded` (já é o caso em `decide_alert_level`).

- [ ] **Step 1: Teste** `test_driftresult_expoe_label_shift_e_alias_legado`: `to_dict()["label_shift"] == to_dict()["concept_drift"]`; o log `drift_detected` usa `label_shift`.
- [ ] **Step 2:** FALHA. **Step 3:** renomear (usar `graphify query`/grep pelo identificador), atualizar `_render_markdown` ("Label shift"), CLI `main`. **Step 4:** `make test lint` verdes. **Step 5: Commit** `refactor(drift): renomear concept_drift para label_shift com alias legado`.

---

### Task 10: `monitor` — avaliação de um lote

**Files:** Create `src/monitoring/monitor.py`; Test `tests/test_monitoring_monitor.py`.

**Interfaces — Produces:** `main(argv=None) -> int` com `--batch PATH`, `--baseline-auc FLOAT` (opcional), `--labeled/--no-labeled` inferido pela coluna `target`. Sai com 0 em `OK`/`ATENCAO`, **2 em `CRITICO`**, **3 em `BLOQUEADO`**, 1 em erro operacional. O baseline de AUC vem do run de lote mais recente com `batch_name=production_t0` e o mesmo `model_version` (filtro no cliente, sem depender de tag ausente), ou de `--baseline-auc`. Executa: `read_batch` → `load_bundle(expected_sha256=<champion>)` → contrato (via `score_batch`) → drift contra a referência → `score_psi` → AUC (se rotulado) → `decide_alert_level` → `log_batch` num run de estágio `monitor`.

- [ ] **Step 1: Testes.**
  - `test_monitor_lote_sem_rotulo_ok`: lote sintético estável sem `target` ⇒ exit 0, nível `OK`, `performance_evaluated=false`.
  - `test_monitor_lote_invalido_e_bloqueado`: exit 3, run `BLOQUEADO`, nenhuma pontuação.
  - `test_monitor_rotulado_sem_baseline_falha_fechado`: sem run `production_t0` e sem `--baseline-auc` ⇒ exit 1, mensagem manda rodar `make demo-drift`.
  - `test_monitor_rotulado_degradado_e_critico`: `--baseline-auc 0.75` com lote cujo AUC cai ≥ 0,05 ⇒ exit 2.
  - `test_monitor_classe_unica_nao_quebra` (Review Focus 1): lote rotulado só com `target=0` ⇒ `performance_evaluated=false`, motivo `single_class`, exit 0 ou 2 apenas por outros sinais, sem `ValueError`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar** reaproveitando `run_drift_detection`, `score_batch`, `evaluate`, `assess_auc_drop` e a seleção de colunas `DEMO_DRIFT_COLUMNS` (exportar de `src/drift/demo.py` como `MONITORED_COLUMNS`, mantendo o nome antigo como alias). **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): comando monitor para avaliar um lote`.

---

### Task 11: Makefile, DVC e aceite real do núcleo

**Files:** Modify `Makefile`; Create `dvc.yaml`, `.dvc/` (por `uv run dvc init`), `tests/test_dvc_yaml.py`; Create `tests/integration/test_acceptance_real.py`; Modify `README.md`.

**Interfaces — Produces (Makefile, §4.8):** alvos `pipeline`, `monitor BATCH=`, `demo-drift`, `tracking-ui`, `repro`, e os apelidos `etapa1`, `etapa2`, `drift-demo`. `pipeline` exporta um único `PIPELINE_RUN_ID` (`date -u +%Y%m%dT%H%M%SZ`) e executa `src.monitoring.stage prepare|train|contract_demo|drift_demo` em sequência; `data`, `train`, `contract-demo` e `drift-demo` passam a chamar o mesmo módulo de estágio. `tracking-ui` = `mlflow ui --backend-store-uri <URI absoluto> --port 5001` (a 5000 colide com o AirPlay no macOS). Alvos: `demo-drift` (principal) e `drift-demo` (apelido). `repro` = `uv run dvc repro`.

- [ ] **Step 1: DVC.** Teste primeiro, em `tests/test_dvc_yaml.py` (lê `dvc.yaml` com `pyyaml`): `test_estagios_na_ordem` (`prepare → train → contract_demo → drift_demo`), `test_outs_nao_se_sobrepoem` (nenhum `out` é igual ou prefixo de outro, entre estágios), `test_outs_de_reports_e_models_sem_cache` (`cache: false`), `test_raw_csv_e_dep_do_prepare` e `test_champion_dir_fora_dos_outs`. Depois: `uv run dvc init`; `dvc.yaml` com `cmd: uv run python -m src.monitoring.stage <nome>` e `outs` arquivo por arquivo (o construtor levanta os arquivos de cada estágio no código); **sem `dvc add`** (o CSV é `deps` do `prepare`; o hash vai ao `dvc.lock`); remoto local só em `.dvc/config.local`, fora do repositório. O construtor confirma que `dvc repro` aceita esses `outs` e que o `.gitignore` gerado pelo DVC não conflita.
- [ ] **Step 2: Teste de aceite real** `tests/integration/test_acceptance_real.py` (`@pytest.mark.skipif` sem `mlflow.db` real **ou** sem `data/raw/application_train.csv`; nunca roda em CI). Consulta o MLflow local e afirma, para o `pipeline_run_id` mais recente: lotes t0/t1/t2 com `alert_level` `OK`/`ATENÇÃO`/`CRÍTICO`; cada métrica da tabela §4.3 presente em pelo menos um run; log anexado a cada run de estágio; run `BLOQUEADO` com `intentional_demo=true`.
- [ ] **Step 3: Makefile** conforme a seção Interfaces; atualizar `help`. Verificar no macOS (make 3.81) que `export` específico de alvo funciona; se não, usar `$(eval export ...)` ou um shell wrapper.
- [ ] **Step 4: Execução real** (exige o dataset): no worktree, ligar o CSV por link simbólico ignorado (`data/raw/application_train.csv` → checkout principal), `make pipeline`, depois `uv run pytest tests/integration/test_acceptance_real.py -v` e `make tracking-ui` (abrir e conferir a UI manualmente). `uv run dvc repro` (ou `make repro`) e `git status`: nada de `data/`, `mlflow.db`, `mlartifacts/`. **Se t0/t1/t2 não saírem `OK`/`ATENÇÃO`/`CRÍTICO`, parar e devolver ao usuário** (a regra é revista, não o dado — §6).
- [ ] **Step 5: Regenerar e conferir os artefatos versionados** (`models/`, `reports/`), incluindo `label_shift` nos relatórios de drift. Rodar `make lint && make test-cov`.
- [ ] **Step 6: Docs do núcleo:** CHANGELOG (Adicionado/Alterado); README (comandos §4.8, como baixar o dataset, `dvc repro`, remoto local); ADR-0006 → `Implementada` (usar `.agents/scripts/adr/validate_adrs.py` e `generate_adr_index.py`); atualizar o relatório de review e o checklist `docs/tech-challenge-fase4-checklist.md` (3.1 a 3.4).
- [ ] **Step 7: Commits** (atômicos): `feat(build): alvos make pipeline/monitor/demo-drift`, `feat(dvc): dvc.yaml e rastreio do dataset bruto`, `test: aceite real do núcleo da etapa 3`, `docs: núcleo da etapa 3 (changelog, readme, adr-0006)`.

**Ponto de parada A:** o núcleo só está pronto quando o critério de aceite 1 a 8 da spec passar. O usuário valida antes da Fase B.

---

# FASE B — Extensão 4.7: retreino, Registry e promoção

### Task 12: Regra de promoção (função pura)

**Files:** Create `src/monitoring/promotion.py`, `tests/test_monitoring_promotion.py`.

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool; reasons: tuple[str, ...]
    gain_on_holdout: float; regression_on_reference: float

def decide_promotion(*, contract_ok: bool, integrity_ok: bool,
                     champion_holdout_auc: float, challenger_holdout_auc: float,
                     champion_reference_auc: float, challenger_reference_auc: float,
                     min_gain: float = PROMOTION_MIN_GAIN,
                     max_regression: float = PROMOTION_MAX_REGRESSION) -> PromotionDecision
```
Motivos estáveis: `contract_failed`, `integrity_failed`, `gain_below_min`, `regression_above_max`.

- [ ] **Step 1: Testes** (um por ramo da §4.7): `test_elegivel` (ganho 0,03, regressão 0,005); `test_ganho_insuficiente` (0,019 ⇒ inelegível, `gain_below_min`; exatamente 0,02 ⇒ elegível); `test_regressao_na_referencia` (queda 0,021 ⇒ `regression_above_max`; 0,02 ⇒ elegível); `test_contrato_falhando`; `test_integridade_falhando`; `test_varios_motivos_acumulam`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar.** **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): regra pura de promoção champion/challenger`.

---

### Task 13: Registry — challenger, consumo de alerta e auditoria

**Files:** Modify `src/monitoring/registry.py`; Test `tests/test_monitoring_registry.py`.

**Interfaces — Produces:**
```python
def register_challenger(*, run_id: str, ..., tags: dict[str, str]) -> int          # alias "challenger" (move se já existir)
def find_unconsumed_critical_run() -> Run | None                                    # run de lote com alert_level=CRÍTICO e sem tag retrain_consumed
def mark_alert_consumed(batch_run_id: str, version: int) -> None
def log_audit_run(*, action: str, version: int, reason: str, operator: str, details: dict) -> str
def promote_alias(version: int) -> None        # champion←version; previous←champion antigo
def rollback_alias() -> int                    # champion←previous; devolve a versão que voltou
```
`operator` vem de `getpass.getuser()`; a hora, do relógio UTC.

- [ ] **Step 1: Testes.** `test_register_challenger_move_alias_e_grava_tags`; `test_find_unconsumed_critical_ignora_consumidos_e_nao_criticos` (filtro de `retrain_consumed` no cliente, não no `search_runs`); `test_promote_move_aliases_e_define_previous`; `test_rollback_devolve_previous_e_falha_sem_previous`; `test_audit_run_registra_quem_quando_por_que`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar.** **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): aliases, consumo de alerta e auditoria no Registry`.

---

### Task 14: Retreino (`make retrain`)

**Files:** Create `src/monitoring/retrain.py`; Test `tests/test_monitoring_retrain.py`.

**Interfaces — Produces:** `main(argv=None) -> int` com `--alert-run ID`, `--force-no-alert` (exige `--reason`), `--allow-reuse` (alerta já consumido; exige `--reason`). `split_critical_batch(df, seed=SEED) -> tuple[pd.DataFrame, pd.DataFrame]` (treino, holdout; estratificado por `target`, 50/50) e `build_challenger_training(reference, critical_train) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`, que chama `split_reference(reference, SEED)` **primeiro** e devolve `(train_ref + critical_train, val_ref, test_ref)`: validação e teste são exatamente os do champion. O comando: acha o alerta, carrega referência + t2, valida contrato (`enforce_contract`) e integridade, treina o challenger com `build_pipeline` e os mesmos hiperparâmetros (`XGB_PARAMS`), escolhendo o *threshold* na validação da referência, avalia **champion e challenger no mesmo holdout** e no split de teste da referência, aplica `decide_promotion`, registra o challenger (alias `challenger`) com tags (run de origem, `reference_sha256`, `model_sha256`, lote, alerta, `origin=retrain`, `bundle_model_version`), marca o alerta como consumido, imprime o veredito e **nunca** move `champion`.

- [ ] **Step 1: Testes.**
  - `test_split_estratificado_e_sem_vazamento`: interseção de `client_id` vazia; proporção de `target` igual em ambos; determinístico com a seed.
  - `test_challenger_nao_treina_com_linhas_do_teste_da_referencia`: nenhum `client_id` do `test_ref` aparece no treino do challenger, e `val_ref`/`test_ref` são iguais aos de `split_reference(reference, SEED)`.
  - `test_sem_alerta_critico_recusa` (Review Focus 5): sai ≠ 0 com mensagem; com `--force-no-alert --reason "..."` segue e grava o motivo no run.
  - `test_alerta_consumido_recusa_sem_flag`; `test_alert_run_de_lote_nao_critico_recusa`.
  - `test_retrain_nao_move_champion`: depois de rodar, `get_alias_version("champion")` é a mesma versão de antes.
  - `test_retrain_marca_alerta_consumido_e_registra_tags`.
  - `test_retrain_falha_fechada_com_tracking_off`.
  - `test_retrain_avalia_champion_e_challenger_no_mesmo_holdout`: o AUC do champion registrado é o recalculado no holdout (não o do lote inteiro).
  - `test_retrain_contrato_falhando_aborta_antes_de_treinar`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar.** **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): retreino do challenger guardado por alerta crítico`.

---

### Task 15: Promoção e rollback humanos + `load_bundle` conferindo o champion

**Files:** Create `src/monitoring/promote.py`; Modify `src/monitoring/stage.py`/`monitor.py` (passam `expected_sha256`); Test `tests/test_monitoring_promote.py`.

**Interfaces — Produces:** `promote.main(argv)` com `--version N` (padrão: versão do alias `challenger`), `--override` (exige `--reason`), `--reason`; `rollback.main(argv)` em `promote.py` (`python -m src.monitoring.promote rollback`). A promoção segue a ordem: `export_champion` para arquivo temporário e conferência do sha256 → mover `previous` → mover `champion` → `os.replace` para `models/champion/` → reconferir (alias e arquivo) e gravar esse estado no run de auditoria (`status=CONSISTENT|INCONSISTENT`); divergência encerra com código ≠ 0.

- [ ] **Step 1: Testes.**
  - `test_promote_reavalia_e_recusa_inelegivel`: challenger com ganho insuficiente ⇒ exit ≠ 0, `champion` inalterado, nenhum arquivo de `models/` tocado.
  - `test_promote_override_exige_motivo_e_audita`: sem `--reason` ⇒ erro; com ele, promove e o run de auditoria guarda o motivo.
  - `test_promote_elegivel_move_aliases_e_exporta_modelo`: `champion` = v2, `previous` = v1, `models/champion/baseline_model.joblib` tem o sha256 da tag `model_sha256`; `models/baseline_model.joblib` fica intacto.
  - `test_promote_falha_no_meio_registra_inconsistente`: falha simulada depois de mover `previous` ⇒ exit ≠ 0, run de auditoria `status=INCONSISTENT`, e `load_bundle` com `expected_sha256` recusa.
  - `test_export_atomico_nao_deixa_arquivo_parcial`: falha simulada no meio da cópia ⇒ `models/` mantém o conteúdo anterior.
  - `test_rollback_restaura_arquivo_e_alias` e `test_rollback_sem_previous_falha`.
  - `test_load_bundle_com_champion_divergente_falha` (integra a Tarefa 7): arquivo de `models/champion/` de uma versão e `champion` do Registry de outra ⇒ `Modelo recusado`.
  - `test_promote_e_rollback_geram_run_de_auditoria`.
- [ ] **Step 2:** FALHA. **Step 3: Implementar.** **Step 4:** PASSA. **Step 5: Commit** `feat(monitoring): comandos humanos de promoção e rollback com auditoria`.

---

### Task 16: Makefile, ADR-0007 e documentação da extensão

**Files:** Modify `Makefile`, `README.md`, `CHANGELOG.md`; Create `docs/adrs/0007-retreino-registry-e-promocao-humana.md`.

- [ ] **Step 1:** alvos `retrain`, `promote`, `rollback` (passam `ARGS=`), `help` atualizado.
- [ ] **Step 2: ADR-0007** (usar o template de `docs/adrs/`): gatilho só por `CRITICO`, dados de retreino sem vazamento, aliases, regra de promoção com os **limiares pré-registrados** (`PROMOTION_MIN_GAIN=0.02`, `PROMOTION_MAX_REGRESSION=0.02`, escritos antes do retreino real), promoção humana, `models/champion/` fora do DVC, `train` repetido sem mover o champion. Emendar também a ADR-0005 (item 4) se ainda não feito na Tarefa 9. Rodar `validate_adrs.py` e `generate_adr_index.py`.
- [ ] **Step 3:** CHANGELOG, README. **Step 4:** `make lint && make test-cov`. **Step 5: Commit** `docs(adr): ADR-0007 e documentação do ciclo de retreino e promoção`.

---

### Task 17: Aceite real da extensão e calibração dos limiares

**Files:** Modify `tests/integration/test_acceptance_real.py`; Create relatório em `docs/ai/reports/`. `src/config.py` **não** muda nesta tarefa: os limiares foram pré-registrados na ADR-0007 antes do retreino real.

- [ ] **Step 1:** `make retrain` com o t2 real (alerta `CRÍTICO` do `pipeline_run_id` mais recente). Registrar **os números medidos**: AUC do champion e do challenger no holdout e na referência, ganho, regressão, veredito.
- [ ] **Step 2:** Se **elegível**: `make promote`, rodar `make demo-drift` de novo e confirmar que o `drift-demo` usa o novo champion; `make rollback` e conferir o retorno; depois repromover para deixar o repositório no estado final escolhido pelo usuário. Se **inelegível**: registrar o veredito como está, **não** usar `--override` e **parar para o usuário decidir** (aceitar o resultado, ou abrir uma nova decisão registrada em ADR para mudar `PROMOTION_*`, com o histórico do que foi medido) (§4.7). Os ramos de promoção e rollback continuam provados pelos testes das Tarefas 12 e 15.
- [ ] **Step 3:** estender o teste de aceite real (campos de Registry: `champion`/`challenger`/`previous`, tags, runs de auditoria). Escrever o relatório `docs/ai/reports/2026-10-xx-etapa3-aceite.md` com evidências (IDs de run, números, comandos).
- [ ] **Step 4:** ADR-0007 e ADR-0006 → `Implementada`; checklist atualizado; `graphify update .`.
- [ ] **Step 5: Commit** `docs(ai): relatório de aceite da etapa 3`.

---

## Cobertura da spec

| Spec | Tarefa |
| :--- | :--- |
| §4.1 runs (estágio, lote, treino) | 5, 6, 7, 8 |
| §4.2 níveis e PSI do score | 2, 3, 7 |
| §4.3 métricas de saúde | 4, 5, 7 (aceite real: 11) |
| §4.4 logs centralizados | 4, 5 |
| §4.5 DVC | 1 (higiene), 11 |
| §4.6 `label_shift` | 9 |
| §4.7 retreino/Registry/promoção | 6 (bootstrap), 12 a 17 |
| §4.8 comandos | 10, 11, 16 |
| §3 falhar fechado, camadas | 1, 5, 6 |
| §8 critérios 1 a 9 | 11 e 16 |
| §8 critério 10 | 14, 15, 17 |

## Pendências para o revisor e o usuário

1. **Bloqueio intencional e `alerts_failure_count`:** aplicado literalmente como na spec §4.1 (conta; `stage_success=1`; tag `intentional_demo=true`). Se o usuário preferir que a demonstração não conte como falha, é mudança de spec.
2. **Decisões 4 e 5** mudam o texto da spec (já atualizada): `train` repetido não move o `champion`, e o champion servido fica em `models/champion/`.
3. **Versões de MLflow e DVC** são sugestões; o construtor confirma a API de aliases e o backend SQLite na versão que o `uv` resolver.
4. **CI:** o workflow roda `uv sync --locked`; `mlflow` e `dvc` aumentam o tempo, e o CI nunca rodou no GitHub (item adiado do review).
