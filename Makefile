.PHONY: help setup download-data pipeline data train contract-demo demo-drift drift-demo monitor retrain promote rollback tracking-ui repro validate predict etapa1 simulate-drift detect-drift etapa2 test test-cov lint format clean

# uv e obrigatorio (mesmo padrao da Fase 3): garante que todo o grupo rode
# no mesmo ambiente, travado pelo uv.lock.
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(strip $(UV)),)
  $(error uv nao encontrado no PATH. Instale: https://docs.astral.sh/uv/getting-started/installation/)
endif
PYTHON := uv run python
# Estagios rastreados no MLflow (spec da Etapa 3, §4.1): o mesmo modulo que o
# dvc.yaml chama. `MONITORING_TRACKING=off` desliga o rastreamento.
STAGE := $(PYTHON) -m src.monitoring.stage
# 5001: a 5000 colide com o AirPlay no macOS.
MLFLOW_UI_PORT := 5001

# Receita que roda o comando $(2) e repassa o codigo de saida dele. O make so sai
# com 0 ou 2: o codigo exato (0/2/3/1) vai na mensagem de erro do make ("Error 3")
# e na linha "$(1): codigo de saida N (...)". A legenda dos codigos vem de
# CODIGOS, definida no alvo (uma virgula num argumento do $(call) o partiria).
com_codigo = @$(2); \
	codigo=$$?; \
	if [ $$codigo -ne 0 ]; then echo "$(1): codigo de saida $$codigo ($(CODIGOS))" >&2; fi; \
	exit $$codigo

help:
	@echo "credit-drift-monitor - Tech Challenge Fase 4"
	@echo "--------------------------------------------"
	@echo "make setup          - Instala o ambiente (uv sync, Python 3.12)"
	@echo "make download-data  - Baixa application_train.csv do Kaggle (exige token do Kaggle)"
	@echo "make pipeline       - Etapa 3: prepare -> train -> contract-demo -> demo-drift no MLflow,"
	@echo "                      com um unico PIPELINE_RUN_ID (sempre executa todos os estagios)"
	@echo "make monitor BATCH=<arquivo> [BASELINE_AUC=<auc>] - Avalia um lote e registra o alerta"
	@echo "                      (codigo do monitor: 0 OK/ATENCAO, 2 CRITICO, 3 BLOQUEADO, 1 erro)"
	@echo "make retrain [ARGS='--alert-run <id>'] - Treina o challenger a partir de um alerta CRITICO"
	@echo "                      e o registra no Model Registry; nunca move o champion"
	@echo "                      (0 registrado, 2 recusa sem alerta, 3 contrato barrou, 1 erro)"
	@echo "make promote [ARGS='--override --reason <motivo>'] - Promove o challenger a champion"
	@echo "                      (comando humano: reavalia a regra; 0 promovido, 2 inelegivel, 1 erro)"
	@echo "make rollback [ARGS='--reason <motivo>'] - Devolve o champion a versao previous (0 feito, 1 erro)"
	@echo "make tracking-ui    - Abre o MLflow UI em http://127.0.0.1:$(MLFLOW_UI_PORT) (banco local)"
	@echo "make repro          - dvc repro: so os estagios com mudanca (sem PIPELINE_RUN_ID comum)"
	@echo "make data           - Estagio prepare: Dataset de Referencia + lote de Producao t0"
	@echo "make train          - Estagio train: baseline (gate do contrato antes) + Model Registry"
	@echo "make contract-demo  - Estagio contract_demo: o contrato bloqueando um lote invalido"
	@echo "make demo-drift     - Estagio drift_demo: simula t1/t2 e avalia t0, t1 e t2 (apelido: drift-demo)"
	@echo "make validate INPUT=<arquivo> [LABELED=1] - Valida um lote (exit 1 se bloqueado)"
	@echo "make predict INPUT=<arquivo>  - Pontua um lote (somente se passar no contrato)"
	@echo "make etapa1         - Apelido: data -> train -> contract-demo"
	@echo "make simulate-drift - Gera fatias t1 (data) e t2 (concept) a partir de production_base"
	@echo "make detect-drift   - Evidently: referencia vs CURRENT (padrao: t1)"
	@echo "make etapa2         - Apelido: demo-drift"
	@echo "make test           - Testes (pytest)"
	@echo "make lint / format  - Ruff"

setup:
	uv sync

download-data:
	$(PYTHON) -m src.data.download

# Um unico PIPELINE_RUN_ID por execucao (export especifico do alvo, avaliado uma
# vez): todos os estagios o recebem como tag `pipeline_run_id` e no log JSONL.
# Encadeado direto (nao via `dvc repro`, que pula estagios sem mudanca); para no
# primeiro estagio que falhar.
pipeline: export PIPELINE_RUN_ID := $(shell date -u +%Y%m%dT%H%M%SZ)
pipeline:
	@echo "PIPELINE_RUN_ID=$(PIPELINE_RUN_ID)"
	$(STAGE) prepare
	$(STAGE) train
	$(STAGE) contract_demo
	$(STAGE) drift_demo

data:
	$(STAGE) prepare

train:
	$(STAGE) train

contract-demo:
	$(STAGE) contract_demo

demo-drift:
	$(STAGE) drift_demo

drift-demo: demo-drift

monitor: CODIGOS = 2 CRITICO, 3 BLOQUEADO, 1 erro operacional
monitor:
	$(if $(strip $(BATCH)),,$(error Informe o lote: make monitor BATCH=<arquivo .parquet ou .csv>))
	$(call com_codigo,monitor,$(PYTHON) -m src.monitoring.monitor --batch "$(BATCH)" $(if $(BASELINE_AUC),--baseline-auc $(BASELINE_AUC),))

# Extensao da spec, §4.7: retreino, promocao e rollback. ARGS vai sem aspas para o
# shell, entao aspas dentro dele valem: ARGS="--override --reason 'motivo'".
# So o retrain e automatizavel; promote e rollback sao decisoes humanas.
retrain: CODIGOS = 2 recusa sem alerta CRITICO, 3 contrato barrou, 1 erro operacional
retrain:
	$(call com_codigo,retrain,$(PYTHON) -m src.monitoring.retrain $(ARGS))

promote: CODIGOS = 2 recusa: challenger inelegivel, 1 erro operacional
promote:
	$(call com_codigo,promote,$(PYTHON) -m src.monitoring.promote $(ARGS))

rollback: CODIGOS = 1 erro operacional, por exemplo sem previous
rollback:
	$(call com_codigo,rollback,$(PYTHON) -m src.monitoring.promote rollback $(ARGS))

# URI absoluto do banco local derivado de src.config (nunca escrito aqui).
tracking-ui:
	@uri="$$($(PYTHON) -c 'from src import config; print(config.MLFLOW_TRACKING_URI)')" || exit 1; \
	echo "MLflow UI em http://127.0.0.1:$(MLFLOW_UI_PORT) (Ctrl+C para sair)"; \
	exec $(UV) run mlflow ui --backend-store-uri "$$uri" --port $(MLFLOW_UI_PORT)

repro:
	$(UV) run dvc repro

validate:
	$(PYTHON) -m src.validation.validate --input $(INPUT) $(if $(LABELED),--labeled,)

predict:
	$(PYTHON) -m src.models.predict --input $(INPUT)

etapa1: data train contract-demo

simulate-drift:
	$(PYTHON) -m src.drift.simulate

detect-drift:
	$(PYTHON) -m src.drift.detect --current $(or $(CURRENT),data/production/production_t1_data_drift.parquet)

etapa2: demo-drift

test:
	$(PYTHON) -m pytest

test-cov:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/
	$(UV) run lint-imports

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ai-guides: memória dos agentes (guia-construcao-harness §5)
pull-memory: ## Sincronizar memória global para o repositório local
	@./.claude/sync-claude-memory.sh pull; ./.gemini/sync-memory.sh pull

push-memory: ## Sincronizar memória local para o sistema global
	@./.claude/sync-claude-memory.sh push; ./.gemini/sync-memory.sh push

setup-gemini-hook: ## Instalar pre-commit de auto-staging de memórias
	@cp ".agents/ai-guides/templates/pre-commit-memory.sh" "$$(git rev-parse --git-path hooks)/pre-commit" && chmod +x "$$(git rev-parse --git-path hooks)/pre-commit"
