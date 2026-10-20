.PHONY: help setup download-data data train contract-demo validate predict etapa1 simulate-drift detect-drift drift-demo etapa2 test test-cov lint format clean

# uv e obrigatorio (mesmo padrao da Fase 3): garante que todo o grupo rode
# no mesmo ambiente, travado pelo uv.lock.
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(strip $(UV)),)
  $(error uv nao encontrado no PATH. Instale: https://docs.astral.sh/uv/getting-started/installation/)
endif
PYTHON := uv run python

help:
	@echo "credit-drift-monitor - Tech Challenge Fase 4"
	@echo "--------------------------------------------"
	@echo "make setup          - Instala o ambiente (uv sync, Python 3.12)"
	@echo "make download-data  - Baixa application_train.csv do Kaggle (exige token do Kaggle)"
	@echo "make data           - Gera Dataset de Referencia + lote de Producao t0"
	@echo "make train          - Treina o baseline (gate do contrato antes do treino)"
	@echo "make contract-demo  - Demonstra o contrato bloqueando um lote invalido"
	@echo "make validate INPUT=<arquivo> [LABELED=1] - Valida um lote (exit 1 se bloqueado)"
	@echo "make predict INPUT=<arquivo>  - Pontua um lote (somente se passar no contrato)"
	@echo "make etapa1         - Executa a Etapa 1 completa (data -> train -> contract-demo)"
	@echo "make simulate-drift - Gera fatias t1 (data) e t2 (concept) a partir de production_base"
	@echo "make detect-drift   - Evidently: referencia vs CURRENT (padrao: t1)"
	@echo "make drift-demo     - Etapa 2: simular + pontuar + detectar + sumario"
	@echo "make etapa2         - Executa a Etapa 2 completa (drift-demo)"
	@echo "make test           - Testes (pytest)"
	@echo "make lint / format  - Ruff"

setup:
	uv sync

download-data:
	$(PYTHON) -m src.data.download

data:
	$(PYTHON) -m src.data.prepare

train:
	$(PYTHON) -m src.models.train

contract-demo:
	$(PYTHON) -m src.validation.demo

validate:
	$(PYTHON) -m src.validation.validate --input $(INPUT) $(if $(LABELED),--labeled,)

predict:
	$(PYTHON) -m src.models.predict --input $(INPUT)

etapa1: data train contract-demo

simulate-drift:
	$(PYTHON) -m src.drift.simulate

detect-drift:
	$(PYTHON) -m src.drift.detect --current $(or $(CURRENT),data/production/production_t1_data_drift.parquet)

drift-demo:
	$(PYTHON) -m src.drift.demo

etapa2: drift-demo

test:
	$(PYTHON) -m pytest

test-cov:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
