"""Alvos do Makefile da Etapa 3 (spec §4.8), exercitados com estágios de mentira.

`STAGE` e `PYTHON` são sobrescritos por um script que só imprime o que recebeu:
nada do pipeline real roda e nenhum `mlflow.db` é criado.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ESTAGIOS = ["prepare", "train", "contract_demo", "drift_demo"]

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None or shutil.which("uv") is None,
    reason="make e uv são necessários para exercitar o Makefile",
)

STUB = """\
import json, os, sys
if "--batch" in sys.argv:
    sys.exit(int(sys.argv[sys.argv.index("--batch") + 1]))
if "--exit" in sys.argv:
    print(json.dumps(sys.argv[1:]))
    sys.exit(int(sys.argv[sys.argv.index("--exit") + 1]))
print(f"{sys.argv[1]} {os.environ.get('PIPELINE_RUN_ID')}")
"""
# Alvos da extensão §4.7 (retreino e promoção) e o módulo que cada um chama.
CICLO = {
    "retrain": ["-m", "src.monitoring.retrain"],
    "promote": ["-m", "src.monitoring.promote"],
    "rollback": ["-m", "src.monitoring.promote", "rollback"],
}


@pytest.fixture
def stub(tmp_path) -> str:
    caminho = tmp_path / "stub.py"
    caminho.write_text(STUB, encoding="utf-8")
    return f"{sys.executable} {caminho}"


def _make(*args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PIPELINE_RUN_ID"}
    return subprocess.run(
        ["make", "--no-print-directory", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_pipeline_encadeia_os_estagios_com_um_unico_run_id(stub):
    resultado = _make("-s", "pipeline", f"STAGE={stub}")

    assert resultado.returncode == 0, resultado.stderr
    linhas = resultado.stdout.splitlines()
    cabecalho, *estagios = linhas
    run_id = cabecalho.removeprefix("PIPELINE_RUN_ID=")
    assert re.fullmatch(r"\d{8}T\d{6}Z", run_id), cabecalho
    assert estagios == [f"{nome} {run_id}" for nome in ESTAGIOS]


def test_pipeline_para_no_primeiro_estagio_que_falha(tmp_path):
    falho = tmp_path / "falho.py"
    falho.write_text(
        "import sys\nprint(sys.argv[1])\nsys.exit(sys.argv[1] == 'train')\n",
        encoding="utf-8",
    )
    resultado = _make("-s", "pipeline", f"STAGE={sys.executable} {falho}")

    assert resultado.returncode != 0
    assert resultado.stdout.splitlines()[1:] == ["prepare", "train"]


def test_alvos_de_estagio_chamam_o_modulo_de_estagio():
    resultado = _make("-n", "data", "train", "contract-demo", "demo-drift")
    comandos = resultado.stdout.splitlines()
    assert comandos == [
        f"uv run python -m src.monitoring.stage {nome}" for nome in ESTAGIOS
    ]


@pytest.mark.parametrize("apelido", ["drift-demo", "etapa2"])
def test_apelidos_da_demo_de_drift(apelido):
    resultado = _make("-n", apelido)
    assert resultado.stdout.splitlines() == [
        "uv run python -m src.monitoring.stage drift_demo"
    ]


def test_etapa1_e_apelido_dos_tres_primeiros_estagios():
    resultado = _make("-n", "etapa1")
    assert resultado.stdout.splitlines() == [
        f"uv run python -m src.monitoring.stage {nome}" for nome in ESTAGIOS[:3]
    ]


def test_monitor_sem_batch_falha_com_mensagem():
    resultado = _make("monitor")
    assert resultado.returncode != 0
    assert "make monitor BATCH=" in resultado.stderr


@pytest.mark.parametrize("codigo", [2, 3, 1])
def test_monitor_repassa_o_codigo_de_saida(stub, codigo):
    resultado = _make("monitor", f"BATCH={codigo}", f"PYTHON={stub}")
    # O make sai com 2 em qualquer falha; o código do monitor vem na mensagem.
    assert resultado.returncode == 2
    assert f"Error {codigo}" in resultado.stderr
    assert f"codigo de saida {codigo}" in resultado.stderr


def test_monitor_ok_sai_com_zero_e_repassa_baseline(stub):
    assert _make("monitor", "BATCH=0", f"PYTHON={stub}").returncode == 0
    seco = _make("-n", "monitor", "BATCH=lote.parquet", "BASELINE_AUC=0.75")
    assert (
        'python -m src.monitoring.monitor --batch "lote.parquet" --baseline-auc 0.75'
        in seco.stdout
    )


def test_tracking_ui_na_porta_5001_com_uri_do_config():
    seco = _make("-n", "tracking-ui").stdout
    assert "config.MLFLOW_TRACKING_URI" in seco
    assert 'mlflow ui --backend-store-uri "$uri" --port 5001' in seco


def test_repro_chama_o_dvc():
    assert _make("-n", "repro").stdout.strip().endswith("uv run dvc repro")


@pytest.mark.parametrize("alvo", sorted(CICLO))
def test_ciclo_sem_args_chama_o_modulo(alvo):
    seco = _make("-n", alvo).stdout
    assert f"uv run python {' '.join(CICLO[alvo])};" in seco.replace(" ;", ";")


@pytest.mark.parametrize("alvo", sorted(CICLO))
def test_ciclo_repassa_args_com_aspas(stub, alvo):
    resultado = _make(
        alvo, f"PYTHON={stub}", "ARGS=--reason 'motivo com espaços' --exit 0"
    )
    assert resultado.returncode == 0, resultado.stderr
    recebido = json.loads(resultado.stdout.splitlines()[-1])
    assert recebido == [
        *CICLO[alvo],
        "--reason",
        "motivo com espaços",
        "--exit",
        "0",
    ]


@pytest.mark.parametrize(
    ("alvo", "codigo"),
    [
        ("retrain", 2),
        ("retrain", 3),
        ("retrain", 1),
        ("promote", 2),
        ("promote", 1),
        ("rollback", 1),
    ],
)
def test_ciclo_mostra_o_codigo_de_saida(stub, alvo, codigo):
    resultado = _make(alvo, f"PYTHON={stub}", f"ARGS=--exit {codigo}")
    # Como no monitor: o make sai com 2 em qualquer falha; o código vem no texto.
    assert resultado.returncode == 2
    assert f"Error {codigo}" in resultado.stderr
    assert f"{alvo}: codigo de saida {codigo}" in resultado.stderr


def test_help_e_phony_listam_o_ciclo():
    ajuda = _make("help").stdout
    phony = next(
        linha
        for linha in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
        if linha.startswith(".PHONY:")
    ).split()
    for alvo in CICLO:
        assert f"make {alvo} [ARGS=" in ajuda
        assert alvo in phony
