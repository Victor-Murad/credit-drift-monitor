"""Higiene do repositorio: dados e artefatos locais fora do git, camadas limpas."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def requires_git() -> None:
    try:
        _git("rev-parse", "--is-inside-work-tree")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("nao e um repositorio git")


def test_nada_de_data_no_indice_do_git(requires_git):
    tracked = _git("ls-files", "data/").stdout.split()
    assert all(Path(p).name == ".gitkeep" for p in tracked), tracked


def test_artefatos_do_mlflow_ignorados(requires_git):
    for nome in ("mlflow.db", "mlartifacts", "mlruns"):
        resultado = _git("check-ignore", nome, check=False)
        assert resultado.returncode == 0, f"{nome} nao esta no .gitignore"


def test_mlflow_so_na_camada_de_monitoring():
    permitidos = (ROOT / "src" / "monitoring", ROOT / "tests")
    infratores = []
    for caminho in ROOT.rglob("*.py"):
        partes = caminho.relative_to(ROOT).parts
        if partes[0] in {".venv", ".claude", "node_modules", "graphify-out"}:
            continue
        if any(caminho.is_relative_to(p) for p in permitidos):
            continue
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            texto = linha.strip()
            if texto.startswith(("import mlflow", "from mlflow")):
                infratores.append(caminho.relative_to(ROOT).as_posix())
                break
    assert not infratores, f"mlflow importado fora de src/monitoring: {infratores}"
