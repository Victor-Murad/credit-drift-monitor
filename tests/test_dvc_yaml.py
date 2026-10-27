"""Forma do `dvc.yaml` (spec da Etapa 3, §4.5): estágios, `deps` e `outs`.

O DVC apaga os `outs` antes de rodar um estágio e rejeita `outs` sobrepostos;
`reports/` e `models/` continuam no git (`cache: false`) e o champion servido
(`models/champion/`) nunca é `out`, porque promoção e rollback o reescrevem.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DVC_YAML = ROOT / "dvc.yaml"
ESTAGIOS = ["prepare", "train", "contract_demo", "drift_demo"]
RAW_CSV = "data/raw/application_train.csv"
CHAMPION_DIR = PurePosixPath("models/champion")


@pytest.fixture(scope="module")
def estagios() -> dict[str, dict]:
    return yaml.safe_load(DVC_YAML.read_text(encoding="utf-8"))["stages"]


def _entradas(itens: list) -> list[tuple[str, dict]]:
    """`deps`/`outs` normalizados em `(caminho, opções)`."""
    normalizados = []
    for item in itens:
        if isinstance(item, str):
            normalizados.append((item, {}))
        else:
            ((caminho, opcoes),) = item.items()
            normalizados.append((caminho, opcoes or {}))
    return normalizados


def _outs(estagios: dict[str, dict]) -> list[tuple[str, str, dict]]:
    return [
        (nome, caminho, opcoes)
        for nome, estagio in estagios.items()
        for caminho, opcoes in _entradas(estagio.get("outs", []))
    ]


def _contem(a: str, b: str) -> bool:
    """`a` é igual a `b` ou um diretório-pai dele (comparação por componentes)."""
    pa, pb = PurePosixPath(a), PurePosixPath(b)
    return pa == pb or pa in pb.parents


def test_estagios_na_ordem(estagios):
    assert list(estagios) == ESTAGIOS


def test_cmd_chama_o_modulo_de_estagio(estagios):
    for nome, estagio in estagios.items():
        assert estagio["cmd"] == f"uv run python -m src.monitoring.stage {nome}"


def test_outs_nao_se_sobrepoem(estagios):
    outs = _outs(estagios)
    assert outs
    sobrepostos = [
        (a, b)
        for (_, a, _), (_, b, _) in combinations(outs, 2)
        if _contem(a, b) or _contem(b, a)
    ]
    assert not sobrepostos, sobrepostos


def test_outs_de_reports_e_models_sem_cache(estagios):
    versionados = [
        (nome, caminho, opcoes)
        for nome, caminho, opcoes in _outs(estagios)
        if PurePosixPath(caminho).parts[0] in {"reports", "models"}
    ]
    assert versionados
    com_cache = [(n, c) for n, c, o in versionados if o.get("cache", True)]
    assert not com_cache, com_cache


def test_raw_csv_e_dep_do_prepare(estagios):
    deps = [caminho for caminho, _ in _entradas(estagios["prepare"]["deps"])]
    assert RAW_CSV in deps


def test_champion_dir_fora_dos_outs(estagios):
    invasores = [
        (nome, caminho)
        for nome, caminho, _ in _outs(estagios)
        if _contem(caminho, CHAMPION_DIR.as_posix())
        or _contem(CHAMPION_DIR.as_posix(), caminho)
    ]
    assert not invasores, invasores
