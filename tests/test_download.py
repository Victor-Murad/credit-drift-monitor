"""Download do Kaggle: caminhos de sucesso e falha, sem rede."""

from __future__ import annotations

import subprocess
import zipfile

import pytest

from src.data import download as dl


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(dl, "RAW_DIR", raw)
    monkeypatch.setattr(dl, "RAW_FILE", raw / dl.RAW_FILE_NAME)
    return raw


def test_download_skips_when_file_exists(raw_dir, monkeypatch):
    raw_dir.mkdir()
    (raw_dir / dl.RAW_FILE_NAME).write_text("x")
    monkeypatch.setattr(dl.shutil, "which", lambda _: pytest.fail("nao deveria rodar"))
    dl.download()


def test_download_requires_kaggle_cli(raw_dir, monkeypatch):
    monkeypatch.setattr(dl.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="CLI do Kaggle"):
        dl.download()


def test_download_exits_on_kaggle_failure(raw_dir, monkeypatch):
    monkeypatch.setattr(dl.shutil, "which", lambda _: "/bin/kaggle")
    monkeypatch.setattr(
        dl.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "403 Forbidden"),
    )
    with pytest.raises(SystemExit, match="403 Forbidden"):
        dl.download()


def test_download_extracts_zip(raw_dir, monkeypatch):
    def fake_run(cmd, **kwargs):
        with zipfile.ZipFile(raw_dir / f"{dl.RAW_FILE_NAME}.zip", "w") as zf:
            zf.writestr(dl.RAW_FILE_NAME, "SK_ID_CURR\n1\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(dl.shutil, "which", lambda _: "/bin/kaggle")
    monkeypatch.setattr(dl.subprocess, "run", fake_run)

    dl.download()

    assert (raw_dir / dl.RAW_FILE_NAME).read_text() == "SK_ID_CURR\n1\n"
    assert not (raw_dir / f"{dl.RAW_FILE_NAME}.zip").exists()
