"""Download do dataset bruto (Home Credit Default Risk / Kaggle).

Pre-requisitos (uma vez por maquina):
  1. Aceitar as regras da competicao em
     https://www.kaggle.com/competitions/home-credit-default-risk/rules
  2. Gerar o token em Kaggle > Settings > API > "Create New Token" e salvar
     o valor em ~/.kaggle/access_token
     (Windows: C:\\Users\\<usuario>\\.kaggle\\access_token) OU exporta-lo na
     variavel de ambiente KAGGLE_API_TOKEN. Credenciais legadas
     (~/.kaggle/kaggle.json) tambem funcionam.

Alternativa sem API: baixar manualmente `application_train.csv` na aba Data
da competicao e colocar em data/raw/.

    uv run python -m src.data.download
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile

from src.config import KAGGLE_COMPETITION, RAW_DIR, RAW_FILE, RAW_FILE_NAME
from src.logging_utils import get_logger, log_event

logger = get_logger("data.download")


def download() -> None:
    if RAW_FILE.exists():
        log_event(logger, "raw_download", status="SKIPPED", reason="arquivo ja existe")
        return

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        sys.exit(
            "CLI do Kaggle nao encontrada. Rode `uv sync` ou baixe o arquivo "
            f"manualmente para {RAW_FILE}."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        kaggle,
        "competitions",
        "download",
        "-c",
        KAGGLE_COMPETITION,
        "-f",
        RAW_FILE_NAME,
        "-p",
        str(RAW_DIR),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        log_event(logger, "raw_download", status="FAILED", stderr=completed.stderr)
        sys.exit(
            "Falha no download pelo Kaggle. Verifique se o token esta em "
            "~/.kaggle/access_token (ou na variavel KAGGLE_API_TOKEN) e se as "
            "regras da competicao foram aceitas.\n" + completed.stderr
        )

    # O Kaggle entrega `application_train.csv.zip` (ou o CSV direto).
    archive = RAW_DIR / f"{RAW_FILE_NAME}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extract(RAW_FILE_NAME, RAW_DIR)
        archive.unlink()

    log_event(
        logger,
        "raw_download",
        status="SUCCESS",
        file=RAW_FILE.name,
        size_mb=round(RAW_FILE.stat().st_size / 1e6, 1),
    )


if __name__ == "__main__":
    download()
