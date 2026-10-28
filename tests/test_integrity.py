"""Testes da verificacao de integridade do artefato do modelo."""

from __future__ import annotations

import json

import joblib
import pytest

from src.models.integrity import (
    ModelIntegrityError,
    file_sha256,
    verify_model_integrity,
)
from src.models.predict import load_bundle


def _write_model(tmp_path, *, sha: str | None = None):
    model = tmp_path / "baseline_model.joblib"
    joblib.dump({"model_version": "t"}, model)
    metadata = tmp_path / "baseline_metadata.json"
    metadata.write_text(
        json.dumps({"artifact": {"sha256": sha or file_sha256(model)}}),
        encoding="utf-8",
    )
    return model, metadata


def test_verify_accepts_matching_hash(tmp_path) -> None:
    model, _ = _write_model(tmp_path)

    verify_model_integrity(model)


def test_verify_rejects_tampered_artifact(tmp_path) -> None:
    model, _ = _write_model(tmp_path)
    model.write_bytes(model.read_bytes() + b"x")

    with pytest.raises(ModelIntegrityError, match="diverge"):
        verify_model_integrity(model)


def test_verify_fails_closed_without_metadata(tmp_path) -> None:
    model, metadata = _write_model(tmp_path)
    metadata.unlink()

    with pytest.raises(ModelIntegrityError, match="ausente"):
        verify_model_integrity(model)


def test_verify_rejects_metadata_without_hash(tmp_path) -> None:
    model, metadata = _write_model(tmp_path)
    metadata.write_text("{}", encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="artifact.sha256"):
        verify_model_integrity(model)


def test_load_bundle_loads_verified_model(tmp_path) -> None:
    model, _ = _write_model(tmp_path)

    assert load_bundle(model)["model_version"] == "t"


def test_load_bundle_refuses_tampered_model(tmp_path) -> None:
    model, _ = _write_model(tmp_path, sha="0" * 64)

    with pytest.raises(SystemExit, match="Modelo recusado"):
        load_bundle(model)
