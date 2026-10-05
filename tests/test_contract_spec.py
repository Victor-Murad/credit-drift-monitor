"""A especificacao YAML e o schema Pandera precisam descrever o mesmo contrato."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.validation import contract as c

SPEC = Path(__file__).resolve().parents[1] / "contracts" / "credit_application.yaml"


def _fields() -> dict:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    return spec["models"]["credit_application"]["fields"]


def test_spec_and_schema_have_same_fields():
    assert set(_fields()) == set(c.contract_columns(labeled=True))


def test_spec_required_flags_match_schema_nullability():
    columns = c.LabeledCreditApplicationSchema.to_schema().columns
    for name, field in _fields().items():
        if name == "target":
            continue  # obrigatorio so no contrato rotulado
        assert field["required"] is (not columns[name].nullable), name


def test_spec_enums_match_schema_domains():
    fields = _fields()
    assert fields["contract_type"]["enum"] == c.CONTRACT_TYPES
    assert fields["income_type"]["enum"] == c.INCOME_TYPES
    assert fields["education_type"]["enum"] == c.EDUCATION_TYPES
    assert fields["family_status"]["enum"] == c.FAMILY_STATUSES
    assert fields["housing_type"]["enum"] == c.HOUSING_TYPES
    assert fields["region_rating"]["enum"] == c.REGION_RATINGS


def test_spec_limits_match_schema_constants():
    fields = _fields()
    assert fields["age_years"]["minimum"] == c.MIN_AGE_YEARS
    assert fields["age_years"]["maximum"] == c.MAX_AGE_YEARS
    assert fields["income_total"]["maximum"] == c.MAX_INCOME_TOTAL
    assert fields["credit_amount"]["maximum"] == c.MAX_CREDIT_AMOUNT
