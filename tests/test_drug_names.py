"""Brand and salt names map to the ingredient via RxNorm replies recorded on 2026-10-12."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src import drug_names
from src.drug_names import normalise_drug_names

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, path: Path):
        self.status_code, self.text = 200, path.read_text()

    def json(self):
        return json.loads(self.text)


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url.rsplit("/REST/", 1)[1], dict(params or {})))
        if url.endswith("/rxcui.json"):
            return _Reply(FIXTURES / ("rxnorm_rxcui_" + re.sub(r"[^A-Za-z0-9_-]", "_", params["name"]) + ".json"))
        rxcui = url.split("/rxcui/")[1].split("/")[0]
        return _Reply(FIXTURES / (f"rxnorm_tty_{rxcui}.json" if url.endswith("property.json") else f"rxnorm_related_IN_{rxcui}.json"))

    monkeypatch.setattr(drug_names.requests, "get", fake_get)
    return calls


def test_brand_becomes_its_ingredient(recorded):
    answer, changes = normalise_drug_names("Ascorbic acid, DDAVP")

    assert answer == "Ascorbic acid, desmopressin"
    assert changes == [{"from": "DDAVP", "to": "desmopressin", "rxcui": "203658", "tty": "BN"}]


def test_salt_form_becomes_its_ingredient(recorded):
    answer, changes = normalise_drug_names("desmopressin acetate")

    assert answer == "desmopressin"
    assert changes[0]["tty"] == "PIN"


def test_ingredient_names_are_left_as_written(recorded):
    # L-serine resolves to the ingredient "serine", but an ingredient is never rewritten.
    answer, changes = normalise_drug_names("L-serine")

    assert answer == "L-serine" and changes == []


def test_non_drug_items_are_skipped_without_a_lookup(recorded):
    answer, changes = normalise_drug_names("None, 18 months, NCT05402384, No, GRD")

    assert answer == "None, 18 months, NCT05402384, No, GRD" and changes == []
    assert recorded == []


def test_unknown_name_is_left_alone(recorded):
    answer, changes = normalise_drug_names("Trikafta")

    assert answer == "Trikafta" and changes == []


def test_rxnorm_failure_leaves_the_answer_untouched(monkeypatch, caplog):
    monkeypatch.setattr(drug_names.time, "sleep", lambda s: None)
    monkeypatch.setattr(drug_names.requests, "get", lambda url, params=None, timeout=None: (_ for _ in ()).throw(drug_names.requests.Timeout("simulated")))

    answer, changes = normalise_drug_names("DDAVP")

    assert answer == "DDAVP" and changes == []
    assert caplog.text.count("RxNorm timeout") == 3
