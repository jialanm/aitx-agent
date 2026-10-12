"""ClinicalTrials.gov adapter: status filter spelling, against replies recorded on 2026-10-11."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from src.tools import clinical_trials
from src.tools.clinical_trials import ClinicalTrialsTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, status, text, ct="application/json"):
        self.status_code, self.text, self.headers = status, text, {"content-type": ct}
        self.url = "https://clinicaltrials.gov/api/v2/studies"

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error for url: {self.url}", response=self)


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params or {}))
        return _Reply(200, (FIXTURES / "clinicaltrials_KCNT1_recruiting.json").read_text())

    monkeypatch.setattr(clinical_trials.requests, "get", fake_get)
    return calls


def test_status_list_with_spaces_is_sent_comma_joined(recorded):
    summary, evidence = ClinicalTrialsTool().execute(
        query="KCNT1", status_filter="RECRUITING, ACTIVE_NOT_RECRUITING, NOT_YET_RECRUITING"
    )

    assert recorded[0]["filter.overallStatus"] == "RECRUITING,ACTIVE_NOT_RECRUITING,NOT_YET_RECRUITING"
    assert "NCT07227857" in summary
    assert len(evidence) == 4


def test_lowercase_and_pipes_are_normalised(recorded):
    ClinicalTrialsTool().execute(query="KCNT1", status_filter="recruiting | not_yet_recruiting")

    assert recorded[0]["filter.overallStatus"] == "RECRUITING,NOT_YET_RECRUITING"


def test_unknown_status_values_are_dropped_and_logged(recorded, caplog):
    ClinicalTrialsTool().execute(query="KCNT1", status_filter="RECRUITING, ONGOING")

    assert recorded[0]["filter.overallStatus"] == "RECRUITING"
    assert "dropped unknown values ['ONGOING']" in caplog.text


def test_all_unknown_sends_no_filter(recorded, caplog):
    ClinicalTrialsTool().execute(query="KCNT1", status_filter="open")

    assert "filter.overallStatus" not in recorded[0]
    assert "dropped unknown values ['OPEN']" in caplog.text


def test_api_rejection_is_reported_not_hidden(monkeypatch):
    body = (FIXTURES / "clinicaltrials_400_status_filter.txt").read_text()
    monkeypatch.setattr(clinical_trials.requests, "get", lambda url, params=None, timeout=None: _Reply(400, body, "text/plain"))

    summary, evidence = ClinicalTrialsTool().execute(query="KCNT1", status_filter="RECRUITING")

    assert summary.startswith("ClinicalTrials.gov API error: 400 Client Error")
    assert evidence == []
