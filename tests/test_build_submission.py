"""The submission file's checks: one report per case, non-empty, under 5 MB."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("build_submission", Path(__file__).parent.parent / "scripts" / "build_submission.py")
build_submission = importlib.util.module_from_spec(_spec)
sys.modules["build_submission"] = build_submission
_spec.loader.exec_module(build_submission)

IDS = ["AITX-40001", "AITX-40002", "AITX-40003", "AITX-40004"]


def write(tmp_path, reports):
    out = tmp_path / "submission.json"
    out.write_text(json.dumps(reports))
    return out


def test_four_good_reports_pass(tmp_path):
    reports = [{"id": i, "report_markdown": f"# Therapeutic Actionability Report for Patient {i}\n\n## Summary\nx"} for i in IDS]
    assert build_submission.check(reports, IDS, write(tmp_path, reports)) == []


def test_missing_and_duplicate_ids_are_caught(tmp_path):
    reports = [{"id": IDS[0], "report_markdown": "# r"}, {"id": IDS[0], "report_markdown": "# r"}]
    problems = build_submission.check(reports, IDS, write(tmp_path, reports))
    assert any("do not match the cases" in p for p in problems)
    assert any("duplicate ids" in p for p in problems)


def test_empty_or_unheaded_report_is_caught(tmp_path):
    reports = [{"id": IDS[0], "report_markdown": ""}, {"id": IDS[1], "report_markdown": "Summary without a heading"},
               {"id": IDS[2], "report_markdown": "# ok"}, {"id": IDS[3], "report_markdown": "# ok"}]
    problems = build_submission.check(reports, IDS, write(tmp_path, reports))
    assert f"{IDS[0]}: empty report_markdown" in problems
    assert f"{IDS[1]}: report does not start with a heading" in problems


def test_oversized_file_is_caught(tmp_path):
    reports = [{"id": i, "report_markdown": "# r\n" + "x" * 1_400_000} for i in IDS]
    problems = build_submission.check(reports, IDS, write(tmp_path, reports))
    assert any("over the 5 MB limit" in p for p in problems)


def test_the_four_downloaded_cases_parse_as_report_inputs():
    from src.schemas import ReportInput
    cases = json.load(open("data/patients/aitx_final_patients.json"))
    assert [c["id"] for c in cases] == IDS
    for c in cases:
        r = ReportInput(**c)
        assert r.patient.genotype and r.patient.clinical_context
