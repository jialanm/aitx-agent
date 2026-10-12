"""ClinVar adapter against searches and summaries recorded on 2026-10-12 for eval variants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools import clinvar
from src.tools.clinvar import ClinVarTool

FIXTURES = Path(__file__).parent / "fixtures"


class _Reply:
    def __init__(self, path: Path):
        self.status_code, self.text, self.url = 200, path.read_text(), "https://eutils.ncbi.nlm.nih.gov/x"

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass


@pytest.fixture
def recorded(monkeypatch):
    """Serve the recorded reply that matches each search term or summary id list."""
    terms = []

    def fake_get(url, params=None, timeout=None):
        if "esearch" in url:
            term = params["term"]; terms.append(term)
            if "[variant name]" in term:
                cdna = term.split('"')[1]; gene = term.split(" AND ")[1].replace("[gene]", "")
                return _Reply(FIXTURES / f"clinvar_esearch_name_{gene}_{cdna.replace('>', '_')}.json")
            if "[chrpos38]" in term:
                chrom = term.split("[chr]")[0]; pos = term.split(" AND ")[1].replace("[chrpos38]", "")
                return _Reply(FIXTURES / f"clinvar_esearch_pos_{chrom}_{pos}.json")
            raise AssertionError(f"unexpected loose search: {term}")
        return _Reply(FIXTURES / f"clinvar_esummary_{params['id'].replace(',', '_')}.json")

    monkeypatch.setattr(clinvar.requests, "get", fake_get)
    return terms


def test_exact_name_search_returns_only_the_patients_record(recorded):
    summary, evidence = ClinVarTool().execute(gene="AGXT", variant="c.508G>A", transcript="NM_000030.3",
                                              genomic_hgvs="NC_000002.12:g.240871433G>A")

    assert summary.startswith("ClinVar record for the patient's variant NM_000030.3:c.508G>A (found by exact variant name):")
    assert "Variant: NM_000030.3(AGXT):c.508G>A (p.Gly170Arg)" in summary
    assert "Clinical Significance: Pathogenic" in summary
    assert "NOT the patient's variant" not in summary
    assert [e.url for e in evidence] == ["https://www.ncbi.nlm.nih.gov/clinvar/variation/40166/"]
    assert recorded == ['"c.508G>A"[variant name] AND AGXT[gene]']


def test_position_search_rejects_neighbouring_variants(recorded, monkeypatch):
    # Force the name search to miss so the position search runs: it returns the
    # patient's record plus c.508G>C and an insertion at the same base.
    real_get = clinvar.requests.get

    def name_misses(url, params=None, timeout=None):
        if "esearch" in url and "[variant name]" in params["term"]:
            return _Reply(FIXTURES / "clinvar_esearch_name_DMD_c.7544_9286del.json")  # an empty id list
        return real_get(url, params=params, timeout=timeout)

    monkeypatch.setattr(clinvar.requests, "get", name_misses)

    summary, evidence = ClinVarTool().execute(gene="AGXT", variant="c.508G>A", transcript="NM_000030.3",
                                              genomic_hgvs="NC_000002.12:g.240871433G>A")

    assert "(found by GRCh38 position)" in summary
    assert "ClinVar ID: 40166" in summary
    assert "2710435" not in summary and "3359182" not in summary  # c.508G>C and c.507_508insA are not reported
    assert [e.url for e in evidence] == ["https://www.ncbi.nlm.nih.gov/clinvar/variation/40166/"]


def test_variant_absent_from_clinvar_is_said_so_and_neighbours_are_marked(recorded):
    summary, evidence = ClinVarTool().execute(gene="DMD", variant="c.7544_9286del", transcript="NM_004006.3",
                                              genomic_hgvs="NC_000023.11:g.31260956_31729748del")

    assert summary.startswith("No ClinVar record found for the patient's exact variant NM_004006.3:c.7544_9286del")
    assert "Other variants in this gene returned by the search, NOT the patient's variant:" in summary
    assert "NM_004006.3(DMD):c.9285A>G" in summary
    assert evidence == []  # a neighbour is shown for transparency but never cited


@pytest.mark.parametrize("gene, cdna, tx, uid, sig", [
    ("GRIN2B", "c.2755C>T", "NM_000834.5", "956942", "Pathogenic"),
    ("ANO10", "c.289del", "NM_018075.5", "503974", "Pathogenic"),
])
def test_other_eval_variants_resolve_to_their_record(recorded, gene, cdna, tx, uid, sig):
    summary, evidence = ClinVarTool().execute(gene=gene, variant=cdna, transcript=tx)

    assert f"ClinVar record for the patient's variant {tx}:{cdna}" in summary
    assert f"ClinVar ID: {uid}" in summary
    assert f"Clinical Significance: {sig}" in summary
    assert [e.url for e in evidence] == [f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid}/"]
