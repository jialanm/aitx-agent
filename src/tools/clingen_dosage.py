"""ClinGen dosage sensitivity tool: haploinsufficiency and triplosensitivity curations.

ClinGen publishes its full gene-level dosage curation as one tab-separated
file, refreshed daily. The adapter downloads it once per process and
looks the gene up locally; there is no per-gene JSON endpoint.
"""

from __future__ import annotations

import csv
import io
import logging
import time

import requests

from .base import BaseTool, EvidenceRecord

logger = logging.getLogger(__name__)

CLINGEN_DOSAGE_TSV = "https://ftp.clinicalgenome.org/ClinGen_gene_curation_list_GRCh38.tsv"
# Per-gene page named in the file's own header as the link to use.
CLINGEN_GENE_URL = "https://www.ncbi.nlm.nih.gov/projects/dbvar/clingen/clingen_gene.cgi?sym={gene}"
TIMEOUT = 60
RETRIES = 5
RETRY_INTERVAL_SECONDS = 5

# ClinGen's score vocabulary, as written in the file's description column.
SCORE_MEANING = {
    "3": "sufficient evidence for dosage pathogenicity",
    "2": "some evidence for dosage pathogenicity",
    "1": "little evidence for dosage pathogenicity",
    "0": "no evidence available",
    "30": "gene associated with autosomal recessive phenotype",
    "40": "dosage sensitivity unlikely",
    "Not yet evaluated": "not yet evaluated",
}


class ClinGenDosageTool(BaseTool):
    """Look up a gene's ClinGen haploinsufficiency and triplosensitivity scores."""

    def __init__(self) -> None:
        self._rows: dict[str, dict] | None = None
        self._file_date: str = ""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "clingen_dosage",
                "description": (
                    "ClinGen Dosage Sensitivity: expert-curated scores for whether "
                    "losing one copy of a gene (haploinsufficiency) or gaining a "
                    "copy (triplosensitivity) causes disease. For a gene it returns "
                    "the haploinsufficiency score and triplosensitivity score on "
                    "ClinGen's scale (3 sufficient evidence, 2 some, 1 little, 0 "
                    "none, 30 autosomal recessive gene, 40 dosage sensitivity "
                    "unlikely, or not yet evaluated), the supporting PMIDs, the "
                    "linked disease, and the evaluation date. Use it to decide "
                    "whether a knockdown or reduced-dosage strategy is safe (a "
                    "score of 3 means the gene is haploinsufficient, so knockdown "
                    "is not eligible unless allele-selective) and whether loss of "
                    "function is the disease mechanism. Covers about 1,500 curated "
                    "genes; an uncurated gene is reported as not curated, which is "
                    "not the same as no evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., DMD, GRIN2B)",
                        },
                    },
                    "required": ["gene"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        gene = (kwargs.get("gene") or "").strip()
        evidence: list[EvidenceRecord] = []
        if not gene:
            return "ClinGen dosage: no gene given.", evidence
        rows, failure = self._table()
        if rows is None:
            return f"ClinGen dosage returned nothing for {gene}: {failure}", evidence
        row = rows.get(gene.upper())
        if row is None:
            return (
                f"ClinGen dosage: {gene} has no dosage curation in ClinGen's list "
                f"({len(rows)} genes, file dated {self._file_date}). Not curated is not evidence of absence.",
                evidence,
            )
        evidence.append(EvidenceRecord(url=CLINGEN_GENE_URL.format(gene=gene)))
        hi, ts = row["Haploinsufficiency Score"], row["Triplosensitivity Score"]
        hi_pmids = [row[f"Haploinsufficiency PMID{i}"] for i in range(1, 7) if row.get(f"Haploinsufficiency PMID{i}")]
        ts_pmids = [row[f"Triplosensitivity PMID{i}"] for i in range(1, 7) if row.get(f"Triplosensitivity PMID{i}")]
        lines = [
            f"ClinGen dosage sensitivity for {row['Gene Symbol']} ({row['cytoBand']}, file dated {self._file_date}):",
            f"Haploinsufficiency score: {hi} ({SCORE_MEANING.get(hi, row['Haploinsufficiency Description'])})",
        ]
        if hi_pmids:
            lines.append("Haploinsufficiency evidence PMIDs: " + ", ".join(hi_pmids))
        if row.get("Haploinsufficiency Disease ID"):
            lines.append(f"Haploinsufficiency disease: {row['Haploinsufficiency Disease ID']}")
        lines.append(f"Triplosensitivity score: {ts} ({SCORE_MEANING.get(ts, row['Triplosensitivity Description'])})")
        if ts_pmids:
            lines.append("Triplosensitivity evidence PMIDs: " + ", ".join(ts_pmids))
        lines.append(f"Last evaluated: {row['Date Last Evaluated']}")
        lines.append(f"URL: {CLINGEN_GENE_URL.format(gene=gene)}")
        return "\n".join(lines), evidence

    def _table(self) -> tuple[dict[str, dict] | None, str]:
        """The curation file as {SYMBOL: row}, downloaded once per process."""
        if self._rows is not None:
            return self._rows, ""
        last = ""
        for attempt in range(1, RETRIES + 1):
            try:
                resp = requests.get(CLINGEN_DOSAGE_TSV, timeout=TIMEOUT)
            except requests.Timeout:
                last = "timeout"
                logger.warning(f"ClinGen dosage download timeout (attempt {attempt}/{RETRIES})")
            except requests.RequestException as e:
                logger.warning(f"ClinGen dosage download failed: {e}")
                return None, f"request failed ({e})"
            else:
                if resp.status_code < 500:
                    if resp.status_code != 200:
                        logger.warning(f"ClinGen dosage download HTTP {resp.status_code}: {resp.text[:200]!r}")
                        return None, f"HTTP {resp.status_code}"
                    return self._parse(resp.text)
                last = f"HTTP {resp.status_code}"
                logger.warning(f"ClinGen dosage server error (attempt {attempt}/{RETRIES}): HTTP {resp.status_code}")
            if attempt < RETRIES:
                time.sleep(RETRY_INTERVAL_SECONDS)
        return None, f"{last} after {RETRIES} attempts"

    def _parse(self, text: str) -> tuple[dict[str, dict] | None, str]:
        lines = text.splitlines()
        # The header row is the last comment line; the first comment line carries the date.
        header_line = next((l for l in lines if l.startswith("#Gene Symbol")), None)
        if header_line is None:
            logger.warning("ClinGen dosage file has no '#Gene Symbol' header line; format changed?")
            return None, "unexpected file format (no header)"
        # The date is the second comment line, e.g. "#07 Oct,2026".
        self._file_date = lines[1].lstrip("#").strip() if len(lines) > 1 and lines[1].startswith("#") else ""
        header = header_line.lstrip("#").split("\t")
        body = [l for l in lines if l and not l.startswith("#")]
        reader = csv.DictReader(io.StringIO("\n".join(body)), fieldnames=header, delimiter="\t")
        rows = {r["Gene Symbol"].upper(): r for r in reader if r.get("Gene Symbol")}
        if not rows:
            logger.warning("ClinGen dosage file parsed to zero rows")
            return None, "file parsed to zero rows"
        self._rows = rows
        return rows, ""
