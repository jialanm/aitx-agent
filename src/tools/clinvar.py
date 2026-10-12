"""ClinVar tool: the record for the patient's exact variant, via NCBI E-utilities."""

from __future__ import annotations

import logging
import re

import requests

from .base import BaseTool, EvidenceRecord, ncbi_get, ncbi_params, redact_ncbi_key

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CLINVAR_WEB = "https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid}/"

TIMEOUT = 15
MAX_RECORDS = 5

# RefSeq chromosome accessions on GRCh38, for the position search.
CHROMOSOMES = {f"NC_{i:06d}": str(i) for i in range(1, 23)} | {"NC_000023": "X", "NC_000024": "Y", "NC_012920": "MT"}
_GENOMIC = re.compile(r"^(NC_\d{6})\.\d+:g\.(\d+)(?:_(\d+))?")


class ClinVarTool(BaseTool):
    """Find the ClinVar record for a variant and report its classification."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_clinvar",
                "description": (
                    "A public archive of interpretations of clinically relevant "
                    "variants and their supporting evidence. Finds the record for "
                    "the patient's exact variant: first by its cDNA name in the gene, "
                    "then by its GRCh38 position, and reports classification, review "
                    "status, conditions and molecular consequence with the record's "
                    "URL. Every returned record is checked against the patient's "
                    "variant; records for other variants in the gene are listed "
                    "separately and marked as not the patient's. Give the validated "
                    "transcript form and genomic position from the variant block."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {"type": "string", "description": "HGNC gene symbol (e.g., CFTR)"},
                        "variant": {
                            "type": "string",
                            "description": "cDNA change without the transcript (e.g., c.1521_1523del)",
                        },
                        "transcript": {
                            "type": "string",
                            "description": "Current transcript from the validated variant block (e.g., NM_000492.4)",
                        },
                        "genomic_hgvs": {
                            "type": "string",
                            "description": "GRCh38 genomic HGVS from the validated variant block (e.g., NC_000007.14:g.117559592_117559594del)",
                        },
                    },
                    "required": ["gene", "variant"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        gene = (kwargs.get("gene") or "").strip()
        cdna = (kwargs.get("variant") or "").strip()
        transcript = (kwargs.get("transcript") or "").strip()
        genomic = (kwargs.get("genomic_hgvs") or "").strip()
        evidence: list[EvidenceRecord] = []
        if not gene or not cdna:
            return "ClinVar: gene and variant are required.", evidence
        # "NM_000030.3:c.508G>A" given as the variant is accepted too.
        if ":" in cdna and not transcript:
            transcript, cdna = cdna.split(":", 1)
        label = f"{transcript}:{cdna}" if transcript else f"{gene} {cdna}"

        try:
            ids, how = self._find(gene, cdna, genomic)
            if not ids:
                return f"No ClinVar record found for the patient's variant {label}: no search matched.", evidence
            records = self._summaries(ids)
            exact = [r for r in records if self._is_patient_variant(r, cdna, transcript, genomic)]
            others = [r for r in records if r not in exact]
            parts = []
            if exact:
                parts.append(f"ClinVar record for the patient's variant {label} (found by {how}):")
                for r in exact:
                    parts.append(self._format(r, evidence))
            else:
                parts.append(
                    f"No ClinVar record found for the patient's exact variant {label} "
                    f"(searched by {how}; the records returned are other variants)."
                )
            if others and not exact:
                parts.append("Other variants in this gene returned by the search, NOT the patient's variant:")
                for r in others[:3]:
                    parts.append(self._format(r, evidence, cite=False))
            return "\n\n".join(parts)[:4000], evidence
        except requests.RequestException as e:
            return redact_ncbi_key(f"ClinVar API error: {e}"), evidence

    def _find(self, gene: str, cdna: str, genomic: str) -> tuple[list[str], str]:
        """Record IDs by exact name, else by GRCh38 position, else by loose text."""
        ids = self._search(f'"{cdna}"[variant name] AND {gene}[gene]')
        if ids:
            return ids, "exact variant name"
        m = _GENOMIC.match(genomic)
        if m and m.group(1) in CHROMOSOMES:
            ids = self._search(f"{CHROMOSOMES[m.group(1)]}[chr] AND {m.group(2)}[chrpos38]")
            if ids:
                return ids, "GRCh38 position"
        ids = self._search(f"{gene}[gene] AND {cdna}")
        return ids, "gene and variant text"

    def _search(self, term: str) -> list[str]:
        resp = ncbi_get(label="ClinVar", url=ESEARCH_URL, params=ncbi_params(db="clinvar", term=term, retmode="json", retmax=MAX_RECORDS), timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])

    def _summaries(self, ids: list[str]) -> list[dict]:
        resp = ncbi_get(label="ClinVar", url=ESUMMARY_URL, params=ncbi_params(db="clinvar", id=",".join(ids[:MAX_RECORDS]), retmode="json"), timeout=TIMEOUT)
        resp.raise_for_status()
        result = resp.json().get("result", {})
        out = []
        for uid in result.get("uids", []):
            entry = result.get(uid, {})
            entry["_uid"] = uid
            out.append(entry)
        return out

    @staticmethod
    def _is_patient_variant(record: dict, cdna: str, transcript: str, genomic: str) -> bool:
        """The record names the same cDNA change on the same transcript accession, or spans the same GRCh38 bases."""
        vs = (record.get("variation_set") or [{}])[0]
        name = vs.get("variation_name") or record.get("title") or ""
        accession = transcript.split(".")[0]
        if cdna and cdna in name and (not accession or accession + "." in name or accession + "(" in name):
            return True
        m = _GENOMIC.match(genomic)
        if m:
            # Same GRCh38 span is not enough: c.508G>C sits on the same base as
            # c.508G>A. The change itself (G>A, del, dup, delinsTA) must also
            # appear in the record's name.
            change = re.sub(r"^c\.[-*]?\d+(?:[+-]\d+)?(?:_[-*]?\d+(?:[+-]\d+)?)?", "", cdna)
            chrom, start, stop = CHROMOSOMES.get(m.group(1), ""), m.group(2), m.group(3) or m.group(2)
            for loc in vs.get("variation_loc") or []:
                if loc.get("assembly_name") == "GRCh38" and str(loc.get("chr")) == chrom \
                        and str(loc.get("start")) == start and str(loc.get("stop")) == stop \
                        and change and change in name:
                    return True
        return False

    @staticmethod
    def _format(entry: dict, evidence: list[EvidenceRecord], cite: bool = True) -> str:
        uid = entry["_uid"]
        # ClinVar's esummary moved the classification to "germline_classification";
        # a record with neither key is reported as such and logged, never blanked.
        sig = entry.get("germline_classification") or entry.get("clinical_significance")
        if isinstance(sig, dict):
            desc = sig.get("description") or "Not provided"
            review = sig.get("review_status") or "Not provided"
            conditions = [t.get("trait_name", "") for t in sig.get("trait_set") or []
                          if t.get("trait_name") and t.get("trait_name") != "not specified"]
        else:
            logger.warning(f"ClinVar record {uid} has no germline_classification; keys: {sorted(entry)}")
            desc, review, conditions = "not returned", "not returned", []
        vs = (entry.get("variation_set") or [{}])[0]
        url = CLINVAR_WEB.format(uid=uid)
        if cite:
            evidence.append(EvidenceRecord(url=url))
        lines = [
            f"ClinVar ID: {uid}",
            f"Variant: {vs.get('variation_name') or entry.get('title', '')}",
            f"Clinical Significance: {desc}",
            f"Review Status: {review}",
        ]
        if conditions:
            lines.append(f"Conditions: {', '.join(conditions[:3])}")
        if entry.get("molecular_consequence_list"):
            lines.append(f"Molecular consequence: {', '.join(entry['molecular_consequence_list'])}")
        lines.append(f"URL: {url}")
        return "\n".join(lines)
