"""OMIM tool for gene-disease relationships and inheritance patterns.

Uses the OMIM search API (no API key required for basic search) and
NCBI E-utilities to look up MIM numbers and gene-disease associations.
"""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
OMIM_WEB = "https://www.omim.org/entry/{mim_number}"

TIMEOUT = 15


class OMIMTool(BaseTool):
    """Search OMIM for gene-disease relationships and inheritance patterns."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_omim",
                "description": (
                    "Search OMIM (Online Mendelian Inheritance in Man) for gene-disease "
                    "relationships, inheritance patterns, and phenotype descriptions. "
                    "Critical for rare/Mendelian disease questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., CFTR, PCCA)",
                        },
                        "condition": {
                            "type": "string",
                            "description": (
                                "Disease or condition name "
                                "(e.g., propionic acidemia, cystic fibrosis)"
                            ),
                        },
                    },
                    "required": ["gene"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        gene = kwargs.get("gene", "")
        condition = kwargs.get("condition", "")
        evidence: list[EvidenceRecord] = []

        try:
            results_parts = []

            # Strategy 1: Search NCBI MedGen/OMIM via E-utilities for the gene
            gene_entries = self._search_omim_gene(gene)
            if gene_entries:
                results_parts.append(self._format_gene_entries(gene_entries, evidence))

            # Strategy 2: Search for phenotype entries
            phenotype_entries = self._search_omim_phenotype(gene, condition)
            if phenotype_entries:
                results_parts.append(
                    self._format_phenotype_entries(phenotype_entries, evidence)
                )

            # Strategy 3: Use MedGen for inheritance pattern
            inheritance = self._get_medgen_inheritance(gene, condition)
            if inheritance:
                results_parts.append(inheritance)

            if not results_parts:
                return f"No OMIM entries found for gene {gene}.", evidence

            summary = "\n\n".join(results_parts)
            return summary[:2000], evidence

        except requests.RequestException as e:
            return f"OMIM lookup error: {e}", evidence

    def _search_omim_gene(self, gene: str) -> list[dict]:
        """Search NCBI for OMIM gene entries."""
        try:
            resp = requests.get(
                ESEARCH_URL,
                params={
                    "db": "omim",
                    "term": f"{gene}[gene symbol]",
                    "retmode": "json",
                    "retmax": 5,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            resp2 = requests.get(
                ESUMMARY_URL,
                params={
                    "db": "omim",
                    "id": ",".join(id_list[:5]),
                    "retmode": "json",
                },
                timeout=TIMEOUT,
            )
            resp2.raise_for_status()
            summary_data = resp2.json().get("result", {})
            uids = summary_data.get("uids", [])

            entries = []
            for uid in uids:
                entry = summary_data.get(uid, {})
                entries.append(entry)
            return entries

        except Exception:
            return []

    def _search_omim_phenotype(self, gene: str, condition: str) -> list[dict]:
        """Search NCBI for OMIM phenotype entries."""
        try:
            query = f"{gene}[gene symbol] AND phenotype"
            if condition:
                query = f"({gene}[gene symbol] OR {condition}) AND phenotype"

            resp = requests.get(
                ESEARCH_URL,
                params={
                    "db": "omim",
                    "term": query,
                    "retmode": "json",
                    "retmax": 5,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            resp2 = requests.get(
                ESUMMARY_URL,
                params={
                    "db": "omim",
                    "id": ",".join(id_list[:5]),
                    "retmode": "json",
                },
                timeout=TIMEOUT,
            )
            resp2.raise_for_status()
            summary_data = resp2.json().get("result", {})
            uids = summary_data.get("uids", [])

            entries = []
            for uid in uids:
                entry = summary_data.get(uid, {})
                entries.append(entry)
            return entries

        except Exception:
            return []

    def _get_medgen_inheritance(self, gene: str, condition: str) -> str | None:
        """Use MedGen to find inheritance pattern information."""
        try:
            query = f"{gene} inheritance"
            if condition:
                query = f"{condition} inheritance"

            resp = requests.get(
                ESEARCH_URL,
                params={
                    "db": "medgen",
                    "term": query,
                    "retmode": "json",
                    "retmax": 3,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return None

            resp2 = requests.get(
                ESUMMARY_URL,
                params={
                    "db": "medgen",
                    "id": ",".join(id_list[:3]),
                    "retmode": "json",
                },
                timeout=TIMEOUT,
            )
            resp2.raise_for_status()
            summary_data = resp2.json().get("result", {})
            uids = summary_data.get("uids", [])

            parts = ["## Inheritance (MedGen)"]
            for uid in uids[:3]:
                entry = summary_data.get(uid, {})
                title = entry.get("title", "")
                definition = entry.get("definition", "")
                if definition and len(definition) > 200:
                    definition = definition[:200] + "..."
                if title:
                    line = f"- {title}"
                    if definition:
                        line += f": {definition}"
                    parts.append(line)

            return "\n".join(parts) if len(parts) > 1 else None

        except Exception:
            return None

    def _format_gene_entries(
        self, entries: list[dict], evidence: list[EvidenceRecord]
    ) -> str:
        """Format OMIM gene entries."""
        parts = ["## OMIM Gene Entries"]
        seen_mims = set()
        for entry in entries:
            mim_number = entry.get("uid", "")
            if mim_number in seen_mims:
                continue
            seen_mims.add(mim_number)

            title = entry.get("title", "Unknown")
            mim_type = entry.get("type", "")

            url = OMIM_WEB.format(mim_number=mim_number)
            evidence.append(EvidenceRecord(url=url))

            line = f"- MIM #{mim_number}: {title}"
            if mim_type:
                line += f" [{mim_type}]"
            line += f"\n  URL: {url}"
            parts.append(line)

        return "\n".join(parts)

    def _format_phenotype_entries(
        self, entries: list[dict], evidence: list[EvidenceRecord]
    ) -> str:
        """Format OMIM phenotype entries."""
        parts = ["## OMIM Phenotype Entries"]
        seen_mims = set()
        for entry in entries:
            mim_number = entry.get("uid", "")
            if mim_number in seen_mims:
                continue
            seen_mims.add(mim_number)

            title = entry.get("title", "Unknown")
            mim_type = entry.get("type", "")

            url = OMIM_WEB.format(mim_number=mim_number)
            evidence.append(EvidenceRecord(url=url))

            line = f"- MIM #{mim_number}: {title}"
            if mim_type:
                line += f" [{mim_type}]"
            line += f"\n  URL: {url}"
            parts.append(line)

        return "\n".join(parts)
