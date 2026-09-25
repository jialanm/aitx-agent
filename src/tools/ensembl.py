"""Ensembl REST API tool for variant annotation and transcript info."""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord

ENSEMBL_REST = "https://rest.ensembl.org"
TIMEOUT = 15


class EnsemblTool(BaseTool):
    """Query Ensembl for variant effect prediction and transcript info."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "query_ensembl",
                "description": (
                    "Query the Ensembl REST API to get variant effect predictions "
                    "(VEP), transcript information, and exon mapping for a variant. "
                    "Provide the transcript ID and HGVS cDNA notation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transcript": {
                            "type": "string",
                            "description": (
                                "RefSeq or Ensembl transcript ID "
                                "(e.g., NM_000492.4, ENST00000003084)"
                            ),
                        },
                        "hgvs_cdna": {
                            "type": "string",
                            "description": (
                                "HGVS cDNA notation including transcript "
                                "(e.g., NM_000492.4:c.1521_1523del)"
                            ),
                        },
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., CFTR)",
                        },
                    },
                    "required": ["hgvs_cdna"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        hgvs_cdna = kwargs.get("hgvs_cdna", "")
        transcript = kwargs.get("transcript", "")
        gene = kwargs.get("gene", "")
        evidence: list[EvidenceRecord] = []
        results = []

        # VEP analysis
        vep_result = self._query_vep(hgvs_cdna, evidence)
        if vep_result:
            results.append(vep_result)

        # Transcript lookup if we have a gene
        if gene:
            tx_result = self._lookup_gene(gene, evidence)
            if tx_result:
                results.append(tx_result)

        if not results:
            return f"No Ensembl data found for {hgvs_cdna}.", evidence

        summary = "\n\n".join(results)
        return summary[:2000], evidence

    def _query_vep(
        self, hgvs_notation: str, evidence: list[EvidenceRecord]
    ) -> str | None:
        """Query VEP for variant consequences."""
        try:
            url = f"{ENSEMBL_REST}/vep/human/hgvs/{hgvs_notation}"
            resp = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                params={"content-type": "application/json"},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            if not data:
                return None

            entry = data[0]
            evidence.append(
                EvidenceRecord(
                    url=f"https://www.ensembl.org/Homo_sapiens/Tools/VEP"
                )
            )

            consequences = []
            for tc in entry.get("transcript_consequences", [])[:3]:
                cons = ", ".join(tc.get("consequence_terms", []))
                impact = tc.get("impact", "")
                biotype = tc.get("biotype", "")
                protein_pos = tc.get("protein_start", "")
                amino_acids = tc.get("amino_acids", "")
                sift = tc.get("sift_prediction", "")
                polyphen = tc.get("polyphen_prediction", "")

                parts = [f"Consequence: {cons}", f"Impact: {impact}"]
                if biotype:
                    parts.append(f"Biotype: {biotype}")
                if protein_pos:
                    parts.append(f"Protein position: {protein_pos}")
                if amino_acids:
                    parts.append(f"Amino acid change: {amino_acids}")
                if sift:
                    parts.append(f"SIFT: {sift}")
                if polyphen:
                    parts.append(f"PolyPhen: {polyphen}")

                consequences.append("\n".join(parts))

            most_severe = entry.get("most_severe_consequence", "unknown")
            header = f"VEP Analysis for {hgvs_notation}:\nMost severe consequence: {most_severe}"
            return header + "\n\n" + "\n---\n".join(consequences)

        except requests.RequestException:
            return None

    def _lookup_gene(
        self, gene: str, evidence: list[EvidenceRecord]
    ) -> str | None:
        """Look up gene info from Ensembl."""
        try:
            url = f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{gene}"
            resp = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                params={"content-type": "application/json", "expand": 1},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            ensembl_id = data.get("id", "")
            description = data.get("description", "")
            biotype = data.get("biotype", "")

            gene_url = f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={ensembl_id}"
            evidence.append(EvidenceRecord(url=gene_url))

            transcripts = data.get("Transcript", [])
            tx_info = ""
            if transcripts:
                canonical = [t for t in transcripts if t.get("is_canonical")]
                tx = canonical[0] if canonical else transcripts[0]
                tx_id = tx.get("id", "")
                exon_count = len(tx.get("Exon", []))
                tx_info = f"\nCanonical transcript: {tx_id}\nExon count: {exon_count}"

            return (
                f"Gene: {gene} ({ensembl_id})\n"
                f"Description: {description}\n"
                f"Biotype: {biotype}"
                f"{tx_info}\n"
                f"URL: {gene_url}"
            )

        except requests.RequestException:
            return None
