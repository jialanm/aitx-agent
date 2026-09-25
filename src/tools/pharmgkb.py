"""PharmGKB tool for drug-gene interactions and pharmacogenomic annotations."""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord

PHARMGKB_API = "https://api.clinpgx.org/v1/data"
PHARMGKB_WEB = "https://www.clinpgx.org"

TIMEOUT = 15


class PharmGKBTool(BaseTool):
    """Search PharmGKB for pharmacogenomic drug-gene relationships."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_pharmgkb",
                "description": (
                    "Search PharmGKB for pharmacogenomic information including "
                    "drug-gene interactions, clinical annotations, dosing guidelines, "
                    "and drug label annotations. Use for questions about how genetic "
                    "variants affect drug response or dosing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., CYP2D6, DPYD)",
                        },
                        "drug": {
                            "type": "string",
                            "description": "Drug name (e.g., warfarin, tamoxifen)",
                        },
                    },
                    "required": ["gene"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        gene = kwargs.get("gene", "")
        drug = kwargs.get("drug", "")
        evidence: list[EvidenceRecord] = []

        try:
            results_parts = []

            # Step 1: Look up the gene in PharmGKB
            gene_data = self._search_gene(gene)
            if gene_data:
                gene_id = gene_data.get("id", "")
                gene_name = gene_data.get("symbol", gene)
                gene_url = f"{PHARMGKB_WEB}/gene/{gene_id}"
                evidence.append(EvidenceRecord(url=gene_url))
                results_parts.append(
                    f"PharmGKB Gene: {gene_name} ({gene_id})\nURL: {gene_url}"
                )

                # Step 2: Get clinical annotations for this gene
                annotations = self._get_clinical_annotations(gene_id)
                if annotations:
                    results_parts.append(self._format_annotations(annotations, drug))

                # Step 3: Get drug labels for this gene
                labels = self._get_drug_labels(gene_id)
                if labels:
                    results_parts.append(self._format_labels(labels, drug, evidence))

                # Step 4: Get guideline annotations
                guidelines = self._get_guidelines(gene_id)
                if guidelines:
                    results_parts.append(
                        self._format_guidelines(guidelines, drug, evidence)
                    )
            else:
                return f"No PharmGKB entry found for gene {gene}.", evidence

            summary = "\n\n".join(results_parts)
            return summary[:2000], evidence

        except requests.RequestException as e:
            return f"PharmGKB API error: {e}", evidence

    def _search_gene(self, gene: str) -> dict | None:
        """Search for a gene in PharmGKB."""
        try:
            resp = requests.get(
                f"{PHARMGKB_API}/gene",
                params={"symbol": gene},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("data", [])
            return results[0] if results else None
        except Exception:
            return None

    def _get_clinical_annotations(self, gene_id: str) -> list[dict]:
        """Get clinical annotations for a gene."""
        try:
            resp = requests.get(
                f"{PHARMGKB_API}/clinicalAnnotation",
                params={"location.genes.accessionId": gene_id, "view": "max"},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("data", [])[:5]
        except Exception:
            return []

    def _get_drug_labels(self, gene_id: str) -> list[dict]:
        """Get FDA drug label annotations for a gene."""
        try:
            resp = requests.get(
                f"{PHARMGKB_API}/drugLabel",
                params={"relatedGenes.accessionId": gene_id},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("data", [])[:5]
        except Exception:
            return []

    def _get_guidelines(self, gene_id: str) -> list[dict]:
        """Get dosing guideline annotations for a gene."""
        try:
            resp = requests.get(
                f"{PHARMGKB_API}/guideline",
                params={"relatedGenes.accessionId": gene_id},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("data", [])[:5]
        except Exception:
            return []

    def _format_annotations(self, annotations: list[dict], drug_filter: str) -> str:
        """Format clinical annotations."""
        parts = ["## Clinical Annotations"]
        for ann in annotations:
            level = ann.get("level", "")
            phenotype_cat = ann.get("phenotypeCategory", "")

            related_chemicals = []
            for chem in ann.get("relatedChemicals", []):
                related_chemicals.append(chem.get("name", ""))

            # Filter by drug if specified
            if drug_filter:
                drug_lower = drug_filter.lower()
                if not any(drug_lower in c.lower() for c in related_chemicals):
                    continue

            chemicals_str = ", ".join(related_chemicals[:3]) or "N/A"
            parts.append(
                f"- Level {level} | Drugs: {chemicals_str} | Category: {phenotype_cat}"
            )
        return "\n".join(parts) if len(parts) > 1 else ""

    def _format_labels(
        self,
        labels: list[dict],
        drug_filter: str,
        evidence: list[EvidenceRecord],
    ) -> str:
        """Format drug label annotations."""
        parts = ["## Drug Labels"]
        for label in labels:
            name = label.get("name", "")
            label_id = label.get("id", "")

            if drug_filter and drug_filter.lower() not in name.lower():
                continue

            source = label.get("source", "")
            url = f"{PHARMGKB_WEB}/drugLabel/{label_id}"
            evidence.append(EvidenceRecord(url=url))
            parts.append(f"- {name} ({source})\n  URL: {url}")

        return "\n".join(parts) if len(parts) > 1 else ""

    def _format_guidelines(
        self,
        guidelines: list[dict],
        drug_filter: str,
        evidence: list[EvidenceRecord],
    ) -> str:
        """Format dosing guidelines."""
        parts = ["## Dosing Guidelines"]
        for gl in guidelines:
            name = gl.get("name", "")
            gl_id = gl.get("id", "")
            source = gl.get("source", "")

            if drug_filter and drug_filter.lower() not in name.lower():
                continue

            url = f"{PHARMGKB_WEB}/guidelineAnnotation/{gl_id}"
            evidence.append(EvidenceRecord(url=url))
            parts.append(f"- {name} ({source})\n  URL: {url}")

        return "\n".join(parts) if len(parts) > 1 else ""
