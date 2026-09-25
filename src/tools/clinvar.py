"""ClinVar tool using NCBI E-utilities."""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord, ncbi_params, redact_ncbi_key

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CLINVAR_WEB = "https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid}/"

TIMEOUT = 15


class ClinVarTool(BaseTool):
    """Search ClinVar for variant clinical significance."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_clinvar",
                "description": (
                    "Search ClinVar for a genetic variant to retrieve clinical "
                    "significance, review status, and associated conditions. "
                    "Provide a gene symbol and variant notation (cDNA or protein)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., CFTR, BRCA1)",
                        },
                        "variant": {
                            "type": "string",
                            "description": (
                                "Variant notation in cDNA or protein HGVS format "
                                "(e.g., c.1521_1523del, p.Phe508del)"
                            ),
                        },
                    },
                    "required": ["gene", "variant"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        gene = kwargs.get("gene", "")
        variant = kwargs.get("variant", "")

        query = f"{gene}[gene] AND {variant}"
        evidence: list[EvidenceRecord] = []

        try:
            # Step 1: search for variant IDs
            resp = requests.get(
                ESEARCH_URL,
                params=ncbi_params(
                    db="clinvar",
                    term=query,
                    retmode="json",
                    retmax=5,
                ),
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            search_data = resp.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                # Try broader search with just gene
                resp2 = requests.get(
                    ESEARCH_URL,
                    params=ncbi_params(
                        db="clinvar",
                        term=f"{gene}[gene] AND {variant.split('.')[0] if '.' in variant else variant}",
                        retmode="json",
                        retmax=5,
                    ),
                    timeout=TIMEOUT,
                )
                resp2.raise_for_status()
                search_data = resp2.json()
                id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return (
                    f"No ClinVar entries found for {gene} {variant}.",
                    evidence,
                )

            # Step 2: get summaries
            resp3 = requests.get(
                ESUMMARY_URL,
                params=ncbi_params(
                    db="clinvar",
                    id=",".join(id_list[:5]),
                    retmode="json",
                ),
                timeout=TIMEOUT,
            )
            resp3.raise_for_status()
            summary_data = resp3.json().get("result", {})

            uids = summary_data.get("uids", [])
            results = []
            for uid in uids[:3]:
                entry = summary_data.get(uid, {})
                title = entry.get("title", "Unknown")
                clinical_sig = entry.get("clinical_significance", {})
                if isinstance(clinical_sig, dict):
                    sig_desc = clinical_sig.get("description", "Not provided")
                    review_status = clinical_sig.get("review_status", "Not provided")
                else:
                    sig_desc = str(clinical_sig)
                    review_status = "Not provided"

                genes_info = entry.get("genes", [])
                gene_name = genes_info[0].get("symbol", gene) if genes_info else gene

                variation_set = entry.get("variation_set", [])
                variant_name = ""
                if variation_set:
                    variant_name = variation_set[0].get("variation_name", "")

                url = CLINVAR_WEB.format(uid=uid)
                evidence.append(EvidenceRecord(url=url))

                results.append(
                    f"ClinVar ID: {uid}\n"
                    f"Title: {title}\n"
                    f"Gene: {gene_name}\n"
                    f"Variant: {variant_name}\n"
                    f"Clinical Significance: {sig_desc}\n"
                    f"Review Status: {review_status}\n"
                    f"URL: {url}"
                )

            summary = "\n\n".join(results)
            return summary[:2000], evidence

        except requests.RequestException as e:
            return redact_ncbi_key(f"ClinVar API error: {e}"), evidence
