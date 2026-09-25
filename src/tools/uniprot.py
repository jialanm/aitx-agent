"""UniProt tool for protein domains, functional annotations, and active sites."""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://www.uniprot.org/uniprotkb/{accession}/entry"

TIMEOUT = 15


class UniProtTool(BaseTool):
    """Query UniProt for protein functional annotations and domain information."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "query_uniprot",
                "description": (
                    "Query UniProt for protein information including functional domains, "
                    "active sites, binding sites, and post-translational modifications. "
                    "Useful for understanding how a variant's protein location affects function."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., CFTR, BRCA1)",
                        },
                        "protein_position": {
                            "type": "integer",
                            "description": (
                                "Amino acid position to check for domain/site overlap "
                                "(e.g., 508 for p.Phe508del)"
                            ),
                        },
                    },
                    "required": ["gene"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        gene = kwargs.get("gene", "")
        protein_position = kwargs.get("protein_position")
        evidence: list[EvidenceRecord] = []

        try:
            # Search UniProt for human reviewed entry
            resp = requests.get(
                UNIPROT_SEARCH_URL,
                params={
                    "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                    "format": "json",
                    "fields": (
                        "accession,gene_names,protein_name,ft_domain,ft_site,"
                        "ft_act_site,ft_binding,ft_motif,cc_function,"
                        "cc_catalytic_activity,cc_subcellular_location,length"
                    ),
                    "size": 1,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            results_list = data.get("results", [])
            if not results_list:
                return f"No UniProt entry found for gene {gene} (Homo sapiens).", evidence

            entry = results_list[0]
            accession = entry.get("primaryAccession", "")
            url = UNIPROT_ENTRY_URL.format(accession=accession)
            evidence.append(EvidenceRecord(url=url))

            # Build summary
            parts = []

            # Protein name
            protein_desc = entry.get("proteinDescription", {})
            rec_name = protein_desc.get("recommendedName", {})
            full_name = rec_name.get("fullName", {}).get("value", "")
            if full_name:
                parts.append(f"Protein: {full_name} ({accession})")

            # Length
            length = entry.get("sequence", {}).get("length")
            if length:
                parts.append(f"Length: {length} aa")

            # Function
            for comment in entry.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        func_text = texts[0].get("value", "")
                        if len(func_text) > 300:
                            func_text = func_text[:300] + "..."
                        parts.append(f"Function: {func_text}")
                elif comment.get("commentType") == "SUBCELLULAR LOCATION":
                    locs = []
                    for sl in comment.get("subcellularLocations", []):
                        loc = sl.get("location", {}).get("value", "")
                        if loc:
                            locs.append(loc)
                    if locs:
                        parts.append(f"Subcellular location: {', '.join(locs[:3])}")

            # Domains and features
            features = entry.get("features", [])
            domains = []
            active_sites = []
            binding_sites = []
            position_hits = []

            for feat in features:
                feat_type = feat.get("type", "")
                desc = feat.get("description", "")
                loc = feat.get("location", {})
                start = loc.get("start", {}).get("value")
                end = loc.get("end", {}).get("value")

                if start is not None and end is not None:
                    range_str = f"{start}-{end}"
                else:
                    range_str = "?"

                if feat_type == "Domain":
                    domains.append(f"{desc} ({range_str})")
                elif feat_type == "Active site":
                    active_sites.append(f"{desc} (pos {range_str})")
                elif feat_type in ("Binding site", "Motif"):
                    binding_sites.append(f"{desc} ({range_str})")

                # Check if protein_position falls in this feature
                if (
                    protein_position is not None
                    and start is not None
                    and end is not None
                    and start <= protein_position <= end
                ):
                    position_hits.append(f"{feat_type}: {desc} ({range_str})")

            if domains:
                parts.append(f"Domains: {'; '.join(domains[:8])}")
            if active_sites:
                parts.append(f"Active sites: {'; '.join(active_sites[:5])}")
            if binding_sites:
                parts.append(f"Binding/Motif sites: {'; '.join(binding_sites[:5])}")

            if protein_position is not None:
                if position_hits:
                    parts.append(
                        f"\nPosition {protein_position} overlaps: "
                        + "; ".join(position_hits)
                    )
                else:
                    parts.append(
                        f"\nPosition {protein_position}: no annotated domain/site overlap"
                    )

            parts.append(f"URL: {url}")
            summary = "\n".join(parts)
            return summary[:2000], evidence

        except requests.RequestException as e:
            return f"UniProt API error: {e}", evidence
