"""UniProt tool for protein domains, functional annotations, and active sites."""

from __future__ import annotations

import logging

import requests

from .base import BaseTool, EvidenceRecord, get_with_retries

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://www.uniprot.org/uniprotkb/{accession}/entry"
# InterPro: every member database's domain match for the same accession.
# UniProt's own Domain feature carries one source's boundary (PROSITE's
# profile for the NF1 Ras-GAP domain, 1251-1482); InterPro, CDD, SMART and
# SUPERFAMILY place the same domain at 1187-1557, which is where the
# literature and the answer key put it (run 5 and 32B runs, AITX-00016,
# position 1243 "outside any domain"). Added 2026-10-13.
INTERPRO_API_URL = "https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{accession}"
INTERPRO_PROTEIN_URL = "https://www.ebi.ac.uk/interpro/protein/UniProt/{accession}/"
INTERPRO_MAX_LISTED = 10

TIMEOUT = 30

logger = logging.getLogger(__name__)


class UniProtTool(BaseTool):
    """Query UniProt for protein functional annotations and domain information."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "query_uniprot",
                "description": (
                    "A comprehensive resource for protein sequence and functional "
                    "information, including isoforms and domain annotations. Returns "
                    "the protein's function, location, domains, active and binding "
                    "sites, and whether a given amino acid position falls inside an "
                    "annotated domain or site, by UniProt's own feature table and by "
                    "every InterPro member database (InterPro, CDD, SMART, Pfam, PROSITE, "
                    "SUPERFAMILY), whose boundaries for the same domain differ; each match "
                    "is listed with its source and residues. Take the position from the VEP result."
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
            resp = get_with_retries(
                label="UniProt", url=UNIPROT_SEARCH_URL,
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

            interpro_lines, interpro_url = self._interpro(accession, protein_position)
            parts.extend(interpro_lines)
            if interpro_url:
                evidence.append(EvidenceRecord(url=interpro_url))

            parts.append(f"URL: {url}")
            summary = "\n".join(parts)
            return summary[:3000], evidence

        except requests.RequestException as e:
            return f"UniProt API error: {e}", evidence

    @staticmethod
    def _interpro(accession: str, protein_position: int | None) -> tuple[list[str], str | None]:
        """Domain matches for the accession from InterPro and its member
        databases, as lines: those covering the position, else the nearest
        ones; without a position, all of them. Returns (lines, page URL)."""
        try:
            resp = get_with_retries(
                label="InterPro", url=INTERPRO_API_URL.format(accession=accession),
                params={"page_size": 200}, timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"InterPro lookup failed for {accession}: {e}")
            return [f"InterPro domain matches: not available ({e})"], None
        matches = []
        for entry in data.get("results", []):
            meta = entry.get("metadata", {})
            for prot in entry.get("proteins", []):
                for loc in prot.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        matches.append({
                            "db": meta.get("source_database", "?"), "acc": meta.get("accession", "?"),
                            "name": meta.get("name") or "", "type": meta.get("type") or "",
                            "start": frag.get("start"), "end": frag.get("end"),
                        })
        matches = [m for m in matches if isinstance(m["start"], int) and isinstance(m["end"], int)]
        if not matches:
            return [f"InterPro domain matches for {accession}: none"], INTERPRO_PROTEIN_URL.format(accession=accession)

        def line(m):
            kind = f", {m['type']}" if m["type"] and m["type"] != "domain" else ""
            return f"{m['db']} {m['acc']} {m['name']} ({m['start']}-{m['end']}{kind})"

        lines = []
        # Family-level entries span most of the protein and say what the
        # protein is, not where a domain lies; listed apart from the domains.
        families = [m for m in matches if m["type"] == "family"]
        matches = [m for m in matches if m["type"] != "family"]
        if families:
            lines.append("Protein family (InterPro): " + "; ".join(line(m) for m in families[:3]))
        if not matches:
            return lines + [f"InterPro domain matches for {accession}: none below family level"], INTERPRO_PROTEIN_URL.format(accession=accession)
        if protein_position is None:
            shown = sorted(matches, key=lambda m: (m["start"], m["db"]))[:INTERPRO_MAX_LISTED]
            lines.append(f"InterPro domain matches for {accession} (source, name, residues): " + "; ".join(line(m) for m in shown))
        else:
            cover = sorted((m for m in matches if m["start"] <= protein_position <= m["end"]), key=lambda m: (m["end"] - m["start"]))
            if cover:
                lines.append(
                    f"InterPro matches covering position {protein_position} (source, name, residues): "
                    + "; ".join(line(m) for m in cover[:INTERPRO_MAX_LISTED])
                )
            else:
                nearest = sorted(matches, key=lambda m: min(abs(m["start"] - protein_position), abs(m["end"] - protein_position)))[:3]
                lines.append(
                    f"InterPro matches covering position {protein_position}: none; nearest: "
                    + "; ".join(line(m) for m in nearest)
                )
        return lines, INTERPRO_PROTEIN_URL.format(accession=accession)
