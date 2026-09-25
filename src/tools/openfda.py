"""OpenFDA Drug Label API tool."""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord

FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
FDA_LABEL_WEB = "https://labels.fda.gov"

TIMEOUT = 15


class OpenFDATool(BaseTool):
    """Search FDA drug labels for approved indications and safety info."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_fda_labels",
                "description": (
                    "Search FDA-approved drug labels for indications, "
                    "contraindications, warnings, and dosing information. "
                    "Useful for verifying whether a drug is approved for a "
                    "specific condition or patient population."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug": {
                            "type": "string",
                            "description": (
                                "Drug generic or brand name "
                                "(e.g., 'olaparib', 'Trikafta', 'osimertinib')"
                            ),
                        },
                        "indication": {
                            "type": "string",
                            "description": (
                                "Condition or gene to search within indications "
                                "(e.g., 'BRCA1', 'cystic fibrosis', 'EGFR')"
                            ),
                        },
                    },
                    "required": ["drug"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        drug = kwargs.get("drug", "")
        indication = kwargs.get("indication", "")
        evidence: list[EvidenceRecord] = []

        try:
            # Search by generic name first, then brand name
            results = self._search_drug(drug)

            if not results:
                return (
                    f"No FDA drug labels found for: {drug}",
                    evidence,
                )

            # Process top result
            label = results[0]
            openfda = label.get("openfda", {})

            # Build evidence URL
            spl_id = label.get("id", "") or label.get("set_id", "")
            if spl_id:
                url = f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={requests.utils.quote(drug)}"
            else:
                url = f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={requests.utils.quote(drug)}"
            evidence.append(EvidenceRecord(url=url))

            # Extract key sections
            sections = []

            # Drug names
            generic_names = openfda.get("generic_name", [])
            brand_names = openfda.get("brand_name", [])
            if generic_names or brand_names:
                names = ", ".join(generic_names[:3]) or ", ".join(brand_names[:3])
                sections.append(f"Drug: {names}")

            # Indications
            indications_text = label.get("indications_and_usage", [""])[0]
            if indications_text:
                # If user specified an indication filter, check relevance
                if indication:
                    indication_lower = indication.lower()
                    if indication_lower in indications_text.lower():
                        sections.append(f"Indications (matches '{indication}'): {indications_text[:600]}")
                    else:
                        sections.append(f"Indications ('{indication}' NOT found in label): {indications_text[:600]}")
                else:
                    sections.append(f"Indications: {indications_text[:600]}")

            # Contraindications
            contras = label.get("contraindications", [""])[0]
            if contras:
                sections.append(f"Contraindications: {contras[:300]}")

            # Warnings
            warnings = label.get("warnings_and_cautions", label.get("warnings", [""]))[0]
            if warnings and len(warnings) > 10:
                sections.append(f"Warnings: {warnings[:300]}")

            # Dosing
            dosage = label.get("dosage_and_administration", [""])[0]
            if dosage:
                sections.append(f"Dosing: {dosage[:300]}")

            if not sections:
                return f"FDA label found for {drug} but no relevant sections extracted.", evidence

            summary = f"FDA Drug Label for {drug}:\n\n" + "\n\n".join(sections)
            return summary[:2000], evidence

        except requests.RequestException as e:
            return f"OpenFDA API error: {e}", evidence

    def _search_drug(self, drug: str) -> list[dict]:
        """Search openFDA for a drug by name."""
        # Try generic name first
        for field in ["openfda.generic_name", "openfda.brand_name"]:
            try:
                resp = requests.get(
                    FDA_LABEL_URL,
                    params={
                        "search": f'{field}:"{drug}"',
                        "limit": 1,
                    },
                    timeout=TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        return results
            except requests.RequestException:
                continue

        # Fallback: broad search
        try:
            resp = requests.get(
                FDA_LABEL_URL,
                params={
                    "search": drug,
                    "limit": 1,
                },
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json().get("results", [])
        except requests.RequestException:
            pass

        return []
