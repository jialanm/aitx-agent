"""ClinicalTrials.gov v2 API tool."""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord

CT_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CT_WEB_URL = "https://clinicaltrials.gov/study/{nct_id}"

TIMEOUT = 15


class ClinicalTrialsTool(BaseTool):
    """Search ClinicalTrials.gov for relevant clinical trials."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_clinical_trials",
                "description": (
                    "Search ClinicalTrials.gov for clinical trials related to a "
                    "gene, condition, or intervention. Returns trial IDs, titles, "
                    "status, and eligibility information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Search query combining gene, condition, and/or "
                                "intervention (e.g., 'BRAF melanoma vemurafenib')"
                            ),
                        },
                        "condition": {
                            "type": "string",
                            "description": "Disease or condition name",
                        },
                        "intervention": {
                            "type": "string",
                            "description": "Drug or therapy name",
                        },
                        "status_filter": {
                            "type": "string",
                            "description": (
                                "Filter by trial status. Options: "
                                "RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, "
                                "NOT_YET_RECRUITING. Leave empty for all."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        query = kwargs.get("query", "")
        condition = kwargs.get("condition", "")
        intervention = kwargs.get("intervention", "")
        status_filter = kwargs.get("status_filter", "")
        evidence: list[EvidenceRecord] = []

        try:
            params: dict = {
                "format": "json",
                "pageSize": 5,
            }

            # Build query
            query_parts = []
            if query:
                query_parts.append(query)
            if condition:
                params["query.cond"] = condition
            if intervention:
                params["query.intr"] = intervention
            if query_parts:
                params["query.term"] = " ".join(query_parts)

            if status_filter:
                params["filter.overallStatus"] = status_filter

            resp = requests.get(CT_API_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            studies = data.get("studies", [])
            if not studies:
                # The search itself is evidence — return the search URL
                search_query = f"{query} {condition} {intervention}".strip()
                search_url = f"https://clinicaltrials.gov/search?term={requests.utils.quote(search_query)}"
                evidence.append(EvidenceRecord(url=search_url))
                return (
                    f"No clinical trials found for query: {search_query}. "
                    f"Search URL: {search_url}",
                    evidence,
                )

            total = data.get("totalCount", len(studies))
            results = [f"Found {total} clinical trials. Showing top {min(5, total)}:\n"]

            for study in studies[:5]:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                desc_module = protocol.get("descriptionModule", {})

                nct_id = id_module.get("nctId", "Unknown")
                title = id_module.get("briefTitle", "No title")
                status = status_module.get("overallStatus", "Unknown")
                phase_info = protocol.get("designModule", {}).get("phases", [])
                phase = ", ".join(phase_info) if phase_info else "Not specified"

                brief_summary = desc_module.get("briefSummary", "")
                if brief_summary and len(brief_summary) > 200:
                    brief_summary = brief_summary[:200] + "..."

                url = CT_WEB_URL.format(nct_id=nct_id)
                evidence.append(EvidenceRecord(url=url))

                entry = (
                    f"NCT ID: {nct_id}\n"
                    f"Title: {title}\n"
                    f"Status: {status}\n"
                    f"Phase: {phase}\n"
                    f"URL: {url}"
                )
                if brief_summary:
                    entry += f"\nSummary: {brief_summary}"
                results.append(entry)

            summary = "\n\n".join(results)
            return summary[:2000], evidence

        except requests.RequestException as e:
            return f"ClinicalTrials.gov API error: {e}", evidence
