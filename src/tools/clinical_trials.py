"""ClinicalTrials.gov v2 API tool."""

from __future__ import annotations

import logging
import re

import requests

from .base import BaseTool, EvidenceRecord

logger = logging.getLogger(__name__)

CT_API_URL = "https://clinicaltrials.gov/api/v2/studies"
CT_WEB_URL = "https://clinicaltrials.gov/study/{nct_id}"

# Overall-status values the v2 API accepts, upper case, comma-joined. Live
# check 2026-10-11: spaces after commas and lower case are rejected with 400.
OVERALL_STATUSES = {
    "ACTIVE_NOT_RECRUITING", "COMPLETED", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING",
    "RECRUITING", "SUSPENDED", "TERMINATED", "WITHDRAWN", "AVAILABLE", "NO_LONGER_AVAILABLE",
    "TEMPORARILY_NOT_AVAILABLE", "APPROVED_FOR_MARKETING", "WITHHELD", "UNKNOWN",
}

TIMEOUT = 15


class ClinicalTrialsTool(BaseTool):
    """Search ClinicalTrials.gov for relevant clinical trials."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_clinical_trials",
                "description": (
                    "A registry of clinical trials, providing information on ongoing "
                    "or completed studies involving genetic conditions and therapies. "
                    "Two uses. Search (query): returns matching trials with NCT ID, "
                    "title, status, phase, age range, sex, interventions and a short "
                    "summary; the query is matched literally against trial records, so "
                    "use the gene symbol or the disease name alone (e.g. 'KCNT1'), never "
                    "a sentence or a list of symptoms. Fetch (nct_id): returns one "
                    "trial in full, including its complete inclusion and exclusion "
                    "criteria, for judging whether this patient's age, findings, labs "
                    "and treatment history qualify. Search to find candidates, then "
                    "fetch each candidate before stating eligibility."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nct_id": {
                            "type": "string",
                            "description": "Trial identifier to fetch in full, e.g. NCT05402384; other arguments are then ignored",
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "Gene symbol or disease name, one or two terms "
                                "(e.g., 'KCNT1' or 'propionic acidemia')"
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
                                "Filter by trial status, one or more of RECRUITING, "
                                "NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, "
                                "ENROLLING_BY_INVITATION, COMPLETED, TERMINATED; "
                                "separate several with commas. Leave empty for all."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    @staticmethod
    def _normalise_statuses(text: str) -> tuple[list[str], list[str]]:
        """Split a status list however it was written; return (accepted, rejected)."""
        raw = [t.strip().upper().replace(" ", "_") for t in re.split(r"[,|;]|\s{2,}", text) if t.strip()]
        # A single space-separated list ("RECRUITING NOT_YET_RECRUITING") also splits on spaces.
        if len(raw) == 1 and " " in text.strip():
            raw = [t.strip().upper() for t in text.split() if t.strip()]
        accepted = [r for r in raw if r in OVERALL_STATUSES]
        rejected = [r for r in raw if r not in OVERALL_STATUSES]
        return accepted, rejected

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        nct_id = (kwargs.get("nct_id") or "").strip().upper()
        if nct_id:
            return self._fetch(nct_id)
        query = kwargs.get("query", "")
        if not query and not kwargs.get("condition") and not kwargs.get("intervention"):
            return "ClinicalTrials.gov: give a query to search, or an nct_id to fetch.", []
        condition = kwargs.get("condition", "")
        intervention = kwargs.get("intervention", "")
        status_filter = kwargs.get("status_filter", "")
        evidence: list[EvidenceRecord] = []

        try:
            params: dict = {
                "format": "json",
                "pageSize": 5,
                # Age range, sex and interventions come back in the search itself.
                "fields": "NCTId,BriefTitle,OverallStatus,Phase,BriefSummary,MinimumAge,MaximumAge,Sex,"
                          "InterventionName,InterventionType,Condition",
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
                statuses, rejected = self._normalise_statuses(status_filter)
                if rejected:
                    logger.warning(f"ClinicalTrials.gov status filter: dropped unknown values {rejected} from {status_filter!r}")
                if statuses:
                    # The API wants upper-case values joined by commas with no spaces;
                    # "RECRUITING, NOT_YET_RECRUITING" was rejected with HTTP 400 (run 2).
                    params["filter.overallStatus"] = ",".join(statuses)

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
                    f"Eligible ages: {self._age_range(protocol)} | Sex: {protocol.get('eligibilityModule', {}).get('sex') or 'not stated'}\n"
                    f"Interventions: {self._interventions(protocol)}\n"
                    f"URL: {url}"
                )
                if brief_summary:
                    entry += f"\nSummary: {brief_summary}"
                results.append(entry)

            summary = "\n\n".join(results)
            return summary[:4000], evidence

        except requests.RequestException as e:
            return f"ClinicalTrials.gov API error: {e}", evidence

    def _fetch(self, nct_id: str) -> tuple[str, list[EvidenceRecord]]:
        """One trial in full, with its complete eligibility criteria."""
        evidence: list[EvidenceRecord] = []
        try:
            resp = requests.get(f"{CT_API_URL}/{nct_id}", params={"format": "json"}, timeout=TIMEOUT)
            if resp.status_code == 404:
                logger.warning(f"ClinicalTrials.gov has no study {nct_id}")
                return f"ClinicalTrials.gov has no study with ID {nct_id}.", evidence
            resp.raise_for_status()
            protocol = resp.json().get("protocolSection", {})
        except requests.RequestException as e:
            return f"ClinicalTrials.gov API error: {e}", evidence
        ident = protocol.get("identificationModule", {})
        status = protocol.get("statusModule", {})
        elig = protocol.get("eligibilityModule", {})
        url = CT_WEB_URL.format(nct_id=nct_id)
        evidence.append(EvidenceRecord(url=url))
        criteria = (elig.get("eligibilityCriteria") or "").strip() or "not stated"
        lines = [
            f"NCT ID: {nct_id}",
            f"Title: {ident.get('briefTitle') or ident.get('officialTitle') or 'No title'}",
            f"Status: {status.get('overallStatus', 'Unknown')}",
            f"Phase: {', '.join(protocol.get('designModule', {}).get('phases') or []) or 'Not specified'}",
            f"Conditions: {', '.join(protocol.get('conditionsModule', {}).get('conditions') or []) or 'not stated'}",
            f"Interventions: {self._interventions(protocol)}",
            f"Eligible ages: {self._age_range(protocol)} | Sex: {elig.get('sex') or 'not stated'}",
            f"URL: {url}",
            "Eligibility criteria:",
            criteria if len(criteria) <= 3000 else criteria[:2997] + "...",
        ]
        return "\n".join(lines), evidence

    @staticmethod
    def _age_range(protocol: dict) -> str:
        elig = protocol.get("eligibilityModule", {})
        lo, hi = elig.get("minimumAge"), elig.get("maximumAge")
        if not lo and not hi:
            return "not stated"
        return f"{lo or 'no minimum'} to {hi or 'no maximum'}"

    @staticmethod
    def _interventions(protocol: dict) -> str:
        items = protocol.get("armsInterventionsModule", {}).get("interventions") or []
        return "; ".join(f"{i.get('name')} ({i.get('type', '').lower()})" for i in items if i.get("name")) or "not stated"
