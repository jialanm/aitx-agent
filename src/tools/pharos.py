"""PHAROS (TCRD) tool: druggability of a gene's protein and the drugs that act on it."""

from __future__ import annotations

import logging
import time

import requests

from .base import BaseTool, EvidenceRecord

logger = logging.getLogger(__name__)

PHAROS_GRAPHQL = "https://pharos-api.ncats.io/graphql"
PHAROS_WEB = "https://pharos.nih.gov/targets/{uniprot}"
TIMEOUT = 30
RETRIES = 5
RETRY_INTERVAL_SECONDS = 5

# Target development level, PHAROS's four-step druggability scale.
TDL_MEANING = {
    "Tclin": "an approved drug acts on this protein",
    "Tchem": "potent small molecules bind it, no approved drug",
    "Tbio": "biology known, no potent binder",
    "Tdark": "little known",
}

TARGET_QUERY = """{ target(q:{sym:"%s"}) { sym name uniprot tdl fam description
  ligandCounts { name value }
  ligands(isdrug:true, top:10) { name isdrug actcnt activities(all:false) { type value moa reference } }
  diseases(top:5) { name } } }"""


class PharosTool(BaseTool):
    """Query PHAROS for a target's druggability, known drugs and linked diseases."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "query_pharos",
                "description": (
                    "PHAROS, the interface to the Target Central Resource Database "
                    "(TCRD), on the druggability and function of human proteins. "
                    "For a gene it returns the target development level (Tclin: an "
                    "approved drug acts on it; Tchem: potent binders exist; Tbio: "
                    "biology known only; Tdark: little known), the protein family, "
                    "a one-paragraph description, up to ten approved drugs that act "
                    "on the protein with their mechanism (inhibitor, blocker, "
                    "antisense, modulator), and the diseases most associated with "
                    "it. Use it for drug repurposing: whether the protein is a "
                    "proven drug target and which existing drugs touch it. It does "
                    "not say whether a drug suits this patient's variant."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., DMD, GRIN2B)",
                        },
                    },
                    "required": ["gene"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        gene = kwargs.get("gene", "")
        evidence: list[EvidenceRecord] = []
        if not gene:
            return "PHAROS: no gene given.", evidence

        body, failure = self._post(TARGET_QUERY % gene)
        if body is None:
            return f"PHAROS returned nothing for {gene}: {failure}", evidence
        if body.get("errors"):
            logger.warning(f"PHAROS query error for {gene}: {body['errors']}")
            return f"PHAROS returned nothing for {gene}: query error {body['errors'][0].get('message', '')[:120]}", evidence
        target = (body.get("data") or {}).get("target")
        if not target:
            return f"PHAROS has no target for gene {gene}.", evidence

        if target.get("uniprot"):
            evidence.append(EvidenceRecord(url=PHAROS_WEB.format(uniprot=target["uniprot"])))
        else:
            logger.warning(f"PHAROS target {gene} has no uniprot accession; no evidence link")

        tdl = target.get("tdl") or "not returned"
        lines = [
            f"PHAROS target {target.get('sym')} ({target.get('name')}, UniProt {target.get('uniprot')}):",
            f"Target development level: {tdl}" + (f" ({TDL_MEANING[tdl]})" if tdl in TDL_MEANING else ""),
            f"Protein family: {target.get('fam') or 'not assigned'}",
        ]
        desc = (target.get("description") or "").strip()
        if desc:
            lines.append("Description: " + (desc if len(desc) <= 500 else desc[:497] + "..."))

        drugs = target.get("ligands") or []
        if drugs:
            lines.append("Approved drugs acting on this target:")
            for lig in drugs:
                acts = lig.get("activities") or []
                moa = next((a.get("moa") for a in acts if a.get("moa")), None)
                lines.append(f"- {lig.get('name')}" + (f": {moa.lower()}" if moa else ": mechanism not annotated"))
        else:
            counts = {c["name"]: c["value"] for c in target.get("ligandCounts") or []}
            lines.append(
                "Approved drugs acting on this target: none in PHAROS"
                + (f" (ligands: {counts})" if counts else "")
            )

        diseases = [d.get("name") for d in target.get("diseases") or [] if d.get("name")]
        if diseases:
            lines.append("Most associated diseases: " + "; ".join(diseases))
        return "\n".join(lines), evidence

    def _post(self, query: str) -> tuple[dict | None, str]:
        """POST the GraphQL query; timeouts and 5xx retried, other failures reported once."""
        last = ""
        for attempt in range(1, RETRIES + 1):
            try:
                resp = requests.post(PHAROS_GRAPHQL, json={"query": query}, timeout=TIMEOUT)
            except requests.Timeout:
                last = "timeout"
                logger.warning(f"PHAROS timeout (attempt {attempt}/{RETRIES})")
            except requests.RequestException as e:
                logger.warning(f"PHAROS request failed: {e}")
                return None, f"request failed ({e})"
            else:
                if resp.status_code < 500:
                    try:
                        return resp.json(), ""
                    except ValueError:
                        logger.warning(f"PHAROS non-JSON reply: HTTP {resp.status_code}: {resp.text[:200]!r}")
                        return None, f"HTTP {resp.status_code}, non-JSON reply"
                last = f"HTTP {resp.status_code}"
                logger.warning(f"PHAROS server error (attempt {attempt}/{RETRIES}): HTTP {resp.status_code}")
            if attempt < RETRIES:
                time.sleep(RETRY_INTERVAL_SECONDS)
        return None, f"{last} after {RETRIES} attempts"
