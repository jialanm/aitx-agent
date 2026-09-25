"""GeneReviews tool via NCBI PubMed + Bookshelf."""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord, ncbi_params, redact_ncbi_key

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

TIMEOUT = 15

# Sections of interest for therapeutic questions
THERAPY_KEYWORDS = [
    "management",
    "targeted therap",
    "treatment",
    "pharmacologic",
    "therapies",
    "agents/circumstances to avoid",
    "avoid",
    "prevention",
    "surveillance",
]


class GeneReviewsTool(BaseTool):
    """Search GeneReviews for gene-specific management information."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_genereviews",
                "description": (
                    "Search NCBI GeneReviews for expert-authored, peer-reviewed "
                    "disease descriptions including management and treatment "
                    "recommendations for genetic conditions. Returns relevant "
                    "management sections."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., CFTR, SCN1A)",
                        },
                        "condition": {
                            "type": "string",
                            "description": (
                                "Disease or condition name "
                                "(e.g., cystic fibrosis, Dravet syndrome)"
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
            # Strategy 1: Search PubMed for GeneReviews entries
            search_term = f"{gene} AND GeneReviews[book]"
            if condition:
                search_term = f"({gene} OR {condition}) AND GeneReviews[book]"

            resp = requests.get(
                ESEARCH_URL,
                params=ncbi_params(
                    db="pubmed",
                    term=search_term,
                    retmode="json",
                    retmax=3,
                ),
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            search_data = resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if id_list:
                return self._process_pubmed_results(id_list, gene, evidence)

            # Strategy 2: Broader PubMed search
            if condition:
                resp2 = requests.get(
                    ESEARCH_URL,
                    params=ncbi_params(
                        db="pubmed",
                        term=f"{condition} AND GeneReviews[book]",
                        retmode="json",
                        retmax=3,
                    ),
                    timeout=TIMEOUT,
                )
                resp2.raise_for_status()
                id_list = resp2.json().get("esearchresult", {}).get("idlist", [])
                if id_list:
                    return self._process_pubmed_results(id_list, gene, evidence)

            return (
                f"No GeneReviews entries found for gene {gene}.",
                evidence,
            )

        except requests.RequestException as e:
            return redact_ncbi_key(f"GeneReviews API error: {e}"), evidence

    def _process_pubmed_results(
        self,
        pmids: list[str],
        gene: str,
        evidence: list[EvidenceRecord],
    ) -> tuple[str, list[EvidenceRecord]]:
        """Process PubMed IDs to get GeneReviews content."""
        # Get PubMed summaries for titles
        resp = requests.get(
            ESUMMARY_URL,
            params=ncbi_params(
                db="pubmed",
                id=",".join(pmids[:3]),
                retmode="json",
            ),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        summary_data = resp.json().get("result", {})

        # Try to get linked books (NBK IDs)
        nbk_urls = self._get_bookshelf_links(pmids[:3])

        # Score and rank articles by title relevance to the gene
        uids = summary_data.get("uids", [])
        scored_uids = []
        for uid in uids[:3]:
            entry = summary_data.get(uid, {})
            title = entry.get("title", "Unknown")
            score = self._title_relevance_score(title, gene)
            scored_uids.append((uid, title, score))

        # Sort by relevance (highest first)
        scored_uids.sort(key=lambda x: x[2], reverse=True)

        results = []
        best_therapy_text = ""

        for uid, title, score in scored_uids:
            # Use NBK link if available, otherwise specific PubMed URL
            if uid in nbk_urls:
                url = nbk_urls[uid]
            else:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"

            evidence.append(EvidenceRecord(url=url))
            results.append(f"GeneReviews: {title}\nPMID: {uid}\nURL: {url}")

        # Fetch abstract ONLY for the most relevant article (not all)
        if scored_uids:
            best_uid = scored_uids[0][0]
            try:
                resp2 = requests.get(
                    EFETCH_URL,
                    params=ncbi_params(
                        db="pubmed",
                        id=best_uid,
                        rettype="abstract",
                        retmode="text",
                    ),
                    timeout=TIMEOUT,
                )
                resp2.raise_for_status()
                best_therapy_text = self._extract_therapy_content(resp2.text)
            except requests.RequestException:
                pass

        # Build final text
        header = f"GeneReviews results for {gene}:\n\n"
        entries = "\n\n".join(results)

        full_text = header + entries
        if best_therapy_text:
            full_text += f"\n\n## Relevant Content\n{best_therapy_text}"

        return full_text[:2000], evidence

    @staticmethod
    def _title_relevance_score(title: str, gene: str) -> int:
        """Score how relevant a GeneReviews article title is to the gene.

        Higher score = more relevant. Penalizes articles that mention the
        gene only tangentially (e.g., Beckwith-Wiedemann mentions KCNQ1
        only because of the 11p15.5 locus).
        """
        title_lower = title.lower()
        gene_lower = gene.lower()
        score = 0

        # Direct gene name in title is a strong signal
        if gene_lower in title_lower:
            score += 10

        # Common disease-gene associations boost relevance
        # Articles with "Overview" or "Syndrome" in the title that also
        # match the gene are likely the primary entry
        if "overview" in title_lower:
            score += 3

        # Penalize titles that are clearly about a different condition
        # (gene appears in the search results but the article is about
        # something else, e.g., imprinting regions)
        if gene_lower not in title_lower:
            score -= 5

        return score

    def _get_bookshelf_links(self, pmids: list[str]) -> dict[str, str]:
        """Try to get Bookshelf NBK links for PubMed IDs."""
        try:
            resp = requests.get(
                ELINK_URL,
                params=ncbi_params(
                    dbfrom="pubmed",
                    db="books",
                    id=",".join(pmids),
                    retmode="json",
                ),
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            result = {}
            for linkset in data.get("linksets", []):
                pmid = str(linkset.get("ids", [None])[0])
                for linksetdb in linkset.get("linksetdbs", []):
                    links = linksetdb.get("links", [])
                    if links:
                        nbk_id = links[0]
                        result[pmid] = f"https://www.ncbi.nlm.nih.gov/books/NBK{nbk_id}/"
            return result
        except Exception:
            return {}

    def _extract_therapy_content(self, text: str) -> str:
        """Extract therapy-relevant lines from abstract text."""
        lines = text.split("\n")
        relevant = []
        capture = False
        for line in lines:
            lower = line.lower().strip()
            if any(kw in lower for kw in THERAPY_KEYWORDS):
                capture = True
            if capture:
                if line.strip():
                    relevant.append(line.strip())
                else:
                    if relevant:
                        capture = False
            if len(relevant) > 20:
                break
        return "\n".join(relevant)
