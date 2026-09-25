"""PubMed tool using NCBI E-utilities."""

from __future__ import annotations

import requests

from .base import BaseTool, EvidenceRecord

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_WEB = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

TIMEOUT = 15


class PubMedTool(BaseTool):
    """Search PubMed for biomedical literature."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_pubmed",
                "description": (
                    "Search PubMed for biomedical research articles. Returns "
                    "article titles, authors, publication dates, and PMIDs. "
                    "Use for finding evidence about treatments, drug efficacy, "
                    "variant pathogenicity, and clinical outcomes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "PubMed search query (e.g., "
                                "'CFTR F508del Trikafta treatment')"
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default 5, max 10)",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        query = kwargs.get("query", "")
        max_results = min(kwargs.get("max_results", 5), 10)
        evidence: list[EvidenceRecord] = []

        try:
            # Step 1: search for PMIDs
            resp = requests.get(
                ESEARCH_URL,
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": max_results,
                    "sort": "relevance",
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            search_data = resp.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return f"No PubMed articles found for: {query}", evidence

            total = search_data.get("esearchresult", {}).get("count", "0")

            # Step 2: get summaries
            resp2 = requests.get(
                ESUMMARY_URL,
                params={
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "json",
                },
                timeout=TIMEOUT,
            )
            resp2.raise_for_status()
            summary_data = resp2.json().get("result", {})

            uids = summary_data.get("uids", [])
            results = [f"PubMed search: {query}\nTotal results: {total}\n"]

            for uid in uids:
                entry = summary_data.get(uid, {})
                title = entry.get("title", "No title")
                authors = entry.get("authors", [])
                author_str = ""
                if authors:
                    first_author = authors[0].get("name", "")
                    author_str = f"{first_author} et al." if len(authors) > 1 else first_author
                pub_date = entry.get("pubdate", "")
                source = entry.get("source", "")

                url = PUBMED_WEB.format(pmid=uid)
                evidence.append(EvidenceRecord(url=url))

                results.append(
                    f"PMID: {uid}\n"
                    f"Title: {title}\n"
                    f"Authors: {author_str}\n"
                    f"Journal: {source}\n"
                    f"Date: {pub_date}\n"
                    f"URL: {url}"
                )

            summary = "\n\n".join(results)
            return summary[:2000], evidence

        except requests.RequestException as e:
            return f"PubMed API error: {e}", evidence
