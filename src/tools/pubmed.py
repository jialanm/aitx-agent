"""PubMed tool using NCBI E-utilities: search, then fetch the articles' abstracts."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import requests

from .base import BaseTool, EvidenceRecord, ncbi_get, ncbi_params, redact_ncbi_key

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_WEB = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

TIMEOUT = 30
MAX_RESULTS = 10
# An abstract is the unit the model must read to judge relevance; titles
# alone could not say which drug suits a loss-of-function variant (run 2,
# AITX-00012). Each abstract is kept whole up to this length.
MAX_ABSTRACT_CHARS = 1500
MAX_SUMMARY_CHARS = 9000


class PubMedTool(BaseTool):
    """Search PubMed and return the matching articles with their abstracts."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_pubmed",
                "description": (
                    "A searchable database of biomedical literature, including "
                    "research articles relevant to genetics and molecular biology. "
                    "Returns up to ten articles ranked by PubMed relevance, each "
                    "with title, first author, journal, year, PMID and its "
                    "abstract, so the article's finding can be read and judged. "
                    "Results are only as specific as the query: name the gene, "
                    "the variant consequence or mechanism, and the therapy or "
                    "question, not a general topic. Use it for evidence on a "
                    "therapy's effect, human treatment precedent for off-label "
                    "use, and mechanistic rationale for repurposing. It is the "
                    "only source here that covers therapies approved outside the "
                    "United States or still investigational: when GeneReviews and "
                    "the FDA labels name no therapy for the variant's consequence, "
                    "search here by gene and consequence before answering None."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Specific PubMed query naming the gene, the variant "
                                "consequence and the therapy, e.g. 'CFTR F508del "
                                "homozygous elexacaftor', not 'CFTR treatment'"
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of articles (default 5, max 10)",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        query = kwargs.get("query", "")
        max_results = max(1, min(int(kwargs.get("max_results") or 5), MAX_RESULTS))
        evidence: list[EvidenceRecord] = []

        try:
            resp = ncbi_get(
                label="PubMed", url=ESEARCH_URL,
                params=ncbi_params(db="pubmed", term=query, retmode="json", retmax=max_results, sort="relevance"),
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            search = resp.json().get("esearchresult", {})
            id_list = search.get("idlist", [])
            if not id_list:
                return f"No PubMed articles found for: {query}", evidence
            total = search.get("count", "0")

            resp2 = ncbi_get(
                label="PubMed", url=EFETCH_URL,
                params=ncbi_params(db="pubmed", id=",".join(id_list), retmode="xml"),
                timeout=TIMEOUT,
            )
            resp2.raise_for_status()
            articles = self._parse(resp2.text)
            if not articles:
                logger.warning(f"PubMed efetch returned no parsable articles for ids {id_list}")
                return f"PubMed search '{query}' found {total} articles but their records could not be read.", evidence

            parts = [f"PubMed search: {query}\nTotal results: {total}; showing {len(articles)} with abstracts\n"]
            for a in articles:
                url = PUBMED_WEB.format(pmid=a["pmid"])
                evidence.append(EvidenceRecord(url=url))
                abstract = a["abstract"] or "(no abstract in PubMed)"
                if len(abstract) > MAX_ABSTRACT_CHARS:
                    abstract = abstract[:MAX_ABSTRACT_CHARS - 3] + "..."
                parts.append(
                    f"PMID: {a['pmid']}\nTitle: {a['title']}\nAuthors: {a['authors']}\n"
                    f"Journal: {a['journal']} ({a['year']})\nURL: {url}\nAbstract: {abstract}"
                )
            summary = "\n\n".join(parts)
            if len(summary) > MAX_SUMMARY_CHARS:
                logger.warning(f"PubMed summary for {query!r} cut from {len(summary)} to {MAX_SUMMARY_CHARS} chars")
                summary = summary[:MAX_SUMMARY_CHARS - 3] + "..."
            return summary, evidence

        except requests.RequestException as e:
            return redact_ncbi_key(f"PubMed API error: {e}"), evidence

    @staticmethod
    def _parse(xml_text: str) -> list[dict]:
        """Articles from an efetch XML reply, in reply order."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"PubMed efetch XML did not parse: {e}")
            return []
        articles = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//MedlineCitation/PMID") or ""
            title_el = art.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else "(no title)"
            # Structured abstracts carry a Label per part (BACKGROUND, RESULTS, ...).
            pieces = []
            for part in art.findall(".//Abstract/AbstractText"):
                text = "".join(part.itertext()).strip()
                label = part.get("Label")
                pieces.append(f"{label}: {text}" if label else text)
            authors = art.findall(".//AuthorList/Author")
            first = ""
            if authors:
                last, init = authors[0].findtext("LastName") or "", authors[0].findtext("Initials") or ""
                first = f"{last} {init}".strip() or (authors[0].findtext("CollectiveName") or "")
            author_str = f"{first} et al." if len(authors) > 1 and first else first
            articles.append({
                "pmid": pmid, "title": title, "abstract": " ".join(pieces), "authors": author_str,
                "journal": art.findtext(".//Journal/Title") or art.findtext(".//MedlineTA") or "",
                "year": art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or "",
            })
        return articles
