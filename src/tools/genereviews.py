"""GeneReviews tool: the chapters for a gene, and a chapter's management text.

GeneReviews is NCBI Bookshelf's expert-written, peer-reviewed reference on
rare genetic diseases, one chapter per disease. Two public index files on
NCBI's FTP site map gene symbols to chapters and chapters to their title,
NBK accession and PMID, so chapter selection needs no search. Bookshelf's
pages refuse scripted access, but each chapter's PubMed record carries a
structured abstract with CLINICAL CHARACTERISTICS, DIAGNOSIS/TESTING,
MANAGEMENT and GENETIC COUNSELING paragraphs, and that is the text served.
"""

from __future__ import annotations

import logging
import re

import requests

from .base import BaseTool, EvidenceRecord, ncbi_get, ncbi_params, redact_ncbi_key

logger = logging.getLogger(__name__)

GR_FTP = "https://ftp.ncbi.nlm.nih.gov/pub/GeneReviews/"
GENE_INDEX = GR_FTP + "NBKid_shortname_genesymbol.txt"
TITLE_INDEX = GR_FTP + "GRtitle_shortname_NBKid.txt"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BOOKSHELF_URL = "https://www.ncbi.nlm.nih.gov/books/{nbk}/"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
TIMEOUT = 60

# Section headings as they appear in the structured abstract.
SECTIONS = {
    "clinical": "CLINICAL CHARACTERISTICS",
    "diagnosis": "DIAGNOSIS/TESTING",
    "management": "MANAGEMENT",
    "counseling": "GENETIC COUNSELING",
}
_HEADINGS = "|".join(re.escape(h) for h in SECTIONS.values())


class GeneReviewsTool(BaseTool):
    """List a gene's GeneReviews chapters, or return one chapter's management text."""

    def __init__(self) -> None:
        self._gene_to_nbks: dict[str, list[str]] | None = None
        self._chapters: dict[str, dict] | None = None   # NBK -> {title, pmid, shortname}

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_genereviews",
                "description": (
                    "GeneReviews: expert-authored, peer-reviewed overviews of "
                    "inherited disorders, one chapter per disease, covering "
                    "diagnosis, management and genetic counseling. Called with "
                    "a gene only, it returns every chapter indexed to that gene "
                    "(accession, title, PMID); a gene can have several, for "
                    "different diseases, so choose the chapter that matches the "
                    "patient's phenotype and call again with its accession. With "
                    "an accession, it returns the chapter's MANAGEMENT text in "
                    "full: targeted therapies and which variants qualify, "
                    "treatment of manifestations, agents to avoid, and therapies "
                    "under investigation; ask for 'clinical', 'diagnosis', "
                    "'counseling' or 'all' for the other sections. A gene with "
                    "one chapter is fetched at once. Use it for Tier 1 targeted "
                    "and supportive therapies and contraindications. Coverage: "
                    "the management text reflects United States practice, so a "
                    "therapy approved only in another country, withdrawn, or "
                    "still investigational may not be named; when no listed "
                    "therapy matches the variant's consequence, search PubMed by "
                    "that consequence before answering None."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., CFTR, SCN1A)",
                        },
                        "nbk": {
                            "type": "string",
                            "description": (
                                "Chapter accession from the list, e.g. NBK1250. "
                                "Give it to fetch that chapter's text."
                            ),
                        },
                        "section": {
                            "type": "string",
                            "description": (
                                "Which part of the chapter: management (default), "
                                "clinical, diagnosis, counseling, or all"
                            ),
                        },
                    },
                    "required": ["gene"],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        gene = (kwargs.get("gene") or "").strip().upper()
        nbk = (kwargs.get("nbk") or "").strip().upper()
        section = (kwargs.get("section") or "management").strip().lower()
        evidence: list[EvidenceRecord] = []
        if not gene and not nbk:
            return "GeneReviews: no gene or chapter given.", evidence

        failure = self._load_index()
        if failure:
            return f"GeneReviews returned nothing: {failure}", evidence

        if not nbk:
            nbks = self._gene_to_nbks.get(gene, [])
            if not nbks:
                return f"GeneReviews has no chapter indexed to gene {gene}.", evidence
            if len(nbks) > 1:
                # A menu, not a record consulted: no evidence links, so nothing is
                # cited until a chapter is fetched (run 3: 6 of 11 unsupported
                # citations were these list links).
                lines = [f"GeneReviews chapters indexed to {gene} ({len(nbks)}); call again with the accession that matches the patient's phenotype:"]
                for n in nbks:
                    ch = self._chapters.get(n, {})
                    lines.append(f"- {n}: {ch.get('title', '(title not in index)')} (PMID {ch.get('pmid', '?')})")
                return "\n".join(lines), evidence
            nbk = nbks[0]

        chapter = self._chapters.get(nbk)
        if chapter is None:
            return f"GeneReviews: {nbk} is not a chapter accession in the index.", evidence
        pmid = chapter.get("pmid", "")
        if not pmid:
            logger.warning(f"GeneReviews chapter {nbk} has no PMID in the index")
            return f"GeneReviews chapter {nbk} ({chapter.get('title')}) has no PubMed record to fetch text from.", evidence

        text, failure = self._abstract(pmid)
        if text is None:
            return f"GeneReviews returned nothing for {nbk}: {failure}", evidence
        evidence.append(EvidenceRecord(url=BOOKSHELF_URL.format(nbk=nbk)))

        header = f"GeneReviews: {chapter.get('title')} ({nbk}, PMID {pmid}, {BOOKSHELF_URL.format(nbk=nbk)})"
        sections = self._split(text)
        if not sections:
            # Overview chapters carry a short unstructured abstract.
            return header + "\nThis chapter's PubMed record has no structured sections; full abstract:\n" + text.strip(), evidence
        if section == "all":
            body = "\n\n".join(f"{SECTIONS[k]}: {v}" for k, v in sections.items())
        elif section in sections:
            body = f"{SECTIONS[section]}: {sections[section]}"
        elif section in SECTIONS:
            logger.warning(f"GeneReviews {nbk}: section {SECTIONS[section]} not in abstract; has {list(sections)}")
            body = f"Section {SECTIONS[section]} not present in this chapter's record. Sections present: " + ", ".join(SECTIONS[k] for k in sections)
        else:
            body = f"Unknown section {section!r}; use management, clinical, diagnosis, counseling or all."
        return header + "\n" + body, evidence

    def _load_index(self) -> str:
        """Both index files, downloaded once per process. Returns '' or a failure reason."""
        if self._gene_to_nbks is not None:
            return ""
        gene_text, failure = self._get(GENE_INDEX, {})
        if gene_text is None:
            return f"gene index unavailable ({failure})"
        title_text, failure = self._get(TITLE_INDEX, {})
        if title_text is None:
            return f"title index unavailable ({failure})"
        gene_to_nbks: dict[str, list[str]] = {}
        for line in gene_text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            gene_to_nbks.setdefault(parts[2].strip().upper(), []).append(parts[0].strip())
        chapters: dict[str, dict] = {}
        for line in title_text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chapters[parts[2].strip()] = {
                "shortname": parts[0].strip(), "title": parts[1].strip(),
                "pmid": parts[3].strip() if len(parts) > 3 else "",
            }
        if not gene_to_nbks or not chapters:
            logger.warning("GeneReviews index files parsed to nothing; format changed?")
            return "index files parsed to nothing"
        self._gene_to_nbks, self._chapters = gene_to_nbks, chapters
        return ""

    def _abstract(self, pmid: str) -> tuple[str | None, str]:
        text, failure = self._get(EFETCH_URL, ncbi_params(db="pubmed", id=pmid, rettype="abstract", retmode="text"))
        if text is None:
            return None, failure
        if not text.strip():
            logger.warning(f"GeneReviews: empty abstract for PMID {pmid}")
            return None, f"empty abstract for PMID {pmid}"
        return text, ""

    @staticmethod
    def _split(text: str) -> dict[str, str]:
        """The structured abstract's sections, keyed by SECTIONS key, in order."""
        found: dict[str, str] = {}
        matches = list(re.finditer(rf"({_HEADINGS}):", text))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = m.group(1)
            key = next(k for k, v in SECTIONS.items() if v == heading)
            body = text[m.end():end]
            body = re.sub(r"\s*\n\s*", " ", body).strip()
            # The abstract ends with copyright and author lines after the last section.
            body = re.split(r"\s*(?:Copyright|©)\s", body)[0].strip()
            found[key] = body
        return found

    def _get(self, url: str, params: dict) -> tuple[str | None, str]:
        """GET text through the shared NCBI retry policy; failures reported once."""
        try:
            resp = ncbi_get(url=url, params=params, timeout=TIMEOUT, label="GeneReviews")
        except requests.RequestException as e:
            msg = redact_ncbi_key(str(e))
            logger.warning(f"GeneReviews request failed: {url}: {msg}")
            return None, f"request failed ({msg})"
        if resp.status_code != 200:
            logger.warning(f"GeneReviews HTTP {resp.status_code}: {url}: {resp.text[:200]!r}")
            return None, f"HTTP {resp.status_code}"
        return resp.text, ""
