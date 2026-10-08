"""N1C VARIANT guidelines tool: ASO eligibility rules, with the paper's text on request.

Source: Cheerie D. et al., "Consensus guidelines for assessing eligibility of
pathogenic DNA variants for antisense oligonucleotide treatments", Am J Hum
Genet 112(5), 2025, doi:10.1016/j.ajhg.2025.02.017, PMC12120168, CC BY 4.0.
The step-by-step rules live in the paper's Data S1 and in the open-source
N1C eligibility calculator; the summary below is written from both.
"""

from __future__ import annotations

import logging
import re
import time

import requests

from .base import BaseTool, EvidenceRecord, ncbi_params, redact_ncbi_key

logger = logging.getLogger(__name__)

PMC_ID = "12120168"
DOI_URL = "https://doi.org/10.1016/j.ajhg.2025.02.017"
PMC_URL = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{PMC_ID}/"
CALCULATOR_URL = "http://eligibilitycalculator.n1collaborative.org/"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TIMEOUT = 60
RETRIES = 5
RETRY_INTERVAL_SECONDS = 5
MAX_SECTION_CHARS = 4000

# Paper sections a caller can ask for, by the title in the article XML.
SECTIONS = {
    "introduction": "Introduction",
    "purpose": "Purpose of guidelines",
    "structure": "Guideline structure",
    "classification": "Classification terms",
    "calculator": "Variant eligibility calculator",
    "discussion": "Discussion",
}

# Written from the paper and the calculator's question flow; every rule
# here is stated in one or the other. No rule is inferred.
OVERVIEW = """N1C VARIANT guidelines (Cheerie et al., AJHG 2025) for assessing a pathogenic variant's eligibility for antisense oligonucleotide (ASO) treatment.

Step 0, variant check. The variant must be correctly written (HGVS), confirmed disease-causing, and of a type the guidelines cover: single-nucleotide variants, small indels, one- or multi-exon deletions or duplications, whole-gene deletions or duplications. Excluded: variants in non-coding genes, deletions or duplications spanning several genes, imprinting defects or uniparental disomy, structural rearrangements. Missing or unsure information at any step gives the verdict "unable to assess".

Step 1-2, context. Record the inheritance pattern and the pathomechanism: loss of function, gain of function, dominant negative, or unknown. The pathomechanism decides which strategies apply.

Strategies assessed:
1. Splice correction: for variants with functional evidence of a splicing effect, an ASO blocks the aberrant splice signal to restore normal splicing. Nonsense or frameshift variants that act through aberrant splicing and cause a gain-of-function or dominant-negative effect can also be corrected this way.
2. Canonical exon skipping: for truncating loss-of-function variants, an ASO skips the variant's exon (or an adjacent one) to restore the reading frame and make a shorter but working protein. An exon is NOT eligible if any of these holds: it is the first coding exon; it is the last coding exon; it is out of frame; skipping it creates a stop codon; the gene has only one coding exon. Exceptions are listed in the guidelines' Section A.
3. Transcript knockdown (ASO or siRNA): for gain-of-function and dominant-negative variants, reduce the transcript. Not eligible when the gene is dosage-sensitive (haploinsufficient), unless an allele-selective approach targets only the variant allele; ClinGen dosage sensitivity is the recommended check.
4. Upregulation from the wild-type allele (e.g. TANGO): for haploinsufficiency; strategies are described but eligibility verdicts are not defined, as the approach is less established.

Verdicts, given per strategy: eligible (functional evidence that the approach works for this variant, exon, or gene, e.g. an ASO already developed, or naturally occurring benign skipping of the exon); likely eligible (molecular criteria met, no direct functional evidence); unlikely eligible (criteria argue against it); not eligible (a disqualifying criterion, or functional evidence of failure); unable to assess (required information missing).

Always check whether an ASO, RNAi or siRNA approach has already been developed for the variant, exon, or gene, clinically or preclinically; that is what moves a verdict from likely eligible to eligible.

The guidelines cover the variant only; disease- and patient-level factors are outside their scope."""


class N1CGuidelinesTool(BaseTool):
    """Return the N1C ASO eligibility rules, or a section of the guideline paper."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "n1c_aso_guidelines",
                "description": (
                    "The N=1 Collaborative (N1C) consensus guidelines for assessing "
                    "whether a pathogenic variant is eligible for antisense "
                    "oligonucleotide (ASO) treatment (Cheerie et al., AJHG 2025). "
                    "With no section given, returns the rules in summary: Step 0 "
                    "checks (correct HGVS, confirmed causal, covered variant type), "
                    "the inheritance and pathomechanism questions (loss of function, "
                    "gain of function, dominant negative), the four strategies "
                    "(splice correction for splicing variants; exon skipping for "
                    "truncating loss-of-function variants, not eligible if the exon "
                    "is the first or last coding exon, out of frame, skipping makes "
                    "a stop codon, or the gene has one coding exon; transcript "
                    "knockdown for gain-of-function or dominant-negative variants, "
                    "not eligible in haploinsufficient genes unless allele-selective; "
                    "wild-type upregulation for haploinsufficiency, no verdicts "
                    "defined), and the five verdicts (eligible, likely eligible, "
                    "unlikely eligible, not eligible, unable to assess). With a "
                    "section name, returns that section of the paper's full text. "
                    "Apply the rules to the facts from the validated variant block, "
                    "VEP (consequence, exon, frame, codon count), and ClinGen dosage."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "description": (
                                "Optional paper section for more detail: "
                                + ", ".join(SECTIONS)
                                + ". Leave empty for the rules summary."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    def __init__(self) -> None:
        self._article_xml: str | None = None

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        section = (kwargs.get("section") or "").strip().lower()
        evidence = [EvidenceRecord(url=DOI_URL), EvidenceRecord(url=CALCULATOR_URL)]
        if not section:
            return OVERVIEW, evidence
        if section not in SECTIONS:
            return (
                f"N1C guidelines: unknown section {section!r}. Available: {', '.join(SECTIONS)}.",
                evidence,
            )
        xml, failure = self._article()
        if xml is None:
            return f"N1C guidelines: could not fetch the paper's text from PubMed Central: {failure}", evidence[:1]
        text = self._section_text(xml, SECTIONS[section])
        if not text:
            logger.warning(f"N1C paper section {SECTIONS[section]!r} not found in PMC{PMC_ID} XML")
            return f"N1C guidelines: section {section!r} not found in the paper's text.", evidence[:1]
        header = f"From Cheerie et al. 2025 (PMC{PMC_ID}, CC BY 4.0), section '{SECTIONS[section]}':\n"
        return header + (text if len(text) <= MAX_SECTION_CHARS else text[:MAX_SECTION_CHARS - 3] + "..."), evidence[:1]

    def _article(self) -> tuple[str | None, str]:
        """The article XML from PubMed Central, fetched once per process."""
        if self._article_xml is not None:
            return self._article_xml, ""
        last = ""
        for attempt in range(1, RETRIES + 1):
            try:
                resp = requests.get(
                    EFETCH_URL, params=ncbi_params(db="pmc", id=PMC_ID, retmode="xml"), timeout=TIMEOUT
                )
            except requests.Timeout:
                last = "timeout"
                logger.warning(f"PMC efetch timeout (attempt {attempt}/{RETRIES})")
            except requests.RequestException as e:
                msg = redact_ncbi_key(str(e))
                logger.warning(f"PMC efetch failed: {msg}")
                return None, f"request failed ({msg})"
            else:
                if resp.status_code < 500:
                    if resp.status_code != 200 or "<article" not in resp.text:
                        logger.warning(f"PMC efetch HTTP {resp.status_code}, no article in body: {resp.text[:200]!r}")
                        return None, f"HTTP {resp.status_code}, no article in reply"
                    self._article_xml = resp.text
                    return self._article_xml, ""
                last = f"HTTP {resp.status_code}"
                logger.warning(f"PMC efetch server error (attempt {attempt}/{RETRIES}): HTTP {resp.status_code}")
            if attempt < RETRIES:
                time.sleep(RETRY_INTERVAL_SECONDS)
        return None, f"{last} after {RETRIES} attempts"

    @staticmethod
    def _section_text(xml: str, title: str) -> str:
        m = re.search(r"<sec[^>]*>\s*<title>" + re.escape(title) + r"</title>(.*?)</sec>", xml, re.S)
        if not m:
            return ""
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", m.group(1), re.S)
        clean = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", p)).strip() for p in paragraphs]
        return "\n\n".join(p for p in clean if p)
