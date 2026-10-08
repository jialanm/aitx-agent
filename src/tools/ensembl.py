"""Ensembl REST API tool for variant effect prediction and transcript info."""

from __future__ import annotations

import logging
import time

import requests

from .base import BaseTool, EvidenceRecord

logger = logging.getLogger(__name__)

ENSEMBL_REST = "https://rest.ensembl.org"
# Browser page for a genomic region on GRCh38, cited as evidence for the VEP
# result; the region comes from VEP's own reply. Reused by the report task.
ENSEMBL_LOCATION_URL = "https://www.ensembl.org/Homo_sapiens/Location/View?r={chrom}:{start}-{end}"

TIMEOUT = 15
# VEP took up to 66 s for a 1.7 kb deletion on 2026-10-07. Timeouts and
# server errors (5xx) are retried because the service answered the same
# request on a later try in 7 of 8 cases that day; 4xx means the request
# itself is wrong and is not retried.
VEP_TIMEOUT = 90
VEP_RETRIES = 5
VEP_RETRY_INTERVAL_SECONDS = 5
# Fields the model needs from the entry for the patient's transcript. A
# missing one is logged by name and shown as "not returned", never blank.
VEP_REQUIRED_FIELDS = ("consequence_terms", "impact", "exon", "protein_start", "amino_acids", "hgvsp")


class EnsemblTool(BaseTool):
    """Query Ensembl for variant effect prediction and transcript info."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "query_ensembl",
                "description": (
                    "Query the Ensembl Variant Effect Predictor (VEP) for a variant's "
                    "consequence on the patient's transcript: consequence terms, "
                    "impact, exon, protein position and amino acid change. Also "
                    "returns the exon's coding length in bases and amino acids and "
                    "its share of the transcript's coding sequence, and the protein "
                    "length. Prefer the genomic HGVS form from the validated variant "
                    "block."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "genomic_hgvs": {
                            "type": "string",
                            "description": (
                                "Genomic HGVS on GRCh38 from the validated variant block "
                                "(e.g., NC_000023.11:g.33020162A>G)"
                            ),
                        },
                        "transcript": {
                            "type": "string",
                            "description": (
                                "Transcript to report on, current version "
                                "(e.g., NM_004006.3)"
                            ),
                        },
                        "hgvs_cdna": {
                            "type": "string",
                            "description": (
                                "Fallback when no genomic form is available: transcript "
                                "HGVS cDNA (e.g., NM_000492.4:c.1521_1523del)"
                            ),
                        },
                        "gene": {
                            "type": "string",
                            "description": "HGNC gene symbol (e.g., CFTR)",
                        },
                    },
                    "required": [],
                },
            },
        }

    def execute(self, **kwargs) -> tuple[str, list[EvidenceRecord]]:
        genomic_hgvs = kwargs.get("genomic_hgvs", "")
        hgvs_cdna = kwargs.get("hgvs_cdna", "")
        transcript = kwargs.get("transcript", "") or hgvs_cdna.split(":")[0]
        gene = kwargs.get("gene", "")
        evidence: list[EvidenceRecord] = []
        results = []

        notation = genomic_hgvs or hgvs_cdna
        if notation:
            vep_result = self._query_vep(notation, transcript, evidence)
            results.append(vep_result)

        # Transcript lookup if we have a gene
        if gene:
            tx_result = self._lookup_gene(gene, evidence)
            if tx_result:
                results.append(tx_result)

        if not results:
            return "No Ensembl data found: no variant or gene given.", evidence

        summary = "\n\n".join(results)
        return summary[:2000], evidence

    def _query_vep(
        self, notation: str, transcript: str, evidence: list[EvidenceRecord]
    ) -> str:
        """Query VEP and report the consequence on the given transcript.

        Always returns text: either the consequence block or a line saying
        VEP returned nothing and why, so the model never sees a silent gap.
        """
        url = f"{ENSEMBL_REST}/vep/human/hgvs/{notation}"
        # refseq=1 lets VEP parse NM_ transcripts and report on them; numbers
        # adds exon/intron numbers; hgvs adds the HGVS protein string.
        resp, failure = self._get_with_retries(url, {"refseq": 1, "numbers": 1, "hgvs": 1}, "VEP")
        if resp is None:
            return f"VEP returned nothing for {notation}: {failure}"

        if resp.status_code != 200 or "json" not in resp.headers.get("content-type", ""):
            reason = f"HTTP {resp.status_code}, content-type {resp.headers.get('content-type', '')!r}"
            logger.warning(f"VEP error: {url}: {reason}: {resp.text[:200]!r}")
            return f"VEP returned nothing for {notation}: {reason}"
        data = resp.json()
        if not data:
            logger.warning(f"VEP empty reply: {url}")
            return f"VEP returned nothing for {notation}: empty reply"

        entry = data[0]
        chrom, start, end = entry.get("seq_region_name"), entry.get("start"), entry.get("end")
        if chrom and start and end:
            # An insertion or duplication is reported with start = end + 1,
            # naming the two bases either side of the gap; the browser link
            # needs the smaller coordinate first.
            lo, hi = min(start, end), max(start, end)
            evidence.append(
                EnsemblTool.location_evidence(chrom, lo, hi)
            )
        else:
            logger.warning(f"VEP reply lacks seq_region_name/start/end: {url}")

        tc = self._transcript_entry(entry.get("transcript_consequences", []), transcript)
        if tc is None:
            have = [t.get("transcript_id") for t in entry.get("transcript_consequences", [])]
            logger.warning(f"VEP reply has no entry for transcript {transcript!r}: {url}; has {have[:8]}")
            return (
                f"VEP for {notation}: most severe consequence across transcripts: "
                f"{entry.get('most_severe_consequence', 'unknown')}; no entry for transcript "
                f"{transcript or '(none given)'}"
            )

        missing = [f for f in VEP_REQUIRED_FIELDS if tc.get(f) in (None, "", [])]
        if missing:
            logger.warning(f"VEP entry for {transcript} lacks {missing}: {url}")

        def show(field: str) -> str:
            value = tc.get(field)
            if value in (None, "", []):
                return "not returned"
            if isinstance(value, list):
                return ", ".join(value)
            text = str(value)
            return text if len(text) <= 60 else text[:57] + "..."

        lines = [
            f"VEP (Ensembl, GRCh38) for {notation} on transcript {tc.get('transcript_id')}:",
            f"Consequence: {show('consequence_terms')}",
            f"Impact: {show('impact')}",
            f"Exon: {show('exon')}" + (f" | Intron: {show('intron')}" if tc.get("intron") else ""),
            self._exon_length_line(tc.get("transcript_id") or transcript, tc.get("exon")),
            f"Protein position: {show('protein_start')}"
            + (f"-{tc['protein_end']}" if tc.get("protein_end") not in (None, tc.get("protein_start")) else ""),
            f"Amino acid change: {show('amino_acids')}",
            f"HGVS protein: {show('hgvsp')}",
            f"HGVS cDNA: {show('hgvsc')}",
            f"Most severe consequence across all transcripts: {entry.get('most_severe_consequence', 'unknown')}",
        ]
        return "\n".join(lines)

    def _get_with_retries(self, url: str, params: dict, label: str) -> tuple[requests.Response | None, str]:
        """GET with the VEP retry policy: timeouts and 5xx retried, 4xx returned at once.

        Returns (response, "") on a reply below 500, or (None, reason) after
        the retries are spent or the request itself failed.
        """
        last_failure = ""
        for attempt in range(1, VEP_RETRIES + 1):
            try:
                resp = requests.get(
                    url, headers={"Content-Type": "application/json"}, params=params, timeout=VEP_TIMEOUT
                )
            except requests.Timeout:
                last_failure = "timeout"
                logger.warning(f"{label} timeout (attempt {attempt}/{VEP_RETRIES}): {url}")
            except requests.RequestException as e:
                logger.warning(f"{label} request failed: {url}: {e}")
                return None, f"request failed ({e})"
            else:
                if resp.status_code < 500:
                    return resp, ""
                last_failure = f"HTTP {resp.status_code}"
                logger.warning(f"{label} server error (attempt {attempt}/{VEP_RETRIES}): {url}: HTTP {resp.status_code}")
            if attempt < VEP_RETRIES:
                time.sleep(VEP_RETRY_INTERVAL_SECONDS)
        return None, f"{last_failure} after {VEP_RETRIES} attempts"

    def _exon_length_line(self, transcript: str, exon_field: str | None) -> str:
        """Coding length of the exon(s) VEP named, from the transcript's exon map.

        VEP gives only the exon index ("3/13" or "52-63/79"). Ensembl's
        transcript record, expanded, lists every exon's genomic span and the
        CDS bounds, so the coding bases per exon follow by overlap. Two
        requests: the RefSeq name to its Ensembl transcript, then the record.
        """
        if not exon_field or "/" not in exon_field:
            return "Exon coding length: not returned (no exon index from VEP)"
        index, _, total = exon_field.partition("/")
        first, _, last = index.partition("-")
        try:
            lo_idx, hi_idx, n_exons = int(first), int(last or first), int(total)
        except ValueError:
            return f"Exon coding length: not returned (unparsed exon field {exon_field!r})"

        accession = transcript.split(".")[0]
        resp, failure = self._get_with_retries(
            f"{ENSEMBL_REST}/xrefs/symbol/homo_sapiens/{transcript}", {"object_type": "transcript"}, "Ensembl xref"
        )
        if resp is None or resp.status_code != 200 or not resp.json():
            # Ensembl may not know this exact version; try the accession alone.
            resp, failure = self._get_with_retries(
                f"{ENSEMBL_REST}/xrefs/symbol/homo_sapiens/{accession}", {"object_type": "transcript"}, "Ensembl xref"
            )
        if resp is None or resp.status_code != 200 or not resp.json():
            logger.warning(f"No Ensembl transcript for {transcript}: {failure or resp.status_code if resp else failure}")
            return f"Exon coding length: not returned (no Ensembl transcript found for {transcript})"
        ensembl_id = resp.json()[0].get("id", "")

        resp, failure = self._get_with_retries(f"{ENSEMBL_REST}/lookup/id/{ensembl_id}", {"expand": 1}, "Ensembl lookup")
        if resp is None or resp.status_code != 200:
            logger.warning(f"Transcript lookup failed for {ensembl_id}: {failure or resp.status_code}")
            return f"Exon coding length: not returned (transcript lookup failed for {ensembl_id})"
        record = resp.json()
        translation = record.get("Translation") or {}
        cds_start, cds_end = translation.get("start"), translation.get("end")
        exons = record.get("Exon") or []
        if not exons or cds_start is None or cds_end is None:
            logger.warning(f"Transcript {ensembl_id} has no exons or no CDS in the lookup reply")
            return f"Exon coding length: not returned ({ensembl_id} has no exon/CDS data)"
        # Exons in transcript order: ascending on the forward strand, descending on the reverse.
        exons = sorted(exons, key=lambda e: e["start"], reverse=(record.get("strand") == -1))
        if len(exons) != n_exons:
            logger.warning(
                f"Exon count differs: VEP says {n_exons} for {transcript}, Ensembl {ensembl_id} has {len(exons)}"
            )
            return (
                f"Exon coding length: not returned (VEP counts {n_exons} exons on {transcript}, "
                f"Ensembl {ensembl_id} has {len(exons)}; the exon maps may differ)"
            )

        def coding(e: dict) -> int:
            lo, hi = max(e["start"], cds_start), min(e["end"], cds_end)
            return max(0, hi - lo + 1)

        total_cds = sum(coding(e) for e in exons)
        bases = sum(coding(e) for e in exons[lo_idx - 1:hi_idx])
        which = f"Exon {lo_idx}" if lo_idx == hi_idx else f"Exons {lo_idx}-{hi_idx}"
        share = 100 * bases / total_cds if total_cds else 0
        # Exon edges rarely fall on codon boundaries, so give codons and the
        # leftover bases rather than a fractional amino-acid count; a remainder
        # of zero means removing these exons keeps the reading frame.
        codons, rest = divmod(bases, 3)
        frame = "in frame" if rest == 0 else f"out of frame by {rest}"
        return (
            f"{which} of {n_exons} ({ensembl_id}): {bases} coding bases = {codons} codons"
            + (f" + {rest} bases" if rest else "")
            + f" ({frame}), {share:.1f}% of the {total_cds}-base coding sequence; "
            f"protein length {translation.get('length')} aa"
        )

    @staticmethod
    def location_evidence(chrom: str, start: int, end: int) -> EvidenceRecord:
        """Evidence record for a GRCh38 region in the Ensembl browser."""
        return EvidenceRecord(url=ENSEMBL_LOCATION_URL.format(chrom=chrom, start=start, end=end))

    @staticmethod
    def _transcript_entry(consequences: list[dict], transcript: str) -> dict | None:
        """The VEP entry for the given transcript: exact ID, else same accession any version."""
        if not transcript:
            return None
        exact = [t for t in consequences if t.get("transcript_id") == transcript]
        if exact:
            return exact[0]
        accession = transcript.split(".")[0]
        same = [t for t in consequences if str(t.get("transcript_id", "")).split(".")[0] == accession]
        return same[0] if same else None

    def _lookup_gene(
        self, gene: str, evidence: list[EvidenceRecord]
    ) -> str | None:
        """Look up gene info from Ensembl."""
        try:
            url = f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{gene}"
            resp = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                params={"content-type": "application/json", "expand": 1},
                timeout=TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(f"Ensembl gene lookup HTTP {resp.status_code}: {url}")
                return None

            data = resp.json()
            ensembl_id = data.get("id", "")
            description = data.get("description", "")
            biotype = data.get("biotype", "")

            gene_url = f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={ensembl_id}"
            evidence.append(EvidenceRecord(url=gene_url))

            transcripts = data.get("Transcript", [])
            tx_info = ""
            if transcripts:
                canonical = [t for t in transcripts if t.get("is_canonical")]
                tx = canonical[0] if canonical else transcripts[0]
                tx_id = tx.get("id", "")
                exon_count = len(tx.get("Exon", []))
                tx_info = f"\nCanonical transcript: {tx_id}\nExon count: {exon_count}"

            return (
                f"Gene: {gene} ({ensembl_id})\n"
                f"Description: {description}\n"
                f"Biotype: {biotype}"
                f"{tx_info}\n"
                f"URL: {gene_url}"
            )

        except requests.RequestException as e:
            logger.warning(f"Ensembl gene lookup failed: {gene}: {e}")
            return None
