"""Ensembl REST API tool for variant effect prediction and transcript info."""

from __future__ import annotations

import logging
import re
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
# Closes the VEP block. The consequence label and the exon line count
# different things, and the model must know which governs when they differ.
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

        exon_line = self._exon_length_line(tc.get("transcript_id") or transcript, tc.get("exon"))
        lines = [
            f"VEP (Ensembl, GRCh38) for {notation} on transcript {tc.get('transcript_id')}:",
            self._frame_line(tc, self._exon_info, entry.get("allele_string") or ""),
            f"Consequence{self._as_written(tc, self._exon_info)}: {show('consequence_terms')}",
            f"Impact: {show('impact')}",
            f"Exon: {show('exon')}" + (f" | Intron: {show('intron')}" if tc.get("intron") else ""),
            exon_line,
            *self._deletion_lines(tc, self._exon_info),
            f"Protein position: {show('protein_start')}"
            + (f"-{tc['protein_end']}" if tc.get("protein_end") not in (None, tc.get("protein_start")) else ""),
            f"Amino acid change: {show('amino_acids')}",
            f"HGVS protein{self._as_written(tc, self._exon_info)}: {show('hgvsp')}",
            f"HGVS cDNA{self._as_written(tc, self._exon_info)}: {show('hgvsc')}",
            *self._other_transcript_line(entry, tc),
        ]
        return "\n".join(lines)

    # A deletion that removes splice sites across one or more exons. What is
    # known from the two replies is stated as fact (which exons are deleted
    # entirely, how much of a partly covered exon is gone); what the cell is
    # expected to do with a remnant too short to be an exon is stated as an
    # expectation, with the reason, and VEP's lines for the letters as
    # written are labelled as such (32B rerun 2026-10-13, AITX-00001: the
    # model read "inframe_deletion" first and never reconciled the rest).
    @staticmethod
    def _splice_deletion(tc: dict, info: dict | None) -> dict | None:
        terms = set(tc.get("consequence_terms") or [])
        if not info or not (terms & {"splice_acceptor_variant", "splice_donor_variant"}) or "del" not in (tc.get("hgvsc") or ""):
            return None
        cs, ce = tc.get("cds_start"), tc.get("cds_end")
        if not isinstance(cs, int) or not isinstance(ce, int):
            return None
        rows = []  # (exon index, coding length, deleted bases)
        for idx in range(info["lo"], info["hi"] + 1):
            span, length = info["spans"][idx - 1], info["lengths"][idx - 1]
            deleted = max(0, min(ce, span[1]) - max(cs, span[0]) + 1) if span else 0
            rows.append((idx, length, deleted))
        whole = info["bases"]
        written = sum(d for _, _, d in rows)
        # Only arithmetic from the two replies is stated: how much of each
        # touched exon the deletion covers, the count as written, and the
        # count if the touched exons were lost whole. Whether a remnant is
        # spliced in is biology, not arithmetic; it is left to the genetics
        # reference, where it carries a citation (decided 2026-10-13).
        return {"rows": rows, "whole": whole, "written": written, "remnant": whole - written,
                "partial": [(i, l, d) for i, l, d in rows if 0 < d < l]}

    @classmethod
    def _as_written(cls, tc: dict, info: dict | None) -> str:
        """Qualifier for VEP's literal lines when they count fewer bases than the touched exons hold."""
        d = cls._splice_deletion(tc, info)
        return " (for the letters as written)" if d and d["remnant"] else ""

    @staticmethod
    def _in_words(n: int) -> str:
        rest = n % 3
        if rest == 0:
            return f"{n} coding bases, a multiple of three"
        return f"{n} coding bases, {'one' if rest == 1 else 'two'} more than a multiple of three"

    @classmethod
    def _frame_line(cls, tc: dict, info: dict | None, allele_string: str = "") -> str:
        """First line of the block: VEP's frame term and the length change, as
        numbers. Present on every variant so the model learns it is always
        there. No judgment is added; what a count means for the protein is
        the genetics reference's to say, with a citation."""
        d = cls._splice_deletion(tc, info)
        if d:
            line = f"Frame, as written: {cls._in_words(d['written'])} deleted (VEP: {cls._vep_frame_term(tc)})"
            if d["remnant"]:
                line += f". If {info['which'].lower()} were lost whole: {cls._in_words(d['whole'])}"
            return line
        ref, _, alt = allele_string.partition("/")
        change = ""
        if ref or alt:
            n = len(alt.replace("-", "")) - len(ref.replace("-", ""))
            change = "no change in length" if n == 0 else f"{abs(n)} base{'s' if abs(n) != 1 else ''} {'inserted' if n > 0 else 'deleted'}"
        term = cls._vep_frame_term(tc)
        first = next(iter(tc.get("consequence_terms") or ["unknown"]))
        label = term if term != "no frame term" else first
        return f"Frame (VEP): {label}" + (f"; {change}" if change else "")

    @staticmethod
    def _vep_frame_term(tc: dict) -> str:
        terms = tc.get("consequence_terms") or []
        return next((t for t in ("inframe_deletion", "frameshift_variant") if t in terms), "no frame term")

    @classmethod
    def _deletion_lines(cls, tc: dict, info: dict | None) -> list[str]:
        """Coverage as fact, outcome as expectation, and what would repair the frame."""
        d = cls._splice_deletion(tc, info)
        if not d:
            return []
        lines = []
        entire = [i for i, l, dd in d["rows"] if dd == l]
        if entire:
            rng = f"exon {entire[0]}" if len(entire) == 1 else f"exons {entire[0]}-{entire[-1]}"
            lines.append(f"Deleted entirely: {rng} ({sum(l for i, l, _ in d['rows'] if i in entire)} coding bases).")
        for i, l, dd in d["partial"]:
            lines.append(f"Exon {i}: {dd} of {l} coding bases deleted.")
        if d["remnant"]:
            lines.append(
                f"The deletion as written covers {d['written']} of the {d['whole']} coding bases of {info['which'].lower()}; "
                f"VEP reports splice acceptor and donor sites among the consequences. What the transcript keeps of a "
                f"partly deleted exon is not stated by VEP; the genetics reference (topic: splicing) gives the rule and its source."
            )
        if d["whole"] % 3 != 0:
            # Arithmetic only: which neighbouring exon, if removed together with
            # the whole touched exons, makes the total a multiple of three.
            parts = []
            for idx in (info["lo"] - 1, info["hi"] + 1):
                if 1 <= idx <= info["n"]:
                    length = info["lengths"][idx - 1]
                    total = d["whole"] + length
                    parts.append(
                        f"exon {idx} ({length} coding bases) gives {total}, "
                        + ("a multiple of three" if total % 3 == 0 else "not a multiple of three")
                    )
            if parts:
                lines.append(
                    f"If {info['which'].lower()} were lost whole ({d['whole']} bases) together with one neighbouring exon: "
                    + "; ".join(parts) + "."
                )
        return lines

    @staticmethod
    def _other_transcript_line(entry: dict, tc: dict) -> list[str]:
        """VEP's most_severe_consequence is the worst effect on any transcript
        overlapping the position, not the patient's; shown only when it differs
        from the patient's transcript, and says whose it is (run 5, AITX-00001:
        "transcript_ablation" was read as the patient's)."""
        severe = entry.get("most_severe_consequence")
        if not severe or severe in set(tc.get("consequence_terms") or []):
            return []
        return [
            f"Note: on another transcript overlapping this position the most severe consequence "
            f"is {severe}; the lines above describe the patient's transcript {tc.get('transcript_id')}."
        ]

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
        self._exon_info = None  # set on success; the Frame line reads it
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
        # leftover bases rather than a fractional amino-acid count. Nothing
        # about the frame is said here (decided 2026-10-13): the codon count
        # shows the arithmetic, and the deletion lines carry the sums that
        # matter for a whole-exon loss.
        codons, rest = divmod(bases, 3)
        # CDS coordinates of each exon (1-based, transcript order), so a
        # deletion's CDS span from VEP can be measured against them.
        spans, pos = [], 0
        for e in exons:
            c = coding(e)
            spans.append((pos + 1, pos + c) if c else None)
            pos += c
        self._exon_info = {
            "which": which, "bases": bases, "rest": rest, "lo": lo_idx, "hi": hi_idx, "n": n_exons,
            "lengths": [coding(e) for e in exons], "spans": spans,
        }
        return (
            f"{which} of {n_exons} ({ensembl_id}): {bases} coding bases = {codons} codons"
            + (f" + {rest} bases" if rest else "")
            + f", {share:.1f}% of the {total_cds}-base coding sequence; "
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
            resp, failure = self._get_with_retries(url, {"content-type": "application/json", "expand": 1}, "Ensembl gene lookup")
            if resp is None:
                logger.warning(f"Ensembl gene lookup gave nothing for {gene}: {failure}")
                return None
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
