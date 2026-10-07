"""Validate a variant and rewrite it onto its current transcript version.

Runs in step 1, before the model, for every variant in the input. Uses the
VariantValidator REST API (https://variantvalidator.org), which keeps every
RefSeq transcript version; Ensembl does not, and rejects the outdated
versions the challenge data uses. Nothing is corrected: an invalid variant
is reported with the validator's own warnings and the strings as given.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

VV_API = "https://rest.variantvalidator.org/VariantValidator/variantvalidator"
GENOME_BUILD = "GRCh38"
TIMEOUT = 60
# Only a timeout is retried: an HTTP error or an unparseable body will not
# change on a second try and is reported at once.
TIMEOUT_RETRIES = 5
RETRY_INTERVAL_SECONDS = 5

_NEWER_VERSION = re.compile(r"A more recent version of .+ is available for genome build \S+ \((\S+)\)")


@dataclass
class ValidatedVariant:
    submitted: str                      # "NM_004006.2:c.7544_9286del", as given
    valid: bool
    warnings: list[str] = field(default_factory=list)
    transcript_variant: str = ""        # on the current transcript version
    genomic_variant: str = ""           # e.g. NC_000023.11:g.31260956_31729748del
    gene_symbol: str = ""
    superseded_transcript: str = ""     # the outdated version, when one was replaced
    source_url: str = ""                # the request that produced transcript_variant


def validate_variant(transcript: str, variant_cdna: str) -> ValidatedVariant:
    """Validate transcript:variant_cdna and return it on the current transcript version.

    One request when the version is current. When the validator warns that a
    newer version exists, a second request maps the genomic position onto
    that version, because cDNA coordinates can shift between versions and
    only the genomic position is version-free.
    """
    submitted = f"{transcript}:{variant_cdna}"
    result = ValidatedVariant(submitted=submitted, valid=False)

    body, url = _get(submitted, transcript)
    if body is None:
        result.warnings.append("validator unavailable: timeout after retries")
        return result
    entry = _entry(body)
    if body.get("flag") != "gene_variant" or entry is None:
        result.warnings = _warnings(body, entry)
        logger.warning(f"Variant {submitted} failed validation: flag={body.get('flag')!r} {result.warnings}")
        return result

    _fill(result, entry, url)
    result.valid = True

    newer = next((m.group(1) for w in result.warnings for m in [_NEWER_VERSION.search(w)] if m), None)
    if newer and result.genomic_variant:
        body2, url2 = _get(result.genomic_variant, newer)
        entry2 = _entry(body2) if body2 else None
        if body2 is None or body2.get("flag") != "gene_variant" or entry2 is None:
            logger.warning(
                f"Variant {submitted}: could not map {result.genomic_variant} onto {newer}; "
                f"keeping {result.transcript_variant}. flag={body2.get('flag') if body2 else 'timeout'}"
            )
            return result
        result.superseded_transcript = transcript
        _fill(result, entry2, url2)
        logger.info(f"Variant {submitted} rewritten to {result.transcript_variant}")
    return result


def _get(variant: str, select_transcript: str) -> tuple[dict | None, str]:
    """GET one validator request; returns (json body or None after timeouts, url)."""
    url = f"{VV_API}/{GENOME_BUILD}/{variant}/{select_transcript}"
    for attempt in range(1, TIMEOUT_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=TIMEOUT,
            )
        except requests.Timeout:
            logger.warning(f"VariantValidator timeout (attempt {attempt}/{TIMEOUT_RETRIES}): {url}")
            if attempt < TIMEOUT_RETRIES:
                time.sleep(RETRY_INTERVAL_SECONDS)
            continue
        except requests.RequestException as e:
            logger.warning(f"VariantValidator request failed: {url}: {e}")
            return {"flag": "request_error", "error": str(e)}, url
        if resp.status_code != 200:
            logger.warning(f"VariantValidator HTTP {resp.status_code}: {url}: {resp.text[:200]!r}")
            return {"flag": f"http_{resp.status_code}", "error": resp.text[:200]}, url
        try:
            return resp.json(), url
        except ValueError:
            logger.warning(f"VariantValidator non-JSON body: {url}: {resp.text[:200]!r}")
            return {"flag": "non_json", "error": resp.text[:200]}, url
    return None, url


def _entry(body: dict) -> dict | None:
    """The single variant entry in a validator reply, or None."""
    entries = [v for k, v in body.items() if k not in ("flag", "metadata") and isinstance(v, dict)]
    return entries[0] if entries else None


def _warnings(body: dict, entry: dict | None) -> list[str]:
    if entry and entry.get("validation_warnings"):
        return list(entry["validation_warnings"])
    if body.get("error"):
        return [f"{body.get('flag')}: {body['error']}"]
    return [f"flag={body.get('flag')!r}, no variant entry in reply"]


def _fill(result: ValidatedVariant, entry: dict, url: str) -> None:
    result.warnings = list(entry.get("validation_warnings") or [])
    result.transcript_variant = entry.get("hgvs_transcript_variant") or ""
    result.gene_symbol = entry.get("gene_symbol") or ""
    result.source_url = url
    result.genomic_variant = (
        ((entry.get("primary_assembly_loci") or {}).get(GENOME_BUILD.lower()) or {})
        .get("hgvs_genomic_description") or ""
    )
