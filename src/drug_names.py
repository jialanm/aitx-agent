"""Map brand and salt names in a string answer to their ingredient, via RxNorm.

RxNorm (NLM) types every concept: BN is a brand name, PIN a salt or other
precise form of an ingredient, IN the ingredient. Only a brand or a salt is
replaced, by its single ingredient; an ingredient is left as written (so
"L-serine" stays, and "No" is never read as nitric oxide), and anything
RxNorm does not know is left alone. Every replacement is returned so the
caller can record it.
"""

from __future__ import annotations

import logging
import re
import time

import requests

logger = logging.getLogger(__name__)

RXNORM_API = "https://rxnav.nlm.nih.gov/REST"
TIMEOUT = 15
RETRIES = 3
RETRY_INTERVAL_SECONDS = 2
REPLACEABLE_TYPES = {"BN", "PIN"}
# Items that are not drug names: identifiers, numbers, very short tokens.
_SKIP = re.compile(r"^(?:NCT\d+|rs\d+|NBK\d+|[A-Z]{2,4}|\d[\d.,% ]*(?:\s*[a-z]+)?|none|not available|n/?a|yes|no)$", re.IGNORECASE)


def normalise_drug_names(answer: str) -> tuple[str, list[dict]]:
    """Return (answer with brands and salts replaced, list of replacements made)."""
    items = [i.strip() for i in answer.split(",")]
    changes: list[dict] = []
    out = []
    for item in items:
        new = item
        if item and len(item) >= 4 and not _SKIP.match(item):
            resolved = _ingredient_for(item)
            if resolved:
                new = resolved["ingredient"]
                changes.append({"from": item, "to": new, "rxcui": resolved["rxcui"], "tty": resolved["tty"]})
                logger.info(f"Drug name normalised via RxNorm: {item!r} -> {new!r} (rxcui {resolved['rxcui']}, {resolved['tty']})")
        out.append(new)
    return ", ".join(out), changes


def is_drug_name(name: str) -> bool | None:
    """True if RxNorm knows the name as a drug concept, False if not, None
    when RxNorm could not be reached. Used to decide whether an answer term
    that no tool text contains is a drug name (removed) or a code such as a
    protein domain or GOF/LOF (kept): the lookup, not the answer format,
    decides."""
    return drug_concept(name) is not None if _reachable(name) else None


def _reachable(name: str) -> bool:
    """Whether a lookup was attempted at all; a name RxNorm cannot be asked
    about returns None from is_drug_name."""
    if not name or len(name) < 3 or _SKIP.match(name):
        return True            # answered without a lookup: not a drug
    return _get(f"{RXNORM_API}/rxcui.json", {"name": name, "search": 2}) is not None


def drug_concept(name: str) -> dict | None:
    """RxNorm's concept for this name, or None.

    Two lookups. The default one returns only concepts in the current
    release, which NLM intends as an approximation of drugs currently
    marketed in the United States. The second passes allsrc=1, which also
    returns a concept RxNorm keeps but does not call current, and ataluren is
    one: concept 1030912, sourced from the World Health Organization's ATC
    classification, released 11/2015, isCurrent NO. Its absence from the
    first lookup made the harness treat an approved European therapy as not a
    drug (32B runs, AITX-00002).

    The returned dict carries "current", so a caller can say which lookup
    found it; pegalysine, which the model invented, is found by neither.
    """
    if not name or len(name) < 3 or _SKIP.match(name):
        return None
    found = _get(f"{RXNORM_API}/rxcui.json", {"name": name, "search": 2})
    if found is None:
        return None
    ids = (found.get("idGroup") or {}).get("rxnormId") or []
    if ids:
        return {"rxcui": ids[0], "current": True, "source": ""}
    wider = _get(f"{RXNORM_API}/rxcui.json", {"name": name, "search": 2, "allsrc": 1})
    ids = ((wider or {}).get("idGroup") or {}).get("rxnormId") or []
    if not ids:
        return None
    history = _get(f"{RXNORM_API}/rxcui/{ids[0]}/historystatus.json", {}) or {}
    meta = (history.get("rxcuiStatusHistory") or {}).get("metaData") or {}
    return {"rxcui": ids[0], "current": False, "source": meta.get("source", "")}


def _ingredient_for(name: str) -> dict | None:
    """The single ingredient when RxNorm types the name as a brand or salt; else None."""
    found = _get(f"{RXNORM_API}/rxcui.json", {"name": name, "search": 2})
    ids = ((found or {}).get("idGroup") or {}).get("rxnormId") or []
    if not ids:
        return None
    rxcui = ids[0]
    props = _get(f"{RXNORM_API}/rxcui/{rxcui}/property.json", {"propName": "TTY"})
    tty = ((((props or {}).get("propConceptGroup") or {}).get("propConcept") or [{}])[0]).get("propValue")
    if tty not in REPLACEABLE_TYPES:
        return None
    related = _get(f"{RXNORM_API}/rxcui/{rxcui}/related.json", {"tty": "IN"})
    ingredients = [
        c["name"] for g in (((related or {}).get("relatedGroup") or {}).get("conceptGroup") or [])
        for c in (g.get("conceptProperties") or [])
    ]
    if len(ingredients) != 1:
        # A combination product, or nothing linked: keep the name as written.
        logger.info(f"RxNorm {name!r} ({tty}) maps to {len(ingredients)} ingredients; left as written")
        return None
    return {"ingredient": ingredients[0], "rxcui": rxcui, "tty": tty}


def _get(url: str, params: dict) -> dict | None:
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
        except requests.Timeout:
            logger.warning(f"RxNorm timeout (attempt {attempt}/{RETRIES}): {url}")
        except requests.RequestException as e:
            logger.warning(f"RxNorm request failed: {url}: {e}")
            return None
        else:
            if resp.status_code < 500:
                if resp.status_code != 200:
                    logger.warning(f"RxNorm HTTP {resp.status_code}: {url}")
                    return None
                try:
                    return resp.json()
                except ValueError:
                    logger.warning(f"RxNorm non-JSON reply: {url}")
                    return None
            logger.warning(f"RxNorm server error HTTP {resp.status_code} (attempt {attempt}/{RETRIES}): {url}")
        if attempt < RETRIES:
            time.sleep(RETRY_INTERVAL_SECONDS)
    return None
