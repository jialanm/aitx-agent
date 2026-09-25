"""Fetch the challenge's Phase 1 validator set and split it into questions and answers.

The upstream file carries the answer key inside each record. We never commit
it in that form: the split files are what the repo holds, and the agent is
only ever given the questions file. Default mode re-downloads the pinned
revision and verifies every checksum in the manifest; --update-manifest
rewrites the manifest's checksums after an intentional refresh.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import file_sha256, split_records

DATA_DIR = Path(__file__).parent.parent / "data"
MANIFEST = DATA_DIR / "phase1_validator.manifest.json"


def _dump(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")


def main(update_manifest: bool) -> int:
    manifest = json.loads(MANIFEST.read_text())
    url = f"{manifest['source']}/resolve/{manifest['revision']}/{manifest['source_file']}"

    print(f"Fetching {url}")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()

    import hashlib
    upstream_sha = hashlib.sha256(raw).hexdigest()
    if upstream_sha != manifest["upstream_sha256"]:
        print(f"UPSTREAM CHECKSUM MISMATCH: got {upstream_sha}, manifest says {manifest['upstream_sha256']}")
        return 1

    questions, answers = split_records(json.loads(raw))
    q_path = DATA_DIR / manifest["questions"]["file"]
    a_path = DATA_DIR / manifest["answers"]["file"]
    _dump(q_path, questions)
    _dump(a_path, answers)

    ok = True
    for key, path in (("questions", q_path), ("answers", a_path)):
        digest = file_sha256(path)
        if update_manifest:
            manifest[key]["sha256"] = digest
        elif digest != manifest[key]["sha256"]:
            print(f"{key.upper()} CHECKSUM MISMATCH: got {digest}, manifest says {manifest[key]['sha256']}")
            ok = False

    if update_manifest:
        manifest["fetched"] = date.today().isoformat()
        manifest["n_records"] = len(questions)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Manifest updated: {MANIFEST}")

    if ok:
        print(f"OK: {len(questions)} questions -> {q_path.name}, answers -> {a_path.name}, checksums verified")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--update-manifest", action="store_true",
                        help="rewrite the split-file checksums and fetch date after an intentional refresh")
    sys.exit(main(parser.parse_args().update_manifest))
