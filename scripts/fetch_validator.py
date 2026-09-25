"""Fetch the challenge's Phase 1 validator set at the revision pinned in its manifest.

The file is committed, so this is only needed to re-verify or re-download it.
A checksum mismatch means the upstream dataset changed under the same
revision, which should never happen, or the local copy was edited.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import file_sha256

DATA_DIR = Path(__file__).parent.parent / "data"
MANIFEST = DATA_DIR / "phase1_validator.manifest.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    target = DATA_DIR / manifest["file"]
    url = f"{manifest['source']}/resolve/{manifest['revision']}/{manifest['source_file']}"

    print(f"Fetching {url}")
    with urllib.request.urlopen(url) as resp:
        target.write_bytes(resp.read())

    digest = file_sha256(target)
    if digest != manifest["sha256"]:
        print(f"CHECKSUM MISMATCH: got {digest}, manifest says {manifest['sha256']}")
        return 1
    n = len(json.loads(target.read_text()))
    print(f"OK: {target} ({n} records, sha256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
