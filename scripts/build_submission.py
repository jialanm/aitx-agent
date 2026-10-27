"""Run the report chain on each challenge patient and write the submission file.

The final task is submitted as one JSON array of {id, report_markdown}, one
per patient case downloaded from the challenge site (no hosted endpoint is
required). This script runs the chain once per case, writes each report and
its trace beside the input for inspection, then assembles and checks the
array: valid JSON, one report per expected id, under the 5 MB limit.

Usage:
  uv run python scripts/build_submission.py data/patients/*.json \
      --model Qwen/Qwen3-32B-FP8 --precision fp8 --out data/submission.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.model import MODEL_ID, PRECISIONS, load_model
from src.report import ReportBuilder
from src.schemas import ReportInput
from src.tools.clinvar import ClinVarTool
from src.tools.clinical_trials import ClinicalTrialsTool
from src.tools.ensembl import EnsemblTool
from src.tools.genereviews import GeneReviewsTool
from src.tools.pubmed import PubMedTool
from src.tools.uniprot import UniProtTool
from src.tools.pharmgkb import PharmGKBTool
from src.tools.omim import OMIMTool
from src.tools.openfda import OpenFDATool
from src.tools.pharos import PharosTool
from src.tools.n1c import N1CGuidelinesTool
from src.tools.genetics_reference import GeneticsReferenceTool
from src.tools.clingen_dosage import ClinGenDosageTool

logger = logging.getLogger(__name__)
MAX_BYTES = 5 * 1024 * 1024


def build_tools() -> list:
    return [ClinVarTool(), EnsemblTool(), GeneReviewsTool(), ClinicalTrialsTool(), PubMedTool(), UniProtTool(),
            PharmGKBTool(), OMIMTool(), OpenFDATool(), PharosTool(), N1CGuidelinesTool(), GeneticsReferenceTool(),
            ClinGenDosageTool()]


def check(reports: list[dict], expected_ids: list[str], out: Path) -> list[str]:
    """The checks the challenge's upload form applies, run before uploading."""
    problems = []
    ids = [r["id"] for r in reports]
    if sorted(ids) != sorted(expected_ids):
        problems.append(f"ids {sorted(ids)} do not match the cases {sorted(expected_ids)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    for r in reports:
        if not r.get("report_markdown", "").strip():
            problems.append(f"{r['id']}: empty report_markdown")
        elif not r["report_markdown"].lstrip().startswith("#"):
            problems.append(f"{r['id']}: report does not start with a heading")
    size = out.stat().st_size if out.exists() else 0
    if size > MAX_BYTES:
        problems.append(f"file is {size} bytes, over the 5 MB limit")
    return problems


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_json", nargs="+", help="One or more patient case files, or one file holding an array")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--precision", default="bf16", choices=sorted(PRECISIONS))
    parser.add_argument("--out", default="data/submission.json")
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    cases: list[ReportInput] = []
    for path in args.patient_json:
        data = json.loads(Path(path).read_text())
        for item in (data if isinstance(data, list) else [data]):
            cases.append(ReportInput(**item))
    print(f"{len(cases)} patient cases: {', '.join(c.id for c in cases)}")

    model, tokenizer = load_model(model_id=args.model, precision=args.precision)
    agent = Agent(model, tokenizer, build_tools(), show_thinking=args.show_thinking)
    builder = ReportBuilder(agent)

    out = Path(args.out)
    reports: list[dict] = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case.id}")
        start = time.time()
        try:
            output = builder.build(case)
            reports.append({"id": case.id, "report_markdown": output.report_markdown})
        except Exception as e:
            logger.error(f"{case.id} failed: {e}", exc_info=True)
            reports.append({"id": case.id, "report_markdown": f"# Therapeutic Actionability Report for Patient {case.id}\n\nReport generation failed: {e}\n"})
        Path(f"{out.with_suffix('')}_{case.id}_trace.json").write_text(json.dumps(agent.last_trace, indent=2, default=str))
        # Written after every case, so an interrupted run keeps what it had.
        out.write_text(json.dumps(reports, indent=2))
        print(f"  {len(reports[-1]['report_markdown'])} chars in {time.time() - start:.0f}s")

    problems = check(reports, [c.id for c in cases], out)
    print(f"\nWrote {out} ({out.stat().st_size} bytes, {len(reports)} reports)")
    print("Checks: " + ("passed" if not problems else "FAILED\n  " + "\n  ".join(problems)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
