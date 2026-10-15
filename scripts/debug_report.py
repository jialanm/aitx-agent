"""Build one actionability report and write its trace.

Usage: uv run python scripts/debug_report.py path/to/patient.json [--model ID --precision bf16|nf4|fp8] [--show-thinking]
Writes the report markdown and the trace beside the input file.
"""

from __future__ import annotations

import json
import sys
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_json")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--precision", default="bf16", choices=sorted(PRECISIONS))
    parser.add_argument("--show-thinking", action="store_true")
    args = parser.parse_args()

    report_input = ReportInput(**json.loads(Path(args.patient_json).read_text()))
    model, tokenizer = load_model(model_id=args.model, precision=args.precision)
    tools = [ClinVarTool(), EnsemblTool(), GeneReviewsTool(), ClinicalTrialsTool(), PubMedTool(), UniProtTool(),
             PharmGKBTool(), OMIMTool(), OpenFDATool(), PharosTool(), N1CGuidelinesTool(), GeneticsReferenceTool(),
             ClinGenDosageTool()]
    agent = Agent(model, tokenizer, tools, show_thinking=args.show_thinking)
    output = ReportBuilder(agent).build(report_input)

    stem = Path(args.patient_json).with_suffix("")
    Path(f"{stem}_report.md").write_text(output.report_markdown)
    Path(f"{stem}_trace.json").write_text(json.dumps(agent.last_trace, indent=2, default=str))
    print(output.report_markdown)
    print(f"\nReport: {stem}_report.md\nTrace: {stem}_trace.json")


if __name__ == "__main__":
    main()
