"""Debug a single question with verbose trace."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.model import load_model
from src.schemas import TaskInput
from src.tools.clinvar import ClinVarTool
from src.tools.clinical_trials import ClinicalTrialsTool
from src.tools.ensembl import EnsemblTool
from src.tools.genereviews import GeneReviewsTool
from src.tools.pubmed import PubMedTool
from src.tools.uniprot import UniProtTool
from src.tools.pharmgkb import PharmGKBTool
from src.tools.omim import OMIMTool
from src.tools.openfda import OpenFDATool

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(message)s")


def debug_question(question_json: str | dict):
    """Run a single question with full debug output."""
    if isinstance(question_json, str):
        data = json.loads(question_json)
    else:
        data = question_json

    task_input = TaskInput(**data)

    print("=" * 60)
    print(f"Task ID: {task_input.id}")
    print(f"Category: {task_input.question.category}")
    print(f"Format: {task_input.question.answer_format}")
    print(f"Question: {task_input.question.prompt}")
    print(f"Gene(s): {', '.join(v.gene for v in task_input.patient.genotype)}")
    print("=" * 60)

    print("\nLoading model...")
    model, tokenizer = load_model()

    tools = [
        ClinVarTool(),
        EnsemblTool(),
        GeneReviewsTool(),
        ClinicalTrialsTool(),
        PubMedTool(),
        UniProtTool(),
        PharmGKBTool(),
        OMIMTool(),
        OpenFDATool(),
    ]
    agent = Agent(model, tokenizer, tools)

    print("\nRunning agent...")
    start = time.time()
    output = agent.run(task_input)
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Response: {output.response}")
    print(f"Evidence items: {len(output.evidence)}")
    for i, ev in enumerate(output.evidence):
        print(f"  [{i+1}] {ev.source}")
        print(f"      {ev.justification[:100]}...")
    print(f"\nTime: {elapsed:.1f}s")
    print(f"\nFull output:\n{output.model_dump_json(indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: load first validation question
        print("Usage: python scripts/debug_question.py <question_json_or_index>")
        print("No argument provided, loading first validation question...")
        with open("data/phase1_questions.json") as f:
            questions = json.load(f)
        debug_question(questions[0])
    else:
        arg = sys.argv[1]
        try:
            # Try as index into validation set
            idx = int(arg)
            with open("data/phase1_questions.json") as f:
                questions = json.load(f)
            if 0 <= idx < len(questions):
                debug_question(questions[idx])
            else:
                print(f"Index {idx} out of range (0-{len(questions)-1})")
        except ValueError:
            # Try as JSON string or file path
            if Path(arg).exists():
                with open(arg) as f:
                    debug_question(json.load(f))
            else:
                debug_question(arg)
