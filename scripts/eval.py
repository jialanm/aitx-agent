"""Run a question set through the agent and report exact-match accuracy.

Defaults to the challenge's Phase 1 validator set, split into a questions file
the agent sees and an answers file it never does. Outputs spec-compliant
TaskOutput JSON per question: {id, response, evidence: [{source, time_accessed,
justification}]}.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.evaluation import file_sha256, is_exact_match, load_answers, load_questions
from src.model import MODEL_ID, load_model
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


DEFAULT_QUESTIONS = "data/phase1_questions.json"
DEFAULT_ANSWERS = "data/phase1_answers.json"


def run_metadata(questions_path: str, answers_path: str | None) -> dict:
    """What is needed to reproduce or compare this run."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "model_id": MODEL_ID,
        "decoding": "greedy (do_sample=False)",
        "questions_path": questions_path,
        "questions_sha256": file_sha256(questions_path),
        "answers_path": answers_path,
        "answers_sha256": file_sha256(answers_path) if answers_path else None,
    }


def validate_output(output_dict: dict) -> list[str]:
    """Validate that output matches the project.md spec.

    Returns list of validation errors (empty = valid).
    """
    errors = []
    if not output_dict.get("id"):
        errors.append("Missing or empty 'id'")
    if not isinstance(output_dict.get("response"), str):
        errors.append("'response' must be a string")
    if not output_dict.get("response"):
        errors.append("Empty 'response'")

    evidence = output_dict.get("evidence", [])
    if not isinstance(evidence, list):
        errors.append("'evidence' must be a list")
    elif len(evidence) == 0:
        errors.append("'evidence' must contain at least one item")
    else:
        for i, ev in enumerate(evidence):
            if not ev.get("source") or not ev["source"].startswith("http"):
                errors.append(f"evidence[{i}].source must be a valid URL, got: {ev.get('source', '')!r}")
            if not isinstance(ev.get("time_accessed"), int) or ev["time_accessed"] < 1_000_000_000:
                errors.append(f"evidence[{i}].time_accessed must be a Unix timestamp")
            justification = ev.get("justification", "")
            if not justification:
                errors.append(f"evidence[{i}].justification is empty")
            elif len(justification) > 500:
                errors.append(f"evidence[{i}].justification exceeds 500 chars ({len(justification)})")
    return errors


def run_evaluation(
    validation_path: str = DEFAULT_QUESTIONS,
    answers_path: str | None = DEFAULT_ANSWERS,
    output_path: str = "data/eval_results.json",
    outputs_path: str = "data/eval_outputs.json",
):
    """Run evaluation on validation set.

    Args:
        validation_path: Path to validation JSON.
        answers_path: Path to the answers JSON, or None to run unscored.
        output_path: Path to save evaluation stats.
        outputs_path: Path to save spec-compliant TaskOutput JSONs.
    """
    print("Loading model...")
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

    print("Loading questions...")
    validation_data = load_questions(validation_path)
    metadata = run_metadata(validation_path, answers_path)

    # Answers are loaded separately and consulted only after the agent has
    # answered. The question records passed to the agent never contain them.
    reference = load_answers(answers_path) if answers_path else {}
    print(f"{len(validation_data)} questions, {len(reference)} with reference answers")

    task_outputs = []  # Spec-compliant TaskOutput dicts
    results = []       # Internal eval results
    stats = {
        "total": 0,
        "correct": 0,
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_format": defaultdict(lambda: {"total": 0, "correct": 0}),
    }

    for i, item in enumerate(validation_data):
        task_input = TaskInput(**item)
        task_id = task_input.id
        category = task_input.question.category
        answer_format = task_input.question.answer_format

        print(f"\n[{i+1}/{len(validation_data)}] Processing {task_id} ({category})...")
        start_time = time.time()

        try:
            output = agent.run(task_input)
            elapsed = time.time() - start_time

            # Serialize to spec-compliant dict
            output_dict = json.loads(output.model_dump_json())
            task_outputs.append(output_dict)

            # Validate output
            validation_errors = validate_output(output_dict)
            if validation_errors:
                print(f"  VALIDATION WARNINGS:")
                for err in validation_errors:
                    print(f"    - {err}")

            result = {
                "id": task_id,
                "category": category,
                "answer_format": answer_format,
                "response": output.response,
                "evidence_count": len(output.evidence),
                "evidence_urls": [e.source for e in output.evidence],
                "time_seconds": round(elapsed, 1),
                "valid": len(validation_errors) == 0,
            }

            # Check against reference if available
            if task_id in reference:
                expected = reference[task_id]
                is_correct = is_exact_match(output.response, expected)
                result["expected"] = expected
                result["correct"] = is_correct

                stats["total"] += 1
                stats["by_category"][category]["total"] += 1
                stats["by_format"][answer_format]["total"] += 1

                if is_correct:
                    stats["correct"] += 1
                    stats["by_category"][category]["correct"] += 1
                    stats["by_format"][answer_format]["correct"] += 1

                status = "CORRECT" if is_correct else "WRONG"
                print(f"  Response: {output.response} | Expected: {expected} | {status}")
            else:
                print(f"  Response: {output.response} | Evidence: {len(output.evidence)} items | {elapsed:.1f}s")

            results.append(result)

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "id": task_id,
                "category": category,
                "answer_format": answer_format,
                "response": "",
                "error": str(e),
            })
            # Still produce a minimal valid output
            task_outputs.append({
                "id": task_id,
                "response": "",
                "evidence": [],
            })

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(validation_data)}")

    # Output validity check
    valid_count = sum(1 for r in results if r.get("valid", False))
    print(f"Valid outputs (spec-compliant): {valid_count}/{len(results)}")

    if stats["total"] > 0:
        accuracy = stats["correct"] / stats["total"] * 100
        print(f"Overall accuracy: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

        print("\nBy category:")
        for cat, cat_stats in sorted(stats["by_category"].items()):
            cat_acc = cat_stats["correct"] / cat_stats["total"] * 100 if cat_stats["total"] > 0 else 0
            print(f"  {cat}: {cat_stats['correct']}/{cat_stats['total']} ({cat_acc:.1f}%)")

        print("\nBy answer format:")
        for fmt, fmt_stats in sorted(stats["by_format"].items()):
            fmt_acc = fmt_stats["correct"] / fmt_stats["total"] * 100 if fmt_stats["total"] > 0 else 0
            print(f"  {fmt}: {fmt_stats['correct']}/{fmt_stats['total']} ({fmt_acc:.1f}%)")

    # Print per-question summary
    print("\nPer-question results:")
    for r in results:
        ev_count = r.get("evidence_count", 0)
        valid_mark = "OK" if r.get("valid") else "INVALID"
        status = ""
        if "correct" in r:
            status = " CORRECT" if r["correct"] else " WRONG"
        print(f"  {r['id']} | {r['response']:<30} | evidence={ev_count} | {valid_mark}{status}")

    # Save spec-compliant outputs
    with open(outputs_path, "w") as f:
        json.dump(task_outputs, f, indent=2)
    print(f"\nSpec-compliant outputs saved to {outputs_path}")

    # Save evaluation stats
    with open(output_path, "w") as f:
        json.dump(
            {"metadata": metadata, "results": results, "stats": dict(stats)},
            f, indent=2, default=dict,
        )
    print(f"Evaluation stats saved to {output_path}")

    # Print example output for verification
    if task_outputs:
        print(f"\nExample output (first question):")
        print(json.dumps(task_outputs[0], indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AI-Tx evaluation")
    parser.add_argument("--validation", default=DEFAULT_QUESTIONS, help="Path to question set JSON")
    parser.add_argument("--answers", default=DEFAULT_ANSWERS,
                        help="Path to answers JSON ({id, answer_expected} records); pass '' to run unscored")
    parser.add_argument("--output", default="data/eval_results.json", help="Path to save eval stats")
    parser.add_argument("--outputs", default="data/eval_outputs.json", help="Path to save spec-compliant outputs")
    args = parser.parse_args()

    run_evaluation(args.validation, args.answers or None, args.output, args.outputs)
