"""Gradio application entry point for AI-Tx Challenge."""

from __future__ import annotations

import json
import logging
import traceback

import gradio as gr

from src.agent import Agent
from src.model import load_model
from src.schemas import TaskInput, TaskOutput
from src.tools.clinvar import ClinVarTool
from src.tools.clinical_trials import ClinicalTrialsTool
from src.tools.ensembl import EnsemblTool
from src.tools.genereviews import GeneReviewsTool
from src.tools.pubmed import PubMedTool
from src.tools.uniprot import UniProtTool
from src.tools.pharmgkb import PharmGKBTool
from src.tools.omim import OMIMTool
from src.tools.openfda import OpenFDATool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent — initialized once at startup
_agent: Agent | None = None


def get_agent() -> Agent:
    """Get or initialize the global agent."""
    global _agent
    if _agent is None:
        logger.info("Loading model and initializing agent...")
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
        _agent = Agent(model, tokenizer, tools)
        logger.info("Agent initialized successfully.")
    return _agent


def predict(input_json: str) -> str:
    """Process a task input JSON and return task output JSON.

    Args:
        input_json: JSON string matching TaskInput schema.

    Returns:
        JSON string matching TaskOutput schema.
    """
    try:
        # Parse input
        data = json.loads(input_json)
        task_input = TaskInput(**data)

        # Run agent
        agent = get_agent()
        task_output = agent.run(task_input)

        # Return JSON
        return task_output.model_dump_json(indent=2)

    except json.JSONDecodeError as e:
        error_output = {
            "id": "error",
            "response": "",
            "evidence": [],
        }
        logger.error(f"JSON parse error: {e}")
        return json.dumps(error_output, indent=2)

    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        # Try to extract ID from input
        task_id = "error"
        try:
            task_id = json.loads(input_json).get("id", "error")
        except Exception:
            pass

        error_output = TaskOutput(
            id=task_id,
            response="",
            evidence=[],
        )
        return error_output.model_dump_json(indent=2)


# Build Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Input JSON",
        placeholder="Paste task input JSON here...",
        lines=20,
    ),
    outputs=gr.Textbox(
        label="Output JSON",
        lines=20,
    ),
    title="AI-Tx Challenge: Precision Medicine QA",
    description=(
        "Enter a task input JSON to get a precision medicine answer. "
        "The system uses Qwen3-8B with RAG-based tool calling to search "
        "ClinVar, PubMed, ClinicalTrials.gov, Ensembl, GeneReviews, "
        "UniProt, PharmGKB, OMIM, and FDA drug labels."
    ),
    api_name="predict",
)

if __name__ == "__main__":
    # Pre-load the agent
    get_agent()
    demo.launch(server_name="0.0.0.0", server_port=7860)
