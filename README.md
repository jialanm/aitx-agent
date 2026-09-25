---
title: AI-Tx Challenge
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
license: apache-2.0
---

# AI-Tx Agent: Rare Disease Therapeutics QA

ReAct agent built on Qwen3-8B for the [AI-Tx Challenge](https://aitxchallenge.org/),
Tier 1 (open weights). Given a rare disease patient's genotype and clinical context, the agent
retrieves evidence from live biomedical APIs and answers with verifiable source URLs.

**Status (September 2026).** The code implements the challenge's Phase 1
question-answering task, which closed on June 30, 2026. The final task,
[Actionability Report Generation](https://aitxchallenge.org/#markdown-report-template),
takes the same patient input without a question and returns a structured markdown
report. That task is not implemented yet. No accuracy numbers are claimed here;
see the evaluation section.

## How it works

1. Parses HGVS variant notation and clinical context deterministically.
2. Routes to biomedical APIs in a category-specific priority order.
3. Iteratively retrieves and reasons over evidence (ReAct loop, max 5 iterations).
4. Normalises the answer to the requested format and validates it, re-prompting once if invalid.
5. Returns the answer with evidence records built only from URLs the tools actually fetched.

## Tools (9)

| Tool | API | Use case |
|------|-----|----------|
| ClinVar | NCBI E-utilities | Variant pathogenicity and clinical significance |
| Ensembl | Ensembl REST | Variant effect prediction (VEP), gene information |
| GeneReviews | NCBI PubMed/Bookshelf | Expert management guidelines for genetic conditions |
| ClinicalTrials.gov | CT.gov v2 API | Active clinical trials for gene/condition |
| PubMed | NCBI E-utilities | Biomedical literature search |
| UniProt | UniProt REST | Protein domains, functional annotations |
| PharmGKB | PharmGKB API | Drug-gene interactions, pharmacogenomics |
| OMIM | NCBI E-utilities | Gene-disease relationships, inheritance patterns |
| FDA Labels | openFDA API | Approved drug indications, contraindications |

No API keys are required. All calls run at each service's unauthenticated rate limit.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer. Running the
agent loads Qwen3-8B in bfloat16, which needs a GPU with at least 24 GB of memory
(about 16.4 GB for the weights plus working memory). The unit tests need no GPU.

```bash
uv sync                      # creates .venv and installs dependencies
uv run pytest -m "not slow"  # unit tests, no GPU or network
uv run pytest -m slow        # integration tests against the live APIs
uv run python app.py         # Gradio app on http://localhost:7860
```

`requirements.txt` duplicates the dependency list because Hugging Face Spaces
installs from it; `pyproject.toml` is the source of truth.

## API

The Gradio app exposes one named endpoint, `/predict`, which takes and returns
JSON strings. With `gradio_client`:

```python
from gradio_client import Client

client = Client("http://localhost:7860")
output_json = client.predict(input_json, api_name="/predict")
```

Input follows the Phase 1 task schema: `id`, `patient` (`genotype` list and
`clinical_context`), and `question` (`category`, `answer_format`, `prompt`,
`date_submitted`). Output is `id`, `response`, and an `evidence` list of
`source`, `time_accessed`, and `justification` records.

## Evaluation

```bash
uv run python scripts/eval.py              # runs the Phase 1 validator set, reports exact-match accuracy
uv run python scripts/debug_question.py 0  # trace a single question by index
uv run python scripts/fetch_validator.py   # re-download the set at its pinned revision and verify checksum
```

The question set is the challenge's public
[Phase1_Model_Validator](https://huggingface.co/datasets/aitxchallenge/Phase1_Model_Validator)
(MIT, 16 questions with reference answers), split into `data/phase1_questions.json`,
which is all the agent ever sees, and `data/phase1_answers.json`, which the harness
consults only after the agent has answered. The manifest beside them records the
source revision and a checksum for each file. Accuracy has not been measured yet.
Each eval run records the git commit, model id, decoding settings, and file
checksums in its results file. Eval outputs are not committed.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgements

Claude Code was used as a coding assistant during development. Design
decisions, clinical review, and the evaluation are the author's own.
