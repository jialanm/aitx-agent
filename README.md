---
title: AI-Tx Challenge
emoji: "\U0001F9EC"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: apache-2.0
hardware: l4
---

# AI-Tx Challenge: Precision Medicine QA

ReAct agent built on Qwen3-8B (Tier 1, ≤8B parameters) for precision medicine question answering.

## How it works

Given a patient's genotype, clinical context, and a therapeutic question, the agent:
1. Parses HGVS variant notation and clinical context
2. Routes to relevant biomedical APIs based on question category
3. Iteratively retrieves and reasons over evidence (ReAct loop, max 5 iterations)
4. Returns an exact-match answer with verifiable evidence URLs

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

## API Endpoint

```
POST /api/predict
Content-Type: application/json

{"data": ["<task_input_json_string>"]}
```

## Local Development

```bash
pip install -r requirements.txt
python app.py
```
