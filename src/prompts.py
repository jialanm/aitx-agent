"""Category-specific system prompts for the ReAct agent."""

from __future__ import annotations

from .preprocessing import ParsedVariant
from .variant_validation import ValidatedVariant

# Tool routing per category
TOOL_PRIORITY = {
    # Phase 1 question categories (retired task, kept for the eval set).
    # VEP is not listed: the consequence, exon and frame are fetched for
    # every variant in step 1, before the model runs.
    "Established_Targeted": ["search_genereviews", "search_clinvar", "genetics_reference", "search_fda_labels", "search_pharmgkb", "search_pubmed"],
    "Established_Supportive": ["search_genereviews", "search_omim", "search_pubmed"],
    "Clinical_Trials": ["search_clinical_trials", "search_omim", "search_pubmed"],
    "Drug_Development_and_Repurposing": ["search_pubmed", "search_pharmgkb", "search_fda_labels", "search_clinical_trials"],
    "Variant_Assessment": ["search_clinvar", "genetics_reference", "query_uniprot", "search_pubmed"],
    # Phase 3 report goals, in the order the report is built. Only tools
    # with an adapter are listed; a name here is a promise the model can
    # call it. Planned sources move in as their adapters are built:
    #   DrugBank needs a licensed key (skipped 2026-10-08)
    "goal1_existing_targeted": ["search_genereviews", "search_pubmed", "search_clinical_trials"],
    "goal2_existing_supportive": ["search_genereviews", "search_pubmed", "search_clinical_trials"],
    "goal3_ongoing_trials": ["search_clinical_trials"],
    "goal4_drug_repurposing": ["query_pharos", "clingen_dosage"],
    "goal5_personalized_therapeutics": ["n1c_aso_guidelines", "genetics_reference", "clingen_dosage", "query_uniprot"],
}

# Asked once per multiple-choice question, with thinking off, before the
# answering loop. The result is verified against the prompt text by code.
OPTION_EXTRACTION_INSTRUCTION = (
    "List the answer choices offered in the following question as a JSON array "
    "of strings, copying each choice exactly as written. Output only the JSON "
    "array, nothing else.\n\nQuestion: "
)

ANSWER_FORMAT_INSTRUCTIONS = {
    "binary": (
        "You MUST respond with EXACTLY 'Yes' or 'No' as your final answer. "
        "Do not include any other text in your final answer."
    ),
    "multiple_choice": (
        "You MUST respond with the EXACT text of one of the provided options. "
        "Do not include the letter prefix (A, B, C, D), just the option text."
    ),
    "numeric_match": (
        "You MUST respond with EXACTLY one number as your final answer. "
        "No units, no text, just the number."
    ),
    "string_match": (
        "You MUST respond with a concise, exact term as your final answer. "
        "Use standard nomenclature (e.g., drug generic names, GOF/LOF)."
    ),
}


def _format_variant_summary(
    variants: list[ParsedVariant],
    validated: list[ValidatedVariant] | None = None,
    consequences: list[str] | None = None,
) -> str:
    """Format parsed variants into a readable summary.

    When validation results are given (one per variant, same order), each
    variant also shows its current transcript form, genomic position, any
    superseded transcript version, and the validator's warnings; when VEP
    consequences are given, each variant shows its predicted consequence.
    The local regex label ("Variant type") is no longer shown: it misread
    bracketed protein strings, and VEP's consequence replaces it.
    """
    parts = []
    for i, v in enumerate(variants):
        desc = (
            f"Gene: {v.gene}\n"
            f"Transcript: {v.transcript}\n"
            f"cDNA: {v.variant_cdna}\n"
            f"Protein: {v.variant_protein}\n"
            f"Zygosity: {v.zygosity}"
        )
        if validated and i < len(validated):
            desc += "\n" + _format_validation(validated[i])
        if consequences and i < len(consequences):
            desc += "\n" + consequences[i]
        parts.append(desc)
    return "\n---\n".join(parts)


def _format_validation(val: ValidatedVariant) -> str:
    if not val.valid:
        return (
            f"VALIDATION FAILED for {val.submitted}: " + "; ".join(val.warnings)
        )
    lines = [
        f"Validated (VariantValidator, GRCh38): {val.transcript_variant}",
        f"Genomic position: {val.genomic_variant}",
    ]
    if val.superseded_transcript:
        lines.append(
            f"Note: submitted transcript {val.superseded_transcript} is superseded; "
            f"the current version is used above"
        )
    if val.warnings:
        lines.append("Validator warnings: " + "; ".join(val.warnings))
    return "\n".join(lines)


# Prompt-sensitivity control. "reword" swaps two headings for synonyms with
# the same meaning; a run under it should match the base run if the model is
# reading content rather than wording. Set once per eval run, recorded in
# the run metadata.
PROMPT_VARIANT = "none"
REWORDINGS = {
    "## Tool Usage Strategy": "## How to use the tools",
    "Recommended tool order for this category:": "Suggested order of tools for this category:",
}


def set_prompt_variant(name: str) -> None:
    global PROMPT_VARIANT
    if name not in ("none", "reword", "shuffle_schemas"):
        raise ValueError(f"unknown prompt variant {name!r}")
    PROMPT_VARIANT = name


def build_system_prompt(
    category: str,
    parsed_variants: list[ParsedVariant],
    clinical_context: str,
    answer_format: str,
    date_submitted: str,
    options: list[str] | None = None,
    validated: list[ValidatedVariant] | None = None,
    consequences: list[str] | None = None,
) -> str:
    """Build the system prompt for the ReAct agent.

    For multiple choice, verified options are spelled out in the format
    instruction so the model sees them exactly as the scorer expects.
    """

    variant_summary = _format_variant_summary(parsed_variants, validated, consequences)
    format_instruction = ANSWER_FORMAT_INSTRUCTIONS.get(answer_format, "")
    if answer_format == "multiple_choice" and options:
        format_instruction = (
            "You MUST respond with exactly one of these options, spelled as given: "
            + "; ".join(options)
            + ". Do not include any other text in your final answer."
        )
    tool_order = TOOL_PRIORITY.get(category, ["search_pubmed"])
    tool_order_str = " → ".join(tool_order)

    base_prompt = f"""You are a precision medicine expert assistant. Your task is to answer a clinical genetics question accurately and concisely using available biomedical tools.

## Patient Information

### Genetic Variants
{variant_summary}

### Clinical Context
{clinical_context}

## Question Category: {category}

## Date Context
The question was submitted on {date_submitted}. When evaluating time-sensitive information (e.g., FDA approvals, clinical trial status), consider this date as the reference point.

## Tool Usage Strategy
Recommended tool order for this category: {tool_order_str}

**Every step your answer rests on needs a tool result that supports that step.** One tool call is the minimum, not the target: a result that names a drug, gene or trial without saying it applies to this patient's condition and variant does not support a step about it. Even if you think you know the answer, verify it with a tool call first; your evidence must come from real API responses, not from memory.

Start by calling the first recommended tool above. After receiving tool results, you may answer or call additional tools.

Your thinking is not kept between turns; only what you write outside it is. So before each tool call, write one line outside your thinking that starts with NOTES: and states what is established so far and what you will check next, then make the call. You may call more than one tool in the same turn when you already know what you need.

## Answer Format
{format_instruction}

## Critical Rules
1. **Call the tools the answer needs, never none.** A step with no supporting tool result is a guess.
2. Use tools to find real evidence. Do NOT hallucinate facts or URLs.
3. After gathering evidence from tools, provide your final answer directly as plain text (no tool calls).
4. Keep your reasoning concise. Focus on answering the specific question asked.
5. Your final response should be ONLY the answer — no explanation, no preamble.
6. A fact you remember is a hypothesis, not evidence. Before a remembered fact decides your answer, confirm it with a tool call and say which tool. A drug, trial, gene or guideline named in your answer must appear in a tool result of this conversation."""

    # Category-specific additions
    category_addendum = _get_category_addendum(category)
    if category_addendum:
        base_prompt += f"\n\n## Category-Specific Guidance\n{category_addendum}"

    if PROMPT_VARIANT == "reword":
        for original, synonym in REWORDINGS.items():
            base_prompt = base_prompt.replace(original, synonym)
    return base_prompt


def _get_category_addendum(category: str) -> str:
    """Get category-specific prompt additions."""
    addenda = {
        "Established_Targeted": (
            "Focus on FDA-approved targeted therapies for this specific variant/condition. "
            "Check GeneReviews for established management guidelines, then verify with ClinVar "
            "for variant pathogenicity. Consider whether the specific variant is covered by "
            "the drug's indication (e.g., specific mutations may or may not be included)."
        ),
        "Established_Supportive": (
            "Focus on supportive care and management recommendations. Check GeneReviews "
            "for standard-of-care guidelines. Consider drugs to AVOID as well as those "
            "recommended. Look for contraindicated medications specific to this condition."
        ),
        "Clinical_Trials": (
            "Search for clinical trials that are ACTIVE as of the submitted date. "
            "A trial is 'active' if its status is Recruiting, Active not recruiting, "
            "or Not yet recruiting. Completed or terminated trials are NOT active. "
            "Be specific about the gene/condition when searching."
        ),
        "Drug_Development_and_Repurposing": (
            "Focus on investigational therapies and drugs being repurposed. "
            "Check for FDA approval status — distinguish between approved and "
            "investigational. Search PubMed for recent drug development research."
        ),
        "Variant_Assessment": (
            "Assess the functional impact of the specific variant. Use Ensembl VEP "
            "for computational predictions. Check ClinVar for clinical significance. "
            "For GOF/LOF determination, consider the variant type (missense, nonsense, "
            "frameshift) and its location in the protein domain."
        ),
    }
    return addenda.get(category, "")


def build_user_message(prompt: str, parsed_variants: list[ParsedVariant]) -> str:
    """Build the user message containing the question."""
    genes = ", ".join({v.gene for v in parsed_variants})
    variants = ", ".join(
        f"{v.variant_cdna} ({v.variant_protein})" for v in parsed_variants
    )

    return (
        f"Patient has variant(s) in {genes}: {variants}\n\n"
        f"Question: {prompt}"
    )
