"""Category-specific system prompts for the ReAct agent."""

from __future__ import annotations

from .preprocessing import ParsedVariant

# Tool routing per category
TOOL_PRIORITY = {
    "Established_Targeted": ["search_genereviews", "search_clinvar", "search_fda_labels", "search_pharmgkb", "search_pubmed"],
    "Established_Supportive": ["search_genereviews", "search_omim", "search_pubmed"],
    "Clinical_Trials": ["search_clinical_trials", "search_omim", "search_pubmed"],
    "Drug_Development_and_Repurposing": ["search_pubmed", "search_pharmgkb", "search_fda_labels", "search_clinical_trials"],
    "Variant_Assessment": ["query_ensembl", "search_clinvar", "query_uniprot", "search_pubmed"],
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


def _format_variant_summary(variants: list[ParsedVariant]) -> str:
    """Format parsed variants into a readable summary."""
    parts = []
    for v in variants:
        desc = (
            f"Gene: {v.gene}\n"
            f"Transcript: {v.transcript}\n"
            f"cDNA: {v.variant_cdna}\n"
            f"Protein: {v.variant_protein}\n"
            f"Zygosity: {v.zygosity}\n"
            f"Variant type: {v.variant_type}"
        )
        if v.protein_position:
            desc += f"\nProtein position: {v.protein_position}"
        parts.append(desc)
    return "\n---\n".join(parts)


def build_system_prompt(
    category: str,
    parsed_variants: list[ParsedVariant],
    clinical_context: str,
    answer_format: str,
    date_submitted: str,
) -> str:
    """Build the system prompt for the ReAct agent."""

    variant_summary = _format_variant_summary(parsed_variants)
    format_instruction = ANSWER_FORMAT_INSTRUCTIONS.get(answer_format, "")
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

**MANDATORY: You MUST call at least one tool before providing your final answer.** Even if you think you know the answer, you must verify it with a tool call first. Your evidence must come from real API responses, not from memory.

Start by calling the first recommended tool above. After receiving tool results, you may answer or call additional tools.

## Answer Format
{format_instruction}

## Critical Rules
1. **ALWAYS call at least one tool before answering.** Never skip tool calls. This is required for evidence collection.
2. Use tools to find real evidence. Do NOT hallucinate facts or URLs.
3. After gathering evidence from tools, provide your final answer directly as plain text (no tool calls).
4. Keep your reasoning concise. Focus on answering the specific question asked.
5. Your final response should be ONLY the answer — no explanation, no preamble."""

    # Category-specific additions
    category_addendum = _get_category_addendum(category)
    if category_addendum:
        base_prompt += f"\n\n## Category-Specific Guidance\n{category_addendum}"

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
