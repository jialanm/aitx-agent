"""HGVS regex parsing and clinical context extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import TaskInput, Variant


@dataclass
class ParsedVariant:
    """Parsed HGVS variant information."""

    gene: str
    transcript: str
    variant_cdna: str
    variant_protein: str
    zygosity: str
    variant_type: str  # missense, nonsense, frameshift, deletion, duplication, splice, delins, unknown
    cdna_position: str  # e.g., "1521_1523", "2836"
    protein_position: str  # e.g., "508", "946"
    amino_acid_change: str  # e.g., "Phe508del", "Arg946Ter"


# cDNA patterns
_CDNA_SUB = re.compile(r"c\.(-?\d+)([ACGT])>([ACGT])")  # c.2836C>T
_CDNA_DEL = re.compile(r"c\.(-?\d+)(?:_(-?\d+))?del(\w*)")  # c.1521_1523del
_CDNA_DUP = re.compile(r"c\.(-?\d+)(?:_(-?\d+))?dup(\w*)")  # c.5266dup
_CDNA_DELINS = re.compile(r"c\.(-?\d+)(?:_(-?\d+))?delins(\w+)")  # c.NNN_NNNdelinsXXX
_CDNA_SPLICE = re.compile(r"c\.(-?\d+)([+-]\d+)([ACGT])>([ACGT])")  # c.NNN+N splice
_CDNA_SPLICE2 = re.compile(r"c\.(-?\d+)([+-]\d+)")  # simpler splice pattern
_CDNA_REPEAT = re.compile(r"c\.(\d+)([A-Z]+)\[(\d+)\]")  # c.52CAG[42]

# Protein patterns
_PROT_MISSENSE = re.compile(
    r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$"
)  # p.Val600Glu
_PROT_NONSENSE = re.compile(
    r"p\.([A-Z][a-z]{2})(\d+)(Ter|\*)"
)  # p.Arg946Ter or p.Arg946*
_PROT_FRAMESHIFT = re.compile(
    r"p\.([A-Z][a-z]{2})(\d+).*fs"
)  # p.Gly407AspfsTer14
_PROT_DEL = re.compile(
    r"p\.([A-Z][a-z]{2})(\d+)del"
)  # p.Phe508del
_PROT_REPEAT = re.compile(
    r"p\.([A-Z][a-z]{2})(\d+)\[(\d+)\]"
)  # p.Gln18[42]
_PROT_UNKNOWN = re.compile(r"p\.\?")  # p.?


def parse_variant(variant: Variant) -> ParsedVariant:
    """Parse HGVS notations from a Variant into structured data."""
    cdna = variant.variant_cdna
    protein = variant.variant_protein
    variant_type = "unknown"
    cdna_position = ""
    protein_position = ""
    amino_acid_change = ""

    # Determine variant type from protein notation first (more informative)
    if _PROT_FRAMESHIFT.match(protein):
        variant_type = "frameshift"
        m = _PROT_FRAMESHIFT.match(protein)
        protein_position = m.group(2)
        amino_acid_change = protein[2:]  # everything after "p."
    elif _PROT_NONSENSE.match(protein):
        variant_type = "nonsense"
        m = _PROT_NONSENSE.match(protein)
        protein_position = m.group(2)
        amino_acid_change = protein[2:]
    elif _PROT_DEL.match(protein):
        variant_type = "in-frame deletion"
        m = _PROT_DEL.match(protein)
        protein_position = m.group(2)
        amino_acid_change = protein[2:]
    elif _PROT_MISSENSE.match(protein):
        variant_type = "missense"
        m = _PROT_MISSENSE.match(protein)
        protein_position = m.group(2)
        amino_acid_change = protein[2:]
    elif _PROT_REPEAT.match(protein):
        variant_type = "repeat expansion"
        m = _PROT_REPEAT.match(protein)
        protein_position = m.group(2)
        amino_acid_change = protein[2:]
    elif _PROT_UNKNOWN.match(protein):
        # Protein effect unknown, determine from cDNA
        variant_type = _classify_cdna(cdna)
    else:
        # Fall back to cDNA classification
        variant_type = _classify_cdna(cdna)

    # Extract cDNA position
    cdna_position = _extract_cdna_position(cdna)

    # Extract protein position if not already set
    if not protein_position:
        prot_m = re.search(r"p\.\w+?(\d+)", protein)
        if prot_m:
            protein_position = prot_m.group(1)

    if not amino_acid_change and protein and protein != "p.?":
        amino_acid_change = protein[2:] if protein.startswith("p.") else protein

    return ParsedVariant(
        gene=variant.gene,
        transcript=variant.transcript,
        variant_cdna=cdna,
        variant_protein=protein,
        zygosity=variant.zygosity,
        variant_type=variant_type,
        cdna_position=cdna_position,
        protein_position=protein_position,
        amino_acid_change=amino_acid_change,
    )


def _classify_cdna(cdna: str) -> str:
    """Classify variant type from cDNA notation."""
    if _CDNA_DELINS.match(cdna):
        return "delins"
    if _CDNA_DEL.match(cdna):
        return "deletion"
    if _CDNA_DUP.match(cdna):
        return "duplication"
    if _CDNA_SPLICE.match(cdna) or _CDNA_SPLICE2.search(cdna):
        return "splice"
    if _CDNA_SUB.match(cdna):
        return "substitution"
    if _CDNA_REPEAT.match(cdna):
        return "repeat expansion"
    return "unknown"


def _extract_cdna_position(cdna: str) -> str:
    """Extract position number(s) from cDNA notation."""
    # Match position patterns like -32-13, 1521_1523, 2836
    m = re.search(r"c\.(-?\d+)(?:([+-]\d+))?(?:_(-?\d+))?", cdna)
    if m:
        start = m.group(1)
        offset = m.group(2) or ""
        end = m.group(3)
        if end:
            return f"{start}{offset}_{end}"
        return f"{start}{offset}"
    return ""


def preprocess_input(task_input: TaskInput) -> dict:
    """Preprocess a TaskInput into a structured context dict.

    Returns a dict with:
        - parsed_variants: list[ParsedVariant]
        - genes: list of gene symbols
        - clinical_context: the patient clinical context string
        - category: question category
        - answer_format: answer format type
        - date_submitted: date string
    """
    parsed_variants = [
        parse_variant(v) for v in task_input.patient.genotype
    ]

    genes = list({v.gene for v in parsed_variants})

    return {
        "parsed_variants": parsed_variants,
        "genes": genes,
        "clinical_context": task_input.patient.clinical_context,
        "category": task_input.question.category,
        "answer_format": task_input.question.answer_format,
        "prompt": task_input.question.prompt,
        "date_submitted": task_input.question.date_submitted,
    }
