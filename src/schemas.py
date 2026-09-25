"""Pydantic models for AI-Tx Challenge I/O JSON."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Variant(BaseModel):
    gene: str
    transcript: str
    variant_cdna: str
    variant_protein: str
    zygosity: str  # "heterozygous", "homozygous", or "hemizygous"


class Patient(BaseModel):
    genotype: list[Variant]
    clinical_context: str


class Question(BaseModel):
    category: str  # one of 5 categories
    answer_format: str  # binary, multiple_choice, numeric_match, string_match
    prompt: str
    date_submitted: str


class TaskInput(BaseModel):
    id: str
    patient: Patient
    question: Question


class EvidenceItem(BaseModel):
    source: str
    time_accessed: int
    justification: str = Field(max_length=500)


class TaskOutput(BaseModel):
    id: str
    response: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
