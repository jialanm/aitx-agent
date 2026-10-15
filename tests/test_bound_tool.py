"""A bound tool hides its fixed arguments from the model and fills them itself."""

from src.tools.base import BaseTool, EvidenceRecord
from src.tools.bound import BoundTool


class _GR(BaseTool):
    def __init__(self):
        self.calls = []

    def schema(self):
        return {"type": "function", "function": {
            "name": "search_genereviews", "description": "GeneReviews chapters.",
            "parameters": {"type": "object", "properties": {
                "gene": {"type": "string"}, "nbk": {"type": "string"}, "section": {"type": "string"}},
                "required": ["gene"]}}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return "chapter text", [EvidenceRecord(url="https://www.ncbi.nlm.nih.gov/books/NBK1283/")]


def test_fixed_argument_is_removed_from_the_schema_and_named_in_the_description():
    bound = BoundTool(_GR(), {"gene": "AGXT", "not_a_param": "x"})
    fn = bound.schema()["function"]
    assert bound.name == "search_genereviews"
    assert "gene" not in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == []
    assert "Fixed for this patient by the harness: gene=AGXT" in fn["description"]
    assert "not_a_param" not in fn["description"]


def test_execute_fills_the_fixed_argument_and_overrides_the_models_value():
    inner = _GR()
    bound = BoundTool(inner, {"gene": "AGXT"})
    bound.execute(nbk="NBK1283", gene="DMD")
    assert inner.calls == [{"nbk": "NBK1283", "gene": "AGXT"}]


def test_nothing_bound_leaves_the_schema_unchanged():
    inner = _GR()
    assert BoundTool(inner, {"gene": None}).schema() == inner.schema()
