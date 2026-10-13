"""The schemas state each source's coverage limit and where to go instead.

GeneReviews management text and FDA labels reflect United States practice;
a therapy approved only elsewhere is absent from both, and only PubMed
will say so. The model must be told this in the schemas it reads, not
learn it from an empty result.
"""

from src.tools.genereviews import GeneReviewsTool
from src.tools.openfda import OpenFDATool
from src.tools.pubmed import PubMedTool


def _description(tool) -> str:
    return tool.schema()["function"]["description"]


def test_us_only_sources_say_so_and_point_to_pubmed():
    assert "United States" in _description(GeneReviewsTool())
    assert "search PubMed" in _description(GeneReviewsTool())
    assert "United States label" in _description(OpenFDATool())
    assert "not evidence that no such drug exists" in _description(OpenFDATool())


def test_pubmed_names_itself_as_the_source_for_non_us_therapies():
    text = _description(PubMedTool())
    assert "approved outside the United States" in text
    assert "before answering None" in text


def test_coverage_notes_name_no_drug():
    # The note is a coverage fact, not a hint: no drug or disease named.
    for tool in (GeneReviewsTool(), OpenFDATool(), PubMedTool()):
        assert "ataluren" not in _description(tool).lower()
        assert "nonsense" not in _description(tool).lower()
