"""The genetics reference: every rule has a source, names no gene or drug in its
body, and comes back verbatim with its PubMed links as evidence."""

from src.tools.genetics_reference import GeneticsReferenceTool, load_sections

TOPICS = ["reading-frame", "splicing", "exon-skipping", "nonsense-mediated-decay", "read-through"]
# Names that would make a rule a hint rather than a rule.
LEAKS = ["eteplirsen", "ataluren", "golodirsen", "viltolarsen", "casimersen", "pyridoxine", "lumasiran",
         "GRIN2B", "AGXT", "ANO10", "KMT2B", "NF1", "KCNT1", "SLC35A2", "COL1A1", "DDC"]


def test_file_has_the_five_rules_each_with_a_pubmed_source():
    sections = load_sections()
    assert list(sections) == TOPICS
    for topic, s in sections.items():
        assert "Applies when:" in s["text"], topic
        assert "Source:" in s["text"], topic
        assert s["sources"], topic


def test_rule_bodies_name_no_gene_or_drug():
    for topic, s in load_sections().items():
        body = s["text"].split("Source:")[0]
        for leak in LEAKS:
            assert leak.lower() not in body.lower(), (topic, leak)


def test_tool_returns_the_section_and_its_sources_as_evidence():
    text, evidence = GeneticsReferenceTool().execute(topic="splicing")
    assert text.startswith("Genetics reference, splicing: What a transcript keeps of a partly deleted exon")
    assert "exon definition" in text
    assert [e.url for e in evidence] == ["https://pubmed.ncbi.nlm.nih.gov/7852296/", "https://pubmed.ncbi.nlm.nih.gov/28188674/"]


def test_no_topic_lists_the_rules_and_unknown_topic_is_refused():
    text, evidence = GeneticsReferenceTool().execute()
    assert all(t in text for t in TOPICS) and evidence == []
    text, evidence = GeneticsReferenceTool().execute(topic="telomeres")
    assert text.startswith("Genetics reference has no rule 'telomeres'") and evidence == []


def test_schema_names_every_topic():
    schema = GeneticsReferenceTool().schema()["function"]
    assert schema["name"] == "genetics_reference"
    assert schema["parameters"]["properties"]["topic"]["enum"] == TOPICS
    for t in TOPICS:
        assert t in schema["description"]
