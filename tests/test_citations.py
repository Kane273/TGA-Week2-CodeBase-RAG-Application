from langchain_core.documents import Document

from app.graph import citation_document, cited_documents, evidence_fallback


def test_only_exact_cited_documents_are_exposed():
    cited = Document(page_content="x", metadata={
        "path": "app/main.py", "start_line": 10, "end_line": 20,
    })
    unrelated = Document(page_content="y", metadata={
        "path": "app/other.py", "start_line": 1, "end_line": 4,
    })
    result = cited_documents("The API starts here [app/main.py:10-20].", [cited, unrelated])
    assert result == [cited]


def test_precise_line_range_inside_retrieved_chunk_is_supported():
    document = Document(page_content="x", metadata={
        "path": "app/main.py", "start_line": 10, "end_line": 30,
        "symbol": "upload_documents",
    })
    assert citation_document("app/main.py", 14, 18, [document]) is document
    assert cited_documents("Upload starts here [app/main.py:14-18].", [document]) == [document]


def test_citation_outside_retrieved_range_is_rejected():
    document = Document(page_content="x", metadata={
        "path": "app/main.py", "start_line": 10, "end_line": 30,
    })
    assert citation_document("app/main.py", 31, 40, [document]) is None


def test_evidence_fallback_contains_retrieved_source_labels():
    document = Document(page_content="def upload_documents(): pass", metadata={
        "path": "app/main.py", "start_line": 10, "end_line": 30,
        "symbol": "upload_documents",
    })
    answer = evidence_fallback([document])
    assert "[app/main.py:10-30]" in answer
    assert "upload_documents" in answer
