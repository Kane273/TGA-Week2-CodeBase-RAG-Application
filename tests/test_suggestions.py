from langchain_core.documents import Document

from app.suggestions import follow_up_questions


def test_architecture_follow_ups_use_retrieved_symbols_and_paths():
    result = {
        "route": "architecture",
        "documents": [
            Document(page_content="", metadata={"path": "app/graph.py", "symbol": "build_graph"}),
            Document(page_content="", metadata={"path": "app/main.py", "symbol": "chat"}),
        ],
    }
    questions = follow_up_questions(result)
    assert len(questions) == 3
    assert "`build_graph`" in questions[0]
    assert "`app/main.py`" in questions[0]


def test_follow_ups_have_safe_defaults_without_documents():
    questions = follow_up_questions({"route": "unknown", "documents": []})
    assert len(questions) == 3
    assert "this component" in questions[0]
