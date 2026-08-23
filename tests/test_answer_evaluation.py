import pytest
from langchain_core.documents import Document

from app.evaluation import (aggregate_answer_evaluations, deterministic_answer_metrics,
                            evaluate_answer, parse_judge_scores,
                            semantic_answer_relevance)


def source(path: str, text: str, start: int = 1, end: int = 20) -> Document:
    return Document(page_content=text, metadata={
        "path": path, "start_line": start, "end_line": end,
        "branch": "main", "chunk_id": path,
    })


def test_deterministic_answer_quality_metrics_cover_context_and_citations():
    documents = [
        source("app/security.py", "validate token authorization access roles claims"),
        source("app/unrelated.py", "unrelated helper function"),
    ]
    answer = "Token authorization validates access roles. [app/security.py:1-20]"
    metrics = deterministic_answer_metrics(
        "How does token authorization validate access roles?", answer, documents,
        ["app/security.py"],
    )
    assert metrics["context_precision"] == 0.5
    assert metrics["context_recall"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_completeness"] == 1.0
    assert metrics["faithfulness"] >= 0.8
    assert metrics["answer_relevance"] >= 0.8


def test_invalid_citation_and_uncited_claim_fail_grounding_metrics():
    documents = [source("app/security.py", "validate token")]
    metrics = deterministic_answer_metrics(
        "How is access validated?",
        "Access is always granted. [missing.py:1-4]\nTokens are accepted automatically.",
        documents, ["app/security.py"],
    )
    assert metrics["citation_precision"] == 0.0
    assert metrics["citation_completeness"] < 1.0
    assert metrics["faithfulness"] == 0.0


def test_semantic_judge_scores_override_lexical_proxies():
    class Judge:
        def invoke(self, payload):
            assert payload["question"] == "What happens?"
            return '{"answer_relevance": 0.91, "faithfulness": 0.87}'

    metrics = evaluate_answer(
        "What happens?", "A supported behavior. [app.py:1-20]",
        [source("app.py", "supported behavior")], ["app.py"], Judge(),
    )
    assert metrics["answer_relevance"] == 0.91
    assert metrics["faithfulness"] == 0.87


def test_judge_requires_valid_bounded_json_scores():
    assert parse_judge_scores('{"answer_relevance": 2, "faithfulness": -1}') == {
        "answer_relevance": 1.0, "faithfulness": 0.0,
    }
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_judge_scores("not-json")


def test_quality_gate_reports_metric_failures():
    details = [{"metrics": {"faithfulness": 0.6, "answer_relevance": 0.9}}]
    result = aggregate_answer_evaluations(
        details, {"faithfulness": 0.8, "answer_relevance": 0.8}
    )
    assert not result["passed"]
    assert result["failures"] == {
        "faithfulness": {"score": 0.6, "threshold": 0.8}
    }


def test_semantic_relevance_recognizes_paraphrases_without_keyword_overlap():
    class Embeddings:
        def embed_documents(self, texts):
            assert texts == ["How is access controlled?", "JWT claims authorize each request."]
            return [[1.0, 0.0], [0.8, 0.6]]

    assert semantic_answer_relevance(
        "How is access controlled?", "JWT claims authorize each request.", Embeddings()
    ) == 0.8


def test_semantic_relevance_falls_back_when_local_embeddings_are_unavailable():
    class BrokenEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("offline")

    assert semantic_answer_relevance("question", "answer", BrokenEmbeddings(), 0.35) == 0.35
