from langchain_core.documents import Document

from app.context import build_bounded_context, retrieval_policy
from app.embedding_cache import CachedEmbeddings
from app.planning import RetrievalPlan, RetrievalPlanCache
from app.retrieval import RepositoryIndex


def doc(identifier: str, content: str) -> Document:
    return Document(page_content=content, metadata={
        "chunk_id": identifier, "path": f"{identifier}.py",
        "start_line": 1, "end_line": 5,
    })


def test_rrf_rewards_documents_found_by_multiple_retrievers():
    shared = doc("shared", "shared")
    dense_only = doc("dense", "dense")
    sparse_only = doc("sparse", "sparse")
    ranked = RepositoryIndex.reciprocal_rank_fusion([
        ([dense_only, shared], 1.0), ([sparse_only, shared], 1.0),
    ])
    assert ranked[0] is shared


def test_context_budget_deduplicates_and_keeps_complete_chunks():
    first = doc("one", "same content")
    duplicate = doc("two", "same   content")
    oversized = doc("three", "x" * 500)
    context, selected = build_bounded_context([first, duplicate, oversized], 100)
    assert selected == [first]
    assert "same content" in context
    assert "three.py" not in context


def test_adaptive_policy_spends_more_work_on_architecture():
    simple = retrieval_policy("code_search", "Where is auth?")
    architecture = retrieval_policy("architecture", "Explain the architecture and data flow")
    assert architecture["hops"] > simple["hops"]
    assert architecture["top_k"] > simple["top_k"]


def test_embedding_cache_avoids_duplicate_model_calls(tmp_path):
    class Delegate:
        def __init__(self): self.calls = 0
        def embed_documents(self, texts):
            self.calls += 1
            return [[float(len(text))] for text in texts]

    delegate = Delegate()
    cache = CachedEmbeddings(delegate, "test-model", tmp_path / "embeddings.db")
    assert cache.embed_documents(["alpha", "beta"]) == [[5.0], [4.0]]
    assert cache.embed_documents(["alpha", "beta"]) == [[5.0], [4.0]]
    assert delegate.calls == 1


def test_query_plan_cache_is_bounded_and_expires():
    plan = RetrievalPlan(rewritten_question="Where is auth?", queries=["auth"])
    cache = RetrievalPlanCache(ttl_seconds=60, max_entries=1)
    cache.put(("first",), plan)
    assert cache.get(("first",)) is plan
    cache.put(("second",), plan)
    assert cache.get(("first",)) is None
    assert cache.get(("second",)) is plan
