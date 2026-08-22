from unittest.mock import Mock
import threading

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableSequence

from app.config import Settings
from app.planning import build_planning_chain, parse_retrieval_plan
from app.retrieval import RepositoryIndex


def document(chunk_id: str, path: str, start: int, symbol: str = "module") -> Document:
    return Document(
        page_content=f"content for {chunk_id}",
        metadata={
            "chunk_id": chunk_id, "path": path, "start_line": start,
            "end_line": start + 9, "repository": "demo", "branch": "main",
            "language": "py", "symbol": symbol,
        },
    )


def bare_index() -> RepositoryIndex:
    index = object.__new__(RepositoryIndex)
    index.settings = Settings(parent_context_chunks=1, max_retrieval_tool_calls=2)
    index._active_collections = {}
    index._cache_lock = threading.RLock()
    index._parent_cache = {}
    return index


def test_planning_chain_is_lcel():
    chain = build_planning_chain(RunnableLambda(lambda messages: "{}"))
    assert isinstance(chain.bound, RunnableSequence)
    assert len(chain.bound.steps) == 3


def test_plan_parser_allow_lists_filters_tools_and_bounds_queries():
    raw = '''```json
    {"rewritten_question":"How does authentication work?",
     "queries":["entry point","token validation","role checks","extra"],
     "filters":{"language":"py","unsafe":"ignored"},
     "tools":["hybrid_search","dependency_search"]}
    ```'''
    plan = parse_retrieval_plan(raw, "How does it work?", max_queries=3)
    assert plan.queries == ["entry point", "token validation", "role checks"]
    assert plan.filters == {"language": "py"}
    assert plan.tools == ["hybrid_search", "dependency_search"]


def test_invalid_plan_falls_back_to_original_question():
    plan = parse_retrieval_plan("not json", "Where is auth handled?")
    assert plan.rewritten_question == "Where is auth handled?"
    assert plan.queries == ["Where is auth handled?"]


def test_self_query_filters_are_combined_with_repository_scope():
    where = RepositoryIndex._filtered_where(
        "demo", "main", {"branch": "release", "language": "py", "bad": "value"}
    )
    assert where == {"$and": [
        {"repository": "demo"}, {"branch": "release"}, {"language": "py"},
    ]}


def test_parent_expansion_adds_neighboring_chunks():
    index = bare_index()
    docs = [document("one", "app.py", 1), document("two", "app.py", 11),
            document("three", "app.py", 21), document("other", "other.py", 1)]
    index._stored_documents = Mock(return_value=docs)
    expanded = index.expand_parent_context([docs[1]], "demo", "main")
    assert [item.metadata["chunk_id"] for item in expanded] == ["one", "two", "three"]


def test_tool_execution_is_allow_listed_and_budgeted():
    index = bare_index()
    hit = document("one", "app.py", 1)
    index.retrieve = Mock(return_value=[hit])
    index._rerank = Mock(return_value=[hit])
    index.expand_parent_context = Mock(return_value=[hit])
    docs, calls = index.retrieve_with_tools(
        "auth", "demo", "main", {},
        ["hybrid_search", "not_a_tool", "symbol_search", "dependency_search"],
        max_calls=2,
    )
    assert docs == [hit]
    assert calls == ["hybrid_search", "symbol_search"]
    assert index.retrieve.call_count == 2
