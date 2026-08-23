import os

import pytest
from langchain_core.runnables import RunnableLambda, RunnableSequence

from app.config import Settings
from app.graph import build_generation_chain
from app.observability import configure_langsmith


def test_generation_is_an_explicit_lcel_sequence():
    chain = build_generation_chain(RunnableLambda(lambda messages: "grounded answer"))
    assert isinstance(chain.bound, RunnableSequence)
    assert len(chain.bound.steps) == 3


def test_generation_prompt_includes_permitted_citation_catalog():
    captured = {}

    def inspect(prompt):
        captured["text"] = prompt.to_string()
        return "grounded answer"

    chain = build_generation_chain(RunnableLambda(inspect))
    chain.invoke({
        "context": "[app/main.py:1-5]\ncode",
        "citation_catalog": "[app/main.py:1-5]",
        "question": "Where is the API?",
        "correction": "",
    })
    assert "Permitted citation catalog" in captured["text"]
    assert "[app/main.py:1-5]" in captured["text"]


def test_langsmith_is_disabled_without_a_key_by_default(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    configure_langsmith(Settings(langsmith_tracing=False))
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_enabled_langsmith_requires_a_key():
    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        configure_langsmith(Settings(langsmith_tracing=True, langsmith_api_key=None))
