import json
import re
import threading
import time
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.performance import CACHE_EVENTS


ALLOWED_TOOLS = {"hybrid_search", "symbol_search", "dependency_search"}
ALLOWED_FILTERS = {"branch", "language", "path"}


class RetrievalPlan(BaseModel):
    rewritten_question: str = Field(min_length=2, max_length=4000)
    queries: list[str] = Field(min_length=1, max_length=5)
    filters: dict[str, str] = Field(default_factory=dict)
    tools: list[Literal["hybrid_search", "symbol_search", "dependency_search"]] = Field(
        default_factory=lambda: ["hybrid_search"]
    )

    @field_validator("queries")
    @classmethod
    def clean_queries(cls, values: list[str]) -> list[str]:
        unique: list[str] = []
        for value in values:
            clean = " ".join(value.split())[:1000]
            if clean and clean not in unique:
                unique.append(clean)
        if not unique:
            raise ValueError("at least one non-empty query is required")
        return unique

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, values: dict[str, str]) -> dict[str, str]:
        return {
            key: str(value).strip()[:300]
            for key, value in values.items()
            if key in ALLOWED_FILTERS and str(value).strip()
        }


PLANNER_PROMPT = """You plan retrieval for a code and document knowledge base.
Rewrite follow-up questions so they stand alone, then produce up to {query_count} focused queries.
For architecture or cross-file questions, decompose the work into dependent queries ordered from
entry point to downstream components. Select only useful tools: hybrid_search, symbol_search,
dependency_search. Filters may contain only branch, language, or path; omit uncertain filters.
Return JSON only with keys rewritten_question, queries, filters, tools.

Recent conversation:
{history}

Current question: {question}
Route: {route}"""


def build_planning_chain(llm):
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    return (prompt | llm | StrOutputParser()).with_config(
        {"run_name": "conversation_aware_retrieval_plan"}
    )


def parse_retrieval_plan(raw: str, question: str, max_queries: int = 3) -> RetrievalPlan:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        payload = json.loads(match.group(0) if match else raw)
        plan = RetrievalPlan.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, AttributeError, TypeError):
        return RetrievalPlan(rewritten_question=question, queries=[question])
    return plan.model_copy(update={"queries": plan.queries[:max_queries]})


class RetrievalPlanCache:
    def __init__(self, ttl_seconds: int = 900, max_entries: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._items: dict[tuple, tuple[float, RetrievalPlan]] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple) -> RetrievalPlan | None:
        with self._lock:
            item = self._items.get(key)
            if item and item[0] > time.monotonic():
                CACHE_EVENTS.labels("query_plan", "hit").inc()
                return item[1]
            self._items.pop(key, None)
        CACHE_EVENTS.labels("query_plan", "miss").inc()
        return None

    def put(self, key: tuple, plan: RetrievalPlan) -> None:
        with self._lock:
            if len(self._items) >= self.max_entries:
                oldest = min(self._items, key=lambda item: self._items[item][0])
                self._items.pop(oldest, None)
            self._items[key] = (time.monotonic() + self.ttl_seconds, plan)
