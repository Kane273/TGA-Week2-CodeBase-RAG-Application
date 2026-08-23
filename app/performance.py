import time
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram


STAGE_LATENCY = Histogram(
    "rag_stage_duration_seconds", "RAG pipeline stage latency", ["stage"]
)
CACHE_EVENTS = Counter(
    "rag_cache_events_total", "RAG cache hits and misses", ["cache", "result"]
)
CONTEXT_CHARACTERS = Histogram(
    "rag_context_characters", "Characters supplied to answer generation"
)
RETRIEVAL_RESULTS = Histogram(
    "rag_retrieval_results", "Documents returned by optimized retrieval"
)
ACTIVE_MODEL_REQUESTS = Gauge(
    "rag_active_model_requests", "Model requests currently executing"
)
ANSWER_QUALITY = Histogram(
    "rag_answer_quality_ratio", "Answer quality score from zero to one", ["metric"],
    buckets=(0.25, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
)
INDEX_JOBS = Counter(
    "rag_index_jobs_total", "Completed indexing job outcomes", ["kind", "status"]
)


@contextmanager
def stage_timer(stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        STAGE_LATENCY.labels(stage).observe(time.perf_counter() - started)
