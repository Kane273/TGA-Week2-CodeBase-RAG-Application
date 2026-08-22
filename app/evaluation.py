import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langsmith import traceable

from app.graph import CITATION_PATTERN, citation_document


WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
JUDGE_PROMPT = """You are a strict RAG evaluator. Score each metric from 0.0 to 1.0.
answer_relevance: the answer directly and completely addresses the question.
faithfulness: every factual answer claim is supported by the supplied context.
Do not reward confident language or unsupported details. Return JSON only:
{{"answer_relevance": number, "faithfulness": number}}

Question: {question}
Answer: {answer}
Context:
{context}"""

@dataclass
class GateResult:
    passed: bool
    document_count: int
    errors: list[str]


def _terms(text: str) -> set[str]:
    stop = {"the", "and", "that", "this", "with", "from", "where", "what", "how",
            "does", "into", "your", "about", "which", "then", "when", "are", "for"}
    return {word.lower() for word in WORD_PATTERN.findall(text) if word.lower() not in stop}


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def build_local_answer_judge(settings):
    """Build a no-key semantic judge using the configured local Ollama model."""
    llm = ChatOllama(model=settings.chat_model, base_url=settings.ollama_base_url,
                     temperature=0)
    return (ChatPromptTemplate.from_template(JUDGE_PROMPT) | llm | StrOutputParser()).with_config(
        {"run_name": "local_answer_quality_judge"}
    )


def parse_judge_scores(raw: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        payload = json.loads(match.group(0) if match else raw)
        return {name: _bounded(float(payload[name]))
                for name in ("answer_relevance", "faithfulness")}
    except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError):
        raise ValueError("Local answer judge returned invalid JSON scores")


def deterministic_answer_metrics(question: str, answer: str, documents: list[Document],
                                 expected_paths: list[str]) -> dict[str, float]:
    """Auditable no-model metrics for citations, context coverage, and lexical support."""
    expected = set(expected_paths)
    retrieved_paths = {str(doc.metadata.get("path", "")) for doc in documents}
    context_precision = (len(retrieved_paths & expected) / len(retrieved_paths)
                         if retrieved_paths else 0.0)
    context_recall = len(retrieved_paths & expected) / max(len(expected), 1)

    matches = CITATION_PATTERN.findall(answer)
    valid = [(path, start, end) for path, start, end in matches
             if citation_document(path, int(start), int(end), documents) is not None]
    citation_precision = len(valid) / max(len(matches), 1)

    citation_group = r"(?:\[[^\[\]:]+:\d+-\d+\]\s*)+"
    normalized_answer = re.sub(
        rf"([.!?])\s*({citation_group})", r" \2\1", answer
    )
    claims = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", normalized_answer)
              if len(_terms(part)) >= 3]
    cited_claims = [claim for claim in claims if CITATION_PATTERN.search(claim)]
    citation_completeness = len(cited_claims) / max(len(claims), 1)

    supported: list[float] = []
    for claim in cited_claims:
        claim_terms = _terms(CITATION_PATTERN.sub("", claim))
        cited_docs = [citation_document(path, int(start), int(end), documents)
                      for path, start, end in CITATION_PATTERN.findall(claim)]
        evidence_terms = set().union(*(_terms(doc.page_content) for doc in cited_docs if doc))
        supported.append(len(claim_terms & evidence_terms) / max(len(claim_terms), 1))
    lexical_faithfulness = sum(supported) / len(supported) if supported else 0.0
    question_terms = _terms(question)
    answer_relevance = len(question_terms & _terms(answer)) / max(len(question_terms), 1)
    return {name: _bounded(value) for name, value in {
        "answer_relevance": answer_relevance,
        "faithfulness": lexical_faithfulness,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "citation_precision": citation_precision,
        "citation_completeness": citation_completeness,
    }.items()}


@traceable(name="answer_quality_evaluation", run_type="chain")
def evaluate_answer(question: str, answer: str, documents: list[Document],
                    expected_paths: list[str], judge=None) -> dict[str, float]:
    metrics = deterministic_answer_metrics(question, answer, documents, expected_paths)
    if judge is not None:
        context = "\n\n".join(
            f'[{doc.metadata["path"]}:{doc.metadata["start_line"]}-{doc.metadata["end_line"]}]\n'
            f"{doc.page_content}" for doc in documents
        )
        semantic = parse_judge_scores(judge.invoke({
            "question": question, "answer": answer, "context": context,
        }))
        metrics.update(semantic)
    return metrics


def aggregate_answer_evaluations(details: list[dict], thresholds: dict[str, float]) -> dict:
    names = tuple(thresholds)
    count = max(len(details), 1)
    scores = {name: _bounded(sum(item["metrics"][name] for item in details) / count)
              for name in names}
    failures = {name: {"score": scores[name], "threshold": threshold}
                for name, threshold in thresholds.items() if scores[name] < threshold}
    return {"scores": scores, "thresholds": thresholds, "passed": not failures,
            "failures": failures, "details": details}


@traceable(name="golden_end_to_end_rag_evaluation", run_type="chain")
def evaluate_rag_dataset(graph, dataset_path: Path, repository: str, branch: str | None,
                         judge=None, thresholds: dict[str, float] | None = None) -> dict:
    cases = json.loads(dataset_path.read_text())
    effective_thresholds = thresholds or {
        "answer_relevance": 0.7,
        "faithfulness": 0.7,
        "context_precision": 0.2,
        "context_recall": 0.8,
        "citation_precision": 1.0,
        "citation_completeness": 0.8,
    }
    details = []
    for case in cases:
        question = case["question"]
        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "question": question, "repository": repository, "branch": branch,
            "documents": [], "answer": "", "route": "", "rewritten_question": "",
            "retrieval_queries": [], "retrieval_filters": {}, "retrieval_tools": [],
            "retrieval_hop": 0, "retrieval_k": 8, "tool_calls": [], "validation_errors": [],
            "generation_attempts": 0,
        }, config={"configurable": {"thread_id": f"evaluation-{uuid4()}"}})
        metrics = evaluate_answer(question, result["answer"], result.get("documents", []),
                                  case["expected_paths"], judge)
        details.append({"question": question, "answer": result["answer"],
                        "metrics": metrics,
                        "retrieved_paths": [doc.metadata.get("path")
                                            for doc in result.get("documents", [])]})
    return aggregate_answer_evaluations(details, effective_thresholds)


def validate_staging_collection(store, expected_chunks: int) -> GateResult:
    data = store.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    errors: list[str] = []
    if not metadatas:
        errors.append("staging collection is empty")
    if len(metadatas) != expected_chunks:
        errors.append(f"expected {expected_chunks} chunks but found {len(metadatas)}")
    required = {"repository", "branch", "path", "start_line", "end_line", "chunk_id"}
    for position, metadata in enumerate(metadatas[:100]):
        missing = required - set(metadata)
        if missing:
            errors.append(f"chunk {position} is missing metadata: {sorted(missing)}")
            break
    return GateResult(not errors, len(metadatas), errors)


@traceable(name="golden_retrieval_evaluation", run_type="chain")
def evaluate_golden_dataset(index, dataset_path: Path, repository: str,
                            branch: str | None, minimum_recall: float = 0.8) -> dict:
    cases = json.loads(dataset_path.read_text())
    hits = 0
    reciprocal_ranks: list[float] = []
    details = []
    for case in cases:
        documents = index.retrieve(case["question"], repository, branch)
        paths = [doc.metadata["path"] for doc in documents]
        expected = set(case["expected_paths"])
        ranks = [position + 1 for position, path in enumerate(paths) if path in expected]
        hit = bool(ranks)
        hits += int(hit)
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0)
        details.append({"question": case["question"], "passed": hit,
                        "retrieved_paths": paths})
    total = max(len(cases), 1)
    recall = hits / total
    result = {"cases": len(cases), "recall_at_k": recall,
              "mean_reciprocal_rank": sum(reciprocal_ranks) / total,
              "threshold": minimum_recall, "passed": recall >= minimum_recall,
              "details": details}
    return result
