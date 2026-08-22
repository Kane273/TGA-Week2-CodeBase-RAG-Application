import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.evaluation import (build_local_answer_judge, evaluate_golden_dataset,
                            evaluate_rag_dataset)
from app.graph import build_graph
from app.observability import configure_langsmith
from app.retrieval import RepositoryIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local retrieval regression evaluation")
    parser.add_argument("--dataset", default="evaluations/golden.json")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--branch")
    parser.add_argument("--minimum-recall", type=float, default=0.8)
    parser.add_argument("--evaluate-answers", action="store_true",
                        help="Run the complete graph and score answer/context/citation quality")
    parser.add_argument("--no-semantic-judge", action="store_true",
                        help="Use deterministic metrics only; do not invoke the local Ollama judge")
    parser.add_argument("--minimum-answer-relevance", type=float, default=0.7)
    parser.add_argument("--minimum-faithfulness", type=float, default=0.7)
    parser.add_argument("--minimum-context-precision", type=float, default=0.2)
    parser.add_argument("--minimum-context-recall", type=float, default=0.8)
    parser.add_argument("--minimum-citation-precision", type=float, default=1.0)
    parser.add_argument("--minimum-citation-completeness", type=float, default=0.8)
    args = parser.parse_args()
    settings = get_settings()
    configure_langsmith(settings)
    index = RepositoryIndex(settings)
    index.set_active_collection(args.repository, args.collection)
    if args.evaluate_answers:
        graph = build_graph(settings, index)
        judge = None if args.no_semantic_judge else build_local_answer_judge(settings)
        thresholds = {
            "answer_relevance": args.minimum_answer_relevance,
            "faithfulness": args.minimum_faithfulness,
            "context_precision": args.minimum_context_precision,
            "context_recall": args.minimum_context_recall,
            "citation_precision": args.minimum_citation_precision,
            "citation_completeness": args.minimum_citation_completeness,
        }
        result = evaluate_rag_dataset(graph, Path(args.dataset), args.repository,
                                      args.branch, judge, thresholds)
    else:
        result = evaluate_golden_dataset(index, Path(args.dataset), args.repository,
                                         args.branch, args.minimum_recall)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
