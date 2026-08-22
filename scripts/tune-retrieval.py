import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.evaluation import evaluate_golden_dataset
from app.retrieval import RepositoryIndex


def integers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune retrieval settings against golden data")
    parser.add_argument("--dataset", default="evaluations/golden.json")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--branch")
    parser.add_argument("--candidate-k", type=integers, default=[16, 24, 32])
    parser.add_argument("--rerank-k", type=integers, default=[16, 24, 32])
    parser.add_argument("--parent-radius", type=integers, default=[1, 2])
    args = parser.parse_args()

    settings = get_settings()
    index = RepositoryIndex(settings)
    index.set_active_collection(args.repository, args.collection)
    runs = []
    for candidate_k, rerank_k, radius in itertools.product(
        args.candidate_k, args.rerank_k, args.parent_radius
    ):
        settings.retrieval_candidate_k = candidate_k
        settings.rerank_candidate_k = rerank_k
        settings.parent_context_chunks = radius
        index.invalidate_caches(args.repository)
        started = time.perf_counter()
        result = evaluate_golden_dataset(
            index, Path(args.dataset), args.repository, args.branch, minimum_recall=0
        )
        runs.append({
            "candidate_k": candidate_k, "rerank_k": rerank_k,
            "parent_radius": radius, "recall_at_k": result["recall_at_k"],
            "mean_reciprocal_rank": result["mean_reciprocal_rank"],
            "latency_seconds": round(time.perf_counter() - started, 4),
        })
    runs.sort(key=lambda item: (-item["recall_at_k"], -item["mean_reciprocal_rank"],
                                item["latency_seconds"]))
    print(json.dumps({"recommended": runs[0] if runs else None, "runs": runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
