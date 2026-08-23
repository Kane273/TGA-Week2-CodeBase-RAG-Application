# Local RAG benchmark

The 30-case held-out benchmark in `golden.json` was run against the
`codebase-rag-benchmark-v3` repository and `benchmark_codebase_rag_v3` Chroma
collection using local `qwen2.5-coder:7b`, `nomic-embed-text`, and the local
cross-encoder reranker. No hosted model or API key was used.

## Observed scores

| Metric | Score | Gate |
|---|---:|---:|
| Answer relevance | 92.67% | 70% |
| Faithfulness | 91.00% | 70% |
| Context precision | 25.07% | 20% |
| Context recall | 86.67% | 80% |
| Citation precision | 100.00% | 100% |
| Citation completeness | 82.39% | 80% |
| Retrieval recall@k | 93.33% | 80% |
| Retrieval mean reciprocal rank | 66.17% | Report-only |

Both retrieval and end-to-end quality gates passed. Local-model scores may vary
slightly across hardware, model builds, and repeated runs, so retain the dataset,
model names, index version, and command with any published result.

## Reproduce

```bash
uv run python scripts/evaluate.py \
  --dataset evaluations/golden.json \
  --repository codebase-rag-benchmark-v3 \
  --collection benchmark_codebase_rag_v3 \
  --branch main

uv run python scripts/evaluate.py \
  --dataset evaluations/golden.json \
  --repository codebase-rag-benchmark-v3 \
  --collection benchmark_codebase_rag_v3 \
  --branch main \
  --evaluate-answers
```
