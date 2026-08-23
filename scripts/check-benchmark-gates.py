#!/usr/bin/env python3
"""Validate a JSON benchmark result against version-controlled quality gates."""
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--gates", type=Path, default=Path("evaluations/gates.json"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    gates = json.loads(args.gates.read_text())
    failures = []
    for metric, threshold in gates.items():
        actual = result.get(metric)
        if actual is None:
            failures.append(f"{metric}: missing")
        elif metric == "p95_latency_ms" and actual > threshold:
            failures.append(f"{metric}: {actual} > {threshold}")
        elif metric != "p95_latency_ms" and actual < threshold:
            failures.append(f"{metric}: {actual} < {threshold}")
    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
