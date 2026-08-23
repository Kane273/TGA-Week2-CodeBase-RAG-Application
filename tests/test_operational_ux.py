import json
from datetime import UTC, datetime

from app.local_workspaces import TemporaryWorkspaceStore


def test_workspace_retention_can_be_extended(tmp_path):
    store = TemporaryWorkspaceStore(tmp_path, ttl_hours=1)
    original = store.save("abc123", "demo", ["README.md"], status="ready")
    extended = store.extend("abc123", 48)
    assert datetime.fromisoformat(extended["expires_at"]) > datetime.fromisoformat(original["expires_at"])


def test_frontend_exposes_accessible_operational_controls():
    html = open("frontend/index.html", encoding="utf-8").read()
    script = open("frontend/app.js", encoding="utf-8").read()
    assert 'id="jobsDialog"' in html
    assert 'aria-live="polite"' in html
    assert "renderJobs" in script
    assert "ranking: data.diagnostics.ranking" in script


def test_benchmark_gates_are_complete():
    gates = json.load(open("evaluations/gates.json", encoding="utf-8"))
    assert set(gates) == {
        "answer_relevance", "faithfulness", "context_precision", "context_recall",
        "citation_precision", "citation_completeness", "retrieval_recall",
        "p95_latency_ms",
    }
