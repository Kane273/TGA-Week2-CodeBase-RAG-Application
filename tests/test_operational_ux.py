import json
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

import app.main as main_module
from app.local_workspaces import TemporaryWorkspaceStore
from app.security import Principal


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


def test_workspace_delete_supports_uploads_and_repository_indexes():
    script = open("frontend/app.js", encoding="utf-8").read()
    assert "/api/local-workspaces/${workspace.id}" in script
    assert "/api/local-sources/${workspace.id}" in script
    assert "/api/repositories/${workspace.id}" in script
    assert "Your original repository files will not be deleted" in script


def test_local_repository_delete_removes_vectors_and_manifest_only(monkeypatch):
    repository, branch = "demo", "main"
    source_id = hashlib.sha256(f"{repository}:{branch}".encode()).hexdigest()[:16]
    deleted = []
    monkeypatch.setattr(main_module, "settings", SimpleNamespace(auth_enabled=False))
    monkeypatch.setattr(main_module, "load_manifest", lambda settings: {
        f"{repository}:{branch}": {"repository": repository, "branch": branch}
    })
    monkeypatch.setattr(main_module.upload_workspace_store, "active", lambda: [])
    monkeypatch.setattr(main_module.index, "delete_repository_documents",
                        lambda repo, ref: deleted.append(("vectors", repo, ref)))
    monkeypatch.setattr(main_module, "remove_manifest_entry",
                        lambda settings, repo, ref: deleted.append(("manifest", repo, ref)))

    main_module.delete_local_source(
        source_id, Principal("local-developer", frozenset({"admin"}))
    )

    assert deleted == [
        ("vectors", repository, branch),
        ("manifest", repository, branch),
    ]


def test_benchmark_gates_are_complete():
    gates = json.load(open("evaluations/gates.json", encoding="utf-8"))
    assert set(gates) == {
        "answer_relevance", "faithfulness", "context_precision", "context_recall",
        "citation_precision", "citation_completeness", "retrieval_recall",
        "p95_latency_ms",
    }
