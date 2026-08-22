from datetime import UTC, datetime, timedelta

from app.local_workspaces import TemporaryWorkspaceStore


def test_temporary_workspace_survives_store_recreation(tmp_path):
    store = TemporaryWorkspaceStore(tmp_path, ttl_hours=72)
    saved = store.save("abcdef1234", "demo", ["app.py", "README.md"], status="ready")
    restored = TemporaryWorkspaceStore(tmp_path, ttl_hours=72).active()
    assert restored[0]["repository"] == "demo"
    assert restored[0]["file_count"] == 2
    assert restored[0]["expires_at"] == saved["expires_at"]


def test_reupload_merges_files_and_renews_expiration(tmp_path):
    store = TemporaryWorkspaceStore(tmp_path, ttl_hours=72)
    first = store.save("abcdef1234", "demo", ["app.py"])
    second = store.save("abcdef1234", "demo", ["README.md"], status="ready")
    assert second["files"] == ["README.md", "app.py"]
    assert second["expires_at"] >= first["expires_at"]


def test_expired_workspace_is_not_restored(tmp_path):
    store = TemporaryWorkspaceStore(tmp_path, ttl_hours=72)
    state = store.save("abcdef1234", "demo", ["app.py"], status="ready")
    state["expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    store._path("abcdef1234").write_text(__import__("json").dumps(state))
    assert store.active() == []
    purged = store.purge_expired()
    assert [item["id"] for item in purged] == ["abcdef1234"]
    assert not store._path("abcdef1234").exists()


def test_interrupted_indexing_is_marked_for_retry(tmp_path):
    store = TemporaryWorkspaceStore(tmp_path)
    store.save("abcdef1234", "demo", ["app.py"], status="indexing")
    store.fail_interrupted()
    restored = store.get("abcdef1234")
    assert restored["status"] == "interrupted"
    assert "upload again" in restored["message"]
