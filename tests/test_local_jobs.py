from app.local_jobs import LocalJobStore


def test_local_job_survives_new_store_instance(tmp_path):
    first = LocalJobStore(tmp_path)
    first.save({"id": "abc-123", "status": "complete", "result": {"chunks": 4}})
    restored = LocalJobStore(tmp_path).get("abc-123")
    assert restored["status"] == "complete"
    assert restored["result"]["chunks"] == 4


def test_incomplete_job_is_marked_failed_after_restart(tmp_path):
    store = LocalJobStore(tmp_path)
    store.save({"id": "abc-123", "status": "running", "progress": 40})
    store.fail_interrupted()
    restored = store.get("abc-123")
    assert restored["status"] == "failed"
    assert "Retry" in restored["message"]
