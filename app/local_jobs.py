import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class LocalJobStore:
    """Thread-safe JSON job state for development workflows across server reloads."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        if not job_id or any(character not in "0123456789abcdef-" for character in job_id):
            raise ValueError("Invalid job id")
        return self.directory / f"{job_id}.json"

    def save(self, state: dict) -> dict:
        saved = dict(state)
        saved["updated_at"] = datetime.now(UTC).isoformat()
        path = self._path(saved["id"])
        temporary = path.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(json.dumps(saved))
            temporary.replace(path)
        return saved

    def get(self, job_id: str) -> dict | None:
        try:
            path = self._path(job_id)
            return json.loads(path.read_text()) if path.exists() else None
        except (ValueError, OSError, json.JSONDecodeError):
            return None

    def update(self, job_id: str, **changes) -> dict:
        state = self.get(job_id)
        if state is None:
            raise KeyError(job_id)
        state.update(changes)
        return self.save(state)

    def all(self) -> list[dict]:
        jobs = []
        for path in self.directory.glob("*.json"):
            try:
                jobs.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return jobs

    def fail_interrupted(self) -> None:
        for job in self.all():
            if job.get("status") in {"queued", "running"}:
                self.update(
                    job["id"], status="failed", progress=100,
                    message="Indexing was interrupted by a server restart. Retry the import.",
                )
