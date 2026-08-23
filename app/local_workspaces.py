import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path


class TemporaryWorkspaceStore:
    """Durable local-upload metadata so browser refreshes do not lose a workspace."""

    def __init__(self, directory: Path, ttl_hours: int = 72):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=max(1, ttl_hours))
        self._lock = threading.Lock()

    def _path(self, workspace_id: str) -> Path:
        if not workspace_id or any(char not in "0123456789abcdef" for char in workspace_id):
            raise ValueError("Invalid workspace id")
        return self.directory / f"{workspace_id}.json"

    def save(self, workspace_id: str, repository: str, files: list[str],
             status: str = "indexing", **extra) -> dict:
        now = datetime.now(UTC)
        existing = self.get(workspace_id, include_expired=True) or {}
        state = {
            **existing, **extra, "id": workspace_id, "repository": repository,
            "files": sorted(set(existing.get("files", [])) | set(files)),
            "file_count": len(set(existing.get("files", [])) | set(files)),
            "status": status, "updated_at": now.isoformat(),
            "expires_at": (now + self.ttl).isoformat(),
        }
        state.setdefault("created_at", now.isoformat())
        path = self._path(workspace_id)
        temporary = path.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(json.dumps(state))
            temporary.replace(path)
        return state

    def get(self, workspace_id: str, include_expired: bool = False) -> dict | None:
        try:
            path = self._path(workspace_id)
            state = json.loads(path.read_text()) if path.exists() else None
        except (ValueError, OSError, json.JSONDecodeError):
            return None
        if state and not include_expired:
            expires = datetime.fromisoformat(state["expires_at"])
            if expires <= datetime.now(UTC):
                return None
        return state

    def update(self, workspace_id: str, **changes) -> dict:
        state = self.get(workspace_id, include_expired=True)
        if state is None:
            raise KeyError(workspace_id)
        files = changes.pop("files", state.get("files", []))
        status = changes.pop("status", state.get("status", "ready"))
        return self.save(workspace_id, state["repository"], files, status, **changes)

    def extend(self, workspace_id: str, hours: int) -> dict:
        state = self.get(workspace_id, include_expired=True)
        if state is None:
            raise KeyError(workspace_id)
        state["expires_at"] = (datetime.now(UTC) + timedelta(hours=hours)).isoformat()
        state["updated_at"] = datetime.now(UTC).isoformat()
        path = self._path(workspace_id)
        temporary = path.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(json.dumps(state))
            temporary.replace(path)
        return state

    def active(self) -> list[dict]:
        workspaces = []
        for path in self.directory.glob("*.json"):
            state = self.get(path.stem)
            if state:
                workspaces.append(state)
        return sorted(workspaces, key=lambda item: item["updated_at"], reverse=True)

    def expired(self) -> list[dict]:
        """Return expired workspace records without deleting cleanup metadata."""
        now = datetime.now(UTC)
        expired: list[dict] = []
        for path in self.directory.glob("*.json"):
            state = self.get(path.stem, include_expired=True)
            if state and datetime.fromisoformat(state["expires_at"]) <= now:
                expired.append(state)
        return expired

    def purge_expired(self) -> list[dict]:
        expired = self.expired()
        for state in expired:
            self._path(state["id"]).unlink(missing_ok=True)
        return expired

    def delete(self, workspace_id: str) -> dict | None:
        state = self.get(workspace_id, include_expired=True)
        if state:
            self._path(workspace_id).unlink(missing_ok=True)
        return state

    def fail_interrupted(self) -> None:
        for workspace in self.active():
            if workspace.get("status") == "indexing":
                self.update(workspace["id"], status="interrupted",
                            message="Indexing was interrupted; upload again to refresh the index.")
