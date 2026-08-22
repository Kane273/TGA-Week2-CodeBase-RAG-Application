from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.checkpoints import build_checkpointer
from app.config import Settings
from app.http_controls import RateLimiter, install_http_controls
from app.malware import MalwareDetected, scan_bytes


class FakeClamConnection:
    def __init__(self, response: bytes):
        self.response = response
        self.sent = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sendall(self, content: bytes):
        self.sent.extend(content)

    def recv(self, size: int) -> bytes:
        return self.response


def test_default_checkpointer_is_local_memory():
    handle = build_checkpointer(Settings(graph_checkpoint_backend="memory"))
    assert isinstance(handle.saver, MemorySaver)


def test_postgres_checkpointer_rejects_non_postgres_url():
    with pytest.raises(ValueError, match="PostgreSQL"):
        build_checkpointer(Settings(
            graph_checkpoint_backend="postgres",
            graph_checkpoint_database_url="sqlite:///wrong.db",
        ))


def test_malware_scanning_is_disabled_in_local_mode():
    with patch("app.malware.socket.create_connection") as connect:
        scan_bytes(b"safe", "safe.txt", Settings(malware_scan_enabled=False))
    connect.assert_not_called()


def test_clamav_detection_rejects_upload():
    connection = FakeClamConnection(b"stream: Eicar-Test-Signature FOUND\0")
    with patch("app.malware.socket.create_connection", return_value=connection):
        with pytest.raises(MalwareDetected):
            scan_bytes(b"test", "unsafe.txt", Settings(malware_scan_enabled=True))
    assert connection.sent.startswith(b"zINSTREAM\0")


def test_security_headers_and_request_id_are_added():
    app = FastAPI()
    install_http_controls(app, Settings(rate_limit_per_minute=0))

    @app.get("/test")
    def test_route():
        return {"ok": True}

    response = TestClient(app).get("/test")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-request-id"]


def test_swagger_docs_receive_scoped_cdn_policy_without_weakening_app_policy():
    app = FastAPI()
    install_http_controls(app, Settings(rate_limit_per_minute=0))

    @app.get("/test")
    def test_route():
        return {"ok": True}

    client = TestClient(app)
    docs = client.get("/docs")
    application = client.get("/test")
    assert docs.status_code == 200
    assert "cdn.jsdelivr.net" in docs.headers["content-security-policy"]
    assert "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net" in (
        docs.headers["content-security-policy"]
    )
    assert "cdn.jsdelivr.net" not in application.headers["content-security-policy"]


def test_rate_limiter_enforces_configured_limit():
    limiter = RateLimiter(Settings(rate_limit_per_minute=1))

    class FakeRedis:
        def __init__(self): self.count = 0
        def incr(self, key): self.count += 1; return self.count
        def expire(self, key, seconds): return True

    limiter.redis = FakeRedis()
    assert limiter.allowed("client")
    assert not limiter.allowed("client")
