import threading
import time
import uuid

import redis
from fastapi import Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram

from app.config import Settings


REQUESTS = Counter("rag_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("rag_http_request_duration_seconds", "HTTP request latency", ["method", "route"])
RATE_LIMITED = Counter("rag_http_rate_limited_total", "Rate-limited HTTP requests")
INDEX_QUEUE_DEPTH = Gauge("rag_index_queue_depth", "Queued indexing jobs")


class RateLimiter:
    def __init__(self, settings: Settings):
        self.limit = settings.rate_limit_per_minute
        self.redis = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.25)
        self._local: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    def allowed(self, identity: str) -> bool:
        if self.limit <= 0:
            return True
        window = int(time.time() // 60)
        key = f"rate:{identity}:{window}"
        try:
            count = self.redis.incr(key)
            if count == 1:
                self.redis.expire(key, 70)
        except redis.RedisError:
            with self._lock:
                self._local = {item: count for item, count in self._local.items()
                               if item[1] >= window - 1}
                local_key = (identity, window)
                count = self._local.get(local_key, 0) + 1
                self._local[local_key] = count
        return count <= self.limit


def install_http_controls(app, settings: Settings) -> None:
    limiter = RateLimiter(settings)

    @app.middleware("http")
    async def enterprise_http_controls(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        client = request.client.host if request.client else "unknown"
        if (request.url.path.startswith("/api/")
                and not request.url.path.startswith("/api/health")
                and not limiter.allowed(client)):
            RATE_LIMITED.inc()
            response = JSONResponse(
                status_code=429, content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "60"},
            )
        else:
            started = time.perf_counter()
            response = await call_next(request)
            route = getattr(request.scope.get("route"), "path", request.url.path)
            LATENCY.labels(request.method, route).observe(time.perf_counter() - started)
            REQUESTS.labels(request.method, route, response.status_code).inc()
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        docs_route = request.url.path in {"/docs", "/redoc", "/openapi.json"}
        response.headers["Content-Security-Policy"] = (
            settings.docs_content_security_policy
            if docs_route else settings.content_security_policy
        )
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
