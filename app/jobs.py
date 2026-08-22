from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import time

import redis
from sqlalchemy import select

from app.config import Settings
from app.database import IndexJob, IndexVersion, RepositoryRecord, audit, session_scope
from app.evaluation import validate_staging_collection
from app.ingestion import detect_branch, safe_repository_path
from app.observability import span
from app.http_controls import INDEX_QUEUE_DEPTH
from app.retrieval import RepositoryIndex


class IndexJobManager:
    def __init__(self, settings: Settings, index: RepositoryIndex):
        self.settings = settings
        self.index = index
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="index-worker")
        self._submitted: set[str] = set()
        self._lock = Lock()
        self.redis = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)

    def _publish(self, job_id: str, status: str, progress: int, message: str) -> None:
        try:
            self.redis.publish(f"index-job:{job_id}", f"{status}|{progress}|{message[:500]}")
        except redis.RedisError:
            pass

    def submit(self, job_id: str) -> None:
        if self.settings.indexing_worker_mode == "redis":
            try:
                self.redis.rpush(self.settings.indexing_queue_name, job_id)
                INDEX_QUEUE_DEPTH.set(self.redis.llen(self.settings.indexing_queue_name))
                return
            except redis.RedisError as exc:
                raise RuntimeError("Redis indexing queue is unavailable") from exc
        if self.settings.indexing_worker_mode != "local":
            raise ValueError("INDEXING_WORKER_MODE must be 'local' or 'redis'")
        with self._lock:
            if job_id in self._submitted:
                return
            self._submitted.add(job_id)
        self.executor.submit(self._run, job_id)

    def recover(self) -> None:
        with session_scope() as session:
            jobs = session.scalars(select(IndexJob).where(
                IndexJob.status.in_(["queued", "running"])
            )).all()
            ids = [job.id for job in jobs]
            for job in jobs:
                job.status = "queued"
                job.message = "Recovered after application restart"
        for job_id in ids:
            self.submit(job_id)

    def _update(self, job_id: str, status: str, progress: int, message: str) -> None:
        with session_scope() as session:
            job = session.get(IndexJob, job_id)
            if job:
                job.status, job.progress, job.message = status, progress, message
        self._publish(job_id, status, progress, message)

    def _run(self, job_id: str) -> None:
        try:
            for attempt in range(1, self.settings.max_index_retries + 1):
                try:
                    with session_scope() as session:
                        job = session.get(IndexJob, job_id)
                        if not job:
                            return
                        repository = session.get(RepositoryRecord, job.repository_id)
                        job.status, job.attempts = "running", attempt
                        repository_id = repository.id
                        repository_name = repository.name
                        repository_path = repository.path
                        branch = repository.default_branch
                        version = repository.active_version + 1
                    with span("index.build", repository_id=repository_id, version=version):
                        self._update(job_id, "running", 10, "Scanning and protecting files")
                        root = safe_repository_path(repository_path)
                        branch = detect_branch(root) if branch == "auto" else branch
                        self._update(job_id, "running", 30, "Parsing AST and creating embeddings")
                        upload_root = self.settings.rag_data_dir / "uploads" / repository_id
                        additional = [(upload_root, "uploads")] if upload_root.exists() else []
                        collection, result = self.index.build_version(
                            root, repository_id, repository_name, branch, version, additional
                        )
                        self._update(job_id, "running", 80, "Running staging quality gate")
                        store = self.index._store(collection)
                        gate = validate_staging_collection(store, result.chunks)
                        if not gate.passed:
                            raise RuntimeError("; ".join(gate.errors))
                        # This transaction is the atomic activation boundary. The previous
                        # collection remains available for rollback.
                        with session_scope() as session:
                            repository = session.get(RepositoryRecord, repository_id)
                            session.query(IndexVersion).filter_by(
                                repository_id=repository_id, status="active"
                            ).update({"status": "retired"})
                            session.add(IndexVersion(repository_id=repository_id,
                                                     version=version, collection=collection,
                                                     status="active", chunk_count=result.chunks))
                            repository.active_collection = collection
                            repository.active_version = version
                        self.index.set_active_collection(repository_name, collection)
                        self._update(job_id, "complete", 100,
                                     f"Activated index v{version}: {result.chunks} chunks")
                        audit(job.created_by, "index.activated", repository_id,
                              f"version={version};collection={collection}")
                        return
                except Exception as exc:
                    if attempt >= self.settings.max_index_retries:
                        self._update(job_id, "failed", 100, str(exc))
                        audit("system", "index.failed", detail=f"job={job_id};error={exc}")
                    else:
                        self._update(job_id, "retrying", 0,
                                     f"Attempt {attempt} failed; retrying: {exc}")
                        time.sleep(min(2 ** attempt, 10))
        finally:
            with self._lock:
                self._submitted.discard(job_id)

    def worker_loop(self) -> None:
        """Consume durable indexing jobs in a dedicated worker process."""
        while True:
            item = self.redis.blpop(self.settings.indexing_queue_name, timeout=10)
            if item is None:
                INDEX_QUEUE_DEPTH.set(0)
                continue
            job_id = item[1].decode("utf-8")
            INDEX_QUEUE_DEPTH.set(self.redis.llen(self.settings.indexing_queue_name))
            self._run(job_id)
