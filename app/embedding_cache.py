import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from app.performance import CACHE_EVENTS, stage_timer


class CachedEmbeddings:
    """Content-addressed, model-versioned SQLite cache around a LangChain embedding model."""

    def __init__(self, delegate, model: str, database: Path, enabled: bool = True):
        self.delegate = delegate
        self.model = model
        self.database = database
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            database.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS embeddings "
                    "(cache_key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
                )

    def _connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\0{text}".encode()).hexdigest()

    def _read(self, key: str):
        if not self.enabled:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT vector FROM embeddings WHERE cache_key = ?", (key,)
            ).fetchone()
        CACHE_EVENTS.labels("embedding", "hit" if row else "miss").inc()
        return json.loads(row[0]) if row else None

    def _write_many(self, items: list[tuple[str, list[float]]]):
        if not self.enabled or not items:
            return
        with self._lock, self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(cache_key, vector) VALUES (?, ?)",
                [(key, json.dumps(vector, separators=(",", ":"))) for key, vector in items],
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        output: list[list[float] | None] = [None] * len(texts)
        missing_positions: list[int] = []
        for position, text in enumerate(texts):
            cached = self._read(self._key(text))
            if cached is None:
                missing_positions.append(position)
            else:
                output[position] = cached
        if missing_positions:
            with stage_timer("embedding"):
                vectors = self.delegate.embed_documents([texts[i] for i in missing_positions])
            self._write_many([(self._key(texts[i]), vector)
                              for i, vector in zip(missing_positions, vectors)])
            for position, vector in zip(missing_positions, vectors):
                output[position] = vector
        return [vector for vector in output if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
