from dataclasses import dataclass
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings


@dataclass
class CheckpointerHandle:
    saver: object
    start: Callable[[], None] = lambda: None
    close: Callable[[], None] = lambda: None


def build_checkpointer(settings: Settings) -> CheckpointerHandle:
    if settings.graph_checkpoint_backend == "memory":
        return CheckpointerHandle(MemorySaver())
    if settings.graph_checkpoint_backend != "postgres":
        raise ValueError("GRAPH_CHECKPOINT_BACKEND must be 'memory' or 'postgres'")

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    connection_url = settings.graph_checkpoint_database_url or settings.database_url
    connection_url = connection_url.replace("postgresql+psycopg://", "postgresql://", 1)
    if not connection_url.startswith("postgresql://"):
        raise ValueError("PostgreSQL checkpointing requires a PostgreSQL database URL")
    pool = ConnectionPool(
        conninfo=connection_url, min_size=1, max_size=10, open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    saver = PostgresSaver(pool)

    def start() -> None:
        pool.open()
        pool.wait()
        saver.setup()

    return CheckpointerHandle(saver, start=start, close=pool.close)
