from app.config import get_settings
from app.database import init_database
from app.jobs import IndexJobManager
from app.retrieval import RepositoryIndex


def main() -> None:
    settings = get_settings()
    if settings.indexing_worker_mode != "redis":
        raise SystemExit("Set INDEXING_WORKER_MODE=redis for the dedicated worker")
    init_database()
    IndexJobManager(settings, RepositoryIndex(settings)).worker_loop()


if __name__ == "__main__":
    main()
