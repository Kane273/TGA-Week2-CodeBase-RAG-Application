import argparse
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete

from app.config import get_settings
from app.database import AuditEvent, IndexJob, session_scope


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply local RAG retention policies")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.days < 1:
        raise SystemExit("--days must be positive")
    cutoff = datetime.now(UTC) - timedelta(days=args.days)
    settings = get_settings()
    stale_paths = []
    for root in (settings.rag_data_dir / "github", settings.rag_data_dir / "jobs"):
        if not root.exists():
            continue
        stale_paths.extend(path for path in root.iterdir()
                           if datetime.fromtimestamp(path.stat().st_mtime, UTC) < cutoff)
    print(f"Would remove {len(stale_paths)} stale filesystem entries before {cutoff.isoformat()}")
    if not args.apply:
        print("Dry run only; pass --apply to execute")
        return 0
    for path in stale_paths:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    with session_scope() as session:
        session.execute(delete(AuditEvent).where(AuditEvent.created_at < cutoff))
        session.execute(delete(IndexJob).where(IndexJob.updated_at < cutoff,
                                                IndexJob.status.in_(["complete", "failed"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
