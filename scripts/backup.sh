#!/usr/bin/env bash
set -euo pipefail

backup_root="${BACKUP_ROOT:-.rag_backups}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
destination="$backup_root/$timestamp"
mkdir -p "$destination"

if [[ "${DATABASE_URL:-}" == postgresql* ]]; then
  pg_dump_url="${DATABASE_URL/postgresql+psycopg:/postgresql:}"
  pg_dump --format=custom --file="$destination/postgres.dump" "$pg_dump_url"
else
  cp "${RAG_DATA_DIR:-.rag_data}/application.db" "$destination/application.db"
fi

if command -v redis-cli >/dev/null 2>&1; then
  redis-cli -u "${REDIS_URL:-redis://localhost:6379/0}" --rdb "$destination/redis.rdb"
fi

tar -czf "$destination/rag-data.tar.gz" "${RAG_DATA_DIR:-.rag_data}"
cp evaluations/gates.json "$destination/benchmark-gates.json"
find "$destination" -type f ! -name SHA256SUMS -exec shasum -a 256 {} \; > "$destination/SHA256SUMS"
echo "Backup created at $destination"
