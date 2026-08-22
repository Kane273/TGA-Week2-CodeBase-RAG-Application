#!/usr/bin/env bash
set -euo pipefail

source_backup="${1:-}"
if [[ -z "$source_backup" || "${CONFIRM_RESTORE:-}" != "RESTORE" ]]; then
  echo "Usage: CONFIRM_RESTORE=RESTORE $0 <backup-directory>" >&2
  exit 2
fi
if [[ ! -d "$source_backup" ]]; then
  echo "Backup directory does not exist: $source_backup" >&2
  exit 2
fi

(cd "$source_backup" && shasum -a 256 -c SHA256SUMS)
if [[ -f "$source_backup/postgres.dump" ]]; then
  pg_restore --clean --if-exists --no-owner --dbname="$DATABASE_URL" "$source_backup/postgres.dump"
elif [[ -f "$source_backup/application.db" ]]; then
  cp "$source_backup/application.db" "${RAG_DATA_DIR:-.rag_data}/application.db"
fi
if [[ -f "$source_backup/rag-data.tar.gz" ]]; then
  rag_target="${RAG_DATA_DIR:-.rag_data}"
  mkdir -p "$rag_target"
  tar -xzf "$source_backup/rag-data.tar.gz" --strip-components=1 -C "$rag_target"
fi
echo "Database and local RAG data restored. Run readiness and golden evaluation checks before traffic."
