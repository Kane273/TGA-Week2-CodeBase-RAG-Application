#!/usr/bin/env bash
set -euo pipefail

backup_directory="${1:-}"
if [[ -z "$backup_directory" || ! -d "$backup_directory" ]]; then
  echo "Usage: $0 <backup-directory>" >&2
  exit 2
fi

(cd "$backup_directory" && shasum -a 256 -c SHA256SUMS)
tar -tzf "$backup_directory/rag-data.tar.gz" >/dev/null
if [[ ! -f "$backup_directory/application.db" && ! -f "$backup_directory/postgres.dump" ]]; then
  echo "Backup has no application database" >&2
  exit 1
fi
echo "Backup integrity and archive readability verified."
