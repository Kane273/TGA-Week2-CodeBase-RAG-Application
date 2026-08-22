#!/usr/bin/env bash
set -euo pipefail

test_root="$(mktemp -d)"
trap 'rm -rf "$test_root"' EXIT
BACKUP_ROOT="$test_root/backups" ./scripts/backup.sh
latest="$(find "$test_root/backups" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
(cd "$latest" && shasum -a 256 -c SHA256SUMS)
test -s "$latest/rag-data.tar.gz"
echo "Disaster-recovery artifact validation passed: $latest"
