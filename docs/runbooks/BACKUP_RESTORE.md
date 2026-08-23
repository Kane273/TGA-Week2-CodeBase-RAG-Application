# Backup and restore

- Run `BACKUP_ROOT=/secure/location ./scripts/backup.sh` on a scheduled host with `pg_dump` and `redis-cli` installed.
- Copy backups off-host using organization-approved encrypted storage and enforce retention separately.
- Verify `SHA256SUMS` daily and run `./scripts/dr-test.sh` in a disposable environment.
- Restore only in a maintenance window: stop API/workers, set `DATABASE_URL`, then run `CONFIRM_RESTORE=RESTORE ./scripts/restore.sh <backup>`.
- Reconcile Chroma/Pinecone indexes from source documents when vector restoration is unavailable, then run the golden evaluation gate.

## Recovery acceptance criteria

1. Verify every file against `SHA256SUMS` before restoration.
2. Restore into a disposable environment first and run `/api/health/ready`.
3. Confirm repository/workspace counts, index versions, temporary-upload expiry, and audit records.
4. Run the held-out retrieval and answer benchmark and require every threshold in `evaluations/gates.json` to pass.
5. Record recovery point and recovery time, then retain the drill evidence with the backup manifest.

Backups include application metadata, local Chroma data, manifests, temporary-workspace metadata, persisted local job history, evaluation data stored under the RAG data directory, and Redis state when available. Pinecone vectors must be rebuilt from the owned source corpus unless the organization maintains a separate Pinecone export process.
