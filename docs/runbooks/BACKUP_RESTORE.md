# Backup and restore

- Run `BACKUP_ROOT=/secure/location ./scripts/backup.sh` on a scheduled host with `pg_dump` and `redis-cli` installed.
- Copy backups off-host using organization-approved encrypted storage and enforce retention separately.
- Verify `SHA256SUMS` daily and run `./scripts/dr-test.sh` in a disposable environment.
- Restore only in a maintenance window: stop API/workers, set `DATABASE_URL`, then run `CONFIRM_RESTORE=RESTORE ./scripts/restore.sh <backup>`.
- Reconcile Chroma/Pinecone indexes from source documents when vector restoration is unavailable, then run the golden evaluation gate.
