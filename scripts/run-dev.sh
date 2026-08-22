#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
args=(app.main:app --host 127.0.0.1 --port "${PORT:-8000}")
if [[ "${RAG_HOT_RELOAD:-false}" == "true" ]]; then
  args+=(--reload --reload-dir app --reload-dir frontend)
fi
exec uv run --frozen uvicorn "${args[@]}"
