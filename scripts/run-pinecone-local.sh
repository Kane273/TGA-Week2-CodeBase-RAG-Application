#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for Pinecone Local. Install and start Docker Desktop first." >&2
  exit 1
fi

docker compose --profile pinecone up -d pinecone-local

export VECTOR_STORE_PROVIDER=pinecone
export PINECONE_CONTROL_HOST=http://localhost:5081
export PINECONE_INDEX_HOST=http://localhost:5081
export PINECONE_INDEX_NAME=codebase-rag
export PINECONE_DIMENSION=768
export PINECONE_AUTO_CREATE=false

exec ./scripts/run-dev.sh
