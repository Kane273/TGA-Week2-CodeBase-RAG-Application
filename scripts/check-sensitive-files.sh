#!/usr/bin/env sh
set -eu

# Fail before CI or a push can publish local state, credentials, key material, or databases.
prohibited_pattern='(^|/)(\.env($|\.)|\.rag_data/|\.venv/|\.uv-cache/|\.pytest_cache/|__pycache__/|secrets/|credentials\.json$|id_rsa$|id_ed25519$)|\.(pem|key|p12|db|sqlite|sqlite3|pyc|pyo|pyd)$'
example_pattern='(^|/)\.env(\.production|\.pinecone-local)?\.example$'

violations="$({ git ls-files | grep -E "$prohibited_pattern" || true; } \
  | grep -Ev "$example_pattern" || true)"

if [ -n "$violations" ]; then
  echo "Sensitive or local-only files are tracked:" >&2
  echo "$violations" >&2
  echo "Remove them from Git with: git rm --cached <path>" >&2
  exit 1
fi

echo "Sensitive-file tracking check passed"
