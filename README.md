# Engineering Codebase RAG Assistant

A fully local, no-API-key assistant for understanding source code. LangChain handles repository ingestion and hybrid retrieval; LangGraph routes and executes the grounded answering workflow; FastAPI connects the workflow to a responsive chatbot UI.

## Features

- Repository-aware chunking with file paths and line-level citations
- Tree-sitter AST-aware chunks for complete functions, classes, and methods
- Dense retrieval through local Ollama embeddings plus BM25 keyword retrieval
- Local cross-encoder reranking with `ms-marco-MiniLM-L-6-v2`
- Git-aware incremental indexing with changed/deleted file cleanup
- LangGraph query routing for architecture, debugging, and general code search
- LangGraph citation validation with a bounded answer-repair pass
- Local generation with `qwen2.5-coder:7b`
- Persistent Chroma by default, with an optional Pinecone vector-store backend
- Prompt-injection boundary: indexed code is explicitly treated as evidence
- Browser chatbot with indexing controls, conversation sessions, and cited sources
- Workspace selector, refresh-persistent chat history, rename/delete/new-topic controls, and context opt-out
- Clickable citation drawer with highlighted source lines, safe Markdown, syntax coloring, and copy controls
- Grounding/relevance/citation/latency indicators plus regenerate, retry, stop, and feedback actions
- Staged indexing progress, upload inventory/expiry, actionable errors, responsive navigation, themes, and accessible live status
- Refresh-safe temporary upload workspaces with configurable 72-hour expiry and data cleanup
- Fail-closed authenticated retrieval with versioned per-repository collection isolation
- Symlink, special-file, and repository-root escape protection at file-read boundaries
- Hardened GitHub clones with hooks/submodules disabled plus time, file-count, and byte limits
- Pre-embedding credential detection for private keys, provider tokens, connection strings, and high-entropy assigned secrets
- Multi-file uploads for source code, PDF, DOCX, Markdown, configuration, and text
- Provider-neutral document parsing with local PDF, DOCX, and UTF-8 implementations
- LangChain `BaseLoader` implementations for local PDF, DOCX, and text sources
- Explicit LCEL prompt → Ollama model → string parser generation chain inside LangGraph
- Conversation-aware LCEL retrieval planning with bounded multi-query decomposition
- Validated self-query filters for branch, language, and exact path metadata
- Parent-document context expansion from matched chunks to neighboring file sections
- Bounded hybrid, symbol, and dependency retrieval tools selected by the local planner
- Iterative LangGraph retrieval hops for architecture and cross-file questions
- Expandable retrieval diagnostics in the chatbot UI
- Version-aware document, BM25, parent lookup, reranker, query-plan, and SQLite embedding caches
- Weighted reciprocal-rank fusion with adaptive retrieval depth and parallel bounded tools
- Batched indexing, context deduplication, hard generation budgets, and model concurrency limits
- True Ollama token streaming with browser cancellation and repair-safe answer replacement
- Per-stage Prometheus latency, cache, context-size, retrieval-count, and model-concurrency metrics
- Server-Sent Events streaming for workflow progress and answer rendering
- No external API keys or source-code uploads
- Keycloak OIDC login with PKCE and repository-level `reader`/`writer`/`owner` grants
- Separate, versioned Chroma collections per repository with atomic activation and rollback
- Secret-pattern, sensitive-file, and symlink protections before embedding
- PostgreSQL-backed repository registry, jobs, grants, versions, and audit events
- Durable indexing jobs with restart recovery, bounded retries, and Redis progress events
- Golden retrieval evaluations with CI quality thresholds
- OpenTelemetry traces, readiness/liveness probes, and a local Jaeger UI
- Optional LangSmith tracing for LangChain/LangGraph runs and golden evaluations

## Run locally

Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
uv sync --frozen
./scripts/run-dev.sh
```

Open `http://localhost:8000`, enter a repository path, index it, and ask a question. API documentation is available at `http://localhost:8000/docs`.

To analyze a public GitHub repository without an API key, paste a canonical URL such as `https://github.com/owner/repository` into **Local path or public GitHub URL**, choose a workspace name, and select **Index repository**. The backend performs a shallow, non-interactive clone into `.rag_data/github`, then applies the same secret scanning, AST chunking, embedding, hybrid retrieval, reranking, and citation validation as a local codebase. Private repositories and arbitrary Git hosts are intentionally rejected.

The development script runs without hot reload so long indexing and GitHub-import jobs are not interrupted by filesystem watchers. When actively editing application code, opt in with `RAG_HOT_RELOAD=true ./scripts/run-dev.sh`; do not index repositories while hot reload is enabled.

To use documents or an entire code folder, select individual files or choose **Choose folder**, then select **Upload and index**. Folder uploads preserve relative paths such as `demo-app/src/main.py`; unsupported and sensitive files are reported as skipped. Local mode creates a searchable upload corpus under the workspace name entered in the sidebar. Files are capped at 20 MB each and pass through the same secret and safety checks as repository content.

## Docker

```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5-coder:7b
docker compose exec ollama ollama pull nomic-embed-text
```

Open `http://localhost:8000` and sign in with the development Keycloak user `demo-admin` / `demo-admin`; the first login requires a password change. Keycloak administration is at `http://localhost:8080`, and traces are available at `http://localhost:16686`.

The compose file mounts this project read-only at `/workspace`. Register that path through the repository API before indexing it. Other repositories must be added as explicit read-only mounts; authenticated deployments reject arbitrary paths from the chat UI. The local reranker downloads once during first retrieval and is then served from `.rag_data/models`; it requires no API key.

Development credentials in `docker-compose.yml` and the imported realm are intentionally local-only. Replace them through secrets management before any shared deployment.

## Vector-store environments

Use one active dense vector store in each environment:

| Environment | Provider | Purpose |
| --- | --- | --- |
| Local development | Chroma | Persistent, offline, no-key daily development |
| Integration and CI | Pinecone Local | Validate Pinecone behavior without credentials |
| Production | Managed Pinecone | Hosted scale, availability, and namespace isolation |

Normal local startup reads `.env` and uses Chroma:

```bash
./scripts/run-dev.sh
```

For a Pinecone compatibility run, Docker Desktop must be running. The helper starts the emulator and overrides the Chroma setting only for that process:

```bash
./scripts/run-pinecone-local.sh
```

The equivalent variables are documented in `.env.pinecone-local.example`. Pinecone Local is intended for development and CI only: it stores vectors in memory, so re-index after restarting it. The configured dimension `768` must match `nomic-embed-text`.

For managed Pinecone, provision a 768-dimensional dense cosine index outside the application and inject the values shown in `.env.production.example` through the deployment secret manager. Never place a real key in any example file. The app continues to use Ollama for local embeddings and generation; only chunk vectors and metadata are sent to Pinecone. Each repository index version is isolated in its own namespace, while LangChain retrieval, BM25 fusion, local reranking, and the LangGraph answering workflow remain unchanged.

`PINECONE_AUTO_CREATE=true` is available for disposable environments, but should remain disabled in production so index lifecycle and deletion protection stay under infrastructure governance. Chroma can be restored at any time by setting `VECTOR_STORE_PROVIDER=chroma`.

## Enterprise repository workflow

1. An administrator registers an allow-listed filesystem path with `POST /api/repositories`.
2. The administrator assigns subject grants with `POST /api/repositories/{id}/grants`.
3. A writer queues indexing with `POST /api/repositories/{id}/index` and polls `/api/index-jobs/{job_id}`.
4. A worker scans secrets and symlinks, parses ASTs, builds a new isolated collection, and runs the staging quality gate.
5. PostgreSQL atomically activates the passing version. Older collections remain available through `POST /api/repositories/{id}/rollback`.

Individual files can be added with multipart `POST /api/uploads`. In authenticated mode the caller needs write access, the upload is associated with a registered repository, and a new combined code-and-document index version is queued automatically.

Supported uploads include Python, JavaScript/TypeScript, Java, Go, Rust, C/C++, C#, Ruby, PHP, SQL, shell, YAML, JSON, TOML, Markdown, RST, text, GraphQL, Protocol Buffers, PDF, and DOCX.

Document extraction is isolated behind `DocumentParser` in `app/parsing.py`. The default `DOCUMENT_PARSER_PROVIDER=local` uses PyPDF, python-docx, and UTF-8 text processing without sending files outside the deployment. A future LlamaParse adapter can implement the same interface while leaving chunking, security checks, indexing, retrieval, and LangGraph unchanged; it should be enabled only in environments that permit managed parsing and provide credentials through a secret manager.

Interactive endpoint documentation is available at `/docs`. In production, obtain an OIDC access token through your identity platform and send it as `Authorization: Bearer <token>`.

## Retrieval evaluation

The version-controlled dataset is in `evaluations/golden.json`. Evaluate an active collection locally:

```bash
uv run python scripts/evaluate.py \
  --repository engineering-rag \
  --collection repo_<repository-hash>_v1 \
  --minimum-recall 0.80
```

The command exits nonzero when recall@k falls below the threshold and reports mean reciprocal rank plus per-question retrieval evidence.

Run the complete RAG graph and enforce answer-quality gates with the local Ollama judge:

```bash
uv run python scripts/evaluate.py \
  --repository <repository-name> \
  --collection <active-collection> \
  --branch main \
  --evaluate-answers
```

This report scores answer relevance, faithfulness, context precision, context recall, citation
precision, and citation completeness. Citation and context metrics are deterministic; semantic
answer relevance and faithfulness use the configured local Ollama model and require no API key.
Use `--no-semantic-judge` for a fully deterministic offline regression run. Each threshold has a
matching `--minimum-*` option, and the command exits nonzero when any quality gate fails.

Tune retrieval quality/latency against the golden dataset before changing production defaults:

```bash
uv run python scripts/tune-retrieval.py \
  --repository <repository-name> \
  --collection <active-collection> \
  --branch main
```

The tuner compares candidate depth, reranker depth, and parent radius, then recommends the
highest-recall/highest-MRR configuration with latency as the tie-breaker.

The evaluation function is instrumented as a LangSmith run, but hosted tracing is disabled by default. To opt in, set `LANGSMITH_TRACING=true`, inject `LANGSMITH_API_KEY` from a secret manager, and optionally configure `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`, and `LANGSMITH_WORKSPACE_ID`. Questions, retrieved code, prompts, outputs, and evaluation data may be transmitted to the configured LangSmith service when enabled; leave it disabled for private/no-key operation. OpenTelemetry and Jaeger remain the local observability path.

## Tests

```bash
uv run pytest
```

## Current production boundaries

The default `./scripts/run-dev.sh` path remains deliberately simple, local, persistent, and free of API keys. The hardened profile adds:

- PostgreSQL-backed LangGraph checkpoints (`GRAPH_CHECKPOINT_BACKEND=postgres`)
- Redis-backed dedicated indexing workers (`INDEXING_WORKER_MODE=redis`)
- self-hosted Chroma over its internal service interface
- request IDs, security headers, distributed rate limiting, and Prometheus `/metrics`
- fail-closed ClamAV upload scanning
- Prometheus, Alertmanager, Grafana, Loki, Promtail, OpenTelemetry, and Jaeger
- backup/restore, retention, access-review, and disaster-recovery scripts
- Kubernetes ingress/TLS, network policies, non-root containers, HPA, PDB, probes, and resource limits
- pytest, pip-audit, Trivy filesystem/image scans, and k6 load gates in CI

Start the complete Compose profile with:

```bash
docker compose -f docker-compose.yml -f docker-compose.enterprise.yml up --build -d
```

Prometheus is at `http://localhost:9090`, Alertmanager at `http://localhost:9093`, Grafana at `http://localhost:3000`, Jaeger at `http://localhost:16686`, and application metrics at `http://localhost:8000/metrics`. Development passwords are placeholders and must be replaced before shared use.

Kubernetes resources are in `deploy/k8s/base.yaml`. They require an ingress controller, metrics server, an RWX storage class, a preloaded Ollama model volume, replacement secrets, and a `rag-tls` secret. Runbooks are under `docs/runbooks/`.

No third-party API key is required for this hardened profile. It still requires operational secrets such as database passwords, OIDC credentials, and TLS private keys. Managed Pinecone and hosted LangSmith remain optional and disabled in the no-key configuration.
