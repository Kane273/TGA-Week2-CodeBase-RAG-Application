# Project Report: Engineering Codebase RAG Assistant

## The primer

This RAG application helps software engineers answer architecture, implementation, debugging, and data-flow questions from source repositories in a browser chatbot, with every supported answer grounded by file-and-line citations.

## Framework

| Field | Decision |
|---|---|
| Use case | Engineers ask how a codebase behaves, where logic lives, and what may cause a failure through an internal chatbot. |
| Corpus | Source code, configuration, SQL, API schemas, engineering documentation, and individually uploaded PDF/DOCX/text files; Git and the managed upload store are the sources of truth. |
| Ingestion and cleaning | A recursive scanner and multipart upload API accept supported formats, extract PDF/DOCX text, reject unsafe names/symlinks/secrets, and ignore generated/vendor directories. |
| Ingestion and freshness | Users trigger indexing from the UI. File hashes, commit SHAs, and stable chunk IDs allow changed files to be re-embedded and deleted-file vectors to be removed. |
| Chunking and embedding | Tree-sitter preserves functions, classes, and methods as semantic units, with recursive splitting as a safe fallback. `nomic-embed-text` creates embeddings locally through Ollama. |
| Retrieval | Chroma (default) or Pinecone dense retrieval and BM25 sparse retrieval produce candidates which a local cross-encoder reranks; results are filtered by repository and branch. Pinecone repository versions use isolated namespaces. |
| Operations | PostgreSQL graph checkpoints, Redis workers, self-hosted Chroma, ClamAV, Prometheus/Grafana/Loki, backup and retention workflows, Kubernetes policies, load tests, and security CI form the no-key hardened deployment profile. |
| Generation | `qwen2.5-coder:7b` runs locally with temperature zero and must cite the retrieved file and line range or refuse. |
| Orchestration | LangGraph classifies questions before retrieval and generation, then validates exact citations and performs one bounded repair pass when grounding fails. Memory is partitioned by browser session. |
| Interface | FastAPI exposes indexing/chat endpoints and serves a responsive, dependency-free chatbot UI. Python and dependencies are reproducibly managed with uv. |
| Security | No model API keys or external inference are used. Keycloak authenticates users, repository grants authorize before retrieval, collections are isolated, and secret/symlink protections run before embedding. |
| Evaluation | A version-controlled golden dataset measures recall@k and mean reciprocal rank; staging integrity gates and CI tests block invalid builds. |

## Datasets used

The initial dataset is the application repository itself, including Python backend code, JavaScript UI code, configuration, and Markdown documentation. This self-indexing dataset makes every answer manually verifiable against the source while avoiding proprietary or externally licensed data.

## Prompts and agent instructions

The system instruction requires answers to use only supplied repository context, cite claims as `[path:start-end]`, explicitly report insufficient evidence, and ignore instructions embedded in retrieved source files. The workflow adds a route label to retrieval queries to distinguish architecture, debugging, and general code-search intent.

## Iterations tried

1. Fixed-size source chunks were replaced with separator-aware chunks that prefer class and function boundaries.
2. Dense-only retrieval was expanded to hybrid dense and BM25 retrieval so exact symbols remain discoverable.
3. Plain source filenames were expanded to line-range metadata and stable chunk IDs for verifiable, repeatable citations.
4. Direct question answering was replaced with a LangGraph routing stage and explicit no-evidence refusal behavior.

## Learnings and observations

Code retrieval needs both semantic matching and exact identifier matching; neither dense nor keyword search is sufficient alone. File path, branch, and line metadata are essential parts of the retrieval result, not UI decoration. Local models protect source privacy but make model sizing, indexing latency, and hardware-aware deployment explicit engineering concerns.

## Next enterprise milestones

- Move LangGraph conversation checkpoints from process memory to PostgreSQL.
- Run workers as separate autoscaled services and add dead-letter administration.
- Add organization-specific retention, backup, and disaster-recovery policies.
- Add local-judge groundedness scoring to the existing retrieval regression suite.
