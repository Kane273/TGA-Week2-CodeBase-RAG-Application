import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import Settings
from app.embedding_cache import CachedEmbeddings
from app.ingestion import (
    IndexResult, chunk_file, file_digest, iter_source_files, load_manifest,
    persist_manifest,
)
from app.parsing import build_document_parser
from app.performance import CACHE_EVENTS, RETRIEVAL_RESULTS, stage_timer


class VectorStore(Protocol):
    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]: ...
    def delete(self, ids: list[str]) -> None: ...
    def similarity_search(self, query: str, k: int, filter: dict) -> list[Document]: ...
    def get(self, where: dict | None = None, include: list[str] | None = None) -> dict: ...


class PineconeStore:
    """Small LangChain-compatible adapter around a Pinecone namespace."""

    text_key = "_chunk_text"

    def __init__(self, settings: Settings, embeddings: OllamaEmbeddings, namespace: str):
        from pinecone import Pinecone, ServerlessSpec

        if not settings.pinecone_api_key and not settings.pinecone_control_host:
            raise ValueError(
                "Pinecone requires PINECONE_API_KEY, or PINECONE_CONTROL_HOST "
                "when using the no-auth Pinecone Local emulator"
            )
        client = Pinecone(
            api_key=settings.pinecone_api_key or "pclocal",
            host=settings.pinecone_control_host,
        )
        if settings.pinecone_auto_create and not client.has_index(settings.pinecone_index_name):
            client.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud, region=settings.pinecone_region
                ),
                deletion_protection="disabled",
                tags={"application": "engineering-codebase-rag"},
            )
        self.index = (
            client.Index(host=settings.pinecone_index_host)
            if settings.pinecone_index_host
            else client.Index(settings.pinecone_index_name)
        )
        self.embeddings = embeddings
        self.namespace = namespace

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        records = []
        for identifier, vector, document in zip(ids, vectors, documents):
            metadata = dict(document.metadata)
            metadata[self.text_key] = document.page_content
            records.append({"id": identifier, "values": vector, "metadata": metadata})
        for start in range(0, len(records), 100):
            self.index.upsert(vectors=records[start:start + 100], namespace=self.namespace)
        return ids

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), 1000):
            self.index.delete(ids=ids[start:start + 1000], namespace=self.namespace)

    def delete_collection(self) -> None:
        self.index.delete(delete_all=True, namespace=self.namespace)

    @staticmethod
    def _document(metadata: dict) -> Document:
        clean = dict(metadata)
        text = clean.pop(PineconeStore.text_key, "")
        return Document(page_content=text, metadata=clean)

    def similarity_search(self, query: str, k: int, filter: dict) -> list[Document]:
        result = self.index.query(
            vector=self.embeddings.embed_query(query), top_k=k, namespace=self.namespace,
            filter=filter, include_metadata=True,
        )
        return [self._document(dict(match.metadata or {})) for match in result.matches]

    def get(self, where: dict | None = None, include: list[str] | None = None) -> dict:
        documents: list[str] = []
        metadatas: list[dict] = []
        for id_page in self.index.list(namespace=self.namespace):
            fetched = self.index.fetch(ids=id_page, namespace=self.namespace)
            for vector in fetched.vectors.values():
                metadata = dict(vector.metadata or {})
                document = self._document(metadata)
                if where and not self._matches(document.metadata, where):
                    continue
                documents.append(document.page_content)
                metadatas.append(document.metadata)
        return {"documents": documents, "metadatas": metadatas}

    @classmethod
    def _matches(cls, metadata: dict, where: dict) -> bool:
        if "$and" in where:
            return all(cls._matches(metadata, clause) for clause in where["$and"])
        return all(metadata.get(key) == value for key, value in where.items())


class RepositoryIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        base_embeddings = OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_base_url
        )
        self.embeddings = CachedEmbeddings(
            base_embeddings, settings.embedding_model,
            settings.rag_data_dir / "cache" / "embeddings.sqlite3",
            settings.embedding_cache_enabled,
        )
        self.document_parser = build_document_parser(settings.document_parser_provider)
        self._chroma_client = None
        self._active_collections: dict[str, str] = {}
        self.vector_store = self._store("codebase")
        self.documents: list[Document] = []
        self._reranker = None
        self._reranker_lock = threading.Lock()
        self._cache_lock = threading.RLock()
        self._document_cache: dict[tuple, list[Document]] = {}
        self._bm25_cache: dict[tuple, BM25Retriever] = {}
        self._parent_cache: dict[tuple, dict[str, list[Document]]] = {}
        self._rerank_cache: dict[tuple, float] = {}

    def _store(self, collection_name: str) -> VectorStore:
        provider = self.settings.vector_store_provider.lower()
        if provider == "pinecone":
            return PineconeStore(self.settings, self.embeddings, collection_name)
        if provider != "chroma":
            raise ValueError("VECTOR_STORE_PROVIDER must be 'chroma' or 'pinecone'")
        if self.settings.chroma_host:
            if self._chroma_client is None:
                self._chroma_client = chromadb.HttpClient(
                    host=self.settings.chroma_host,
                    port=self.settings.chroma_port,
                    ssl=self.settings.chroma_ssl,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            return Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                client=self._chroma_client,
            )
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.settings.rag_data_dir / "chroma"),
            client_settings=ChromaSettings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl="app.chroma_telemetry.NoopProductTelemetry",
            ),
        )

    @staticmethod
    def collection_name(repository_id: str, version: int) -> str:
        digest = hashlib.sha256(repository_id.encode()).hexdigest()[:16]
        return f"repo_{digest}_v{version}"

    def set_active_collection(self, repository: str, collection: str) -> None:
        if self._active_collections.get(repository) != collection:
            self._active_collections[repository] = collection
            self.invalidate_caches(repository)

    def invalidate_caches(self, repository: str | None = None) -> None:
        with self._cache_lock:
            if repository is None:
                self._document_cache.clear(); self._bm25_cache.clear(); self._parent_cache.clear()
                self._rerank_cache.clear()
                return
            for cache in (self._document_cache, self._bm25_cache, self._parent_cache):
                for key in list(cache):
                    if repository in key:
                        cache.pop(key, None)
            for key in list(self._rerank_cache):
                if repository in key:
                    self._rerank_cache.pop(key, None)

    def delete_collection(self, collection: str) -> None:
        store = self._store(collection)
        if isinstance(store, PineconeStore):
            store.delete_collection()
        else:
            store.delete_collection()

    def delete_repository_documents(self, repository: str, branch: str | None = None) -> int:
        store = self._active_store(repository)
        documents = self._stored_documents(repository, branch, store)
        identifiers = [str(doc.metadata.get("chunk_id")) for doc in documents
                       if doc.metadata.get("chunk_id")]
        if identifiers:
            store.delete(ids=identifiers)
            self.invalidate_caches(repository)
        return len(identifiers)

    def _active_store(self, repository: str) -> VectorStore:
        collection = self._active_collections.get(repository)
        if self.settings.auth_enabled and not collection:
            raise PermissionError("No authorized active collection is bound to this repository")
        return self._store(collection) if collection else self.vector_store

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            cache = self.settings.rag_data_dir / "models"
            cache.mkdir(parents=True, exist_ok=True)
            self._reranker = CrossEncoder(self.settings.reranker_model,
                                          cache_dir=str(cache))
        return self._reranker

    @staticmethod
    def _where(repository: str, branch: str | None) -> dict:
        if branch:
            return {"$and": [{"repository": repository}, {"branch": branch}]}
        return {"repository": repository}

    @staticmethod
    def _filtered_where(repository: str, branch: str | None,
                        filters: dict[str, str] | None = None) -> dict:
        clauses = [{"repository": repository}]
        effective_branch = (filters or {}).get("branch") or branch
        if effective_branch:
            clauses.append({"branch": effective_branch})
        for key in ("language", "path"):
            value = (filters or {}).get(key)
            if value:
                clauses.append({key: value})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _stored_documents(self, repository: str, branch: str | None,
                          store: VectorStore | None = None) -> list[Document]:
        return self._documents_for_where(repository, branch, self._where(repository, branch), store)

    def _cache_version(self, repository: str) -> str:
        return self._active_collections.get(repository, "codebase")

    def cache_token(self, repository: str) -> str:
        return self._cache_version(repository)

    def _documents_for_where(self, repository: str, branch: str | None, where: dict,
                             store: VectorStore | None = None) -> list[Document]:
        key = (self._cache_version(repository), repository, branch, repr(where))
        with self._cache_lock:
            cached = self._document_cache.get(key)
        if cached is not None:
            CACHE_EVENTS.labels("documents", "hit").inc()
            return cached
        CACHE_EVENTS.labels("documents", "miss").inc()
        with stage_timer("document_load"):
            result = (store or self._active_store(repository)).get(
                where=where, include=["documents", "metadatas"]
            )
        documents = [Document(page_content=text, metadata=metadata)
                     for text, metadata in zip(result.get("documents") or [],
                                               result.get("metadatas") or [])]
        with self._cache_lock:
            self._document_cache[key] = documents
        return documents

    def _bm25(self, repository: str, branch: str | None, where: dict,
              documents: list[Document]) -> BM25Retriever | None:
        if not documents:
            return None
        key = (self._cache_version(repository), repository, branch, repr(where))
        with self._cache_lock:
            retriever = self._bm25_cache.get(key)
            if retriever is None:
                CACHE_EVENTS.labels("bm25", "miss").inc()
                retriever = BM25Retriever.from_documents(documents)
                self._bm25_cache[key] = retriever
            else:
                CACHE_EVENTS.labels("bm25", "hit").inc()
        return retriever

    @staticmethod
    def reciprocal_rank_fusion(rankings: list[tuple[list[Document], float]],
                               constant: int = 60) -> list[Document]:
        scores: dict[str, float] = {}
        documents: dict[str, Document] = {}
        for ranking, weight in rankings:
            for rank, document in enumerate(ranking, start=1):
                identifier = str(document.metadata.get("chunk_id"))
                documents[identifier] = document
                scores[identifier] = scores.get(identifier, 0.0) + weight / (constant + rank)
        return [documents[key] for key in sorted(scores, key=scores.get, reverse=True)]

    def _rerank(self, query: str, repository: str, documents: list[Document], k: int) -> list[Document]:
        candidates = documents[:self.settings.rerank_candidate_k]
        scores: list[float | None] = []
        missing: list[Document] = []
        keys = []
        version = self._cache_version(repository)
        for doc in candidates:
            key = (version, repository, query, str(doc.metadata.get("chunk_id")),
                   self.settings.reranker_model)
            keys.append(key)
            cached = self._rerank_cache.get(key)
            scores.append(cached)
            if cached is None:
                missing.append(doc)
        if missing:
            CACHE_EVENTS.labels("reranker", "miss").inc(len(missing))
            with self._reranker_lock, stage_timer("reranking"):
                predicted = self.reranker.predict([(query, doc.page_content) for doc in missing])
            iterator = iter(float(value) for value in predicted)
            for position, value in enumerate(scores):
                if value is None:
                    value = next(iterator)
                    scores[position] = value
                    self._rerank_cache[keys[position]] = value
        else:
            CACHE_EVENTS.labels("reranker", "hit").inc(len(candidates))
        if len(self._rerank_cache) > 50_000:
            self._rerank_cache.clear()
        return [doc for _, doc in sorted(zip(scores, candidates),
                                        key=lambda item: float(item[0]), reverse=True)[:k]]

    def index(self, root: Path, repository: str, branch: str,
              collection_name: str | None = None, full_rebuild: bool = False,
              additional_roots: list[tuple[Path, str]] | None = None,
              ast_enabled: bool = True) -> IndexResult:
        store = self._store(collection_name) if collection_name else self.vector_store
        key = f"{repository}:{branch}"
        previous = {} if full_rebuild else load_manifest(self.settings).get(key, {})
        previous_files: dict = previous.get("files", {})
        current_files: dict = {}
        changed_docs: list[Document] = []
        changed_paths: set[str] = set()

        sources = [(root, "")] + (additional_roots or [])
        remaining = self.settings.max_repository_files
        for source_root, prefix in sources:
            if not source_root.exists():
                continue
            for path in iter_source_files(source_root, remaining):
                relative = path.relative_to(source_root).as_posix()
                rel_path = f"{prefix.rstrip('/')}/{relative}" if prefix else relative
                digest = file_digest(path)
                old = previous_files.get(rel_path, {})
                if old.get("digest") == digest:
                    current_files[rel_path] = old
                    continue
                file_docs = chunk_file(
                    path, source_root, repository, branch, prefix, self.document_parser,
                    ast_enabled,
                )
                if file_docs:
                    changed_paths.add(rel_path)
                    changed_docs.extend(file_docs)
                    current_files[rel_path] = {
                        "digest": digest,
                        "chunk_ids": [doc.metadata["chunk_id"] for doc in file_docs],
                    }
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        deleted_paths = set(previous_files) - set(current_files)
        stale_paths = changed_paths | deleted_paths
        stale_ids = [chunk_id for path in stale_paths
                     for chunk_id in previous_files.get(path, {}).get("chunk_ids", [])]
        if stale_ids:
            store.delete(ids=stale_ids)
        if changed_docs:
            batch_size = max(1, self.settings.index_batch_size)
            for start in range(0, len(changed_docs), batch_size):
                batch = changed_docs[start:start + batch_size]
                with stage_timer("index_upsert"):
                    store.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch])

        if not current_files:
            raise ValueError("No supported text or source files were found")
        self.invalidate_caches(repository)
        self.documents = self._stored_documents(repository, branch, store)
        where = self._where(repository, branch)
        self._bm25(repository, branch, where, self.documents)
        parent_key = (self._cache_version(repository), repository, branch)
        parent_map: dict[str, list[Document]] = {}
        for document in self.documents:
            parent_map.setdefault(str(document.metadata.get("path", "")), []).append(document)
        for siblings in parent_map.values():
            siblings.sort(key=lambda doc: int(doc.metadata.get("start_line", 0)))
        with self._cache_lock:
            self._parent_cache[parent_key] = parent_map
        total_chunks = sum(len(item.get("chunk_ids", [])) for item in current_files.values())
        result = IndexResult(repository, branch, len(current_files), total_chunks,
                             len(changed_paths), len(deleted_paths))
        persist_manifest(self.settings, result, current_files, root)
        return result

    def build_version(self, root: Path, repository_id: str, repository: str,
                      branch: str, version: int,
                      additional_roots: list[tuple[Path, str]] | None = None) -> tuple[str, IndexResult]:
        collection = self.collection_name(repository_id, version)
        result = self.index(root, repository, branch, collection_name=collection,
                            full_rebuild=True, additional_roots=additional_roots)
        return collection, result

    def retrieve(self, query: str, repository: str, branch: str | None, k: int = 8,
                 filters: dict[str, str] | None = None,
                 tool: str = "hybrid_search", rerank: bool = True):
        candidate_k = max(k * 2, self.settings.retrieval_candidate_k)
        store = self._active_store(repository)
        where = self._filtered_where(repository, branch, filters)
        dense = []
        if tool not in {"symbol_search", "dependency_search"}:
            with stage_timer("dense_retrieval"):
                dense = store.similarity_search(query, k=candidate_k, filter=where)

        candidates = self._documents_for_where(repository, branch, where, store)
        sparse: list[Document] = []
        bm25 = self._bm25(repository, branch, where, candidates)
        if bm25:
            bm25.k = candidate_k
            with stage_timer("sparse_retrieval"):
                sparse = bm25.invoke(query)

        if tool == "symbol_search":
            terms = {term.lower() for term in query.replace(".", " ").split() if len(term) > 2}
            sparse = sorted(
                candidates,
                key=lambda doc: len(terms & set(str(doc.metadata.get("symbol", "")).lower().split())),
                reverse=True,
            )[:candidate_k]
        elif tool == "dependency_search":
            dependency_terms = ("import", "from", "require", "include", "depends", "uses")
            dependency_docs = [doc for doc in candidates
                               if any(term in doc.page_content.lower()
                                      for term in dependency_terms)]
            if dependency_docs:
                bm25 = BM25Retriever.from_documents(dependency_docs)
                bm25.k = candidate_k
                sparse = bm25.invoke(query)

        fused = self.reciprocal_rank_fusion([(dense, 1.0), (sparse, 0.85)])
        if not fused:
            return []
        return self._rerank(query, repository, fused, k) if rerank else fused[:candidate_k]

    def expand_parent_context(self, documents: list[Document], repository: str,
                              branch: str | None, neighbors: int | None = None) -> list[Document]:
        """Add nearby chunks from each hit's file, treating the file as the parent document."""
        if not documents:
            return []
        radius = self.settings.parent_context_chunks if neighbors is None else neighbors
        parent_key = (self._cache_version(repository), repository, branch)
        with self._cache_lock:
            by_path = self._parent_cache.get(parent_key)
        if by_path is None:
            by_path = {}
            for candidate in self._stored_documents(repository, branch):
                by_path.setdefault(str(candidate.metadata.get("path", "")), []).append(candidate)
            for siblings in by_path.values():
                siblings.sort(key=lambda doc: int(doc.metadata.get("start_line", 0)))
            with self._cache_lock:
                self._parent_cache[parent_key] = by_path
        expanded: list[Document] = []
        seen: set[str] = set()
        for hit in documents:
            siblings = by_path.get(str(hit.metadata.get("path", "")), [hit])
            try:
                position = next(i for i, doc in enumerate(siblings)
                                if doc.metadata.get("chunk_id") == hit.metadata.get("chunk_id"))
            except StopIteration:
                position = 0
            for doc in siblings[max(0, position - radius):position + radius + 1]:
                identifier = str(doc.metadata.get("chunk_id", id(doc)))
                if identifier not in seen:
                    seen.add(identifier)
                    expanded.append(doc)
        return expanded

    def retrieve_with_tools(self, query: str, repository: str, branch: str | None,
                            filters: dict[str, str] | None, tools: list[str],
                            k: int = 8, max_calls: int | None = None) -> tuple[list[Document], list[str]]:
        """Run only allow-listed retrieval tools under an explicit call budget."""
        allowed = {"hybrid_search", "symbol_search", "dependency_search"}
        budget = (self.settings.max_retrieval_tool_calls
                  if max_calls is None else max(0, max_calls))
        selected = [tool for tool in tools if tool in allowed][:budget] or ["hybrid_search"]
        calls: list[str] = []
        def run(tool: str):
            return tool, self.retrieve(query, repository, branch, k=k,
                                       filters=filters, tool=tool, rerank=False)
        workers = max(1, min(len(selected), self.settings.retrieval_parallelism))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rankings = list(pool.map(run, selected))
        calls.extend(tool for tool, _ in rankings)
        fused = self.reciprocal_rank_fusion([
            (documents, 1.0 if tool == "hybrid_search" else 0.75)
            for tool, documents in rankings
        ])
        primary = self._rerank(query, repository, fused, k)
        expanded = self.expand_parent_context(primary, repository, branch)
        RETRIEVAL_RESULTS.observe(len(expanded))
        return expanded, calls

    def source_preview(self, repository: str, branch: str | None, path: str,
                       start_line: int, end_line: int, padding_chunks: int = 1) -> dict:
        documents = self._stored_documents(repository, branch)
        matches = [doc for doc in documents if doc.metadata.get("path") == path]
        if not matches:
            raise KeyError(path)
        matches.sort(key=lambda doc: int(doc.metadata.get("start_line", 0)))
        position = next((i for i, doc in enumerate(matches)
                         if int(doc.metadata.get("start_line", 0)) <= start_line
                         and int(doc.metadata.get("end_line", 0)) >= end_line), None)
        if position is None:
            position = min(range(len(matches)),
                           key=lambda i: abs(int(matches[i].metadata.get("start_line", 0))
                                             - start_line))
        selected = matches[max(0, position - padding_chunks):position + padding_chunks + 1]
        lines: dict[int, str] = {}
        for document in selected:
            first = int(document.metadata.get("start_line", 1))
            for offset, content in enumerate(document.page_content.splitlines()):
                lines.setdefault(first + offset, content)
        ordered = sorted(lines.items())
        return {
            "path": path, "branch": branch or str(matches[position].metadata.get("branch", "")),
            "language": str(matches[position].metadata.get("language", "text")),
            "start_line": ordered[0][0], "end_line": ordered[-1][0],
            "highlight_start": start_line, "highlight_end": end_line,
            "lines": [{"number": number, "content": content} for number, content in ordered],
        }
