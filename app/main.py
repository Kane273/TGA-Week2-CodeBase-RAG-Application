import json
import hashlib
import uuid
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Iterator

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.checkpoints import build_checkpointer
from app.database import (IndexJob, IndexVersion, RepositoryGrant, RepositoryRecord, audit,
                          init_database, session_scope)
from app.evaluation import deterministic_answer_metrics, semantic_answer_relevance
from app.graph import build_graph, cited_documents
from app.github import clone_public_repository, validate_public_github_url
from app.ingestion import (SENSITIVE_NAMES, SUPPORTED_EXTENSIONS, detect_branch,
                           LANGUAGE_BY_EXTENSION, load_manifest, remove_manifest_entry, safe_repository_path,
                           sensitive_content_reason)
from app.jobs import IndexJobManager
from app.http_controls import install_http_controls
from app.local_jobs import LocalJobStore
from app.local_workspaces import TemporaryWorkspaceStore
from app.malware import MalwareDetected, scan_bytes
from app.observability import configure_langsmith, configure_telemetry
from app.performance import ANSWER_QUALITY
from app.retrieval import RepositoryIndex
from app.schemas import (ChatRequest, ChatResponse, Citation, GitHubIndexRequest,
                         FeedbackRequest, IndexRequest, IndexResponse, GrantCreate, JobResponse,
                         RepositoryCreate, RepositoryResponse,
                         RollbackRequest, WorkspaceRetentionRequest)
from app.security import (Principal, authorize_repository, current_principal,
                          require_admin)
from app.suggestions import follow_up_questions

settings = get_settings()
configure_langsmith(settings)
index = RepositoryIndex(settings)
checkpoint_handle = build_checkpointer(settings)
graph = build_graph(settings, index, checkpoint_handle.saver)
job_manager = IndexJobManager(settings, index)
local_upload_job_store = LocalJobStore(settings.rag_data_dir / "jobs" / "uploads")
github_job_store = LocalJobStore(settings.rag_data_dir / "jobs" / "github")
feedback_store = LocalJobStore(settings.rag_data_dir / "feedback")
upload_workspace_store = TemporaryWorkspaceStore(
    settings.rag_data_dir / "temporary_workspaces", settings.temporary_upload_ttl_hours
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="Codebase RAG Assistant", version="1.0.0")
install_http_controls(app, settings)
configure_telemetry(app, settings)
if not settings.auth_enabled:
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"], allow_credentials=False)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/metrics", make_asgi_app(), name="metrics")


def cleanup_expired_workspaces() -> int:
    expired = upload_workspace_store.expired()
    for workspace in expired:
        vector_cleanup_succeeded = True
        try:
            index.delete_repository_documents(workspace["repository"], workspace.get("branch"))
        except Exception:
            # Uploaded source files must still expire if a damaged/unavailable vector store
            # prevents deletion of their derived embeddings. Keep the expired metadata so
            # a later cleanup pass can retry the vector deletion.
            vector_cleanup_succeeded = False
        remove_manifest_entry(
            settings, workspace["repository"], workspace.get("branch", "uploads")
        )
        upload_root = settings.rag_data_dir / "uploads" / workspace["id"]
        if upload_root.exists():
            shutil.rmtree(upload_root)
        if vector_cleanup_succeeded:
            upload_workspace_store.delete(workspace["id"])
    return len(expired)


def migrate_existing_upload_workspaces() -> int:
    """Register upload indexes created before refresh-safe workspace metadata existed."""
    migrated = 0
    for entry in load_manifest(settings).values():
        if entry.get("branch") != "uploads" or not entry.get("repository"):
            continue
        repository = entry["repository"]
        workspace_id = hashlib.sha256(repository.encode()).hexdigest()[:16]
        upload_root = settings.rag_data_dir / "uploads" / workspace_id
        if (upload_root.exists()
                and upload_workspace_store.get(workspace_id, include_expired=True) is None):
            upload_workspace_store.save(
                workspace_id, repository, list(entry.get("files", {})), status="ready",
                branch="uploads", chunks=entry.get("chunks", 0),
                message="Existing uploaded workspace restored",
            )
            migrated += 1
    return migrated


@app.on_event("startup")
def startup():
    init_database()
    checkpoint_handle.start()
    github_job_store.fail_interrupted()
    local_upload_job_store.fail_interrupted()
    migrate_existing_upload_workspaces()
    upload_workspace_store.fail_interrupted()
    cleanup_expired_workspaces()
    with session_scope() as session:
        repositories = session.query(RepositoryRecord).all()
        for repository in repositories:
            if repository.active_collection:
                index.set_active_collection(repository.name, repository.active_collection)
    job_manager.recover()


@app.on_event("shutdown")
def shutdown():
    checkpoint_handle.close()


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "vector_store_provider": settings.vector_store_provider,
            "langsmith_tracing": settings.langsmith_tracing}


@app.get("/api/auth/config")
def auth_config():
    return {"enabled": settings.auth_enabled, "issuer": settings.oidc_issuer,
            "client_id": settings.oidc_audience}


@app.get("/api/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/api/health/ready")
def readiness():
    checks = {"database": False, "ollama": False}
    try:
        with session_scope() as session:
            session.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["database"] = True
    except Exception:
        pass
    try:
        checks["ollama"] = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2).is_success
    except httpx.HTTPError:
        pass
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ready", "checks": checks}


def ollama_model_status() -> dict:
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        response.raise_for_status()
        installed = {item.get("name", "") for item in response.json().get("models", [])}
        installed_aliases = installed | {
            name.removesuffix(":latest") for name in installed if name.endswith(":latest")
        }
        required = {settings.chat_model, settings.embedding_model}
        missing = sorted(model for model in required if model not in installed_aliases)
        return {"available": not missing, "installed": sorted(installed),
                "required": sorted(required), "missing": missing}
    except (httpx.HTTPError, ValueError) as exc:
        return {"available": False, "installed": [],
                "required": sorted({settings.chat_model, settings.embedding_model}),
                "missing": sorted({settings.chat_model, settings.embedding_model}),
                "error": f"Cannot connect to Ollama at {settings.ollama_base_url}"}


@app.get("/api/health/models")
def model_readiness():
    status = ollama_model_status()
    if not status["available"]:
        raise HTTPException(status_code=503, detail=status)
    return status


def require_local_models() -> None:
    status = ollama_model_status()
    if not status["available"]:
        missing = ", ".join(status["missing"])
        raise HTTPException(status_code=503, detail=(
            f"Ollama is unavailable or missing models: {missing}. Start `ollama serve`, "
            "then run `ollama pull qwen2.5-coder:7b` and `ollama pull nomic-embed-text`."
        ))


@app.post("/api/repositories", response_model=RepositoryResponse)
def create_repository(payload: RepositoryCreate,
                      principal: Principal = Depends(require_admin)):
    root = safe_repository_path(payload.path)
    with session_scope() as session:
        repository = RepositoryRecord(name=payload.name, path=str(root),
                                      default_branch=payload.default_branch)
        session.add(repository)
        session.flush()
        session.add(RepositoryGrant(repository_id=repository.id,
                                    subject=principal.subject, role="owner"))
        response = RepositoryResponse.model_validate(repository, from_attributes=True)
    audit(principal.subject, "repository.created", response.id)
    return response


@app.get("/api/repositories", response_model=list[RepositoryResponse])
def list_repositories(principal: Principal = Depends(current_principal)):
    with session_scope() as session:
        query = session.query(RepositoryRecord)
        if "admin" not in principal.roles:
            allowed = session.query(RepositoryGrant.repository_id).filter_by(
                subject=principal.subject).subquery()
            query = query.filter(RepositoryRecord.id.in_(allowed))
        return [RepositoryResponse.model_validate(item, from_attributes=True)
                for item in query.all()]


@app.delete("/api/repositories/{repository_id}", status_code=204)
def delete_repository(repository_id: str,
                      principal: Principal = Depends(require_admin)):
    with session_scope() as session:
        repository = session.get(RepositoryRecord, repository_id)
        if not repository:
            raise HTTPException(status_code=404, detail="Repository not found")
        collections = [item.collection for item in session.query(IndexVersion).filter_by(
            repository_id=repository_id).all()]
        repository_name = repository.name
        session.query(RepositoryGrant).filter_by(repository_id=repository_id).delete()
        session.query(IndexJob).filter_by(repository_id=repository_id).delete()
        session.query(IndexVersion).filter_by(repository_id=repository_id).delete()
        session.delete(repository)
    for collection in collections:
        index.delete_collection(collection)
    upload_root = settings.rag_data_dir / "uploads" / repository_id
    if upload_root.exists():
        shutil.rmtree(upload_root)
    audit(principal.subject, "repository.deleted", repository_id,
          f"repository={repository_name};collections={len(collections)}")


@app.post("/api/repositories/{repository_id}/grants", status_code=204)
def create_repository_grant(repository_id: str, payload: GrantCreate,
                            principal: Principal = Depends(require_admin)):
    with session_scope() as session:
        if not session.get(RepositoryRecord, repository_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        grant = session.query(RepositoryGrant).filter_by(
            repository_id=repository_id, subject=payload.subject).one_or_none()
        if grant:
            grant.role = payload.role
        else:
            session.add(RepositoryGrant(repository_id=repository_id,
                                        subject=payload.subject, role=payload.role))
    audit(principal.subject, "repository.grant.updated", repository_id,
          f"subject={payload.subject};role={payload.role}")


@app.post("/api/repositories/{repository_id}/rollback", response_model=RepositoryResponse)
def rollback_index(repository_id: str, payload: RollbackRequest,
                   principal: Principal = Depends(require_admin)):
    with session_scope() as session:
        repository = session.get(RepositoryRecord, repository_id)
        version = session.query(IndexVersion).filter_by(
            repository_id=repository_id, version=payload.version).one_or_none()
        if not repository or not version:
            raise HTTPException(status_code=404, detail="Index version not found")
        session.query(IndexVersion).filter_by(
            repository_id=repository_id, status="active").update({"status": "retired"})
        version.status = "active"
        repository.active_collection = version.collection
        repository.active_version = version.version
        response = RepositoryResponse.model_validate(repository, from_attributes=True)
    index.set_active_collection(response.name, response.active_collection)
    audit(principal.subject, "index.rolled_back", repository_id,
          f"version={payload.version}")
    return response


@app.post("/api/repositories/{repository_id}/index", response_model=JobResponse)
def queue_repository_index(repository_id: str,
                           principal: Principal = Depends(current_principal)):
    authorize_repository(principal, repository_id, write=True)
    with session_scope() as session:
        if not session.get(RepositoryRecord, repository_id):
            raise HTTPException(status_code=404, detail="Repository not found")
        existing = session.query(IndexJob).filter(
            IndexJob.repository_id == repository_id,
            IndexJob.status.in_(["queued", "running", "retrying"]),
        ).first()
        if existing:
            raise HTTPException(status_code=409,
                                detail=f"Index job {existing.id} is already active")
        job = IndexJob(repository_id=repository_id, created_by=principal.subject)
        session.add(job)
        session.flush()
        response = JobResponse.model_validate(job, from_attributes=True)
    job_manager.submit(response.id)
    audit(principal.subject, "index.queued", repository_id, f"job={response.id}")
    return response


@app.get("/api/index-jobs/{job_id}", response_model=JobResponse)
def get_index_job(job_id: str, principal: Principal = Depends(current_principal)):
    with session_scope() as session:
        job = session.get(IndexJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Index job not found")
        authorize_repository(principal, job.repository_id)
        return JobResponse.model_validate(job, from_attributes=True)


@app.get("/api/index-jobs", response_model=list[JobResponse])
def list_index_jobs(principal: Principal = Depends(current_principal)):
    with session_scope() as session:
        query = session.query(IndexJob).order_by(IndexJob.created_at.desc())
        jobs = query.limit(100).all()
        visible = []
        for job in jobs:
            try:
                authorize_repository(principal, job.repository_id)
                visible.append(JobResponse.model_validate(job, from_attributes=True))
            except HTTPException:
                continue
        return visible


@app.post("/api/uploads")
async def upload_documents(
    files: list[UploadFile] = File(...),
    repository: str = Form(...),
    repository_id: str | None = Form(None),
    principal: Principal = Depends(current_principal),
):
    if settings.auth_enabled and not repository_id:
        raise HTTPException(status_code=400, detail="repository_id is required")
    if not settings.auth_enabled:
        require_local_models()
    repository_record = None
    if repository_id:
        authorize_repository(principal, repository_id, write=True)
        with session_scope() as session:
            repository_record = session.get(RepositoryRecord, repository_id)
            if not repository_record:
                raise HTTPException(status_code=404, detail="Repository not found")
            repository = repository_record.name

    storage_key = repository_id or hashlib.sha256(repository.encode()).hexdigest()[:16]
    upload_root = settings.rag_data_dir / "uploads" / storage_key
    upload_root.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    skipped: list[str] = []
    duplicates: list[str] = []
    for uploaded in files:
        original = uploaded.filename or ""
        if "\\" in original:
            raise HTTPException(status_code=400, detail=f"Unsafe filename: {original}")
        relative = PurePosixPath(original)
        parts = relative.parts
        if (not original or relative.is_absolute() or not parts
                or any(part in {"", ".", ".."} for part in parts)):
            raise HTTPException(status_code=400, detail=f"Unsafe filename: {original}")
        relative_name = relative.as_posix()
        if any(part.lower() in SENSITIVE_NAMES or part.startswith(".env.") for part in parts):
            skipped.append(relative_name)
            continue
        if relative.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(relative_name)
            continue
        content = await uploaded.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds upload limit: {relative_name}")
        sensitive_reason = sensitive_content_reason(content.decode("utf-8", "ignore"))
        if sensitive_reason:
            skipped.append(relative_name)
            audit(principal.subject, "documents.sensitive.rejected", repository_id,
                  f"file={relative_name};reason={sensitive_reason}")
            continue
        try:
            scan_bytes(content, relative_name, settings)
        except MalwareDetected as exc:
            audit(principal.subject, "documents.malware.rejected", repository_id,
                  f"file={relative_name}")
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        destination = upload_root.joinpath(*parts)
        try:
            destination.resolve().relative_to(upload_root.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsafe filename: {original}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(content).digest():
            duplicates.append(relative_name)
            continue
        temporary = destination.with_name(f".{destination.name}.uploading")
        temporary.write_bytes(content)
        temporary.replace(destination)
        saved.append(relative_name)

    if not saved and duplicates:
        return {"files": [], "skipped": skipped, "duplicates": duplicates,
                "repository": repository,
                "workspace": upload_workspace_store.get(storage_key),
                "message": "All selected files already exist with identical content"}
    if not saved:
        raise HTTPException(status_code=415,
                            detail="No supported, non-sensitive files were selected")

    if not repository_id:
        upload_workspace_store.save(
            storage_key, repository, saved, status="indexing", branch="uploads",
            message="Upload saved; indexing is in progress",
        )

    audit(principal.subject, "documents.uploaded", repository_id,
          f"count={len(saved)};files={','.join(saved)}")
    if repository_id:
        with session_scope() as session:
            active = session.query(IndexJob).filter(
                IndexJob.repository_id == repository_id,
                IndexJob.status.in_(["queued", "running", "retrying"]),
            ).first()
            if active:
                raise HTTPException(status_code=409,
                                    detail=f"Files saved; index job {active.id} is already active")
            job = IndexJob(repository_id=repository_id, created_by=principal.subject)
            session.add(job)
            session.flush()
            job_response = JobResponse.model_validate(job, from_attributes=True)
        job_manager.submit(job_response.id)
        return {"files": saved, "skipped": skipped, "duplicates": duplicates,
                "repository": repository, "job": job_response.model_dump(),
                "status_url": f"/api/index-jobs/{job_response.id}"}

    local_job_id = str(uuid.uuid4())
    local_upload_job_store.save({
        "id": local_job_id, "status": "queued", "progress": 0,
        "message": "Upload saved; waiting to index", "repository": repository,
        "kind": "upload", "retryable": True,
    })

    def run_local_upload_index():
        local_upload_job_store.update(local_job_id, status="running", progress=20,
                                      message="Creating local embeddings")
        try:
            result = index.index(upload_root, repository, "uploads")
            state = local_upload_job_store.update(
                local_job_id, status="complete", progress=100,
                message=f"Indexed {result.files} files into {result.chunks} chunks",
                result=result.__dict__, retryable=False,
            )
            upload_workspace_store.update(
                storage_key, status="ready", branch="uploads",
                message=state["message"], chunks=result.chunks,
            )
        except Exception as exc:
            local_upload_job_store.update(local_job_id, status="failed", progress=100,
                                          message=str(exc), retryable=True)
            upload_workspace_store.update(storage_key, status="failed", message=str(exc))

    job_manager.executor.submit(run_local_upload_index)
    return {"files": saved, "skipped": skipped, "duplicates": duplicates,
            "repository": repository,
            "workspace": upload_workspace_store.get(storage_key),
            "job": local_upload_job_store.get(local_job_id),
            "status_url": f"/api/upload-jobs/{local_job_id}"}


@app.get("/api/local-workspaces")
def list_local_workspaces(principal: Principal = Depends(current_principal)):
    if settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Local workspaces are disabled")
    cleanup_expired_workspaces()
    return upload_workspace_store.active()


@app.get("/api/local-sources")
def list_local_sources(principal: Principal = Depends(current_principal)):
    if settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Local sources are disabled")
    cleanup_expired_workspaces()
    temporary = {
        (item["repository"], item.get("branch", "uploads")): item
        for item in upload_workspace_store.active()
    }
    sources = []
    for entry in load_manifest(settings).values():
        repository = entry.get("repository")
        branch = entry.get("branch")
        if not repository or not branch:
            continue
        workspace = temporary.get((repository, branch), {})
        identifier = hashlib.sha256(f"{repository}:{branch}".encode()).hexdigest()[:16]
        file_manifest = entry.get("files", {})
        file_details = []
        for path, metadata in sorted(file_manifest.items()):
            extension = Path(path).suffix.lower()
            chunk_types = metadata.get("chunk_types", [])
            if any(kind not in {"text_chunk", "module_context"} for kind in chunk_types):
                strategy = "Tree-sitter AST symbols"
            elif extension == ".pdf":
                strategy = "PDF text extraction"
            elif extension == ".docx":
                strategy = "DOCX text extraction"
            elif extension in LANGUAGE_BY_EXTENSION:
                strategy = "AST-aware code parsing"
            else:
                strategy = "Recursive text splitting"
            file_details.append({
                "path": path,
                "type": extension.lstrip(".").upper() or "TEXT",
                "parser": ", ".join(metadata.get("parsers", [])) or "local",
                "strategy": strategy,
                "chunks": len(metadata.get("chunk_ids", [])),
            })
        sources.append({
            "id": workspace.get("id", identifier), "repository": repository,
            "branch": branch, "file_count": entry.get("file_count", 0),
            "chunks": entry.get("chunks", 0), "status": workspace.get("status", "ready"),
            "expires_at": workspace.get("expires_at"),
            "files": sorted(file_manifest), "file_details": file_details,
            "temporary": bool(workspace),
            "storage_bytes": sum(
                (settings.rag_data_dir / "uploads" / workspace.get("id", "") / path).stat().st_size
                for path in file_manifest
                if workspace and (settings.rag_data_dir / "uploads" / workspace.get("id", "") / path).is_file()
            ),
        })
    return sorted(sources, key=lambda item: (item["repository"], item["branch"]))


@app.delete("/api/local-sources/{source_id}", status_code=204)
def delete_local_source(source_id: str,
                        principal: Principal = Depends(current_principal)):
    """Delete a local repository index without deleting its original source files."""
    if settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Local source not found")
    match = None
    for entry in load_manifest(settings).values():
        repository, branch = entry.get("repository"), entry.get("branch")
        if not repository or not branch:
            continue
        identifier = hashlib.sha256(f"{repository}:{branch}".encode()).hexdigest()[:16]
        if identifier == source_id:
            match = (repository, branch)
            break
    if not match:
        raise HTTPException(status_code=404, detail="Local source not found")
    repository, branch = match
    if any(workspace["repository"] == repository and workspace.get("branch") == branch
           for workspace in upload_workspace_store.active()):
        raise HTTPException(status_code=409, detail="Use the uploaded workspace delete action")
    index.delete_repository_documents(repository, branch)
    remove_manifest_entry(settings, repository, branch)


@app.delete("/api/local-workspaces/{workspace_id}", status_code=204)
def delete_local_workspace(workspace_id: str,
                           principal: Principal = Depends(current_principal)):
    if settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Local workspace not found")
    workspace = upload_workspace_store.delete(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Local workspace not found")
    index.delete_repository_documents(workspace["repository"], workspace.get("branch"))
    remove_manifest_entry(settings, workspace["repository"], workspace.get("branch", "uploads"))
    upload_root = settings.rag_data_dir / "uploads" / workspace_id
    if upload_root.exists():
        shutil.rmtree(upload_root)


@app.patch("/api/local-workspaces/{workspace_id}/retention")
def extend_local_workspace(workspace_id: str, payload: WorkspaceRetentionRequest,
                           principal: Principal = Depends(current_principal)):
    if settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Local workspace not found")
    try:
        return upload_workspace_store.extend(workspace_id, payload.hours)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Local workspace not found") from exc


@app.post("/api/local-workspaces/{workspace_id}/reindex", status_code=202)
def reindex_local_workspace(workspace_id: str,
                            principal: Principal = Depends(current_principal)):
    if settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Local workspace not found")
    workspace = upload_workspace_store.get(workspace_id, include_expired=True)
    upload_root = settings.rag_data_dir / "uploads" / workspace_id
    if not workspace or not upload_root.exists():
        raise HTTPException(status_code=404, detail="Uploaded source files are no longer available")
    job_id = str(uuid.uuid4())
    local_upload_job_store.save({"id": job_id, "kind": "reindex", "status": "queued",
                                 "progress": 0, "message": "Reindex queued",
                                 "repository": workspace["repository"], "retryable": True})
    def run_reindex():
        try:
            local_upload_job_store.update(job_id, status="running", progress=20,
                                          message="Rebuilding uploaded workspace")
            result = index.index(upload_root, workspace["repository"],
                                 workspace.get("branch", "uploads"), full_rebuild=True)
            local_upload_job_store.update(job_id, status="complete", progress=100,
                                          message="Reindex complete", result=result.__dict__,
                                          retryable=False)
            upload_workspace_store.update(workspace_id, status="ready", chunks=result.chunks,
                                          message="Reindex complete")
        except Exception as exc:
            local_upload_job_store.update(job_id, status="failed", progress=100,
                                          message=str(exc), retryable=True)
    job_manager.executor.submit(run_reindex)
    return {"job": local_upload_job_store.get(job_id),
            "status_url": f"/api/upload-jobs/{job_id}"}


@app.get("/api/local-jobs")
def list_local_jobs(principal: Principal = Depends(current_principal)):
    if settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Local jobs are disabled")
    jobs = [*github_job_store.all(), *local_upload_job_store.all()]
    return sorted(jobs, key=lambda item: item.get("updated_at", ""), reverse=True)[:100]


@app.get("/api/upload-jobs/{job_id}")
def get_local_upload_job(job_id: str,
                         principal: Principal = Depends(current_principal)):
    if settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Local upload job not found")
    job = local_upload_job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Local upload job not found")
    return job


@app.post("/api/index", response_model=IndexResponse)
def index_repository(request: IndexRequest,
                     principal: Principal = Depends(current_principal)):
    try:
        if settings.auth_enabled:
            raise HTTPException(status_code=403, detail=(
                "Direct filesystem indexing is disabled; use the repository registry"
            ))
        if not request.repository_path:
            raise ValueError("repository_path is required in development mode")
        require_local_models()
        root = safe_repository_path(request.repository_path)
        repository = request.repository_name or root.name
        branch = request.branch or detect_branch(root)
        result = index.index(root, repository, branch)
        return IndexResponse(**result.__dict__)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/github/index", status_code=202)
def index_public_github_repository(request: GitHubIndexRequest,
                                   principal: Principal = Depends(current_principal)):
    try:
        if settings.auth_enabled:
            raise HTTPException(status_code=403, detail=(
                "Direct GitHub indexing is disabled in authenticated mode; an administrator "
                "must register an approved checkout"
            ))
        require_local_models()
        remote = validate_public_github_url(request.url)
        repository = request.repository_name or f"{remote.owner}-{remote.name}"
        for existing in github_job_store.all():
            if (existing["status"] in {"queued", "running"}
                    and existing["url"] == remote.url
                    and existing["repository"] == repository):
                return {"job": existing,
                        "status_url": f"/api/github-jobs/{existing['id']}"}

        job_id = str(uuid.uuid4())
        state = github_job_store.save({
            "id": job_id, "status": "queued", "progress": 5,
            "message": "GitHub import queued", "url": remote.url,
            "repository": repository, "result": None,
        })

        def run_github_index():
            try:
                github_job_store.update(
                    job_id, status="running", progress=15,
                    message="Cloning public GitHub repository",
                )
                root, cloned = clone_public_repository(
                    request.url, settings.rag_data_dir / "github", request.branch,
                    settings.github_clone_timeout_seconds,
                    settings.github_max_checkout_bytes,
                    settings.github_max_checkout_files,
                )
                branch = request.branch or detect_branch(root)
                github_job_store.update(
                    job_id, progress=40, message="Parsing and embedding repository"
                )
                result = index.index(root, repository, branch, ast_enabled=False)
                response = IndexResponse(**result.__dict__).model_dump()
                github_job_store.update(
                    job_id, status="complete", progress=100,
                    message="GitHub repository indexed", result=response,
                )
                audit(principal.subject, "github.public_repository.indexed", None,
                      f"repository={cloned.owner}/{cloned.name};branch={branch}")
            except Exception as exc:
                github_job_store.update(
                    job_id, status="failed", progress=100, message=str(exc)
                )

        job_manager.executor.submit(run_github_index)
        return {"job": state,
                "status_url": f"/api/github-jobs/{job_id}"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/github-jobs/{job_id}")
def get_local_github_job(job_id: str,
                         principal: Principal = Depends(current_principal)):
    if settings.auth_enabled:
        raise HTTPException(status_code=403, detail="Local GitHub jobs are disabled")
    job = github_job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="GitHub indexing job not found")
    return job


def graph_input(request: ChatRequest) -> dict:
    return {"messages": [HumanMessage(content=request.question)],
            "question": request.question, "repository": request.repository,
            "branch": request.branch, "documents": [], "answer": "", "route": "",
            "rewritten_question": "", "retrieval_queries": [], "retrieval_filters": {},
            "retrieval_tools": [], "retrieval_hop": 0, "tool_calls": [],
            "retrieval_k": 8,
            "validation_errors": [], "generation_attempts": 0}


def retrieval_diagnostics(result: dict) -> dict:
    score_fields = ("dense_rank", "sparse_rank", "lexical_rank", "sparse_score",
                    "fusion_score", "reranker_score", "final_score")
    quality = {
        "rewritten_question": result.get("rewritten_question", ""),
        "queries": result.get("retrieval_queries", []),
        "filters": result.get("retrieval_filters", {}),
        "tools": result.get("tool_calls", []),
        "hops": result.get("retrieval_hop", 0),
        "top_k": result.get("retrieval_k", 0),
        "retrieved_chunks": len(result.get("documents", [])),
        "ranking": [
            {"path": document.metadata.get("path"),
             "start_line": document.metadata.get("start_line"),
             **{field: document.metadata.get(field) for field in score_fields
                if document.metadata.get(field) is not None}}
            for document in result.get("documents", [])[:12]
        ],
    }


def answer_quality(result: dict, question: str, latency_ms: int | None = None) -> dict:
    documents = result.get("documents", [])
    expected_paths = sorted({str(doc.metadata.get("path", "")) for doc in documents
                             if doc.metadata.get("path")})
    metrics = deterministic_answer_metrics(
        question, result.get("answer", ""), documents, expected_paths
    )
    relevance = semantic_answer_relevance(
        question, result.get("answer", ""), index.embeddings,
        fallback=metrics["answer_relevance"],
    )
    return {
        "grounded": (not result.get("validation_errors")
                     and metrics["citation_precision"] == 1.0
                     and metrics["citation_completeness"] >= 0.8),
        "answer_relevance": relevance,
        "answer_relevance_method": "local_embedding_cosine",
        "faithfulness": metrics["faithfulness"],
        "citation_precision": metrics["citation_precision"],
        "citation_completeness": metrics["citation_completeness"],
        "latency_ms": latency_ms,
    }
    for metric in ("answer_relevance", "faithfulness", "citation_precision",
                   "citation_completeness"):
        ANSWER_QUALITY.labels(metric).observe(quality[metric])
    return quality


def citations_for(result: dict) -> list[Citation]:
    score_fields = ("dense_rank", "sparse_rank", "lexical_rank", "sparse_score",
                    "fusion_score", "reranker_score", "final_score")
    return [Citation(path=d.metadata["path"], start_line=d.metadata["start_line"],
                     end_line=d.metadata["end_line"], branch=d.metadata["branch"],
                     **{field: d.metadata.get(field) for field in score_fields})
            for d in cited_documents(result.get("answer", ""), result.get("documents", []))]


def authorized_chat_request(request: ChatRequest, principal: Principal) -> ChatRequest:
    if not request.repository_id:
        if settings.auth_enabled:
            raise HTTPException(status_code=400, detail="repository_id is required")
        return request
    authorize_repository(principal, request.repository_id)
    with session_scope() as session:
        repository = session.get(RepositoryRecord, request.repository_id)
        if not repository or not repository.active_collection:
            raise HTTPException(status_code=409, detail="Repository has no active index")
        index.set_active_collection(repository.name, repository.active_collection)
        return request.model_copy(update={"repository": repository.name,
                                          "branch": request.branch or repository.default_branch})


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest,
         principal: Principal = Depends(current_principal)):
    try:
        request = authorized_chat_request(request, principal)
        started = time.perf_counter()
        result = graph.invoke(
            graph_input(request),
            config={"configurable": {"thread_id": request.session_id}},
        )
        response = ChatResponse(answer=result["answer"], citations=citations_for(result),
                                route=result["route"],
                                diagnostics=retrieval_diagnostics(result),
                                suggested_questions=follow_up_questions(result),
                                quality=answer_quality(
                                    result, request.question,
                                    round((time.perf_counter() - started) * 1000),
                                ))
        audit(principal.subject, "chat.completed", request.repository_id,
              f"question_sha256={hashlib.sha256(request.question.encode()).hexdigest()}")
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest,
                principal: Principal = Depends(current_principal)):
    request = authorized_chat_request(request, principal)
    def event_stream() -> Iterator[str]:
        started = time.perf_counter()
        state = graph_input(request)
        labels = {"classify": "Classifying question…",
                  "plan": "Planning conversation-aware retrieval…",
                  "retrieve": "Searching, expanding context, and reranking…",
                  "generate": "Generating grounded answer…", "validate": "Validating citations…"}
        try:
            streamed_answer = False
            streamed_content = ""
            generation_step = None
            for mode, update in graph.stream(
                state, config={"configurable": {"thread_id": request.session_id}},
                stream_mode=["updates", "messages"],
            ):
                if mode == "messages":
                    message, metadata = update
                    if metadata.get("langgraph_node") != "generate" or not message.content:
                        continue
                    step = metadata.get("langgraph_step")
                    if generation_step is not None and step != generation_step:
                        yield f"data: {json.dumps({'type': 'answer_reset'})}\n\n"
                        streamed_content = ""
                    generation_step = step
                    streamed_answer = True
                    token = str(message.content)
                    streamed_content += token
                    yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
                    continue
                for node, values in update.items():
                    state.update(values)
                    yield f"data: {json.dumps({'type': 'status', 'text': labels.get(node, node)})}\n\n"
            answer = state.get("answer", "")
            if not streamed_answer:
                yield f"data: {json.dumps({'type': 'token', 'text': answer})}\n\n"
            elif streamed_content != answer:
                yield f"data: {json.dumps({'type': 'answer_reset'})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'text': answer})}\n\n"
            payload = {"type": "done", "route": state.get("route", ""),
                       "citations": [citation.model_dump() for citation in citations_for(state)],
                       "diagnostics": retrieval_diagnostics(state),
                       "suggested_questions": follow_up_questions(state),
                       "quality": answer_quality(
                           state, request.question,
                           round((time.perf_counter() - started) * 1000),
                       ),
                       "validation_errors": state.get("validation_errors", [])}
            yield f"data: {json.dumps(payload)}\n\n"
            audit(principal.subject, "chat.stream.completed", request.repository_id,
                  f"question_sha256={hashlib.sha256(request.question.encode()).hexdigest()}")
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/sources/preview")
def preview_source(repository: str, path: str, start_line: int, end_line: int,
                   branch: str | None = None, repository_id: str | None = None,
                   principal: Principal = Depends(current_principal)):
    if start_line < 1 or end_line < start_line or end_line - start_line > 500:
        raise HTTPException(status_code=400, detail="Invalid source line range")
    if settings.auth_enabled:
        if not repository_id:
            raise HTTPException(status_code=400, detail="repository_id is required")
        authorize_repository(principal, repository_id)
        with session_scope() as session:
            record = session.get(RepositoryRecord, repository_id)
            if not record or not record.active_collection:
                raise HTTPException(status_code=404, detail="Repository index not found")
            repository = record.name
            branch = branch or record.default_branch
            index.set_active_collection(record.name, record.active_collection)
    try:
        return index.source_preview(repository, branch, path, start_line, end_line)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Indexed source not found") from exc


@app.post("/api/feedback", status_code=202)
def save_feedback(payload: FeedbackRequest,
                  principal: Principal = Depends(current_principal)):
    if payload.repository_id:
        authorize_repository(principal, payload.repository_id)
    feedback_id = str(uuid.uuid4())
    feedback_store.save({
        "id": feedback_id, "subject": principal.subject,
        **payload.model_dump(),
    })
    audit(principal.subject, "chat.feedback.recorded", payload.repository_id,
          f"rating={payload.rating};feedback={feedback_id}")
    return {"id": feedback_id, "status": "recorded"}
