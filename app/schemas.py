from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    repository_path: str | None = Field(None, description="Development-only filesystem path")
    repository_id: str | None = None
    repository_name: str | None = None
    branch: str | None = None


class IndexResponse(BaseModel):
    repository: str
    branch: str
    files: int
    chunks: int
    changed_files: int = 0
    deleted_files: int = 0


class GitHubIndexRequest(BaseModel):
    url: str = Field(..., min_length=19, max_length=500)
    repository_name: str | None = Field(None, min_length=2, max_length=100)
    branch: str | None = Field(None, max_length=200)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=4000)
    repository: str
    repository_id: str | None = None
    branch: str | None = None
    session_id: str = "default"


class RepositoryCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    path: str
    default_branch: str = "main"


class RepositoryResponse(BaseModel):
    id: str
    name: str
    path: str
    default_branch: str
    active_collection: str | None = None
    active_version: int = 0


class JobResponse(BaseModel):
    id: str
    repository_id: str
    status: str
    progress: int
    message: str
    attempts: int


class GrantCreate(BaseModel):
    subject: str = Field(..., min_length=1, max_length=255)
    role: str = Field(pattern="^(reader|writer|owner)$")


class RollbackRequest(BaseModel):
    version: int = Field(..., ge=1)


class Citation(BaseModel):
    path: str
    start_line: int
    end_line: int
    branch: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    route: str
    diagnostics: dict = Field(default_factory=dict)
    suggested_questions: list[str] = Field(default_factory=list)
    quality: dict = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    repository: str = Field(..., min_length=1, max_length=100)
    repository_id: str | None = None
    rating: int = Field(..., ge=-1, le=1)
    comment: str = Field("", max_length=1000)
    answer_sha256: str = Field(..., min_length=64, max_length=64)
