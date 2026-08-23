import hashlib
import json
import subprocess
import re
import threading
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tree_sitter_language_pack import get_parser
from app.config import Settings
from app.parsing import DocumentParser, DocumentParsingError, LocalDocumentParser


SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb",
    ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".sql", ".sh", ".yaml",
    ".yml", ".json", ".toml", ".md", ".rst", ".txt", ".graphql", ".proto",
    ".pdf", ".docx",
}
IGNORED_DIRS = {
    ".git", ".rag_data", ".venv", "venv", ".uv-cache", ".pytest_cache",
    "node_modules", "dist", "build", "coverage", ".next", "target", "vendor",
    "evaluations",
    "__pycache__",
}
SENSITIVE_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519", "credentials", "credentials.json"}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"glpat-[A-Za-z0-9_-]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{20,}"),
    re.compile(r"(?i)(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s:/]+:[^\s@]{4,}@"),
    re.compile(r"(?i)(?:AccountKey|SharedAccessKey)=[A-Za-z0-9+/=]{20,}"),
]
ASSIGNED_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|token|credential)\s*[:=]\s*['\"]([^'\"]{16,})['\"]"
)


@dataclass
class IndexResult:
    repository: str
    branch: str
    files: int
    chunks: int
    changed_files: int = 0
    deleted_files: int = 0


LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".java": "java", ".go": "go",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".cs": "c_sharp",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
}
_PARSER_LOCKS: dict[str, threading.Lock] = {}


@lru_cache(maxsize=None)
def _language_parser(language: str):
    return get_parser(language)
SYMBOL_NODE_TYPES = {
    "class_definition", "function_definition", "decorated_definition",
    "function_declaration", "class_declaration", "method_definition",
    "method_declaration", "interface_declaration", "type_alias_declaration",
    "struct_item", "enum_item", "trait_item", "impl_item", "function_item",
}


def safe_repository_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError("Repository path must be an existing directory")
    return path


def detect_branch(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "branch", "--show-current"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "detached"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unversioned"


def iter_source_files(root: Path, limit: int):
    count = 0
    for path in root.rglob("*"):
        if count >= limit:
            break
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if path.name.lower() in SENSITIVE_NAMES or path.name.startswith(".env."):
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.stat().st_size > 20_000_000:
            continue
        count += 1
        yield path


def validate_source_path(path: Path, root: Path) -> Path:
    """Resolve a regular source file and prove it remains beneath its approved root."""
    if path.is_symlink():
        raise DocumentParsingError("Symbolic links are not accepted")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise DocumentParsingError("Source path escapes the repository root") from exc
    if not resolved.is_file():
        raise DocumentParsingError("Source path is not a regular file")
    return resolved


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {character: value.count(character) for character in set(value)}
    return -sum((count / len(value)) * math.log2(count / len(value))
                for count in counts.values())


def sensitive_content_reason(text: str) -> str | None:
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return "known credential pattern"
    for match in ASSIGNED_SECRET.finditer(text):
        value = match.group(1)
        if len(value) >= 20 and _entropy(value) >= 3.5:
            return "high-entropy assigned credential"
    return None


def _symbol_name(node, source: bytes) -> str:
    name = node.child_by_field_name("name")
    return source[name.start_byte:name.end_byte].decode("utf-8", "replace") if name else "module"


def _ast_chunks(text: str, extension: str) -> list[tuple[str, int, int, str, str]]:
    language = LANGUAGE_BY_EXTENSION.get(extension)
    if not language:
        return []
    source = text.encode("utf-8")
    try:
        lock = _PARSER_LOCKS.setdefault(language, threading.Lock())
        with lock:
            tree = _language_parser(language).parse(source)
    except Exception:
        return []
    chunks: list[tuple[str, int, int, str, str]] = []
    stack = list(reversed(tree.root_node.children))
    while stack:
        node = stack.pop()
        if node.type in SYMBOL_NODE_TYPES and node.end_byte - node.start_byte >= 10:
            content = source[node.start_byte:node.end_byte].decode("utf-8", "replace")
            start_row = (node.start_point.row if hasattr(node.start_point, "row")
                         else node.start_point[0])
            end_row = (node.end_point.row if hasattr(node.end_point, "row")
                       else node.end_point[0])
            chunks.append((content, start_row + 1, end_row + 1,
                           _symbol_name(node, source), node.type))
            continue
        stack.extend(reversed(node.children))
    return chunks


def chunk_file(path: Path, root: Path, repository: str, branch: str,
               path_prefix: str = "", parser: DocumentParser | None = None,
               ast_enabled: bool = True) -> list[Document]:
    try:
        safe_path = validate_source_path(path, root)
        parsed = (parser or LocalDocumentParser()).parse(safe_path)
        text = parsed.text
    except DocumentParsingError:
        return []
    if not text.strip():
        return []
    if sensitive_content_reason(text):
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=250,
        separators=["\nclass ", "\ndef ", "\nfunction ", "\nexport ", "\n\n", "\n", " "],
        add_start_index=True,
    )
    ast_chunks = _ast_chunks(text, path.suffix.lower()) if ast_enabled else []
    raw_chunks: list[tuple[str, int, int, str, str]] = ast_chunks
    if not raw_chunks:
        raw_chunks = []
        for chunk in splitter.create_documents([text]):
            start_char = int(chunk.metadata.get("start_index", 0))
            start_line = text.count("\n", 0, start_char) + 1
            raw_chunks.append((chunk.page_content, start_line,
                               start_line + chunk.page_content.count("\n"),
                               "module", "text_chunk"))
    else:
        # Preserve imports, constants, and module-level configuration alongside symbols.
        header = "\n".join(text.splitlines()[:80])
        if header.strip():
            raw_chunks.insert(0, (header, 1, min(80, len(text.splitlines())),
                                  "module", "module_context"))

    documents = []
    for content, start_line, end_line, symbol, symbol_type in raw_chunks:
        relative = path.relative_to(root).as_posix()
        rel_path = f"{path_prefix.rstrip('/')}/{relative}" if path_prefix else relative
        metadata = {
            "repository": repository,
            "branch": branch,
            "path": rel_path,
            "language": path.suffix.lstrip(".") or "text",
            "start_line": start_line,
            "end_line": end_line,
            "symbol": symbol,
            "symbol_type": symbol_type,
            "parser": parsed.parser,
            "chunk_id": hashlib.sha256(
                f"{repository}:{branch}:{rel_path}:{start_line}:{end_line}:"
                f"{symbol_type}:{symbol}:{hashlib.sha256(content.encode()).hexdigest()}".encode()
            ).hexdigest(),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(settings: Settings) -> dict:
    manifest_path = settings.rag_data_dir / "repositories.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def persist_manifest(settings: Settings, result: IndexResult, files: dict, root: Path) -> None:
    settings.rag_data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.rag_data_dir / "repositories.json"
    manifest = load_manifest(settings)
    key = f"{result.repository}:{result.branch}"
    manifest[key] = {
        "repository": result.repository, "branch": result.branch,
        "files": files, "file_count": result.files, "chunks": result.chunks,
        "commit": current_commit(root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def remove_manifest_entry(settings: Settings, repository: str, branch: str) -> None:
    manifest_path = settings.rag_data_dir / "repositories.json"
    manifest = load_manifest(settings)
    manifest.pop(f"{repository}:{branch}", None)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)


def current_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unversioned"
