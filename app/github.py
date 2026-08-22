import hashlib
import os
import re
import subprocess
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


GITHUB_PATH = re.compile(r"^/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")
BRANCH_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


@dataclass(frozen=True)
class PublicGitHubRepository:
    url: str
    owner: str
    name: str


def validate_public_github_url(raw_url: str) -> PublicGitHubRepository:
    parsed = urlsplit(raw_url.strip())
    if (parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.port
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use a public URL in the form https://github.com/owner/repository")
    match = GITHUB_PATH.fullmatch(parsed.path)
    if not match:
        raise ValueError("Use a public URL in the form https://github.com/owner/repository")
    owner, name = match.groups()
    return PublicGitHubRepository(f"https://github.com/{owner}/{name}.git", owner, name)


def validate_branch(branch: str | None) -> str | None:
    if branch is None or not branch.strip():
        return None
    value = branch.strip()
    if (not BRANCH_NAME.fullmatch(value) or ".." in value or "//" in value
            or "@{" in value or value.endswith(("/", ".", ".lock"))):
        raise ValueError("Invalid Git branch name")
    return value


def validate_checkout_limits(root: Path, max_bytes: int, max_files: int) -> tuple[int, int]:
    """Measure a checkout without following links and reject unsafe or oversized trees."""
    total_bytes = 0
    file_count = 0
    resolved_root = root.resolve(strict=True)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"GitHub checkout contains a symbolic link: {path.relative_to(root)}")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("GitHub checkout escapes its isolated root") from exc
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += path.stat().st_size
        if file_count > max_files:
            raise ValueError(f"GitHub checkout exceeds the {max_files}-file limit")
        if total_bytes > max_bytes:
            raise ValueError(f"GitHub checkout exceeds the {max_bytes}-byte limit")
    return file_count, total_bytes


def clone_public_repository(raw_url: str, destination_root: Path,
                            branch: str | None = None, timeout: int = 180,
                            max_bytes: int = 250_000_000,
                            max_files: int = 10_000) -> tuple[Path, PublicGitHubRepository]:
    repository = validate_public_github_url(raw_url)
    branch = validate_branch(branch)
    digest = hashlib.sha256(repository.url.encode()).hexdigest()[:12]
    parent = destination_root / f"{repository.owner}-{repository.name}-{digest}"
    parent.mkdir(parents=True, exist_ok=True)
    checkout = parent / uuid.uuid4().hex
    command = ["git", "-c", "protocol.file.allow=never", "-c", "core.hooksPath=/dev/null",
               "clone", "--depth", "1", "--single-branch", "--no-tags",
               "--no-recurse-submodules", "--filter=blob:limit=20m"]
    if branch:
        command.extend(["--branch", branch])
    command.extend([repository.url, str(checkout)])
    environment = os.environ.copy()
    environment.update({
        "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    })
    try:
        subprocess.run(command, check=True, capture_output=True, text=True,
                       timeout=timeout, env=environment, stdin=subprocess.DEVNULL)
        validate_checkout_limits(checkout, max_bytes, max_files)
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"GitHub clone exceeded the {timeout}-second limit") from error
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        detail = getattr(error, "stderr", "") or "git is unavailable"
        summary = detail.strip().splitlines()[-1][:300]
        raise ValueError(f"Could not clone public GitHub repository: {summary}") from error
    except ValueError:
        if checkout.exists():
            shutil.rmtree(checkout)
        raise
    return checkout, repository
