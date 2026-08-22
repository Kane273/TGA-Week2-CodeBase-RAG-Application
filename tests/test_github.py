from pathlib import Path
from unittest.mock import patch

import pytest

from app.github import (clone_public_repository, validate_branch,
                        validate_checkout_limits, validate_public_github_url)


def test_canonical_public_github_url_is_normalized():
    repository = validate_public_github_url("https://github.com/openai/openai-python")
    assert repository.owner == "openai"
    assert repository.name == "openai-python"
    assert repository.url == "https://github.com/openai/openai-python.git"


@pytest.mark.parametrize("url", [
    "http://github.com/owner/repo",
    "https://gitlab.com/owner/repo",
    "https://user:token@github.com/owner/repo",
    "https://github.com/owner/repo?token=secret",
    "https://github.com/owner/repo/tree/main",
])
def test_unsafe_or_noncanonical_github_urls_are_rejected(url):
    with pytest.raises(ValueError):
        validate_public_github_url(url)


def test_unsafe_branch_is_rejected():
    with pytest.raises(ValueError):
        validate_branch("--upload-pack=malicious")


def test_clone_is_shallow_noninteractive_and_uses_argument_list(tmp_path: Path):
    with (patch("app.github.subprocess.run") as run,
          patch("app.github.validate_checkout_limits")):
        checkout, repository = clone_public_repository(
            "https://github.com/owner/repo", tmp_path, branch="main"
        )
    command = run.call_args.args[0]
    assert command[:5] == ["git", "-c", "protocol.file.allow=never", "-c",
                           "core.hooksPath=/dev/null"]
    assert "--no-recurse-submodules" in command
    assert "--filter=blob:limit=20m" in command
    assert "--branch" in command
    assert repository.name == "repo"
    assert checkout.parent.exists()
    assert run.call_args.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert run.call_args.kwargs["stdin"] is not None


def test_checkout_limits_reject_oversized_repository(tmp_path: Path):
    (tmp_path / "large.py").write_bytes(b"x" * 11)
    with pytest.raises(ValueError, match="byte limit"):
        validate_checkout_limits(tmp_path, max_bytes=10, max_files=10)


def test_checkout_limits_reject_symlinks(tmp_path: Path):
    target = tmp_path / "target.py"
    target.write_text("safe")
    (tmp_path / "linked.py").symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        validate_checkout_limits(tmp_path, max_bytes=1000, max_files=10)
