from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.retrieval import RepositoryIndex
from app.security import Principal, authorize_repository


class FakeSession:
    def __init__(self, grant):
        self.grant = grant

    def scalar(self, statement):
        return self.grant


@contextmanager
def fake_session(grant):
    yield FakeSession(grant)


class Grant:
    def __init__(self, role):
        self.role = role


def test_reader_cannot_write_repository():
    principal = Principal("reader-user", frozenset())
    with patch("app.security.session_scope", return_value=fake_session(Grant("reader"))):
        with pytest.raises(HTTPException) as error:
            authorize_repository(principal, "repo-a", write=True)
    assert error.value.status_code == 403


def test_reader_can_read_only_granted_repository():
    principal = Principal("reader-user", frozenset())
    with patch("app.security.session_scope", return_value=fake_session(Grant("reader"))):
        authorize_repository(principal, "repo-a")
    with patch("app.security.session_scope", return_value=fake_session(None)):
        with pytest.raises(HTTPException) as error:
            authorize_repository(principal, "repo-b")
    assert error.value.status_code == 403


def test_authenticated_retrieval_fails_without_bound_collection():
    index = object.__new__(RepositoryIndex)
    index.settings = Settings(auth_enabled=True)
    index._active_collections = {"authorized-repository": "repo_collection"}
    with pytest.raises(PermissionError, match="authorized active collection"):
        index._active_store("different-tenant-repository")


def test_repository_filter_cannot_be_overridden_by_self_query():
    where = RepositoryIndex._filtered_where(
        "authorized-repository", "main",
        {"repository": "different-tenant-repository", "language": "py"},
    )
    assert {"repository": "authorized-repository"} in where["$and"]
    assert all(clause.get("repository") != "different-tenant-repository"
               for clause in where["$and"])
