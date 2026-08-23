import re
import threading
from pathlib import Path

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.config import Settings
from app.retrieval import RepositoryIndex
from app.schemas import FeedbackRequest


def test_source_preview_returns_numbered_context_and_highlight_range():
    index = object.__new__(RepositoryIndex)
    index.settings = Settings()
    index._active_collections = {}
    index._cache_lock = threading.RLock()
    index._document_cache = {}
    document = Document(page_content="first\nsecond\nthird", metadata={
        "path": "app/main.py", "branch": "main", "language": "py",
        "start_line": 10, "end_line": 12, "chunk_id": "one",
    })
    index._stored_documents = lambda repository, branch: [document]
    result = index.source_preview("demo", "main", "app/main.py", 11, 11)
    assert result["highlight_start"] == 11
    assert result["lines"][1] == {"number": 11, "content": "second"}


def test_source_preview_does_not_return_another_path():
    index = object.__new__(RepositoryIndex)
    index._stored_documents = lambda repository, branch: []
    with pytest.raises(KeyError):
        index.source_preview("demo", "main", "private.py", 1, 2)


def test_feedback_payload_rejects_invalid_rating_and_digest():
    with pytest.raises(ValidationError):
        FeedbackRequest(session_id="chat", repository="demo", rating=2,
                        answer_sha256="short")


def test_frontend_script_references_existing_dom_ids():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend/index.html").read_text()
    javascript = (root / "frontend/app.js").read_text()
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    referenced = set(re.findall(r'\$\("([A-Za-z][A-Za-z0-9_-]*)"\)', javascript))
    assert referenced - html_ids == set()
    assert len(html_ids) == len(re.findall(r'id="([^"]+)"', html))


def test_frontend_accepts_stream_and_persisted_suggestion_field_names():
    javascript = (Path(__file__).resolve().parents[1] / "frontend/app.js").read_text()
    assert "data.suggestedQuestions || data.suggested_questions" in javascript


def test_enterprise_ux_controls_are_wired_without_blocking_browser_dialogs():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend/index.html").read_text()
    javascript = (root / "frontend/app.js").read_text()
    for control_id in ("conversationSearch", "workspaceGate", "uploadReview",
                       "actionDialog", "indexSummary", "toastRegion"):
        assert f'id="{control_id}"' in html
    assert "createStreamProgress" in javascript
    assert "How this answer was generated" in javascript
    assert "function selectedUploads" in javascript
    assert not re.search(r"\b(?:alert|confirm|prompt)\(", javascript)


def test_frontend_supports_reduced_motion_and_mobile_backdrop():
    root = Path(__file__).resolve().parents[1]
    css = (root / "frontend/styles.css").read_text()
    html = (root / "frontend/index.html").read_text()
    assert "prefers-reduced-motion: reduce" in css
    assert 'id="sidebarBackdrop"' in html


def test_knowledge_exploration_and_productivity_controls_are_wired():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend/index.html").read_text()
    javascript = (root / "frontend/app.js").read_text()
    for control_id in ("explorerDialog", "compareDialog", "templatesDialog",
                       "analyticsDialog", "reportDialog", "commandDialog"):
        assert f'id="{control_id}"' in html
        assert f'aria-labelledby="{control_id.removesuffix("Dialog")}Title"' in html
    for behavior in ("decorateInlineCitations", "renderFileExplorer", "compareFiles",
                     "renderTemplates", "renderAnalytics", "renderReport",
                     "renderCommands"):
        assert f"function {behavior}" in javascript


def test_interactive_feature_controls_have_accessible_names():
    html = (Path(__file__).resolve().parents[1] / "frontend/index.html").read_text()
    assert 'aria-label="Open command palette"' in html
    assert 'aria-label="Filter repository files"' in html
    assert 'aria-label="Search commands"' in html
    close_targets = set(re.findall(r'data-close="([^"]+)"', html))
    assert close_targets == {"explorerDialog", "compareDialog", "templatesDialog",
                             "analyticsDialog", "jobsDialog", "reportDialog"}
