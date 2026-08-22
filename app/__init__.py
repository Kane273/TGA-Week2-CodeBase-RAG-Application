"""Local engineering codebase RAG assistant."""

import os
from pathlib import Path


_data_dir = Path(os.getenv("RAG_DATA_DIR", ".rag_data"))
os.environ.setdefault("XDG_CACHE_HOME", str(_data_dir / "cache"))
os.environ.setdefault("HF_HOME", str(_data_dir / "models"))
