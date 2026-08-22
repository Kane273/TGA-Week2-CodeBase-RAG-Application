from unittest.mock import patch

from app.config import Settings
from app.retrieval import RepositoryIndex


def test_cross_encoder_uses_supported_cache_dir(tmp_path):
    index = object.__new__(RepositoryIndex)
    index.settings = Settings(rag_data_dir=tmp_path)
    index._reranker = None

    with patch("sentence_transformers.CrossEncoder") as cross_encoder:
        result = index.reranker

    assert result is cross_encoder.return_value
    cross_encoder.assert_called_once_with(
        index.settings.reranker_model,
        cache_dir=str(tmp_path / "models"),
    )
