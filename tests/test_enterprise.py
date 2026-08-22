from app.retrieval import PineconeStore, RepositoryIndex
from app.security import Principal
from langchain_core.documents import Document


def test_collection_names_are_repository_isolated_and_versioned():
    first = RepositoryIndex.collection_name("repository-a", 1)
    second = RepositoryIndex.collection_name("repository-b", 1)
    assert first != second
    assert RepositoryIndex.collection_name("repository-a", 2) != first


def test_principal_roles_are_immutable():
    principal = Principal("user-1", frozenset({"reader"}))
    assert "reader" in principal.roles


def test_pinecone_metadata_filter_matches_repository_and_branch():
    metadata = {"repository": "demo", "branch": "main", "path": "app.py"}
    where = {"$and": [{"repository": "demo"}, {"branch": "main"}]}
    assert PineconeStore._matches(metadata, where)
    assert not PineconeStore._matches(metadata, {"branch": "release"})


def test_unknown_vector_store_provider_is_rejected(tmp_path):
    from app.config import Settings

    settings = Settings(vector_store_provider="unknown", rag_data_dir=tmp_path)
    try:
        RepositoryIndex(settings)
    except ValueError as error:
        assert "VECTOR_STORE_PROVIDER" in str(error)
    else:
        raise AssertionError("unknown vector store provider should fail fast")


def test_pinecone_store_keeps_chunk_text_out_of_returned_metadata():
    document = PineconeStore._document({
        "_chunk_text": "def answer(): return 42", "path": "answer.py"
    })
    assert document == Document(
        page_content="def answer(): return 42", metadata={"path": "answer.py"}
    )
