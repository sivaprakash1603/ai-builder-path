"""
Unit tests for NumPy + SQLite Vector Store.
"""

import os
import tempfile
import pytest
from src.core.types import TextChunk
from src.storage.numpy_vector_store import NumPyVectorStore


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)


def test_vector_store_add_and_search(temp_db_path):
    vs = NumPyVectorStore(db_path=temp_db_path)
    assert vs.get_chunk_count() == 0

    chunk1 = TextChunk(
        doc_id="doc-1",
        text="The quick brown fox jumps over the lazy dog.",
        chunk_index=0,
        metadata={"title": "Fox Story", "source": "story.txt"},
        embedding=[1.0, 0.0, 0.0]
    )
    chunk2 = TextChunk(
        doc_id="doc-2",
        text="Deep learning models utilize artificial neural networks.",
        chunk_index=0,
        metadata={"title": "AI Guide", "source": "ai.txt"},
        embedding=[0.0, 1.0, 0.0]
    )

    added = vs.add_chunks([chunk1, chunk2])
    assert added == 2
    assert vs.get_chunk_count() == 2

    # Search query identical to chunk1 embedding
    results = vs.search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == chunk1.chunk_id
    assert results[0].score == pytest.approx(1.0, 0.001)
    assert results[1].chunk_id == chunk2.chunk_id
    assert results[1].score == pytest.approx(0.0, 0.001)


def test_vector_store_delete(temp_db_path):
    vs = NumPyVectorStore(db_path=temp_db_path)
    chunk = TextChunk(
        doc_id="doc-del",
        text="Temporary text to delete.",
        chunk_index=0,
        metadata={},
        embedding=[0.5, 0.5]
    )
    vs.add_chunks([chunk])
    assert vs.get_chunk_count() == 1

    deleted = vs.delete_document("doc-del")
    assert deleted == 1
    assert vs.get_chunk_count() == 0
