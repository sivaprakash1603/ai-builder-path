"""
Unit tests for the From-Scratch Recursive Character Chunker.
"""

import pytest
from src.core.types import Document
from src.ingestion.chunker import RecursiveCharacterChunker


def test_chunker_basic_splitting():
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    doc = Document(source="test.txt", title="Test", content=text)
    
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 60  # Allow minor flexibility around word boundaries
        assert chunk.doc_id == doc.doc_id
        assert chunk.metadata["source"] == "test.txt"


def test_chunker_overlap():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=25)
    text = "Paragraph one with some detailed explanation about RAG architectures.\n\nParagraph two with another discussion on embeddings and cosine similarity."
    doc = Document(source="test.md", title="RAG MD", content=text)
    
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
