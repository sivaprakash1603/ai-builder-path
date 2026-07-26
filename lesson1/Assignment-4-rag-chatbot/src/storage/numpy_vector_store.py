"""
From-scratch Vector Store using NumPy matrix operations and SQLite.
Built without external vector DB libraries (no Chroma, no FAISS, no LangChain/LlamaIndex).
Provides fast cosine similarity calculations in NumPy and persistent ACID metadata storage in SQLite.
"""

import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.storage.base import BaseVectorStore
from src.core.types import TextChunk, SearchResult
from src.core.exceptions import VectorStoreError


class NumPyVectorStore(BaseVectorStore):
    """
    Custom Vector Database from scratch.
    - Uses SQLite for transactional document and chunk text/metadata storage.
    - Uses NumPy 2D ndarray for high-speed in-memory cosine similarity matrix math.
    """

    def __init__(self, db_path: str = "./data/vector_store.db"):
        self.db_path = str(Path(db_path).resolve())
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # In-memory vector matrix and ID index mapping
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._chunk_ids: List[str] = []
        self._doc_ids: List[str] = []
        
        self._init_db()
        self._load_embeddings_into_memory()

    def _get_connection(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to SQLite vector store at {self.db_path}: {str(e)}")

    def _init_db(self) -> None:
        """Create tables for chunks and embeddings if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
            conn.commit()

    def _load_embeddings_into_memory(self) -> None:
        """Load all embeddings from SQLite into a high-speed NumPy ndarray."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, doc_id, embedding_json FROM chunks ORDER BY rowid")
            rows = cursor.fetchall()
            
            if not rows:
                self._embeddings_matrix = None
                self._chunk_ids = []
                self._doc_ids = []
                return

            self._chunk_ids = [r["chunk_id"] for r in rows]
            self._doc_ids = [r["doc_id"] for r in rows]
            
            # Parse vectors and convert to float32 matrix
            vectors = [json.loads(r["embedding_json"]) for r in rows]
            self._embeddings_matrix = np.array(vectors, dtype=np.float32)

            # Pre-normalize row vectors for ultra-fast dot product cosine similarity
            norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10  # Avoid division by zero
            self._embeddings_matrix = self._embeddings_matrix / norms

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """Add text chunks with embeddings to SQLite and refresh NumPy index."""
        if not chunks:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            added_count = 0
            for chunk in chunks:
                if chunk.embedding is None:
                    raise VectorStoreError(f"Chunk {chunk.chunk_id} has no embedding generated.")
                
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, doc_id, text, chunk_index, metadata_json, embedding_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.text,
                    chunk.chunk_index,
                    json.dumps(chunk.metadata),
                    json.dumps(chunk.embedding)
                ))
                added_count += 1
            conn.commit()

        # Reload memory matrix for updated indexing
        self._load_embeddings_into_memory()
        return added_count

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search top-K most similar chunks using vectorized NumPy cosine similarity:
        cos_sim = (A . B) / (||A|| * ||B||)
        """
        if self._embeddings_matrix is None or len(self._chunk_ids) == 0:
            return []

        # Convert query to float32 and normalize
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        # Compute cosine similarities in one single vectorized matrix dot product!
        similarities = np.dot(self._embeddings_matrix, query_vec)

        # Sort by descending similarity score
        sorted_indices = np.argsort(similarities)[::-1]

        results: List[SearchResult] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for idx in sorted_indices:
                score = float(similarities[idx])
                if score < similarity_threshold:
                    break
                
                chunk_id = self._chunk_ids[idx]
                cursor.execute("SELECT doc_id, text, metadata_json FROM chunks WHERE chunk_id = ?", (chunk_id,))
                row = cursor.fetchone()
                if not row:
                    continue
                
                metadata = json.loads(row["metadata_json"])
                
                # Apply optional metadata filtering (e.g. format, doc_id, source)
                if filter_metadata:
                    match = True
                    for k, v in filter_metadata.items():
                        if metadata.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue
                
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    doc_id=row["doc_id"],
                    text=row["text"],
                    score=round(score, 4),
                    metadata=metadata
                ))
                
                if len(results) >= top_k:
                    break

        return results

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a specific document ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            deleted_count = cursor.rowcount
            conn.commit()
        
        self._load_embeddings_into_memory()
        return deleted_count

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return summary of all indexed documents in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doc_id, 
                       json_extract(metadata_json, '$.title') as title, 
                       json_extract(metadata_json, '$.source') as source, 
                       json_extract(metadata_json, '$.format') as format,
                       COUNT(*) as chunk_count
                FROM chunks
                GROUP BY doc_id
            """)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_chunk_count(self) -> int:
        """Return total number of chunks in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM chunks")
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def clear(self) -> None:
        """Drop table and reset vector store."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks")
            conn.commit()
        self._embeddings_matrix = None
        self._chunk_ids = []
        self._doc_ids = []
