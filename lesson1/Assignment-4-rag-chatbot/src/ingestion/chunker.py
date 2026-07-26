"""
From-scratch Recursive Text Chunker.
Splits documents into manageable semantic segments with configurable overlap
without using LangChain or LlamaIndex wrappers.
"""

from typing import List, Optional
from src.core.types import Document, TextChunk


class RecursiveCharacterChunker:
    """
    Splits text recursively using a hierarchy of separators:
    1. Paragraph breaks ('\\n\\n')
    2. Line breaks ('\\n')
    3. Sentence endings ('. ')
    4. Words (' ')
    5. Characters ('')
    
    Ensures each chunk is bounded by target `chunk_size` characters with `chunk_overlap`.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        if separator == "":
            return list(text)
        return text.split(separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits into chunks respecting chunk_size and chunk_overlap."""
        merged_chunks = []
        current_chunk_pieces = []
        current_length = 0

        for split in splits:
            piece_len = len(split) + (len(separator) if current_chunk_pieces else 0)
            
            if current_length + piece_len > self.chunk_size and current_chunk_pieces:
                # Emit current chunk
                chunk_text = separator.join(current_chunk_pieces).strip()
                if chunk_text:
                    merged_chunks.append(chunk_text)
                
                # Apply sliding window overlap by keeping trailing pieces that fit within overlap
                while current_chunk_pieces and current_length > self.chunk_overlap:
                    removed_piece = current_chunk_pieces.pop(0)
                    current_length -= len(removed_piece) + (len(separator) if current_chunk_pieces else 0)
            
            current_chunk_pieces.append(split)
            current_length += len(split) + (len(separator) if len(current_chunk_pieces) > 1 else 0)

        # Emit any remaining text
        if current_chunk_pieces:
            chunk_text = separator.join(current_chunk_pieces).strip()
            if chunk_text:
                merged_chunks.append(chunk_text)

        return merged_chunks

    def split_text(self, text: str) -> List[str]:
        """Recursively split text into strings of target chunk_size."""
        final_chunks = []
        
        # Find the best separator that exists in the text
        active_separator = ""
        for sep in self.separators:
            if sep == "" or sep in text:
                active_separator = sep
                break
        
        splits = self._split_text_with_separator(text, active_separator)
        
        # Check if any individual split is still too large
        good_splits = []
        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                # Recursively split oversized segment with next separators
                next_separators = [s for s in self.separators if s != active_separator]
                sub_chunker = RecursiveCharacterChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=next_separators
                )
                sub_chunks = sub_chunker.split_text(split)
                good_splits.extend(sub_chunks)

        return self._merge_splits(good_splits, active_separator)

    def chunk_document(self, document: Document) -> List[TextChunk]:
        """Convert a Document into a list of TextChunk objects with metadata preservation."""
        text_segments = self.split_text(document.content)
        chunks = []
        for i, segment in enumerate(text_segments):
            chunk_meta = document.metadata.copy()
            chunk_meta.update({
                "source": document.source,
                "title": document.title,
                "chunk_index": i,
                "total_chunks": len(text_segments)
            })
            chunks.append(TextChunk(
                doc_id=document.doc_id,
                text=segment,
                chunk_index=i,
                metadata=chunk_meta
            ))
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """Chunk multiple documents into a flat list of TextChunk objects."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
