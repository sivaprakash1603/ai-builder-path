"""
From-scratch document ingestion loaders.
Built without LangChain or LlamaIndex to extract text and metadata from files.
Supports PDF, Markdown, Text, and CSV files.
"""

import os
import csv
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

try:
    import pypdf
except ImportError:
    pypdf = None

from src.core.types import Document
from src.core.exceptions import IngestionError


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""
    
    def __init__(self, file_path: str):
        self.file_path = str(Path(file_path).resolve())
        if not os.path.exists(self.file_path):
            raise IngestionError(f"File not found: {self.file_path}")
        self.filename = os.path.basename(self.file_path)

    @abstractmethod
    def load(self) -> List[Document]:
        """Parse file and return a list of Document objects."""
        pass


class TextLoader(BaseLoader):
    """Loader for plain text (.txt) files."""
    
    def load(self) -> List[Document]:
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [Document(
                source=self.file_path,
                title=self.filename,
                content=content,
                metadata={"format": "txt", "file_size_bytes": os.path.getsize(self.file_path)}
            )]
        except Exception as e:
            raise IngestionError(f"Failed to load text file {self.file_path}: {str(e)}")


class MarkdownLoader(BaseLoader):
    """Loader for Markdown (.md) files."""
    
    def load(self) -> List[Document]:
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [Document(
                source=self.file_path,
                title=self.filename,
                content=content,
                metadata={"format": "markdown", "file_size_bytes": os.path.getsize(self.file_path)}
            )]
        except Exception as e:
            raise IngestionError(f"Failed to load markdown file {self.file_path}: {str(e)}")


class CSVLoader(BaseLoader):
    """Loader for CSV files, converting rows into structured readable text."""
    
    def load(self) -> List[Document]:
        try:
            rows_text = []
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                    rows_text.append(f"[Row {i+1}] {row_str}")
            
            content = "\n\n".join(rows_text)
            return [Document(
                source=self.file_path,
                title=self.filename,
                content=content,
                metadata={"format": "csv", "row_count": len(rows_text), "file_size_bytes": os.path.getsize(self.file_path)}
            )]
        except Exception as e:
            raise IngestionError(f"Failed to load CSV file {self.file_path}: {str(e)}")


class PDFLoader(BaseLoader):
    """Loader for PDF documents using pypdf without external framework wrappers."""
    
    def load(self) -> List[Document]:
        if pypdf is None:
            raise IngestionError("pypdf library is required to load PDF documents.")
        try:
            reader = pypdf.PdfReader(self.file_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(f"--- Page {i+1} ---\n{page_text.strip()}")
            
            content = "\n\n".join(pages_text)
            return [Document(
                source=self.file_path,
                title=self.filename,
                content=content,
                metadata={
                    "format": "pdf",
                    "page_count": len(reader.pages),
                    "file_size_bytes": os.path.getsize(self.file_path)
                }
            )]
        except Exception as e:
            raise IngestionError(f"Failed to load PDF file {self.file_path}: {str(e)}")


class DocumentLoaderFactory:
    """Factory to instantiate the appropriate loader based on file extension."""
    
    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".txt":
            return TextLoader(file_path)
        elif ext in (".md", ".markdown"):
            return MarkdownLoader(file_path)
        elif ext == ".csv":
            return CSVLoader(file_path)
        elif ext == ".pdf":
            return PDFLoader(file_path)
        else:
            # Fallback to TextLoader for unknown text extensions
            return TextLoader(file_path)

    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """Convenience method to load any supported file."""
        loader = cls.get_loader(file_path)
        return loader.load()
