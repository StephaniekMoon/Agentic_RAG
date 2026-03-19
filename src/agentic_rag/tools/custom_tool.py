from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence, Type

from chonkie import SemanticChunker
from crewai.tools import BaseTool
from markitdown import MarkItDown
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""

    query: str = Field(..., description="Query to search the indexed knowledge base.")


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = (
        "Search one or more indexed enterprise knowledge base PDFs and return the "
        "most relevant excerpts with source references."
    )
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        file_path: str | None = None,
        file_paths: Sequence[str] | None = None,
        top_k: int = 5,
    ):
        """Initialize the search tool with one PDF or a list of PDFs."""
        super().__init__()
        self.file_paths = self._normalize_paths(file_path=file_path, file_paths=file_paths)
        self.top_k = max(1, top_k)
        self.client = QdrantClient(":memory:")
        self.collection_name = "knowledge_base"
        self.source_names = [os.path.basename(path) for path in self.file_paths]
        self._process_documents()

    def _normalize_paths(
        self,
        file_path: str | None,
        file_paths: Sequence[str] | None,
    ) -> list[str]:
        candidate_paths: list[str] = []
        if file_path:
            candidate_paths.append(file_path)
        if file_paths:
            candidate_paths.extend(file_paths)

        normalized_paths: list[str] = []
        seen_paths: set[str] = set()
        for path in candidate_paths:
            resolved = os.path.abspath(path)
            if resolved in seen_paths:
                continue
            if not os.path.isfile(resolved):
                raise FileNotFoundError(f"Knowledge base PDF not found: {resolved}")
            seen_paths.add(resolved)
            normalized_paths.append(resolved)

        if not normalized_paths:
            raise ValueError("DocumentSearchTool requires at least one PDF file.")
        return normalized_paths

    def _extract_text(self, path: str) -> str:
        """Extract raw text from a PDF using MarkItDown."""
        md = MarkItDown()
        result = md.convert(path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        )
        return chunker.chunk(raw_text)

    def _source_slug(self, path: str) -> str:
        stem = Path(path).stem.lower()
        slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        return slug or "doc"

    def _process_documents(self) -> None:
        """Process all documents and add chunks to the in-memory Qdrant collection."""
        docs: list[str] = []
        metadata: list[dict[str, str | int]] = []
        ids: list[int] = []
        next_id = 0

        for file_path in self.file_paths:
            raw_text = self._extract_text(file_path)
            chunks = self._create_chunks(raw_text)
            source_name = os.path.basename(file_path)
            source_slug = self._source_slug(file_path)

            for chunk_index, chunk in enumerate(chunks, 1):
                chunk_text = getattr(chunk, "text", str(chunk)).strip()
                if not chunk_text:
                    continue

                docs.append(chunk_text)
                metadata.append(
                    {
                        "source": source_name,
                        "chunk_id": f"{source_slug}_c{chunk_index:03d}",
                    }
                )
                ids.append(next_id)
                next_id += 1

        if not docs:
            raise ValueError("No usable text was extracted from the uploaded PDFs.")

        self.client.add(
            collection_name=self.collection_name,
            documents=docs,
            metadata=metadata,
            ids=ids,
        )

    def describe_sources(self) -> list[str]:
        """Return the indexed source file names for UI display."""
        return list(self.source_names)

    def _run(self, query: str) -> str:
        """Search the indexed knowledge base and return excerpts with source tags."""
        relevant_chunks = self.client.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=self.top_k,
        )

        formatted_chunks: list[str] = []
        for chunk in relevant_chunks:
            chunk_text = getattr(chunk, "document", "") or ""
            chunk_meta = getattr(chunk, "metadata", {}) or {}
            source_name = chunk_meta.get("source", "unknown_source.pdf")
            chunk_id = chunk_meta.get("chunk_id", "chunk_unknown")
            if not chunk_text:
                continue
            formatted_chunks.append(
                f"[Source: {source_name} | Ref: {chunk_id}]\n{chunk_text}"
            )

        if not formatted_chunks:
            return "No relevant knowledge base excerpts were found for this query."
        return "\n___\n".join(formatted_chunks)
