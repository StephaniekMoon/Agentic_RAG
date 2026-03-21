from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence, Type

from chonkie import SemanticChunker
from crewai.tools import BaseTool
from markitdown import MarkItDown
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient, models


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
        self.embedding_model_name = self.client.embedding_model_name
        self.vector_name = self.client.get_vector_field_name()
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
            #跳过复制文件
            if resolved in seen_paths:
                continue
            #跳过不存在的文件
            if not os.path.isfile(resolved):
                raise FileNotFoundError(f"Knowledge base PDF not found: {resolved}")
            #跳过重复文件
            seen_paths.add(resolved)
            #跳过非PDF文件
            normalized_paths.append(resolved)

        if not normalized_paths:
            raise ValueError("DocumentSearchTool requires at least one PDF file.")
        return normalized_paths
    # PDF处理
    def _extract_text(self, path: str) -> str:
        """Extract raw text from a PDF using MarkItDown."""
        # 使用MarkItDown提取文本
        md = MarkItDown()
        #   提取PDF文本
        result = md.convert(path)
        # 返回文本
        return result.text_content
    # 语义分块
    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text."""
        # 创建语义分块
        chunker = SemanticChunker(
            # 使用minishlab/potion-base-8M模型进行文本嵌入
            embedding_model="minishlab/potion-base-8M",
            # 设置阈值为0.5，以控制分块的相似度
            threshold=0.5,
            # 设置分块大小为512字节
            chunk_size=512,
            # 设置最小句子数为1，以确保每个分块至少包含一个完整的句子
            min_sentences=1,
        )
        return chunker.chunk(raw_text)

    def _source_slug(self, path: str) -> str:
        # 生成源文件的Slug
        stem = Path(path).stem.lower()
        #   删除特殊字符
        slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        # 返回Slug
        return slug or "doc"
    # 创建索引
    def _ensure_collection(self) -> None:
        """Create the in-memory Qdrant collection with fastembed-aware vector settings."""
        try:
            # 检查是否存在指定的集合
            self.client.get_collection(collection_name=self.collection_name)
            return
        except Exception:
            pass
        # 创建集合
        sparse_vectors_config = None
        #   检查是否存在稀疏嵌入模型
        if getattr(self.client, "sparse_embedding_model_name", None) is not None:
            # 创建稀疏向量配置
            sparse_vectors_config = self.client.get_fastembed_sparse_vector_params()
        # 创建集合
        self.client.create_collection(
            # 集合名称
            collection_name=self.collection_name,
            # 向量配置
            vectors_config=self.client.get_fastembed_vector_params(),
            # 稀疏向量配置
            sparse_vectors_config=sparse_vectors_config,
        )

    def _process_documents(self) -> None:
        """Process all documents and upload chunks with the newer Qdrant inference API."""
        self._ensure_collection()

        points: list[models.PointStruct] = []
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

                points.append(
                    models.PointStruct(
                        id=next_id,
                        vector={
                            self.vector_name: models.Document(
                                text=chunk_text,
                                model=self.embedding_model_name,
                            )
                        },
                        payload={
                            "source": source_name,
                            "chunk_id": f"{source_slug}_c{chunk_index:03d}",
                            "text": chunk_text,
                        },
                    )
                )
                next_id += 1

        if not points:
            raise ValueError("No usable text was extracted from the uploaded PDFs.")

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def describe_sources(self) -> list[str]:
        """Return the indexed source file names for UI display."""
        return list(self.source_names)

    def _run(self, query: str) -> str:
        """Search the indexed knowledge base and return excerpts with source tags."""
        relevant_chunks = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(text=query, model=self.embedding_model_name),
            using=self.vector_name,
            limit=self.top_k,
            with_payload=True,
        )

        formatted_chunks: list[str] = []
        for chunk in relevant_chunks.points:
            chunk_meta = getattr(chunk, "payload", {}) or {}
            chunk_text = chunk_meta.get("text", "") or ""
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
