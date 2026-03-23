from __future__ import annotations

import importlib
import io
import math
import os
import re
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Type

from crewai.tools import BaseTool
from markitdown import MarkItDown
from pydantic import BaseModel, ConfigDict, Field


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def _int_from_env(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        warnings.warn(
            f"Invalid integer value for {name}: {value!r}. Falling back to {default}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default


def _float_from_env(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except ValueError:
        warnings.warn(
            f"Invalid float value for {name}: {value!r}. Falling back to {default}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return default


SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".docx")
DEFAULT_PDF_IMAGE_OCR_ENABLED = _bool_from_env("AGENTIC_RAG_ENABLE_PDF_IMAGE_OCR", True)
DEFAULT_OCR_LANG = os.getenv("AGENTIC_RAG_OCR_LANG", "chi_sim+eng")
DEFAULT_OCR_MIN_TEXT_LENGTH = _int_from_env("AGENTIC_RAG_OCR_MIN_TEXT_LENGTH", 10)
DEFAULT_OCR_MIN_IMAGE_DIMENSION = _int_from_env("AGENTIC_RAG_OCR_MIN_IMAGE_DIMENSION", 64)
DEFAULT_CHUNK_TARGET_CHARS = _int_from_env("AGENTIC_RAG_CHUNK_TARGET_CHARS", 900)
DEFAULT_CHUNK_MIN_CHARS = _int_from_env("AGENTIC_RAG_CHUNK_MIN_CHARS", 180)
DEFAULT_BM25_K1 = _float_from_env("AGENTIC_RAG_BM25_K1", 1.5)
DEFAULT_BM25_B = _float_from_env("AGENTIC_RAG_BM25_B", 0.75)

_TEXT_BLOCK_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
_HAS_MEANINGFUL_TEXT_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


@dataclass
class ChunkRecord:
    source: str
    chunk_id: str
    text: str
    page: int | None = None
    block_type: str = "text"
    section_title: str | None = None
    tokens: list[str] = field(default_factory=list)
    term_freqs: Counter[str] = field(default_factory=Counter)
    length: int = 0


@dataclass
class SearchHit:
    chunk: ChunkRecord
    score: float
    matched_terms: list[str] = field(default_factory=list)
    lexical_coverage: float = 0.0


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""

    query: str = Field(..., description="Query to search the indexed knowledge base.")


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = (
        "Search one or more indexed enterprise knowledge base documents and return the "
        "most relevant excerpts with source references."
    )
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        file_path: str | None = None,
        file_paths: list[str] | None = None,
        top_k: int = 5,
        enable_pdf_image_ocr: bool | None = None,
        ocr_lang: str | None = None,
        ocr_min_text_length: int | None = None,
        ocr_min_image_dimension: int | None = None,
        chunk_target_chars: int | None = None,
        chunk_min_chars: int | None = None,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
    ):
        """Initialize the search tool with one document or a list of documents."""
        super().__init__()
        self.file_paths = self._normalize_paths(file_path=file_path, file_paths=file_paths)
        self.top_k = max(1, top_k)
        self.enable_pdf_image_ocr = (
            DEFAULT_PDF_IMAGE_OCR_ENABLED if enable_pdf_image_ocr is None else enable_pdf_image_ocr
        )
        self.ocr_lang = (ocr_lang or DEFAULT_OCR_LANG).strip() or DEFAULT_OCR_LANG
        self.ocr_min_text_length = max(
            1,
            DEFAULT_OCR_MIN_TEXT_LENGTH if ocr_min_text_length is None else ocr_min_text_length,
        )
        self.ocr_min_image_dimension = max(
            1,
            DEFAULT_OCR_MIN_IMAGE_DIMENSION
            if ocr_min_image_dimension is None
            else ocr_min_image_dimension,
        )
        self.chunk_target_chars = max(
            100,
            DEFAULT_CHUNK_TARGET_CHARS if chunk_target_chars is None else chunk_target_chars,
        )
        self.chunk_min_chars = min(
            self.chunk_target_chars,
            max(40, DEFAULT_CHUNK_MIN_CHARS if chunk_min_chars is None else chunk_min_chars),
        )
        self.bm25_k1 = max(0.1, DEFAULT_BM25_K1 if bm25_k1 is None else bm25_k1)
        self.bm25_b = min(1.0, max(0.0, DEFAULT_BM25_B if bm25_b is None else bm25_b))
        self._ocr_runtime_cache: tuple[Any, Any, Any] | None = None
        self._ocr_runtime_checked = False
        self._warning_messages_emitted: set[str] = set()
        self.source_names = [os.path.basename(path) for path in self.file_paths]
        self.chunks: list[ChunkRecord] = []
        self.document_frequencies: Counter[str] = Counter()
        self.average_document_length = 0.0
        self._process_documents()

    def _normalize_paths(
        self,
        file_path: str | None,
        file_paths: list[str] | None,
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
                raise FileNotFoundError(f"Knowledge base document not found: {resolved}")
            if Path(resolved).suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
                supported = ", ".join(SUPPORTED_DOCUMENT_EXTENSIONS)
                raise ValueError(
                    f"Unsupported knowledge base document type: {resolved}. "
                    f"Supported types: {supported}"
                )
            seen_paths.add(resolved)
            normalized_paths.append(resolved)

        if not normalized_paths:
            raise ValueError("DocumentSearchTool requires at least one supported document file.")
        return normalized_paths

    def _process_documents(self) -> None:
        """Process all documents and build a local lexical index."""
        all_chunks: list[ChunkRecord] = []
        for file_path in self.file_paths:
            if Path(file_path).suffix.lower() == ".pdf":
                all_chunks.extend(self._extract_pdf_chunks(file_path))
            else:
                all_chunks.extend(self._extract_generic_document_chunks(file_path))

        indexed_chunks: list[ChunkRecord] = []
        document_frequencies: Counter[str] = Counter()
        total_length = 0

        for chunk in all_chunks:
            tokens = self._tokenize_text(chunk.text)
            if not tokens:
                continue
            chunk.tokens = tokens
            chunk.term_freqs = Counter(tokens)
            chunk.length = len(tokens)
            indexed_chunks.append(chunk)
            total_length += chunk.length
            document_frequencies.update(set(tokens))

        if not indexed_chunks:
            raise ValueError("No usable text was extracted from the uploaded documents.")

        self.chunks = indexed_chunks
        self.document_frequencies = document_frequencies
        self.average_document_length = total_length / len(indexed_chunks)

    def _extract_pdf_chunks(self, path: str) -> list[ChunkRecord]:
        """Extract PDF-specific chunks for headings, TOC blocks, body text, tables, and OCR."""
        fitz = self._load_pymupdf()
        source_name = os.path.basename(path)
        source_slug = self._source_slug(path)
        chunks: list[ChunkRecord] = []
        current_section_title: str | None = None

        with fitz.open(path) as pdf_document:
            for page_index in range(pdf_document.page_count):
                page = pdf_document.load_page(page_index)
                page_number = page_index + 1
                page_dict = page.get_text("dict")
                table_chunks, table_regions = self._extract_pdf_table_chunks(
                    page=page,
                    source_name=source_name,
                    source_slug=source_slug,
                    page_number=page_number,
                    current_section_title=current_section_title,
                )
                chunks.extend(table_chunks)
                block_infos = self._build_pdf_text_blocks(page_dict=page_dict, table_regions=table_regions)
                page_font_sizes = [block["avg_font_size"] for block in block_infos if block["avg_font_size"] > 0]
                page_base_font_size = median(page_font_sizes) if page_font_sizes else 11.0
                text_segments: list[str] = []
                chunk_serial = 1

                for block in block_infos:
                    block_text = block["text"]
                    block_kind = self._classify_pdf_block(
                        block=block,
                        page_number=page_number,
                        page_count=pdf_document.page_count,
                        page_base_font_size=page_base_font_size,
                    )

                    if block_kind in {"pdf_heading", "pdf_toc"}:
                        if text_segments:
                            chunks.append(
                                self._build_chunk_record(
                                    source_name=source_name,
                                    chunk_id=f"{source_slug}_p{page_number:03d}_b{chunk_serial:03d}",
                                    text=self._compose_section_chunk_text(
                                        section_title=current_section_title,
                                        body_text="\n".join(text_segments),
                                    ),
                                    page=page_number,
                                    block_type="pdf_section_text",
                                    section_title=current_section_title,
                                )
                            )
                            chunk_serial += 1
                            text_segments = []

                        if block_kind == "pdf_heading":
                            current_section_title = block_text

                        chunks.append(
                            self._build_chunk_record(
                                source_name=source_name,
                                chunk_id=f"{source_slug}_p{page_number:03d}_b{chunk_serial:03d}",
                                text=block_text,
                                page=page_number,
                                block_type=block_kind,
                                section_title=current_section_title if block_kind != "pdf_heading" else block_text,
                            )
                        )
                        chunk_serial += 1
                        continue

                    for piece in self._split_text_to_segments(block_text):
                        if not piece:
                            continue
                        if text_segments and self._joined_length(text_segments, piece) > self.chunk_target_chars:
                            chunks.append(
                                self._build_chunk_record(
                                    source_name=source_name,
                                    chunk_id=f"{source_slug}_p{page_number:03d}_b{chunk_serial:03d}",
                                    text=self._compose_section_chunk_text(
                                        section_title=current_section_title,
                                        body_text="\n".join(text_segments),
                                    ),
                                    page=page_number,
                                    block_type="pdf_section_text",
                                    section_title=current_section_title,
                                )
                            )
                            chunk_serial += 1
                            text_segments = []

                        text_segments.append(piece)

                        if self._segments_length(text_segments) >= self.chunk_target_chars:
                            chunks.append(
                                self._build_chunk_record(
                                    source_name=source_name,
                                    chunk_id=f"{source_slug}_p{page_number:03d}_b{chunk_serial:03d}",
                                    text=self._compose_section_chunk_text(
                                        section_title=current_section_title,
                                        body_text="\n".join(text_segments),
                                    ),
                                    page=page_number,
                                    block_type="pdf_section_text",
                                    section_title=current_section_title,
                                )
                            )
                            chunk_serial += 1
                            text_segments = []

                if text_segments:
                    chunks.append(
                        self._build_chunk_record(
                            source_name=source_name,
                            chunk_id=f"{source_slug}_p{page_number:03d}_b{chunk_serial:03d}",
                            text=self._compose_section_chunk_text(
                                section_title=current_section_title,
                                body_text="\n".join(text_segments),
                            ),
                            page=page_number,
                            block_type="pdf_section_text",
                            section_title=current_section_title,
                        )
                    )

                chunks.extend(
                    self._extract_pdf_image_ocr_chunks(
                        pdf_document=pdf_document,
                        page=page,
                        source_name=source_name,
                        source_slug=source_slug,
                        page_number=page_number,
                    )
                )

        return chunks

    def _extract_generic_document_chunks(self, path: str) -> list[ChunkRecord]:
        """Extract paragraph-aware chunks from non-PDF documents."""
        md = MarkItDown()
        result = md.convert(path)
        raw_text = self._normalize_extracted_text(result.text_content or "")
        if not raw_text:
            return []

        source_name = os.path.basename(path)
        source_slug = self._source_slug(path)
        paragraphs = [
            self._normalize_extracted_text(part)
            for part in re.split(r"\n\s*\n", raw_text)
            if self._normalize_extracted_text(part)
        ]

        chunks: list[ChunkRecord] = []
        text_segments: list[str] = []
        chunk_serial = 1

        for paragraph in paragraphs:
            for piece in self._split_text_to_segments(paragraph):
                if not piece:
                    continue
                if text_segments and self._joined_length(text_segments, piece) > self.chunk_target_chars:
                    chunks.append(
                        self._build_chunk_record(
                            source_name=source_name,
                            chunk_id=f"{source_slug}_c{chunk_serial:03d}",
                            text="\n".join(text_segments),
                            page=None,
                            block_type="document_text",
                            section_title=None,
                        )
                    )
                    chunk_serial += 1
                    text_segments = []

                text_segments.append(piece)

        if text_segments:
            chunks.append(
                self._build_chunk_record(
                    source_name=source_name,
                    chunk_id=f"{source_slug}_c{chunk_serial:03d}",
                    text="\n".join(text_segments),
                    page=None,
                    block_type="document_text",
                    section_title=None,
                )
            )

        return chunks

    def _extract_pdf_image_ocr_chunks(
        self,
        pdf_document: Any,
        page: Any,
        source_name: str,
        source_slug: str,
        page_number: int,
    ) -> list[ChunkRecord]:
        """Extract OCR text from embedded PDF images and keep it page-local."""
        if not self.enable_pdf_image_ocr:
            return []

        ocr_runtime = self._load_pdf_image_ocr_runtime()
        if ocr_runtime is None:
            return []

        _, image_module, pytesseract = ocr_runtime
        chunks: list[ChunkRecord] = []

        for image_number, image_info in enumerate(page.get_images(full=True), 1):
            xref = image_info[0]
            try:
                base_image = pdf_document.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image.get("image")
            image_width = int(base_image.get("width") or 0)
            image_height = int(base_image.get("height") or 0)
            if not image_bytes:
                continue
            if min(image_width, image_height) < self.ocr_min_image_dimension:
                continue

            try:
                image = image_module.open(io.BytesIO(image_bytes)).convert("RGB")
                ocr_text = pytesseract.image_to_string(image, lang=self.ocr_lang)
            except Exception as exc:
                self._warn_once(
                    f"PDF image OCR encountered an image it could not process in {source_name}: {exc}"
                )
                continue

            normalized_ocr_text = self._normalize_ocr_text(ocr_text)
            if len(normalized_ocr_text) < self.ocr_min_text_length:
                continue

            chunks.append(
                self._build_chunk_record(
                    source_name=source_name,
                    chunk_id=f"{source_slug}_p{page_number:03d}_img{image_number:03d}",
                    text=f"[OCR Image Text | Page {page_number} | Image {image_number}]\n{normalized_ocr_text}",
                    page=page_number,
                    block_type="pdf_image_ocr",
                    section_title=None,
                )
            )

        return chunks

    def _extract_pdf_table_chunks(
        self,
        page: Any,
        source_name: str,
        source_slug: str,
        page_number: int,
        current_section_title: str | None,
    ) -> tuple[list[ChunkRecord], list[tuple[float, float, float, float]]]:
        """Extract table-shaped regions as dedicated chunks."""
        table_chunks: list[ChunkRecord] = []
        table_regions: list[tuple[float, float, float, float]] = []

        try:
            table_finder = page.find_tables()
        except Exception:
            return table_chunks, table_regions

        for table_index, table in enumerate(getattr(table_finder, "tables", []), 1):
            bbox = tuple(table.bbox)
            table_regions.append(bbox)
            try:
                table_text = self._normalize_extracted_text(table.to_markdown())
            except Exception:
                try:
                    extracted_rows = table.extract()
                except Exception:
                    extracted_rows = []
                table_text = self._normalize_extracted_text(
                    "\n".join(" | ".join(cell or "" for cell in row) for row in extracted_rows)
                )

            if not self._is_meaningful_block(table_text):
                continue

            composed_table_text = self._compose_section_chunk_text(
                section_title=current_section_title,
                body_text=f"[Table | Page {page_number}]\n{table_text}",
            )
            table_chunks.append(
                self._build_chunk_record(
                    source_name=source_name,
                    chunk_id=f"{source_slug}_p{page_number:03d}_tbl{table_index:03d}",
                    text=composed_table_text,
                    page=page_number,
                    block_type="pdf_table",
                    section_title=current_section_title,
                )
            )

        return table_chunks, table_regions

    def _load_pdf_image_ocr_runtime(self) -> tuple[Any, Any, Any] | None:
        """Load OCR dependencies lazily so text-only PDF retrieval still works without OCR extras."""
        if self._ocr_runtime_checked:
            return self._ocr_runtime_cache

        self._ocr_runtime_checked = True
        missing_packages: list[str] = []

        try:
            fitz = importlib.import_module("fitz")
        except ImportError:
            fitz = None
            missing_packages.append("pymupdf")

        try:
            image_module = importlib.import_module("PIL.Image")
        except ImportError:
            image_module = None
            missing_packages.append("pillow")

        try:
            pytesseract = importlib.import_module("pytesseract")
        except ImportError:
            pytesseract = None
            missing_packages.append("pytesseract")

        if missing_packages:
            self._warn_once(
                "PDF image OCR is disabled because optional packages are missing: "
                + ", ".join(missing_packages)
            )
            return None

        if shutil.which("tesseract") is None:
            self._warn_once(
                "PDF image OCR is disabled because the 'tesseract' binary is not installed or not on PATH."
            )
            return None

        self._ocr_runtime_cache = (fitz, image_module, pytesseract)
        return self._ocr_runtime_cache

    def _load_pymupdf(self) -> Any:
        """Load PyMuPDF only when PDF processing is needed."""
        try:
            return importlib.import_module("fitz")
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF-aware chunking and retrieval.") from exc

    def _build_chunk_record(
        self,
        source_name: str,
        chunk_id: str,
        text: str,
        page: int | None,
        block_type: str,
        section_title: str | None,
    ) -> ChunkRecord:
        return ChunkRecord(
            source=source_name,
            chunk_id=chunk_id,
            text=self._normalize_extracted_text(text),
            page=page,
            block_type=block_type,
            section_title=section_title,
        )

    def _build_pdf_text_blocks(
        self,
        page_dict: dict[str, Any],
        table_regions: list[tuple[float, float, float, float]],
    ) -> list[dict[str, Any]]:
        """Convert page layout blocks into text blocks enriched with font and line metadata."""
        text_blocks: list[dict[str, Any]] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0 or not block.get("lines"):
                continue

            bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
            if self._bbox_overlaps_any_region(bbox, table_regions):
                continue

            line_texts: list[str] = []
            font_sizes: list[float] = []
            fonts: list[str] = []

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans)
                line_text = self._normalize_extracted_text(line_text)
                if not line_text:
                    continue
                line_texts.append(line_text)
                for span in spans:
                    if span.get("text", "").strip():
                        font_sizes.append(float(span.get("size") or 0.0))
                        fonts.append(str(span.get("font") or ""))

            block_text = self._normalize_extracted_text("\n".join(line_texts))
            if not self._is_meaningful_block(block_text):
                continue

            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
            max_font_size = max(font_sizes) if font_sizes else 0.0
            text_blocks.append(
                {
                    "text": block_text,
                    "bbox": bbox,
                    "line_count": len(line_texts),
                    "line_texts": line_texts,
                    "avg_font_size": avg_font_size,
                    "max_font_size": max_font_size,
                    "fonts": fonts,
                }
            )

        return text_blocks

    def _classify_pdf_block(
        self,
        block: dict[str, Any],
        page_number: int,
        page_count: int,
        page_base_font_size: float,
    ) -> str:
        """Classify a PDF text block into heading, TOC, or body text."""
        text = block["text"]
        if self._looks_like_toc_block(text=text, line_texts=block["line_texts"], page_number=page_number, page_count=page_count):
            return "pdf_toc"
        if self._looks_like_heading(
            text=text,
            avg_font_size=block["avg_font_size"],
            max_font_size=block["max_font_size"],
            line_count=block["line_count"],
            page_base_font_size=page_base_font_size,
        ):
            return "pdf_heading"
        return "pdf_body"

    def _looks_like_toc_block(
        self,
        text: str,
        line_texts: list[str],
        page_number: int,
        page_count: int,
    ) -> bool:
        """Heuristics for table-of-contents blocks common in SOPs and manuals."""
        normalized = self._normalize_search_text(text)
        early_pages = page_number <= max(6, math.ceil(page_count * 0.2))
        toc_title = "目录" in normalized or "contents" in normalized
        toc_lines = sum(
            1
            for line in line_texts
            if re.search(r"(\.{2,}|…{2,}|·{2,}|-{2,}|\s{3,})\s*\d+\s*$", line)
            or re.search(r".+[章节条部分节].*\d+\s*$", line)
        )
        return early_pages and (toc_title or toc_lines >= 2)

    def _looks_like_heading(
        self,
        text: str,
        avg_font_size: float,
        max_font_size: float,
        line_count: int,
        page_base_font_size: float,
    ) -> bool:
        """Heuristics for section headings in SOPs, policies, and product documents."""
        compact_text = text.replace("\n", " ").strip()
        short_block = len(compact_text) <= 90 and line_count <= 3
        numbered_heading = bool(
            re.match(r"^(\d+(\.\d+){0,4}|[IVXLC]+\.)\s+\S+", compact_text, flags=re.IGNORECASE)
            or re.match(r"^第[一二三四五六七八九十百零0-9]+[章节条部分篇]\s*\S*", compact_text)
            or re.match(r"^[（(]?[一二三四五六七八九十0-9]+[)）、.]\s*\S+", compact_text)
        )
        heading_like_font = max_font_size >= page_base_font_size * 1.18 or avg_font_size >= page_base_font_size * 1.12
        uppercase_heading = compact_text.isupper() and len(compact_text.split()) <= 8
        colon_heading = compact_text.endswith(":") or compact_text.endswith("：")
        return short_block and (numbered_heading or heading_like_font or uppercase_heading or colon_heading)

    def _compose_section_chunk_text(self, section_title: str | None, body_text: str) -> str:
        """Keep section title attached to the body chunk for structure-aware retrieval."""
        normalized_body = self._normalize_extracted_text(body_text)
        if not normalized_body:
            return ""
        if not section_title:
            return normalized_body
        normalized_heading = self._normalize_extracted_text(section_title)
        if not normalized_heading:
            return normalized_body
        if normalized_body.startswith(normalized_heading):
            return normalized_body
        return f"[Section]\n{normalized_heading}\n\n{normalized_body}"

    def _bbox_overlaps_any_region(
        self,
        bbox: tuple[float, float, float, float],
        regions: list[tuple[float, float, float, float]],
    ) -> bool:
        """Return True when a text block intersects a detected table region."""
        for region in regions:
            if self._bbox_overlap_ratio(bbox, region) > 0.2:
                return True
        return False

    def _bbox_overlap_ratio(
        self,
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        x0 = max(left[0], right[0])
        y0 = max(left[1], right[1])
        x1 = min(left[2], right[2])
        y1 = min(left[3], right[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        intersection = (x1 - x0) * (y1 - y0)
        left_area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
        return intersection / left_area

    def _split_text_to_segments(self, text: str) -> list[str]:
        """Split long extracted text into retrieval-friendly segments without semantic models."""
        normalized_text = self._normalize_extracted_text(text)
        if not normalized_text:
            return []
        if len(normalized_text) <= self.chunk_target_chars:
            return [normalized_text]

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized_text) if part.strip()]
        if len(paragraphs) == 1:
            paragraphs = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", normalized_text) if part.strip()]
        if len(paragraphs) == 1:
            paragraphs = [part.strip() for part in re.split(r"\n+", normalized_text) if part.strip()]
        if len(paragraphs) == 1:
            paragraphs = [normalized_text[i : i + self.chunk_target_chars] for i in range(0, len(normalized_text), self.chunk_target_chars)]

        segments: list[str] = []
        current_parts: list[str] = []

        for paragraph in paragraphs:
            if current_parts and self._joined_length(current_parts, paragraph) > self.chunk_target_chars:
                segments.append("\n".join(current_parts))
                current_parts = []
            current_parts.append(paragraph)

            if self._segments_length(current_parts) >= self.chunk_target_chars:
                segments.append("\n".join(current_parts))
                current_parts = []

        if current_parts:
            segments.append("\n".join(current_parts))

        return [segment.strip() for segment in segments if segment.strip()]

    def _segments_length(self, parts: list[str]) -> int:
        return sum(len(part) for part in parts)

    def _joined_length(self, parts: list[str], new_part: str) -> int:
        separator_cost = 1 if parts else 0
        return self._segments_length(parts) + separator_cost + len(new_part)

    def _normalize_extracted_text(self, text: str) -> str:
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _normalize_ocr_text(self, text: str) -> str:
        return self._normalize_extracted_text(text)

    def _normalize_search_text(self, text: str) -> str:
        normalized = self._normalize_extracted_text(text).lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _is_meaningful_block(self, text: str) -> bool:
        return bool(text and len(text) >= 3 and _HAS_MEANINGFUL_TEXT_RE.search(text))

    def _tokenize_text(self, text: str) -> list[str]:
        tokens: list[str] = []
        normalized = self._normalize_search_text(text)
        for match in _TEXT_BLOCK_RE.finditer(normalized):
            token = match.group(0)
            if not token:
                continue
            if "\u4e00" <= token[0] <= "\u9fff":
                if len(token) == 1:
                    tokens.append(token)
                    continue
                tokens.append(token)
                tokens.extend(list(token))
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
            else:
                tokens.append(token)
        return tokens

    def _idf(self, token: str) -> float:
        document_count = len(self.chunks)
        document_frequency = self.document_frequencies.get(token, 0)
        if document_frequency == 0 or document_count == 0:
            return 0.0
        return math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))

    def _score_chunk(self, chunk: ChunkRecord, query: str, query_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0

        score = 0.0
        chunk_length = max(1, chunk.length)
        avg_length = max(1.0, self.average_document_length)
        query_term_frequencies = Counter(query_tokens)

        for token, query_frequency in query_term_frequencies.items():
            term_frequency = chunk.term_freqs.get(token, 0)
            if term_frequency == 0:
                continue

            denominator = term_frequency + self.bm25_k1 * (
                1.0 - self.bm25_b + self.bm25_b * (chunk_length / avg_length)
            )
            bm25_term_score = self._idf(token) * (
                term_frequency * (self.bm25_k1 + 1.0) / max(1e-9, denominator)
            )
            score += bm25_term_score * (1.0 + 0.1 * (query_frequency - 1))

        normalized_query = self._normalize_search_text(query)
        normalized_chunk_text = self._normalize_search_text(chunk.text)
        if normalized_query and normalized_query in normalized_chunk_text:
            score += 2.0

        unique_query_tokens = set(query_tokens)
        if unique_query_tokens and unique_query_tokens.issubset(set(chunk.tokens)):
            score += 0.75

        if score > 0 and chunk.block_type == "pdf_heading":
            score += 0.2
        if score > 0 and chunk.block_type == "pdf_section_text":
            score += 0.1
        if score > 0 and chunk.block_type == "pdf_table":
            score += 0.15
        if score > 0 and chunk.block_type == "pdf_toc":
            score += 0.03
        if score > 0 and chunk.block_type == "pdf_image_ocr":
            score += 0.05

        return score

    def _source_slug(self, path: str) -> str:
        stem = Path(path).stem.lower()
        slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        return slug or "doc"

    def _warn_once(self, message: str) -> None:
        if message in self._warning_messages_emitted:
            return
        self._warning_messages_emitted.add(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def describe_sources(self) -> list[str]:
        """Return the indexed source file names for UI display."""
        return list(self.source_names)

    def classify_query_type(self, query: str) -> dict[str, bool]:
        """Classify the query so the app can prefer rule extraction for precision-heavy asks."""
        normalized = self._normalize_search_text(query)
        return {
            "numeric": bool(
                re.search(r"\d", query)
                or any(
                    keyword in normalized
                    for keyword in [
                        "多少",
                        "几",
                        "编号",
                        "版本",
                        "日期",
                        "金额",
                        "端口",
                        "ip",
                        "序列号",
                        "number",
                        "version",
                        "date",
                        "amount",
                        "price",
                        "port",
                    ]
                )
            ),
            "table": bool(
                any(keyword in normalized for keyword in ["表", "表格", "列表", "清单", "table", "matrix", "catalog"])
            ),
            "steps": bool(
                any(
                    keyword in normalized
                    for keyword in ["步骤", "流程", "如何", "怎么", "procedure", "steps", "workflow", "how to", "configure"]
                )
            ),
        }

    def retrieve_hits(self, query: str, limit: int | None = None) -> list[SearchHit]:
        """Return structured retrieval hits with scores for downstream routing and validation."""
        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            return []

        unique_query_tokens = sorted(set(query_tokens))
        ranked_hits: list[SearchHit] = []
        for chunk in self.chunks:
            score = self._score_chunk(chunk=chunk, query=query, query_tokens=query_tokens)
            if score <= 0:
                continue
            matched_terms = [token for token in unique_query_tokens if chunk.term_freqs.get(token, 0) > 0]
            lexical_coverage = len(matched_terms) / max(1, len(unique_query_tokens))
            ranked_hits.append(
                SearchHit(
                    chunk=chunk,
                    score=score,
                    matched_terms=matched_terms,
                    lexical_coverage=lexical_coverage,
                )
            )

        ranked_hits.sort(key=lambda item: item.score, reverse=True)
        return ranked_hits[: limit or self.top_k]

    def assess_confidence(self, query: str, hits: list[SearchHit]) -> dict[str, Any]:
        """Estimate retrieval confidence so low-confidence cases can automatically fall back."""
        if not hits:
            return {
                "level": "low",
                "score": 0.0,
                "reasons": ["No evidence chunks matched the query."],
            }

        top_score = hits[0].score
        second_score = hits[1].score if len(hits) > 1 else 0.0
        average_score = sum(hit.score for hit in hits) / len(hits)
        top_coverage = hits[0].lexical_coverage
        query_tokens = sorted(set(self._tokenize_text(query)))
        top_matched_term_count = len(hits[0].matched_terms)
        evidence_sources = {hit.chunk.source for hit in hits}
        non_toc_hits = [hit for hit in hits if hit.chunk.block_type != "pdf_toc"]
        only_toc = bool(hits) and not non_toc_hits

        confidence_score = 0.0
        confidence_score += min(1.0, top_score / 6.0) * 0.45
        confidence_score += top_coverage * 0.25
        confidence_score += min(1.0, max(0.0, top_score - second_score) / max(1.0, top_score)) * 0.10
        confidence_score += min(1.0, average_score / 5.0) * 0.10
        confidence_score += 0.10 if len(evidence_sources) <= 2 else 0.0
        confidence_score = max(0.0, min(1.0, confidence_score))

        reasons: list[str] = []
        if only_toc:
            reasons.append("Only table-of-contents style chunks matched the query.")
            confidence_score *= 0.4
        if top_coverage < 0.35:
            reasons.append("Top hit covers only a small portion of the query terms.")
        if top_score < 1.2:
            reasons.append("Top retrieval score is weak.")
        if len(query_tokens) >= 4 and top_matched_term_count <= 2:
            reasons.append("Too few distinct query terms were matched by the top evidence chunk.")
        if len(non_toc_hits) == 0:
            reasons.append("No body, heading, table, or OCR evidence was found.")

        if top_coverage < 0.3:
            confidence_score *= 0.6
        if len(query_tokens) >= 4 and top_matched_term_count <= 2:
            confidence_score *= 0.55
        if not any(self._normalize_search_text(query) in self._normalize_search_text(hit.chunk.text) for hit in hits[:2]):
            confidence_score *= 0.9

        if confidence_score >= 0.72 and not only_toc:
            level = "high"
        elif confidence_score >= 0.42 and not only_toc:
            level = "medium"
        else:
            level = "low"

        return {
            "level": level,
            "score": round(confidence_score, 3),
            "top_score": round(top_score, 3),
            "average_score": round(average_score, 3),
            "top_coverage": round(top_coverage, 3),
            "reasons": reasons,
        }

    def prepare_answer_bundle(self, query: str, limit: int | None = None) -> dict[str, Any]:
        """Build a structured retrieval package used by fallback and validation logic."""
        hits = self.retrieve_hits(query=query, limit=limit or max(self.top_k, 5))
        return {
            "query": query,
            "query_type": self.classify_query_type(query),
            "hits": hits,
            "confidence": self.assess_confidence(query=query, hits=hits),
        }

    def extract_rule_based_answer(self, bundle: dict[str, Any]) -> str | None:
        """Prefer extractive answers for numeric, table, and step-oriented questions."""
        hits: list[SearchHit] = bundle["hits"]
        query: str = bundle["query"]
        query_type: dict[str, bool] = bundle["query_type"]
        if not hits:
            return None

        if query_type["table"]:
            table_lines = self._collect_relevant_lines(
                hits=hits,
                query=query,
                prefer_types={"pdf_table"},
                require_query_overlap=False,
            )
            if table_lines:
                return self._format_extractive_answer(
                    title="Rule-Based Table Answer",
                    query=query,
                    lines=table_lines[:6],
                    note="Returned the most relevant table-like rows directly from the knowledge base.",
                )

        if query_type["steps"]:
            step_lines = self._collect_step_lines(hits=hits, query=query)
            if step_lines:
                return self._format_extractive_answer(
                    title="Rule-Based Procedure Answer",
                    query=query,
                    lines=step_lines[:8],
                    note="Returned procedural steps directly from the retrieved evidence.",
                )

        if query_type["numeric"]:
            numeric_lines = self._collect_relevant_lines(
                hits=hits,
                query=query,
                prefer_types={"pdf_table", "pdf_section_text", "pdf_heading", "pdf_image_ocr", "document_text"},
                require_query_overlap=True,
                require_numbers=True,
            )
            if numeric_lines:
                return self._format_extractive_answer(
                    title="Rule-Based Exact Answer",
                    query=query,
                    lines=numeric_lines[:6],
                    note="Returned lines with exact numbers or identifiers to avoid hallucinated values.",
                )

        return None

    def build_low_confidence_fallback(self, bundle: dict[str, Any]) -> str:
        """Return an evidence-first fallback answer when retrieval confidence is too low."""
        confidence = bundle["confidence"]
        hits: list[SearchHit] = bundle["hits"]
        lines = self._collect_relevant_lines(
            hits=hits,
            query=bundle["query"],
            prefer_types={"pdf_heading", "pdf_section_text", "pdf_table", "pdf_image_ocr", "document_text"},
            require_query_overlap=False,
        )

        response_lines = [
            "I could not answer this confidently from the current evidence, so I am falling back to direct excerpts.",
            f"Retrieval confidence: `{confidence['level']}` ({confidence['score']}).",
        ]
        for reason in confidence.get("reasons", [])[:3]:
            response_lines.append(f"- {reason}")

        if lines:
            response_lines.append("")
            response_lines.append("Most relevant excerpts:")
            response_lines.extend(lines[:5])
        else:
            response_lines.append("")
            response_lines.append("No sufficiently relevant excerpt was found in the indexed documents.")
        return "\n".join(response_lines)

    def validate_generated_answer(self, answer: str, bundle: dict[str, Any]) -> dict[str, Any]:
        """Check whether the model answer stays within the retrieved evidence."""
        evidence_text = "\n".join(hit.chunk.text for hit in bundle["hits"])
        evidence_normalized = self._normalize_search_text(evidence_text)
        answer_normalized = self._normalize_search_text(answer)
        if not answer_normalized:
            return {"is_valid": False, "issues": ["The answer was empty."], "score": 0.0}

        issues: list[str] = []
        support_score = 0.0

        answer_numbers = set(re.findall(r"\d+(?:[./:-]\d+)*", answer))
        evidence_numbers = set(re.findall(r"\d+(?:[./:-]\d+)*", evidence_text))
        unsupported_numbers = sorted(answer_numbers - evidence_numbers)
        if unsupported_numbers:
            issues.append(
                "The answer introduced numbers or identifiers not seen in the evidence: "
                + ", ".join(unsupported_numbers[:5])
            )
        else:
            support_score += 0.35

        answer_tokens = set(self._tokenize_text(answer))
        evidence_tokens = set(self._tokenize_text(evidence_text))
        content_tokens = {
            token
            for token in answer_tokens
            if len(token) > 1 and token not in {"source", "page", "section", "ref"}
        }
        unsupported_tokens = sorted(token for token in content_tokens if token not in evidence_tokens)
        unsupported_ratio = len(unsupported_tokens) / max(1, len(content_tokens))
        if unsupported_ratio > 0.45:
            issues.append("A large portion of the answer vocabulary is unsupported by the evidence.")
        else:
            support_score += max(0.0, 0.45 - unsupported_ratio)

        supporting_sentences = 0
        answer_sentences = [segment.strip() for segment in re.split(r"[\n。！？!?]+", answer) if segment.strip()]
        for sentence in answer_sentences:
            normalized_sentence = self._normalize_search_text(sentence)
            if not normalized_sentence:
                continue
            if normalized_sentence in evidence_normalized:
                supporting_sentences += 1
                continue
            sentence_tokens = set(self._tokenize_text(sentence))
            if sentence_tokens and len(sentence_tokens & evidence_tokens) / max(1, len(sentence_tokens)) >= 0.5:
                supporting_sentences += 1

        sentence_support_ratio = supporting_sentences / max(1, len(answer_sentences))
        if sentence_support_ratio < 0.5:
            issues.append("Too few answer sentences are clearly supported by the retrieved evidence.")
        else:
            support_score += min(0.4, sentence_support_ratio * 0.4)

        return {
            "is_valid": not issues,
            "issues": issues,
            "score": round(max(0.0, min(1.0, support_score)), 3),
        }

    def build_validation_fallback(self, answer: str, bundle: dict[str, Any], validation: dict[str, Any]) -> str:
        """Replace unsupported LLM output with evidence-first excerpts."""
        lines = self._collect_relevant_lines(
            hits=bundle["hits"],
            query=bundle["query"],
            prefer_types={"pdf_heading", "pdf_section_text", "pdf_table", "pdf_image_ocr", "document_text"},
            require_query_overlap=False,
        )
        response_lines = [
            "The generated answer could not be fully verified against the retrieved evidence, so I am falling back to grounded excerpts.",
        ]
        for issue in validation.get("issues", [])[:3]:
            response_lines.append(f"- {issue}")
        response_lines.append("")
        response_lines.append("Verified evidence:")
        response_lines.extend(lines[:6] or ["No verified excerpt was available."])
        return "\n".join(response_lines)

    def format_hits_for_prompt(self, hits: list[SearchHit], limit: int | None = None) -> str:
        """Serialize retrieved evidence for prompting or fallback display."""
        formatted_chunks: list[str] = []
        for hit in hits[: limit or len(hits)]:
            chunk = hit.chunk
            page_tag = f" | Page: {chunk.page}" if chunk.page is not None else ""
            section_tag = f" | Section: {chunk.section_title}" if chunk.section_title else ""
            formatted_chunks.append(
                f"[Source: {chunk.source}{page_tag}{section_tag} | Ref: {chunk.chunk_id} | Score: {hit.score:.2f}]\n{chunk.text}"
            )
        return "\n___\n".join(formatted_chunks)

    def _run(self, query: str) -> str:
        """Search the indexed knowledge base and return excerpts with source tags."""
        hits = self.retrieve_hits(query=query, limit=self.top_k)
        formatted_chunks: list[str] = []
        for hit in hits:
            chunk = hit.chunk
            page_tag = f" | Page: {chunk.page}" if chunk.page is not None else ""
            section_tag = f" | Section: {chunk.section_title}" if chunk.section_title else ""
            formatted_chunks.append(
                f"[Source: {chunk.source}{page_tag}{section_tag} | Ref: {chunk.chunk_id}]\n{chunk.text}"
            )
        if not formatted_chunks:
            return "No relevant knowledge base excerpts were found for this query."
        return "\n___\n".join(formatted_chunks)

    def _collect_relevant_lines(
        self,
        hits: list[SearchHit],
        query: str,
        prefer_types: set[str],
        require_query_overlap: bool,
        require_numbers: bool = False,
    ) -> list[str]:
        """Collect the most useful evidence lines for fallback answers."""
        query_tokens = set(self._tokenize_text(query))
        lines: list[str] = []
        seen_lines: set[str] = set()

        ordered_hits = sorted(
            hits,
            key=lambda hit: (hit.chunk.block_type in prefer_types, hit.score),
            reverse=True,
        )

        for hit in ordered_hits:
            for raw_line in [part.strip() for part in hit.chunk.text.splitlines() if part.strip()]:
                normalized_line = self._normalize_search_text(raw_line)
                if normalized_line in seen_lines:
                    continue
                line_tokens = set(self._tokenize_text(raw_line))
                if require_query_overlap and query_tokens and not (line_tokens & query_tokens):
                    continue
                if require_numbers and not re.search(r"\d", raw_line):
                    continue
                seen_lines.add(normalized_line)
                page_tag = f" | Page: {hit.chunk.page}" if hit.chunk.page is not None else ""
                section_tag = f" | Section: {hit.chunk.section_title}" if hit.chunk.section_title else ""
                lines.append(
                    f"- [Source: {hit.chunk.source}{page_tag}{section_tag} | Ref: {hit.chunk.chunk_id}] {raw_line}"
                )
        return lines

    def _collect_step_lines(self, hits: list[SearchHit], query: str) -> list[str]:
        """Extract step-like lines for workflow and procedure questions."""
        step_pattern = re.compile(
            r"^((step\s*\d+)|(\d+[.)、])|(第[一二三四五六七八九十百零0-9]+步)|([一二三四五六七八九十]+[、.]))",
            flags=re.IGNORECASE,
        )
        lines = self._collect_relevant_lines(
            hits=hits,
            query=query,
            prefer_types={"pdf_section_text", "document_text", "pdf_heading"},
            require_query_overlap=False,
        )
        step_lines = [line for line in lines if step_pattern.search(line.split("] ", 1)[-1])]
        return step_lines or lines

    def _format_extractive_answer(self, title: str, query: str, lines: list[str], note: str) -> str:
        """Render a rule-based extractive answer."""
        response_lines = [f"**{title}**", f"Question: {query}", note, ""]
        response_lines.extend(lines or ["- No matching evidence line was found."])
        return "\n".join(response_lines)
