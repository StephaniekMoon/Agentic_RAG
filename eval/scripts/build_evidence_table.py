import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from markitdown import MarkItDown
from chonkie import SemanticChunker


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    md = MarkItDown()
    result = md.convert(pdf_path)
    return result.text_content


def chunk_text(
    raw_text: str,
    embedding_model: str,
    threshold: float,
    chunk_size: int,
    min_sentences: int,
    overlap: int,
) -> List[str]:
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        threshold=threshold,
        chunk_size=chunk_size,
        min_sentences=min_sentences,
        overlap=overlap,
    )
    chunks = chunker.chunk(raw_text)
    return [c.text for c in chunks]


_RE_ALNUM = re.compile(r"[A-Za-z0-9]")


def is_noisy_chunk(
    text: str,
    *,
    min_chars: int,
    min_alnum_ratio: float,
    max_single_char_line_ratio: float,
    max_newline_ratio: float,
) -> Tuple[bool, str]:
    """
    Heuristic noise filter for PDF extraction artifacts.
    Typical noise: one character per line, or mostly whitespace/punctuation.
    """
    s = (text or "").strip()
    if len(s) < min_chars:
        return True, "too_short"

    newline_ratio = s.count("\n") / max(1, len(s))
    if newline_ratio > max_newline_ratio:
        return True, "newline_heavy"

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        single_char_lines = sum(1 for ln in lines if len(ln) <= 2)
        ratio = single_char_lines / max(1, len(lines))
        if ratio > max_single_char_line_ratio:
            return True, "single_char_lines"

    alnum_count = len(_RE_ALNUM.findall(s))
    alnum_ratio = alnum_count / max(1, len(s))
    if alnum_ratio < min_alnum_ratio:
        return True, "low_alnum_ratio"

    return False, ""


def build_rows(
    doc_id: str,
    source_doc: str,
    chunk_texts: Iterable[str],
    page: Optional[int] = None,
    *,
    min_chars: int,
    min_alnum_ratio: float,
    max_single_char_line_ratio: float,
    max_newline_ratio: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skipped_by_reason: Dict[str, int] = {}
    for idx, text in enumerate(chunk_texts, 1):
        text = (text or "").strip()
        if not text:
            continue
        noisy, reason = is_noisy_chunk(
            text,
            min_chars=min_chars,
            min_alnum_ratio=min_alnum_ratio,
            max_single_char_line_ratio=max_single_char_line_ratio,
            max_newline_ratio=max_newline_ratio,
        )
        if noisy:
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "source_doc": source_doc,
                "page": page,
                "chunk_id": f"{doc_id}_c{idx}",
                "text": text,
            }
        )
    if skipped_by_reason:
        skipped_total = sum(skipped_by_reason.values())
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(skipped_by_reason.items()))
        print(f"Skipped {skipped_total} noisy chunks ({reasons})")
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dumps_compact(row))
            f.write("\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build evidence table JSONL from a PDF using the current chunking logic.")
    parser.add_argument("--pdf", required=True, help="Path to a PDF file.")
    parser.add_argument("--output", required=True, help="Output evidence table JSONL path.")
    parser.add_argument("--doc-id", default="", help="Override doc_id (default: filename without extension).")
    parser.add_argument("--embedding-model", default="minishlab/potion-base-8M")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-sentences", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=200, help="Skip chunks shorter than this.")
    parser.add_argument(
        "--min-alnum-ratio",
        type=float,
        default=0.35,
        help="Skip chunks with too few letters/digits relative to length.",
    )
    parser.add_argument(
        "--max-single-char-line-ratio",
        type=float,
        default=0.45,
        help="Skip chunks where too many lines are 1-2 characters long.",
    )
    parser.add_argument(
        "--max-newline-ratio",
        type=float,
        default=0.08,
        help="Skip chunks with excessive newlines (often broken PDF extraction).",
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    source_doc = os.path.basename(pdf_path)
    doc_id = args.doc_id.strip() or os.path.splitext(source_doc)[0]

    raw_text = extract_text_from_pdf(pdf_path)
    chunk_texts = chunk_text(
        raw_text=raw_text,
        embedding_model=args.embedding_model,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        min_sentences=args.min_sentences,
        overlap=args.overlap,
    )

    rows = build_rows(
        doc_id=doc_id,
        source_doc=source_doc,
        chunk_texts=chunk_texts,
        page=None,
        min_chars=args.min_chars,
        min_alnum_ratio=args.min_alnum_ratio,
        max_single_char_line_ratio=args.max_single_char_line_ratio,
        max_newline_ratio=args.max_newline_ratio,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} chunks to {args.output}")


if __name__ == "__main__":
    main()
