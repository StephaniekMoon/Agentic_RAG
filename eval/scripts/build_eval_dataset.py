import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def flatten_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    qa_index = 1

    for row in rows:
        evidence = row.get("evidence") or {}
        candidates = row.get("candidates") or []
        if not candidates:
            continue

        for candidate_index, candidate in enumerate(candidates, 1):
            qa_id = f"{evidence.get('chunk_id', 'unknown')}_q{candidate_index}"
            out.append(
                {
                    "qa_id": qa_id,
                    "query": candidate["query"],
                    "gold_answer": candidate["gold_answer"],
                    "question_type": candidate["question_type"],
                    "difficulty": candidate["difficulty"],
                    "keywords": candidate["keywords"],
                    "answer_points": candidate["answer_points"],
                    "evidence": candidate["evidence"],
                    "doc_id": evidence.get("doc_id"),
                    "chunk_id": evidence.get("chunk_id"),
                    "page": evidence.get("page"),
                    "source_doc": evidence.get("source_doc"),
                    "source_text": evidence.get("text"),
                    "source_candidate_index": candidate_index,
                    "source_row_model": row.get("model"),
                    "source_row_base_url": row.get("base_url"),
                    "sample_index": qa_index,
                }
            )
            qa_index += 1

    return out


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dumps_compact(row))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten candidate JSONL into a final QA evaluation dataset.")
    parser.add_argument("--input", required=True, help="Input candidates JSONL path.")
    parser.add_argument("--output", required=True, help="Output final eval dataset JSONL path.")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    flattened = flatten_candidates(rows)
    write_jsonl(args.output, flattened)
    print(f"Wrote {len(flattened)} QA samples to {args.output}")


if __name__ == "__main__":
    main()
