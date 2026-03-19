import argparse
import json
import math
import os
import re
import ssl
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(dotenv_path: str = ".env") -> None:
        if not os.path.isfile(dotenv_path):
            return
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


_RE_WORD = re.compile(r"[a-z0-9]+")


class LocalPdfRetriever:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.documents: List[Dict[str, Any]] = []
        self.doc_freq: Counter[str] = Counter()
        self.avg_doc_len = 0.0
        self._index_rows(rows)

    def _index_rows(self, rows: List[Dict[str, Any]]) -> None:
        seen_chunk_ids = set()
        total_doc_len = 0

        for row in rows:
            chunk_id = row.get("chunk_id")
            source_text = row.get("source_text")
            if not chunk_id or not source_text or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            tokens = tokenize(source_text)
            token_counts = Counter(tokens)
            total_doc_len += len(tokens)
            for token in token_counts:
                self.doc_freq[token] += 1
            self.documents.append(
                {
                    "doc_id": row.get("doc_id"),
                    "chunk_id": chunk_id,
                    "source_doc": row.get("source_doc"),
                    "text": source_text,
                    "tokens": tokens,
                    "token_counts": token_counts,
                    "doc_len": len(tokens),
                }
            )

        self.avg_doc_len = total_doc_len / len(self.documents) if self.documents else 0.0

    def search(self, query: str) -> str:
        scored = sorted(
            ((self._bm25_score(query, doc), doc["text"]) for doc in self.documents),
            key=lambda item: item[0],
            reverse=True,
        )
        docs = [text for score, text in scored[:5] if score > 0]
        return "\n___\n".join(docs)

    def _bm25_score(self, query: str, doc: Dict[str, Any], k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        query_terms = tokenize(query)
        if not query_terms:
            return score
        num_docs = max(1, len(self.documents))
        for term in query_terms:
            term_freq = doc["token_counts"].get(term, 0)
            if term_freq == 0:
                continue
            doc_freq = self.doc_freq.get(term, 0)
            idf = math.log(1 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            doc_len = max(1, doc["doc_len"])
            denom = term_freq + k1 * (1 - b + b * doc_len / max(1.0, self.avg_doc_len))
            score += idf * (term_freq * (k1 + 1)) / denom
        return score


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


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dumps_compact(row))
            f.write("\n")


def write_json(path: str, value: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_text(text: Optional[str]) -> str:
    s = (text or "").lower()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("-\n", "").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(text: Optional[str]) -> List[str]:
    return _RE_WORD.findall(normalize_text(text))


def token_recall(reference_text: Optional[str], candidate_text: Optional[str]) -> float:
    ref_tokens = tokenize(reference_text)
    if not ref_tokens:
        return 0.0
    candidate_token_set = set(tokenize(candidate_text))
    hits = sum(1 for token in ref_tokens if token in candidate_token_set)
    return hits / len(ref_tokens)


def phrase_in_text(phrase: Optional[str], text: Optional[str]) -> bool:
    phrase_norm = normalize_text(phrase)
    text_norm = normalize_text(text)
    return bool(phrase_norm) and phrase_norm in text_norm


def answer_point_coverage(answer: Optional[str], answer_points: List[str]) -> float:
    if not answer_points:
        return 0.0
    covered = sum(1 for point in answer_points if phrase_in_text(point, answer))
    return covered / len(answer_points)


def answer_exact_match(pred_answer: Optional[str], gold_answer: Optional[str]) -> bool:
    pred_norm = normalize_text(pred_answer)
    gold_norm = normalize_text(gold_answer)
    if not pred_norm or not gold_norm:
        return False
    return pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm


def resolve_ca_bundle(explicit_path: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []

    for value in [
        explicit_path,
        os.getenv("SSL_CERT_FILE"),
        os.getenv("REQUESTS_CA_BUNDLE"),
        os.getenv("CURL_CA_BUNDLE"),
    ]:
        if value:
            candidates.append(value)

    try:
        import certifi  # type: ignore

        candidates.append(certifi.where())
    except Exception:
        pass

    try:
        from pip._vendor import certifi as pip_certifi  # type: ignore

        candidates.append(pip_certifi.where())
    except Exception:
        pass

    default_paths = ssl.get_default_verify_paths()
    for value in [default_paths.openssl_cafile, default_paths.cafile]:
        if value:
            candidates.append(value)

    seen = set()
    for path in candidates:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isfile(normalized):
            return normalized

    return None


def openai_compatible_answer(
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    retrieved_context: str,
    ca_bundle: Optional[str] = None,
    timeout_s: int = 120,
) -> str:
    import requests

    session = requests.Session()
    session.trust_env = False

    prompt = (
        "You are evaluating a PDF-only RAG system.\n"
        "Answer the user question using only the retrieved context.\n"
        "If the retrieved context is insufficient, say exactly: I couldn't find the answer in the retrieved context.\n"
        "Keep the answer concise.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved context:\n{retrieved_context}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer briefly and use only the provided context."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "enable_thinking": False,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    verify: bool | str = ca_bundle if ca_bundle else True
    resp = session.post(
        url,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout_s,
        verify=verify,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    retrieval_hits = sum(1 for row in rows if row["retrieval_hit"])
    question_type = Counter(row["question_type"] for row in rows)
    difficulty = Counter(row["difficulty"] for row in rows)

    summary: Dict[str, Any] = {
        "num_samples": len(rows),
        "retrieval_hit_count": retrieval_hits,
        "retrieval_hit_rate": retrieval_hits / len(rows) if rows else 0.0,
        "avg_evidence_token_recall": sum(row["evidence_token_recall"] for row in rows) / len(rows) if rows else 0.0,
        "question_type": dict(question_type),
        "difficulty": dict(difficulty),
    }

    answer_rows = [row for row in rows if row.get("pred_answer") is not None]
    if answer_rows:
        answer_exact = sum(1 for row in answer_rows if row["answer_exact_match"])
        summary.update(
            {
                "num_answered_samples": len(answer_rows),
                "answer_exact_match_count": answer_exact,
                "answer_exact_match_rate": answer_exact / len(answer_rows) if answer_rows else 0.0,
                "avg_answer_point_coverage": (
                    sum(row["answer_point_coverage"] for row in answer_rows) / len(answer_rows) if answer_rows else 0.0
                ),
            }
        )

    return summary


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run offline evaluation on the flattened QA dataset.")
    parser.add_argument("--dataset", required=True, help="Final eval dataset JSONL path.")
    parser.add_argument("--results-output", required=True, help="Per-sample results JSONL path.")
    parser.add_argument("--summary-output", required=True, help="Summary JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N samples (0 = all).")
    parser.add_argument(
        "--retrieval-hit-threshold",
        type=float,
        default=0.6,
        help="Token recall threshold for marking retrieval_hit.",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Also generate an answer from the retrieved context using an OpenAI-compatible model.",
    )
    parser.add_argument("--model", default=os.getenv("MODEL") or "qwen3.5-flash", help="Answer model name.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        help="API key for answer generation.",
    )
    parser.add_argument("--ca-bundle", default=None, help="Optional CA bundle PEM path.")
    args = parser.parse_args()

    if args.generate_answers and not args.api_key:
        raise SystemExit("Missing API key: set OPENAI_API_KEY or DASHSCOPE_API_KEY, or pass --api-key.")

    rows = load_jsonl(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]

    ca_bundle = resolve_ca_bundle(args.ca_bundle)
    if args.ca_bundle and not ca_bundle:
        raise SystemExit(f"Invalid --ca-bundle path: {args.ca_bundle}")

    retriever = LocalPdfRetriever(rows=rows)

    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        query = row["query"]
        gold_evidence_text = row["evidence"][0]["text"] if row.get("evidence") else ""
        retrieved_context = retriever.search(query)
        evidence_recall = token_recall(gold_evidence_text, retrieved_context)
        retrieval_hit = evidence_recall >= args.retrieval_hit_threshold

        result: Dict[str, Any] = {
            "qa_id": row["qa_id"],
            "query": query,
            "gold_answer": row["gold_answer"],
            "question_type": row["question_type"],
            "difficulty": row["difficulty"],
            "doc_id": row.get("doc_id"),
            "chunk_id": row.get("chunk_id"),
            "retrieved_context": retrieved_context,
            "retrieval_hit": retrieval_hit,
            "evidence_token_recall": evidence_recall,
            "gold_evidence_text": gold_evidence_text,
            "answer_points": row.get("answer_points", []),
        }

        if args.generate_answers:
            pred_answer = openai_compatible_answer(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                query=query,
                retrieved_context=retrieved_context,
                ca_bundle=ca_bundle,
            )
            result["pred_answer"] = pred_answer
            result["answer_exact_match"] = answer_exact_match(pred_answer, row["gold_answer"])
            result["answer_point_coverage"] = answer_point_coverage(pred_answer, row.get("answer_points", []))
        else:
            result["pred_answer"] = None
            result["answer_exact_match"] = None
            result["answer_point_coverage"] = None

        results.append(result)

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)}")

    write_jsonl(args.results_output, results)
    write_json(args.summary_output, build_summary(results))
    print(f"Wrote {len(results)} rows to {args.results_output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
