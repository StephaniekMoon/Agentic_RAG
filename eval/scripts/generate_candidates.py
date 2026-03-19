import argparse
import json
import os
import ssl
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

_REQUESTS_SESSION = requests.Session()
_REQUESTS_SESSION.trust_env = False


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


PROMPT_TEMPLATE = """You are an offline evaluation dataset annotator for RAG systems.
Given ONE evidence chunk, generate 3 candidate QA samples. Each sample MUST be directly answerable using ONLY the evidence text.

Hard rules (must follow):
1) Use ONLY the evidence text. Do NOT add any external knowledge, assumptions, or inferred facts.
2) The question must be answerable directly from the evidence. The gold_answer must be short, precise, and fully supported.
3) Avoid subjective/opinion questions. Avoid overly open-ended questions.
4) Output MUST be strict JSON, and nothing else.

Output format (a JSON array of length 3). Each element must follow:
{{
  "query": "...",
  "gold_answer": "...",
  "question_type": "definition|factoid|comparison|application|summary",
  "difficulty": "easy|medium|hard",
  "keywords": ["..."],
  "answer_points": ["..."],
  "evidence": [
    {{
      "doc_id": "{doc_id}",
      "page": {page_json},
      "chunk_id": "{chunk_id}",
      "text": "{evidence_text_json}",
      "must_hit": true
    }}
  ]
}}

Evidence (do NOT change the evidence text; copy it verbatim into evidence[0].text):
doc_id: {doc_id}
page: {page_str}
chunk_id: {chunk_id}
evidence_text:
{evidence_text}
"""


@dataclass(frozen=True)
class EvidenceRow:
    doc_id: str
    chunk_id: str
    text: str
    page: Optional[int] = None
    source_doc: Optional[str] = None


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_evidence_jsonl(path: str) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e

            text = obj.get("text") or obj.get("evidence_text")
            if not text or not isinstance(text, str):
                raise ValueError(f"Missing 'text' in evidence row at {path}:{line_no}")

            doc_id = obj.get("doc_id")
            if not doc_id or not isinstance(doc_id, str):
                raise ValueError(f"Missing 'doc_id' in evidence row at {path}:{line_no}")

            chunk_id = obj.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str):
                raise ValueError(f"Missing 'chunk_id' in evidence row at {path}:{line_no}")

            page = obj.get("page")
            if page is not None and not isinstance(page, int):
                # Keep it strict; page may be null/None
                raise ValueError(f"Field 'page' must be int or null at {path}:{line_no}")

            source_doc = obj.get("source_doc")
            if source_doc is not None and not isinstance(source_doc, str):
                raise ValueError(f"Field 'source_doc' must be str or null at {path}:{line_no}")

            rows.append(EvidenceRow(doc_id=doc_id, chunk_id=chunk_id, text=text, page=page, source_doc=source_doc))
    return rows


def build_prompt(row: EvidenceRow) -> str:
    page_json = "null" if row.page is None else str(row.page)
    page_str = "null" if row.page is None else str(row.page)
    return PROMPT_TEMPLATE.format(
        doc_id=row.doc_id,
        chunk_id=row.chunk_id,
        page_json=page_json,
        page_str=page_str,
        evidence_text=row.text,
        evidence_text_json=_json_dumps_compact(row.text),
    )


def extract_json_from_text(text: str) -> Any:
    """
    We ask for strict JSON, but models sometimes wrap it.
    Try to parse whole text first, then fall back to the first JSON array/object block.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first '[' ... matching last ']' (best-effort)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    # Find first '{' ... matching last '}' (best-effort)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("Model output is not valid JSON")


def validate_candidates(value: Any, expected_doc_id: str, expected_chunk_id: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Expected a JSON array of length 3")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(value, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate #{i} must be an object")

        for key in ["query", "gold_answer", "question_type", "difficulty", "keywords", "answer_points", "evidence"]:
            if key not in item:
                raise ValueError(f"Candidate #{i} missing field '{key}'")

        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValueError(f"Candidate #{i} field 'query' must be non-empty string")
        if not isinstance(item["gold_answer"], str) or not item["gold_answer"].strip():
            raise ValueError(f"Candidate #{i} field 'gold_answer' must be non-empty string")

        if item["question_type"] not in {"definition", "factoid", "comparison", "application", "summary"}:
            raise ValueError(f"Candidate #{i} invalid 'question_type': {item['question_type']}")
        if item["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError(f"Candidate #{i} invalid 'difficulty': {item['difficulty']}")

        if not isinstance(item["keywords"], list) or not all(isinstance(x, str) for x in item["keywords"]):
            raise ValueError(f"Candidate #{i} field 'keywords' must be list[str]")
        if not isinstance(item["answer_points"], list) or not all(isinstance(x, str) for x in item["answer_points"]):
            raise ValueError(f"Candidate #{i} field 'answer_points' must be list[str]")

        evidence = item["evidence"]
        if not isinstance(evidence, list) or len(evidence) < 1 or not isinstance(evidence[0], dict):
            raise ValueError(f"Candidate #{i} field 'evidence' must be a non-empty list[object]")

        ev0 = evidence[0]
        if ev0.get("doc_id") != expected_doc_id:
            raise ValueError(f"Candidate #{i} evidence[0].doc_id mismatch")
        if ev0.get("chunk_id") != expected_chunk_id:
            raise ValueError(f"Candidate #{i} evidence[0].chunk_id mismatch")

        # We don't enforce page/text equality strictly here, but keep must_hit type check.
        if "must_hit" in ev0 and not isinstance(ev0["must_hit"], bool):
            raise ValueError(f"Candidate #{i} evidence[0].must_hit must be boolean")

        out.append(item)

    return out


def openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    ca_bundle: Optional[str] = None,
    timeout_s: int = 120,
) -> str:
    """
    Calls OpenAI-compatible /chat/completions endpoint.
    Works for providers like DashScope compatible-mode, OpenAI, etc.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        # DashScope OpenAI-compatible HTTP requests accept enable_thinking as a top-level field.
        "enable_thinking": False,
    }
    verify: bool | str = ca_bundle if ca_bundle else True
    resp = _REQUESTS_SESSION.post(url, headers=headers, json=payload, timeout=timeout_s, verify=verify)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


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


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dumps_compact(row))
            f.write("\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate 3 candidate QA samples per evidence chunk.")
    parser.add_argument("--input", required=True, help="Evidence table JSONL path (one chunk per line).")
    parser.add_argument("--output", required=True, help="Output candidates JSONL path.")
    parser.add_argument("--model", default=os.getenv("MODEL") or "qwen3.5-flash", help="Chat model name.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL (no trailing /chat/completions).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        help="API key. Defaults to OPENAI_API_KEY, then DASHSCOPE_API_KEY.",
    )
    parser.add_argument(
        "--ca-bundle",
        default=None,
        help="Optional CA bundle PEM path for HTTPS requests. Useful when certifi is broken in the current env.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows (0 = all).")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between requests to avoid rate limits.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per row on JSON validation failure.")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key: set OPENAI_API_KEY or DASHSCOPE_API_KEY, or pass --api-key.")

    ca_bundle = resolve_ca_bundle(args.ca_bundle)
    if args.ca_bundle and not ca_bundle:
        raise SystemExit(f"Invalid --ca-bundle path: {args.ca_bundle}")
    if ca_bundle:
        print(f"Using CA bundle: {ca_bundle}")
    else:
        print("Using default TLS certificate discovery from requests/OS.")

    evidence_rows = load_evidence_jsonl(args.input)
    if args.limit and args.limit > 0:
        evidence_rows = evidence_rows[: args.limit]

    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(evidence_rows, 1):
        prompt = build_prompt(row)
        last_error: Optional[str] = None

        for attempt in range(args.max_retries + 1):
            try:
                content = openai_compatible_chat(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=prompt,
                    ca_bundle=ca_bundle,
                )
                parsed = extract_json_from_text(content)
                candidates = validate_candidates(parsed, expected_doc_id=row.doc_id, expected_chunk_id=row.chunk_id)

                results.append(
                    {
                        "evidence": {
                            "doc_id": row.doc_id,
                            "source_doc": row.source_doc,
                            "page": row.page,
                            "chunk_id": row.chunk_id,
                            "text": row.text,
                        },
                        "candidates": candidates,
                        "model": args.model,
                        "base_url": args.base_url,
                    }
                )
                last_error = None
                break
            except Exception as e:
                last_error = str(e)
                if "TLS CA certificate bundle" in last_error and not ca_bundle:
                    last_error += (
                        " | Hint: pass --ca-bundle /path/to/cacert.pem or repair the current environment's certifi package."
                    )
                # Retry with a shorter "fix JSON" instruction.
                prompt = (
                    "Return STRICT JSON only. Fix any formatting errors. "
                    "Output must be a JSON array of length 3. No extra text.\n\n"
                    + build_prompt(row)
                )

        if last_error is not None:
            results.append(
                {
                    "evidence": {
                        "doc_id": row.doc_id,
                        "source_doc": row.source_doc,
                        "page": row.page,
                        "chunk_id": row.chunk_id,
                        "text": row.text,
                    },
                    "error": last_error,
                    "model": args.model,
                    "base_url": args.base_url,
                }
            )

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

        if idx % 10 == 0:
            # Best-effort progress print for long runs.
            print(f"Processed {idx}/{len(evidence_rows)}")

    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
