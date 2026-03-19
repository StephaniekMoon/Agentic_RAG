import argparse
import json
import os
import ssl
from typing import Any, Dict, Iterable, List, Optional

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


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj

    raise ValueError("Judge output is not valid JSON")


def judge_answer_semantics(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    gold_answer: str,
    answer_points: List[str],
    pred_answer: str,
    ca_bundle: Optional[str] = None,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    import requests

    session = requests.Session()
    session.trust_env = False

    prompt = (
        "You are grading a RAG answer for semantic correctness.\n"
        "Use the question, gold answer, and answer points as the reference.\n"
        "The predicted answer can be longer or phrased differently and still be correct.\n"
        "Mark semantic_correct as true if the predicted answer conveys the same essential meaning as the gold answer.\n"
        "Use answer_points to judge whether the key facts are covered.\n"
        "Return strict JSON only with this schema:\n"
        '{'
        '"semantic_correct": true or false, '
        '"score": 0.0 to 1.0, '
        '"covered_points": ["..."], '
        '"missing_points": ["..."], '
        '"reason": "short explanation"'
        '}\n\n'
        f"Question: {question}\n"
        f"Gold answer: {gold_answer}\n"
        f"Answer points: {json.dumps(answer_points, ensure_ascii=False)}\n"
        f"Predicted answer: {pred_answer}\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict semantic answer judge. Output JSON only."},
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
    content = data["choices"][0]["message"]["content"]
    parsed = parse_json_response(content)

    semantic_correct = bool(parsed.get("semantic_correct"))
    score = parsed.get("score", 0.0)
    if not isinstance(score, (int, float)):
        score = 0.0

    covered_points = parsed.get("covered_points") or []
    missing_points = parsed.get("missing_points") or []
    if not isinstance(covered_points, list):
        covered_points = []
    if not isinstance(missing_points, list):
        missing_points = []

    return {
        "semantic_correct": semantic_correct,
        "semantic_score": max(0.0, min(float(score), 1.0)),
        "semantic_covered_points": [str(x) for x in covered_points],
        "semantic_missing_points": [str(x) for x in missing_points],
        "semantic_reason": str(parsed.get("reason", "")),
    }


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    judged_rows = [row for row in rows if row.get("pred_answer") is not None]
    semantic_correct_count = sum(1 for row in judged_rows if row.get("semantic_correct"))
    return {
        "num_samples": len(rows),
        "num_judged_samples": len(judged_rows),
        "semantic_correct_count": semantic_correct_count,
        "semantic_correct_rate": semantic_correct_count / len(judged_rows) if judged_rows else 0.0,
        "avg_semantic_score": (
            sum(float(row.get("semantic_score", 0.0)) for row in judged_rows) / len(judged_rows) if judged_rows else 0.0
        ),
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Judge semantic correctness for generated answers.")
    parser.add_argument("--input", required=True, help="Input answer-results JSONL path.")
    parser.add_argument("--output", required=True, help="Output judged-results JSONL path.")
    parser.add_argument("--summary-output", required=True, help="Output summary JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows (0 = all).")
    parser.add_argument("--model", default=os.getenv("MODEL") or "qwen3.5-flash", help="Judge model name.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        help="API key for semantic judging.",
    )
    parser.add_argument("--ca-bundle", default=None, help="Optional CA bundle PEM path.")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key: set OPENAI_API_KEY or DASHSCOPE_API_KEY, or pass --api-key.")

    ca_bundle = resolve_ca_bundle(args.ca_bundle)
    if args.ca_bundle and not ca_bundle:
        raise SystemExit(f"Invalid --ca-bundle path: {args.ca_bundle}")

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    judged_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        pred_answer = row.get("pred_answer")
        if pred_answer is None:
            row.update(
                {
                    "semantic_correct": None,
                    "semantic_score": None,
                    "semantic_covered_points": [],
                    "semantic_missing_points": [],
                    "semantic_reason": "No predicted answer available.",
                }
            )
        else:
            row.update(
                judge_answer_semantics(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    question=row["query"],
                    gold_answer=row["gold_answer"],
                    answer_points=row.get("answer_points", []),
                    pred_answer=pred_answer,
                    ca_bundle=ca_bundle,
                )
            )

        judged_rows.append(row)

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(rows)}")

    write_jsonl(args.output, judged_rows)
    write_json(args.summary_output, build_summary(judged_rows))
    print(f"Wrote {len(judged_rows)} rows to {args.output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
