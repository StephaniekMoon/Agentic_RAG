import argparse
import json
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool

from src.agentic_rag.tools.custom_tool import DocumentSearchTool

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


def normalize_text(text: Optional[str]) -> str:
    s = (text or "").lower()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("-\n", "").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()


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


def normalize_litellm_model(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return "openai/qwen3.5-flash"
    if "/" in model:
        return model
    # CrewAI uses LiteLLM under the hood and expects provider/model style names.
    return f"openai/{model}"


def model_requires_disable_thinking(model: str) -> bool:
    normalized = normalize_litellm_model(model).lower()
    return normalized.endswith("/qwen3-32b")


def setup_llm_env(base_url: str, api_key: Optional[str], model: str) -> str:
    # CrewAI defaults to OpenAI env names; map DashScope key/base URL into those names.
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        # Keep both names for compatibility across SDK wrappers.
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url
    # Keep model explicit for reproducibility in end-to-end eval runs.
    normalized_model = normalize_litellm_model(model)
    os.environ["MODEL"] = normalized_model
    os.environ["OPENAI_MODEL_NAME"] = normalized_model
    return normalized_model


def build_agent_llm(base_url: str, api_key: Optional[str], model: str) -> LLM:
    normalized_model = normalize_litellm_model(model)
    extra_params: Dict[str, Any] = {}
    if model_requires_disable_thinking(normalized_model):
        extra_params["enable_thinking"] = False
    return LLM(
        model=normalized_model,
        base_url=base_url,
        api_key=api_key,
        **extra_params,
    )


def build_web_tool(enable_web_search: bool) -> Optional[SerperDevTool]:
    if not enable_web_search:
        return None
    if not os.getenv("SERPER_API_KEY"):
        print("SERPER_API_KEY is missing; web search tool is disabled for this run.")
        return None
    try:
        return SerperDevTool()
    except Exception as e:
        print(f"Failed to initialize SerperDevTool ({e}); web search tool is disabled for this run.")
        return None


def create_end_to_end_crew(pdf_path: str, enable_web_search: bool, verbose: bool, llm: LLM) -> Crew:
    pdf_tool = DocumentSearchTool(file_path=pdf_path)
    web_search_tool = build_web_tool(enable_web_search)

    retriever_agent = Agent(
        role="Enterprise knowledge base retriever for the user query: {query}",
        goal=(
            "Retrieve the most relevant evidence from the indexed knowledge base first. "
            "Preserve all source tags such as [Source: ... | Ref: ...]. Only use web search "
            "if the PDF knowledge base is insufficient."
        ),
        backstory=(
            "You help employees search internal enterprise documents while keeping evidence traceable."
        ),
        verbose=verbose,
        tools=[tool for tool in [pdf_tool, web_search_tool] if tool],
        llm=llm,
    )

    response_synthesizer_agent = Agent(
        role="Enterprise knowledge assistant for the user query: {query}",
        goal=(
            "Answer the user query using only the retrieved evidence. Cite material claims "
            "with the provided source tags and avoid unsupported statements."
        ),
        backstory=(
            "You turn retrieved enterprise knowledge into concise and auditable answers."
        ),
        verbose=verbose,
        llm=llm,
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most relevant evidence from the indexed knowledge base for the "
            "user query: {query}. Return excerpts with source tags intact."
        ),
        expected_output="Relevant excerpts with source labels and chunk references.",
        agent=retriever_agent,
    )

    response_task = Task(
        description="Answer the user query: {query} using only the retrieved evidence and cite it.",
        expected_output="A concise answer with inline source citations.",
        agent=response_synthesizer_agent,
    )

    return Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=verbose,
    )


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    answered_rows = [row for row in rows if row.get("pred_answer") is not None]
    exact_correct = sum(1 for row in answered_rows if row.get("answer_exact_match"))

    question_type = Counter(row["question_type"] for row in rows)
    difficulty = Counter(row["difficulty"] for row in rows)

    return {
        "num_samples": len(rows),
        "num_answered_samples": len(answered_rows),
        "num_error_samples": sum(1 for row in rows if row.get("error")),
        "answer_exact_match_count": exact_correct,
        "answer_exact_match_rate": exact_correct / len(answered_rows) if answered_rows else 0.0,
        "avg_answer_point_coverage": (
            sum(float(row.get("answer_point_coverage", 0.0)) for row in answered_rows) / len(answered_rows)
            if answered_rows
            else 0.0
        ),
        "question_type": dict(question_type),
        "difficulty": dict(difficulty),
    }


def kickoff_with_retry(crew: Crew, query: str, max_retries: int) -> str:
    last_error = "Unknown Agent error"
    for attempt in range(max_retries + 1):
        try:
            result = crew.kickoff(inputs={"query": query}).raw
            if result is None or not str(result).strip():
                raise ValueError("Invalid response from LLM call - None or empty.")
            return str(result)
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last_error)


def kickoff_with_retry_and_timeout(crew: Crew, query: str, max_retries: int, timeout_s: int) -> str:
    if timeout_s <= 0:
        return kickoff_with_retry(crew=crew, query=query, max_retries=max_retries)

    def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
        raise TimeoutError(f"Agent kickoff timed out after {timeout_s}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        return kickoff_with_retry(crew=crew, query=query, max_retries=max_retries)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run end-to-end evaluation using the real Agent app workflow.")
    parser.add_argument("--dataset", required=True, help="Final eval dataset JSONL path.")
    parser.add_argument("--pdf", required=True, help="PDF knowledge file path for DocumentSearchTool.")
    parser.add_argument("--results-output", required=True, help="Per-sample results JSONL path.")
    parser.add_argument("--summary-output", required=True, help="Summary JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N samples (0 = all).")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL used by the underlying Agent LLM.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        help="API key for Agent LLM calls (OPENAI_API_KEY or DASHSCOPE_API_KEY).",
    )
    parser.add_argument("--model", default=os.getenv("MODEL") or "qwen3.5-flash", help="LLM model name.")
    parser.add_argument(
        "--disable-web-search",
        action="store_true",
        help="Disable Serper web fallback for a pure PDF end-to-end run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce Crew verbose logs during evaluation.",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per sample for transient Agent failures.")
    parser.add_argument(
        "--sample-timeout-s",
        type=int,
        default=180,
        help="Per-sample timeout in seconds. Use 0 to disable timeout.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key: set OPENAI_API_KEY or DASHSCOPE_API_KEY, or pass --api-key.")

    normalized_model = setup_llm_env(base_url=args.base_url, api_key=args.api_key, model=args.model)
    agent_llm = build_agent_llm(base_url=args.base_url, api_key=args.api_key, model=args.model)
    print(f"Using end-to-end model: {normalized_model}")

    rows = load_jsonl(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]

    crew = create_end_to_end_crew(
        pdf_path=args.pdf,
        enable_web_search=not args.disable_web_search,
        verbose=not args.quiet,
        llm=agent_llm,
    )

    results: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        query = row["query"]
        gold_answer = row["gold_answer"]
        answer_points = row.get("answer_points", [])

        result: Dict[str, Any] = {
            "qa_id": row["qa_id"],
            "query": query,
            "gold_answer": gold_answer,
            "answer_points": answer_points,
            "question_type": row["question_type"],
            "difficulty": row["difficulty"],
            "doc_id": row.get("doc_id"),
            "chunk_id": row.get("chunk_id"),
            "pred_answer": None,
            "answer_exact_match": None,
            "answer_point_coverage": None,
            "error": None,
        }

        try:
            pred_answer = kickoff_with_retry_and_timeout(
                crew=crew,
                query=query,
                max_retries=args.max_retries,
                timeout_s=args.sample_timeout_s,
            )
            result["pred_answer"] = pred_answer
            result["answer_exact_match"] = answer_exact_match(pred_answer, gold_answer)
            result["answer_point_coverage"] = answer_point_coverage(pred_answer, answer_points)
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(rows)}")

    write_jsonl(args.results_output, results)
    write_json(args.summary_output, build_summary(results))
    print(f"Wrote {len(results)} rows to {args.results_output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
