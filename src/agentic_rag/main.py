#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import warnings

from agentic_rag.crew import AgenticRag

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# CLI：运行、训练、回放和测试功能
def _run_parser() -> argparse.ArgumentParser:
    # CLI解析器设置
    parser = argparse.ArgumentParser(
        description="Run the Agentic RAG crew against a PDF knowledge base."
    )
    parser.add_argument(
        "--query",
        default="What is DSPy?",
        help="Question to ask the knowledge base.",
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_path",
        default=None,
        help=(
            "Path to the PDF used by DocumentSearchTool. "
            "Defaults to AGENTIC_RAG_PDF_PATH or knowledge/dspy.pdf."
        ),
    )
    parser.add_argument(
        "--disable-web-search",
        action="store_true",
        help="Disable the optional Serper web fallback even if SERPER_API_KEY is set.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce CrewAI verbosity.",
    )
    return parser


def run() -> None:
    """Run the crew from the packaged CLI."""
    args = _run_parser().parse_args(sys.argv[1:])
    inputs = {"query": args.query}

    result = AgenticRag(
        knowledge_file=args.pdf_path,
        enable_web_search=not args.disable_web_search,
        verbose=not args.quiet,
    ).crew().kickoff(inputs=inputs)

    print(result.raw)


def train() -> None:
    """Train the crew for a given number of iterations."""
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        AgenticRag().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay() -> None:
    """Replay the crew execution from a specific task."""
    try:
        AgenticRag().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test() -> None:
    """Test the crew execution and return the results."""
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        AgenticRag().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
