from __future__ import annotations

import argparse
from pathlib import Path

from src.agentic_rag.tools.custom_tool import DocumentSearchTool


def _default_pdf_path() -> Path:
    return Path(__file__).resolve().parent / "knowledge" / "dspy.pdf"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test for the current DocumentSearchTool implementation."
    )
    parser.add_argument(
        "--pdf",
        default=str(_default_pdf_path()),
        help="PDF path to index for the smoke test.",
    )
    parser.add_argument(
        "--query",
        default="What is DSPy?",
        help="Query to run against the indexed PDF.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of chunks to retrieve.",
    )
    return parser


def run_smoke_test(pdf_path: str, query: str, top_k: int) -> None:
    print("Initializing DocumentSearchTool...")
    tool = DocumentSearchTool(file_path=pdf_path, top_k=top_k)

    print("Indexed sources:")
    for source_name in tool.describe_sources():
        print(f"- {source_name}")

    print(f"\nQuery: {query}")
    result = tool._run(query)

    print("\nResult:")
    print("-" * 50)
    print(result)
    print("-" * 50)


if __name__ == "__main__":
    args = _parser().parse_args()
    run_smoke_test(pdf_path=args.pdf, query=args.query, top_k=args.top_k)
