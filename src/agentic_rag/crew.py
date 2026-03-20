from __future__ import annotations

import os
from pathlib import Path

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from agentic_rag.tools.custom_tool import DocumentSearchTool


def _default_knowledge_file() -> str:
    """Resolve the default demo PDF relative to the repository."""
    env_path = os.getenv("AGENTIC_RAG_PDF_PATH")
    if env_path:
        return str(Path(env_path).expanduser().resolve())

    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / "knowledge" / "dspy.pdf").resolve())


def _build_web_search_tool() -> SerperDevTool | None:
    """Create the optional web fallback only when credentials are available."""
    if not os.getenv("SERPER_API_KEY"):
        return None
    try:
        return SerperDevTool()
    except Exception:
        return None


@CrewBase
class AgenticRag:
    """CrewAI workflow for enterprise knowledge base QA."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        knowledge_file: str | None = None,
        *,
        enable_web_search: bool = True,
        verbose: bool = True,
    ) -> None:
        self.knowledge_file = str(
            Path(knowledge_file or _default_knowledge_file()).expanduser().resolve()
        )
        self.enable_web_search = enable_web_search
        self.verbose = verbose

    @agent
    def retriever_agent(self) -> Agent:
        knowledge_tool = DocumentSearchTool(file_path=self.knowledge_file)
        web_search_tool = _build_web_search_tool() if self.enable_web_search else None
        tools = [tool for tool in [knowledge_tool, web_search_tool] if tool is not None]

        return Agent(
            config=self.agents_config["retriever_agent"],
            verbose=self.verbose,
            tools=tools,
        )

    @agent
    def response_synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["response_synthesizer_agent"],
            verbose=self.verbose,
        )

    @task
    def retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieval_task"],
        )

    @task
    def response_task(self) -> Task:
        return Task(
            config=self.tasks_config["response_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Create the sequential retrieval + synthesis workflow."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
        )
