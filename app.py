import gc
import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

from src.agentic_rag.tools.custom_tool import DocumentSearchTool, SUPPORTED_DOCUMENT_EXTENSIONS

SUPPORTED_UPLOAD_TYPES = [extension.removeprefix(".") for extension in SUPPORTED_DOCUMENT_EXTENSIONS]
KNOWLEDGE_ROOT = Path(__file__).resolve().parent / "knowledge"


def build_web_search_tool() -> Optional[SerperDevTool]:
    """Create the optional web fallback only when credentials are available."""
    if not os.getenv("SERPER_API_KEY"):
        return None
    try:
        return SerperDevTool()
    except Exception:
        return None


def answer_query_with_guardrails(knowledge_base_tool: DocumentSearchTool, query: str, crew: Crew | None) -> tuple[str, Crew | None]:
    """Route the query through retrieval confidence, fallback, and answer validation."""
    bundle = knowledge_base_tool.prepare_answer_bundle(query=query, limit=5)
    confidence = bundle["confidence"]

    rule_based_answer = knowledge_base_tool.extract_rule_based_answer(bundle)
    if rule_based_answer:
        return rule_based_answer, crew

    if confidence["level"] == "low":
        return knowledge_base_tool.build_low_confidence_fallback(bundle), crew

    if crew is None:
        crew = create_agents_and_tasks(knowledge_base_tool)

    llm_answer = crew.kickoff(inputs={"query": query}).raw
    validation = knowledge_base_tool.validate_generated_answer(llm_answer, bundle)
    if not validation["is_valid"]:
        return knowledge_base_tool.build_validation_fallback(llm_answer, bundle, validation), crew
    return llm_answer, crew


def create_agents_and_tasks(knowledge_base_tool: DocumentSearchTool) -> Crew:
    """Create a two-agent workflow for enterprise knowledge base QA."""
    web_search_tool = build_web_search_tool()

    retriever_agent = Agent(
        role="Enterprise knowledge base retriever for the user query: {query}",
        goal=(
            "Retrieve the most relevant evidence for the user query from the indexed "
            "knowledge base first. Preserve all source tags such as [Source: ... | Ref: ...]. "
            "Only use web search if the knowledge base is insufficient."
        ),
        backstory=(
            "You help employees navigate internal documents such as SOPs, product manuals, "
            "and policy documents. You are careful about evidence quality and never drop source tags."
        ),
        verbose=True,
        tools=[tool for tool in [knowledge_base_tool, web_search_tool] if tool],
    )

    response_synthesizer_agent = Agent(
        role="Enterprise knowledge assistant for the user query: {query}",
        goal=(
            "Write a concise answer grounded in the retrieved evidence. Cite every material "
            "claim with the provided source tags like [filename.pdf | chunk_id]. If the retrieved "
            "evidence is insufficient, say so clearly instead of guessing."
        ),
        backstory=(
            "You answer employee questions using enterprise knowledge documents and must keep "
            "responses auditable by citing the exact supporting snippets."
        ),
        verbose=True,
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most relevant evidence from the indexed knowledge base for the user "
            "query: {query}. Return excerpts with their source tags preserved."
        ),
        expected_output=(
            "A set of relevant excerpts, each including its source label and chunk reference."
        ),
        agent=retriever_agent,
    )

    response_task = Task(
        description=(
            "Answer the user query: {query} using only the retrieved evidence. Include source "
            "citations in the answer."
        ),
        expected_output=(
            "A concise answer with inline source citations. If evidence is insufficient, say so."
        ),
        agent=response_synthesizer_agent,
    )

    return Crew(
        agents=[retriever_agent, response_synthesizer_agent],
        tasks=[retrieval_task, response_task],
        process=Process.sequential,
        verbose=True,
    )


def reset_chat() -> None:
    """Clear only the conversation history."""
    st.session_state.messages = []
    gc.collect()


def reset_knowledge_base() -> None:
    """Clear the indexed documents and reset the chat workflow state."""
    st.session_state.messages = []
    st.session_state.knowledge_base_tool = None
    st.session_state.crew = None
    st.session_state.knowledge_base_signature = None
    st.session_state.knowledge_sources = []
    st.session_state.active_library_name = None
    gc.collect()


def discover_knowledge_libraries() -> list[Path]:
    """Return named knowledge libraries from the knowledge root."""
    if not KNOWLEDGE_ROOT.is_dir():
        return []
    return sorted(path for path in KNOWLEDGE_ROOT.iterdir() if path.is_dir())


def list_library_documents(library_dir: Path) -> list[str]:
    """Return supported documents contained in a knowledge library folder."""
    supported_suffixes = {suffix.lower() for suffix in SUPPORTED_DOCUMENT_EXTENSIONS}
    file_paths = [
        str(path.resolve())
        for path in library_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    return sorted(file_paths)


def build_library_signature(library_name: str, file_paths: list[str]) -> tuple:
    """Track the selected library and file metadata to know when a refresh is required."""
    signature_items = []
    for file_path in file_paths:
        stat_result = os.stat(file_path)
        signature_items.append(
            (
                os.path.relpath(file_path, KNOWLEDGE_ROOT),
                stat_result.st_size,
                stat_result.st_mtime_ns,
            )
        )
    return (library_name, tuple(signature_items))


def load_selected_library(library_name: str, force_reload: bool = False) -> None:
    """Load one folder-backed knowledge library into the active session."""
    library_dir = KNOWLEDGE_ROOT / library_name
    file_paths = list_library_documents(library_dir)
    if not file_paths:
        reset_chat()
        st.session_state.knowledge_base_tool = None
        st.session_state.crew = None
        st.session_state.knowledge_base_signature = None
        st.session_state.knowledge_sources = []
        st.session_state.active_library_name = library_name
        return

    library_signature = build_library_signature(library_name=library_name, file_paths=file_paths)
    if not force_reload and library_signature == st.session_state.knowledge_base_signature:
        return

    with st.spinner(f"Loading knowledge library '{library_name}'... Please wait..."):
        st.session_state.knowledge_base_tool = DocumentSearchTool(file_paths=file_paths)

    st.session_state.knowledge_base_signature = library_signature
    st.session_state.knowledge_sources = st.session_state.knowledge_base_tool.describe_sources()
    st.session_state.active_library_name = library_name
    st.session_state.crew = None
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_base_tool" not in st.session_state:
    st.session_state.knowledge_base_tool = None

if "crew" not in st.session_state:
    st.session_state.crew = None

if "knowledge_base_signature" not in st.session_state:
    st.session_state.knowledge_base_signature = None

if "knowledge_sources" not in st.session_state:
    st.session_state.knowledge_sources = []

if "active_library_name" not in st.session_state:
    st.session_state.active_library_name = None


with st.sidebar:
    st.header("Enterprise Knowledge Base")
    libraries = discover_knowledge_libraries()

    if libraries:
        library_names = [path.name for path in libraries]
        active_library = st.session_state.active_library_name
        default_library_index = library_names.index(active_library) if active_library in library_names else 0
        selected_library = st.radio(
            "Select Library",
            options=library_names,
            index=default_library_index,
            help="Each subfolder under knowledge/ is treated as an independent long-term knowledge base.",
        )
        reload_library = st.button("Reload Selected Library")
        load_selected_library(selected_library, force_reload=reload_library)

        library_document_paths = list_library_documents(KNOWLEDGE_ROOT / selected_library)
        if library_document_paths:
            st.success(
                f"Loaded library '{selected_library}' with {len(library_document_paths)} documents."
            )
            st.caption("Current knowledge base:")
            for source_name in st.session_state.knowledge_sources:
                st.write(f"- {source_name}")
        else:
            st.warning(
                f"Library '{selected_library}' has no PDF or Word documents yet. "
                "Add files under knowledge/{selected_library} and click reload."
            )
    else:
        st.info("Create subfolders under knowledge/ to register independent knowledge libraries.")

    root_level_documents = [
        path.name
        for path in KNOWLEDGE_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in {suffix.lower() for suffix in SUPPORTED_DOCUMENT_EXTENSIONS}
    ] if KNOWLEDGE_ROOT.is_dir() else []
    if root_level_documents:
        st.caption(
            "Root-level files under knowledge/ are not part of any named library. "
            "Move them into a subfolder if you want them selectable in the UI."
        )

    st.button("Clear Chat", on_click=reset_chat)
    st.button("Reset Knowledge Base", on_click=reset_knowledge_base)


st.title("Enterprise Knowledge Base Agentic RAG")
st.caption(
    "Ask questions over multiple internal documents. Answers are grounded in retrieved evidence "
    "and include source references for auditability."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about the indexed knowledge base...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.knowledge_base_tool is None:
        warning_message = "Please upload at least one document before asking a question."
        with st.chat_message("assistant"):
            st.markdown(warning_message)
        st.session_state.messages.append({"role": "assistant", "content": warning_message})
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Searching the knowledge base..."):
                result, st.session_state.crew = answer_query_with_guardrails(
                    knowledge_base_tool=st.session_state.knowledge_base_tool,
                    query=prompt,
                    crew=st.session_state.crew,
                )

            lines = result.split("\n")
            for index, line in enumerate(lines):
                full_response += line
                if index < len(lines) - 1:
                    full_response += "\n"
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.08)

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": result})
