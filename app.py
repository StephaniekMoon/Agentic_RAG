import gc
import os
import tempfile
import time
from typing import Optional

import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

from src.agentic_rag.tools.custom_tool import DocumentSearchTool


def build_web_search_tool() -> Optional[SerperDevTool]:
    """Create the optional web fallback only when credentials are available."""
    if not os.getenv("SERPER_API_KEY"):
        return None
    try:
        return SerperDevTool()
    except Exception:
        return None


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
            "and policy PDFs. You are careful about evidence quality and never drop source tags."
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
    """Clear the indexed PDFs and reset the chat workflow state."""
    st.session_state.messages = []
    st.session_state.pdf_tool = None
    st.session_state.crew = None
    st.session_state.knowledge_base_signature = None
    st.session_state.knowledge_sources = []
    gc.collect()


def index_uploaded_pdfs(uploaded_files) -> None:
    """Index uploaded PDFs once per unique upload set."""
    file_signature = tuple((uploaded_file.name, uploaded_file.size) for uploaded_file in uploaded_files)
    if file_signature == st.session_state.knowledge_base_signature:
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as handle:
                handle.write(uploaded_file.getvalue())
            file_paths.append(temp_file_path)

        with st.spinner("Indexing enterprise knowledge base... Please wait..."):
            st.session_state.pdf_tool = DocumentSearchTool(file_paths=file_paths)

    st.session_state.knowledge_base_signature = file_signature
    st.session_state.knowledge_sources = st.session_state.pdf_tool.describe_sources()
    st.session_state.crew = None
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None

if "crew" not in st.session_state:
    st.session_state.crew = None

if "knowledge_base_signature" not in st.session_state:
    st.session_state.knowledge_base_signature = None

if "knowledge_sources" not in st.session_state:
    st.session_state.knowledge_sources = []


with st.sidebar:
    st.header("Enterprise Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        index_uploaded_pdfs(uploaded_files)
        st.success(f"Indexed {len(st.session_state.knowledge_sources)} documents.")
        st.caption("Current knowledge base:")
        for source_name in st.session_state.knowledge_sources:
            st.write(f"- {source_name}")
    else:
        st.info("Add internal docs, SOPs, manuals, or policy PDFs to build the knowledge base.")

    st.button("Clear Chat", on_click=reset_chat)
    st.button("Reset Knowledge Base", on_click=reset_knowledge_base)


st.title("Enterprise Knowledge Base Agentic RAG")
st.caption(
    "Ask questions over multiple internal PDFs. Answers are grounded in retrieved evidence "
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

    if st.session_state.pdf_tool is None:
        warning_message = "Please upload at least one PDF before asking a question."
        with st.chat_message("assistant"):
            st.markdown(warning_message)
        st.session_state.messages.append({"role": "assistant", "content": warning_message})
    else:
        if st.session_state.crew is None:
            st.session_state.crew = create_agents_and_tasks(st.session_state.pdf_tool)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Searching the knowledge base..."):
                result = st.session_state.crew.kickoff(inputs={"query": prompt}).raw

            lines = result.split("\n")
            for index, line in enumerate(lines):
                full_response += line
                if index < len(lines) - 1:
                    full_response += "\n"
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.08)

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": result})
