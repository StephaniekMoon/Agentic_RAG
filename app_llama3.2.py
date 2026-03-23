import gc
import os
import tempfile
import time
from typing import Optional

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool

from src.agentic_rag.tools.custom_tool import DocumentSearchTool, SUPPORTED_DOCUMENT_EXTENSIONS

SUPPORTED_UPLOAD_TYPES = [extension.removeprefix(".") for extension in SUPPORTED_DOCUMENT_EXTENSIONS]


@st.cache_resource
def load_llm() -> LLM:
    return LLM(
        model="ollama/llama3.2",
        base_url=os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434",
    )


def build_web_search_tool() -> Optional[SerperDevTool]:
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
    web_search_tool = build_web_search_tool()
    llm = load_llm()

    retriever_agent = Agent(
        role="Enterprise knowledge base retriever for the user query: {query}",
        goal=(
            "Retrieve the most relevant evidence from the indexed knowledge base first and "
            "preserve source tags such as [Source: ... | Ref: ...]. Use web search only as "
            "a fallback when the document knowledge base is insufficient."
        ),
        backstory=(
            "You help employees search internal SOPs, manuals, and policy documents while "
            "keeping all evidence traceable."
        ),
        verbose=True,
        tools=[tool for tool in [knowledge_base_tool, web_search_tool] if tool],
        llm=llm,
    )

    response_synthesizer_agent = Agent(
        role="Enterprise knowledge assistant for the user query: {query}",
        goal=(
            "Write a concise answer grounded in the retrieved evidence. Cite every material "
            "claim with the provided source tags and avoid unsupported statements."
        ),
        backstory=(
            "You turn retrieved enterprise knowledge into auditable answers that employees can trust."
        ),
        verbose=True,
        llm=llm,
    )

    retrieval_task = Task(
        description=(
            "Retrieve the most relevant evidence from the indexed knowledge base for the user "
            "query: {query}. Keep source tags intact."
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
        verbose=True,
    )


def reset_chat() -> None:
    st.session_state.messages = []
    gc.collect()


def reset_knowledge_base() -> None:
    st.session_state.messages = []
    st.session_state.knowledge_base_tool = None
    st.session_state.crew = None
    st.session_state.knowledge_base_signature = None
    st.session_state.knowledge_sources = []
    gc.collect()


def index_uploaded_documents(uploaded_files) -> None:
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
            st.session_state.knowledge_base_tool = DocumentSearchTool(file_paths=file_paths)

    st.session_state.knowledge_base_signature = file_signature
    st.session_state.knowledge_sources = st.session_state.knowledge_base_tool.describe_sources()
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


with st.sidebar:
    st.header("Enterprise Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF or Word documents",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
    )

    if uploaded_files:
        index_uploaded_documents(uploaded_files)
        st.success(f"Indexed {len(st.session_state.knowledge_sources)} documents.")
        for source_name in st.session_state.knowledge_sources:
            st.write(f"- {source_name}")
    else:
        st.info("Add internal PDFs or Word documents to build the knowledge base.")

    st.button("Clear Chat", on_click=reset_chat)
    st.button("Reset Knowledge Base", on_click=reset_knowledge_base)


st.title("Enterprise Knowledge Base Agentic RAG")
st.caption(
    "Local Llama-backed assistant for enterprise knowledge base QA over multiple documents "
    "with inline source references."
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
