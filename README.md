# Enterprise Knowledge Base Agentic RAG

This project packages the original document Q&A prototype as an enterprise knowledge base assistant for internal documents such as SOPs, policy manuals, onboarding handbooks, and product guides.

It focuses on two practical product requirements:

- answering questions over a multi-document knowledge base built from PDF and Word files
- evaluating system quality with an evidence-driven offline benchmark

## What It Does

- uploads and indexes one or more PDF or Word documents as a lightweight knowledge base
- runs OCR over embedded PDF images when OCR dependencies are available, so screenshot text and scanned inserts can also become retrievable
- uses a two-agent CrewAI workflow:
  - a retriever agent searches the indexed documents first and falls back to web search when needed
  - a response agent synthesizes the final answer
- returns answers with inline source references so users can trace claims back to document chunks
- supports offline dataset construction and multi-stage evaluation for retrieval, reader, and end-to-end agent quality

## Enterprise Use Case

The default business framing is an internal knowledge assistant for teams that work across fragmented internal documents. Typical examples include:

- HR or admin policy handbooks
- product requirement documents and user manuals
- operations SOPs
- technical onboarding documents

Instead of positioning the project as a generic PDF chatbot, the repository now reflects a more realistic enterprise workflow: knowledge ingestion, source-grounded QA, and measurable evaluation.

## Repository Highlights

- `app.py`: default Streamlit app with multi-document upload and source-aware answers
- `app_deep_seek.py`: DashScope/Qwen-backed variant
- `app_llama3.2.py`: local Ollama Llama 3.2 variant
- `src/agentic_rag/tools/custom_tool.py`: custom multi-document search tool with source tags
- `eval/README.md`: evaluation pipeline notes

## Setup

Install the core dependencies:

```bash
pip install -e .
```

To enable OCR for images embedded inside PDFs, also make sure the `tesseract` system binary is installed and available on `PATH`.

Example OCR setup on macOS:

```bash
brew install tesseract tesseract-lang
```

Optional environment variables:

- `SERPER_API_KEY` for web fallback
- `AGENTIC_RAG_PDF_PATH` to set the default PDF used by the packaged CLI
- `AGENTIC_RAG_ENABLE_PDF_IMAGE_OCR=0` to disable embedded PDF image OCR
- `AGENTIC_RAG_BM25_K1` to tune lexical BM25 retrieval, default `1.5`
- `AGENTIC_RAG_BM25_B` to tune lexical BM25 length normalization, default `0.75`
- `AGENTIC_RAG_CHUNK_TARGET_CHARS` to control PDF/document chunk size, default `900`
- `AGENTIC_RAG_CHUNK_MIN_CHARS` to control the minimum preferred chunk size before flushing, default `180`
- `AGENTIC_RAG_OCR_LANG` to override OCR languages, for example `chi_sim+eng`
- `AGENTIC_RAG_OCR_MIN_TEXT_LENGTH` to ignore very short OCR snippets
- `AGENTIC_RAG_OCR_MIN_IMAGE_DIMENSION` to skip tiny embedded images such as icons
- `OPENAI_API_KEY` or `DASHSCOPE_API_KEY` for the DashScope/Qwen app
- `OPENAI_BASE_URL` to override the OpenAI-compatible endpoint
- `OLLAMA_BASE_URL` to override the local Ollama endpoint

## Run The App

Default app:

```bash
streamlit run app.py
```

Packaged CLI smoke run:

```bash
agentic_rag --pdf knowledge/dspy.pdf --query "What is DSPy?"
```

DashScope/Qwen variant:

```bash
streamlit run app_deep_seek.py
```

Local Llama 3.2 variant:

```bash
streamlit run app_llama3.2.py
```

After startup:

1. Upload one or more PDF or Word documents in the sidebar.
2. Wait for indexing to finish.
3. Ask questions about the indexed knowledge base.
4. Inspect inline source references in the answer.

## Evaluation Workflow

The repository also includes an evidence-driven offline evaluation pipeline. At a high level:

1. chunk a source PDF into evidence rows
2. generate candidate QA pairs from evidence chunks
3. flatten the candidates into an evaluation dataset
4. run retrieval, reader, or end-to-end agent evaluation

See `eval/README.md` for the current scripts and data format.

## Suggested Resume Positioning

If you want to align this project with an enterprise knowledge assistant story, a concise version is:

> Built an Agentic RAG prototype for enterprise knowledge base QA over internal PDF and Word documents, and added an evidence-driven evaluation pipeline to diagnose retrieval, reader, and end-to-end agent performance.
