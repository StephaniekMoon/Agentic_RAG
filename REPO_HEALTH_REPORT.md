# Repository Health Report And Study Notes

## 1. What This Repo Is Doing

This repository currently has two main usage paths:

1. A Streamlit app entrypoint in `app.py` for multi-PDF question answering.
2. A packaged CrewAI path under `src/agentic_rag/` intended to support CLI-based execution.

The retrieval core is `src/agentic_rag/tools/custom_tool.py`, which:

- extracts text from PDFs with `MarkItDown`
- chunks text semantically with `chonkie`
- stores searchable chunks in an in-memory `QdrantClient`
- returns source-tagged excerpts for downstream answer generation

## 2. Problems Found

### Problem A: `crew.py` used a hard-coded absolute PDF path

Original issue:

- The file referenced a PDF on another developer's machine.
- That made the packaged Crew path non-portable.

Why it breaks:

- Absolute local paths only work on the original machine.
- As soon as the repository moves to another computer, `FileNotFoundError` is likely.

Fix:

- Replaced the hard-coded path with a repository-relative default.
- Added support for overriding it with `AGENTIC_RAG_PDF_PATH`.

Files touched:

- `src/agentic_rag/crew.py`

### Problem B: `DocumentSearchTool` was initialized with the wrong parameter name

Original issue:

- Old code used `DocumentSearchTool(pdf=...)`.
- The current tool constructor accepts `file_path` or `file_paths`.

Why it breaks:

- This is an interface mismatch between the caller and the callee.
- Python raises an unexpected keyword argument error when parameter names do not match.

Fix:

- Updated the packaged Crew path to use `DocumentSearchTool(file_path=...)`.

Files touched:

- `src/agentic_rag/crew.py`

### Problem C: web search tool initialization happened too early

Original issue:

- `SerperDevTool()` was created at import time.

Why it breaks:

- Import-time side effects are risky.
- If environment variables are missing or tool setup fails, merely importing the module can fail before any user action happens.

Fix:

- Moved web tool creation into a helper that only runs when the Crew instance is built.
- Added a guard so web fallback is skipped when `SERPER_API_KEY` is missing.

Files touched:

- `src/agentic_rag/crew.py`

### Problem D: packaged CLI entrypoint was not truly usable

Original issue:

- `src/agentic_rag/main.py` always asked a hard-coded demo question.
- It had no CLI arguments for choosing the query or the PDF path.

Why it matters:

- A package script is most useful when it can be configured from the command line.
- Hard-coded defaults are fine for demos, but not for real use or testing.

Fix:

- Added `--query`, `--pdf`, `--disable-web-search`, and `--quiet`.
- Printed the final Crew result so the CLI is actually useful.

Files touched:

- `src/agentic_rag/main.py`

### Problem E: packaging metadata did not reflect actual runtime dependencies

Original issue:

- `pyproject.toml` only declared `crewai[tools]`.
- Runtime code also imports `streamlit`, `markitdown`, `chonkie`, `qdrant-client`, `fastembed`, and `python-dotenv`.

Why it breaks:

- A package installer trusts `pyproject.toml`.
- If runtime imports are missing from the dependency list, install success does not guarantee runtime success.

Fix:

- Added the missing runtime dependencies.
- Added Hatch build target configuration so the wheel points to `src/agentic_rag`.

Files touched:

- `pyproject.toml`

### Problem F: the root `test.py` script was stale

Original issue:

- It imported `FireCrawlWebSearchTool`, which no longer exists in the current codebase.

Why it breaks:

- A smoke test must reflect the current implementation.
- Otherwise the test becomes misleading noise rather than a safety net.

Fix:

- Replaced it with a `DocumentSearchTool` smoke test that indexes a PDF and runs a sample query.

Files touched:

- `test.py`

### Problem G: Qdrant code used deprecated APIs

Original issue:

- `custom_tool.py` used `client.add(...)` and `client.query(...)`.
- These methods now emit deprecation warnings in the installed Qdrant version.

Why it matters:

- Deprecation warnings are early signals that a future upgrade will break your code.
- Fixing them early is easier than waiting until the old API is removed.

Fix:

- Switched collection writes to:
  - `create_collection(...)`
  - `upload_points(...)`
- Switched reads to:
  - `query_points(...)`
- Stored chunk text inside payload so the newer query response still has the excerpt text needed for answer synthesis.

Files touched:

- `src/agentic_rag/tools/custom_tool.py`

## 3. Why The Qdrant Migration Needed Extra Care

The tricky part of the Qdrant upgrade was not the method rename itself.

The real issue was response shape:

- old `query(...)` returned objects with a document-like field that directly exposed the chunk text
- new `query_points(...)` returns scored points, where the useful fields usually come from payload

That means a naive migration like this is incomplete:

1. replace `query(...)` with `query_points(...)`
2. keep the rest of the formatter unchanged

If you do only that, the system may still retrieve the correct points but lose the displayed chunk text.

So the correct migration was:

1. create the collection explicitly with fastembed-aware vector config
2. upload each chunk as a point
3. store `source`, `chunk_id`, and `text` in payload
4. query with `models.Document(...)`
5. read the final text from `point.payload["text"]`

This is a good general lesson:

- when replacing an API, check not only the input method name
- also check the output object shape, because formatting logic often breaks there

## 4. Repair Process Used In This Session

### Step 1: inspect the real active path

I first checked:

- `app.py`
- `src/agentic_rag/crew.py`
- `src/agentic_rag/main.py`
- `src/agentic_rag/tools/custom_tool.py`
- `pyproject.toml`
- `test.py`

Goal:

- identify which code path was actually current
- separate active code from stale scaffold code

### Step 2: fix the packaged Crew path

I updated the packaged implementation so it:

- no longer depends on another machine's path
- no longer calls `DocumentSearchTool` with the wrong parameter
- does not initialize the web tool before it is needed

### Step 3: fix CLI usability and packaging metadata

I updated:

- `src/agentic_rag/main.py`
- `pyproject.toml`
- `README.md`

Goal:

- make the CLI testable
- make package installation better match runtime behavior
- reduce onboarding confusion

### Step 4: replace stale smoke test

I rewrote `test.py` to validate the current retrieval implementation instead of a removed web tool.

### Step 5: migrate Qdrant calls to the newer API

I updated `custom_tool.py` to:

- create the collection explicitly
- upload `PointStruct` objects with `models.Document`
- query through `query_points`
- reconstruct excerpts from payload

### Step 6: verify behavior

I ran:

- `python3 -m compileall src/agentic_rag test.py`
- `./agentic_rag/bin/python test.py --top-k 1 --query "What is DSPy?"`
- `./agentic_rag/bin/python -W error::UserWarning -c "...DocumentSearchTool..."`

Observed result:

- the smoke test succeeded
- retrieval returned a real excerpt from `knowledge/dspy.pdf`
- the previous Qdrant deprecation warnings no longer appeared

## 5. What You Can Learn From These Failures

### Lesson 1: avoid import-time side effects

Bad pattern:

- creating networked tools or file-dependent objects at module import time

Better pattern:

- create them inside functions, factories, or instance constructors

Reason:

- imports should be lightweight and predictable

### Lesson 2: treat `pyproject.toml` as executable truth

If code imports a package at runtime, the dependency metadata should usually declare it.

Otherwise you get:

- "works on my machine"
- broken clean installs
- confusing CI failures

### Lesson 3: stale tests are worse than missing tests

A stale test gives false confidence and wastes debugging time.

A good smoke test should:

- exercise the current code path
- run fast
- fail for real regressions, not historical leftovers

### Lesson 4: when upgrading APIs, compare both sides

Always compare:

1. how data goes in
2. how results come out

Many migrations fail because people only update the call site, not the result parsing.

### Lesson 5: repository-relative defaults are usually safer than absolute paths

Good default:

- derive sample assets relative to the current file or repository root

Better still:

- allow an environment variable or CLI flag to override the default

That gives you both:

- portability
- flexibility

## 6. Remaining Risks And Follow-Up Ideas

These are not blocking for the fixes above, but they are worth noting:

### Remaining risk 1: checked-in virtualenv directories add noise

The repository contains a checked-in `agentic_rag/` virtual environment and a backup directory.

Why it matters:

- code search becomes noisy
- packaging can get confused
- repository size grows unnecessarily

### Remaining risk 2: build validation was environment-limited

I attempted a wheel build, but the current environment did not have `hatchling` available for the build command I tested.

Meaning:

- source code and runtime verification passed
- full wheel-build verification was not completed inside this session

### Remaining risk 3: some CLI scaffold commands still look template-like

The `train`, `replay`, and `test` functions in `src/agentic_rag/main.py` still follow the original CrewAI scaffold style.

They are not the main broken path we repaired, but they may deserve another cleanup pass later.

## 7. Recommended Personal Review Order

If you want to study this repo efficiently, review files in this order:

1. `src/agentic_rag/tools/custom_tool.py`
2. `app.py`
3. `src/agentic_rag/crew.py`
4. `src/agentic_rag/main.py`
5. `pyproject.toml`

Why this order:

- first understand retrieval
- then understand how the app uses retrieval
- then understand how the packaged Crew path is wired
- then understand packaging and CLI behavior

## 8. Practical Commands To Remember

Smoke test:

```bash
./agentic_rag/bin/python test.py --top-k 1 --query "What is DSPy?"
```

CLI help:

```bash
PYTHONPATH=src ./agentic_rag/bin/python -c "import sys; from agentic_rag.main import run; sys.argv=['agentic_rag','--help']; run()"
```

Main app:

```bash
streamlit run app.py
```
