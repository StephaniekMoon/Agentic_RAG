# Offline Evaluation

This folder contains scripts and datasets for evaluating the enterprise knowledge base version of the Agentic RAG workflow.

The current pipeline is designed for PDF knowledge documents and supports:

- building evidence tables from source PDFs
- generating candidate QA samples grounded in evidence chunks
- flattening the generated data into an evaluation dataset
- running retrieval / reader / end-to-end agent experiments

## Evidence Table (Input)

`eval/scripts/generate_candidates.py` expects an evidence table JSONL where each line is a single chunk:

Required fields:
- `doc_id` (string)
- `chunk_id` (string)
- `text` (string)

Optional fields:
- `page` (int or null)
- `source_doc` (string or null)

Example line:

```json
{"doc_id":"dspy","chunk_id":"dspy_c1","page":null,"source_doc":"dspy.pdf","text":"..."}
```

## Candidate Generation

Generate 3 candidate QA samples per chunk:

```bash
python3 eval/scripts/build_evidence_table.py \
  --pdf knowledge/dspy.pdf \
  --output eval/evidence/evidence_table.jsonl \
  --min-chars 200 \
  --min-alnum-ratio 0.35 \
  --max-single-char-line-ratio 0.45 \
  --max-newline-ratio 0.08

python3 eval/scripts/generate_candidates.py \
  --input eval/evidence/evidence_table.jsonl \
  --output eval/dataset/candidates.jsonl \
  --model qwen3.5-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
```

Environment variables:
- `OPENAI_API_KEY` (or `DASHSCOPE_API_KEY`)
- `OPENAI_BASE_URL` (optional)
- `MODEL` (optional)

The output `eval/dataset/candidates.jsonl` stores one line per evidence chunk with:
- the original evidence chunk
- `candidates` (length 3) on success, or `error` on failure
