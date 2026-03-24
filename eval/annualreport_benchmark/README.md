## Annual Reports Benchmark Draft

This folder contains a first-pass benchmark draft for the `annual reports` knowledge library.

Files:

- `annual_reports_benchmark_draft.jsonl`: 100 draft benchmark questions for manual review.
- `annual_reports_benchmark_v1.jsonl`: filtered benchmark v1 after high-risk draft removal.
- `removed_high_risk_draft_ids_v1.json`: draft ids removed from v1 with the current filtering pass.

Each JSONL row includes:

- `draft_id`: stable draft sample id
- `library`: target knowledge library name
- `query`: benchmark question draft
- `query_scope`: `single_doc` or `cross_doc`
- `target_docs`: intended document(s) to answer from
- `question_type`: one of `factoid|comparison|summary|application|definition`
- `difficulty`: one of `easy|medium|hard`
- `keywords`: intended retrieval keywords
- `expected_answer_shape`: suggested answer style
- `needs_manual_validation`: always `true` for this draft set
- `notes`: short review guidance

Recommended manual review workflow:

1. Drop questions that are too vague or not directly supported by the source document.
2. Rewrite questions that rely on external knowledge instead of document evidence.
3. For approved questions, add:
   - `gold_answer`
   - `answer_points`
   - `evidence`
   - `gold_source_doc`
   - `gold_chunk_id` or equivalent evidence ids
4. Separate:
   - single-document answerable questions
   - multi-document comparison questions
5. Prefer questions with explicit sourceable evidence over highly subjective prompts.

Suggested first filtering priority:

- keep numeric / segment / strategy / reopening / digital-channel questions
- de-prioritize overly open-ended cross-document comparison prompts

Current v1 filtering rule of thumb:

- keep single-document questions with relatively explicit evidence boundaries
- remove broad synthesis questions that likely need many sections/pages
- remove high-ambiguity cross-document comparison questions for now
