# Offline Evaluation Set v1-final

This directory contains the frozen `v1-final` offline evaluation set for `research_paper_agent`.

Files:

- `eval_cases_v1.jsonl`: frozen v1-final benchmark data.
- `eval_cases_draft.jsonl`: synchronized editable mirror kept for traceability.
- `eval_schema.json`: schema for one evaluation case.
- `validate_eval_cases.py`: deterministic dataset, manifest, and invariant validator.
- `verify_pdf_anchors.py`: deterministic local PDF page-anchor verifier.
- `eval_manifest.json`: dataset and PDF hash manifest generated after validation.
- `final_audit_report.md`: final audit summary generated after validation.

## Freeze Rules

Each case must run with a new `thread_id`:

```json
"thread_mode": "fresh_per_case"
```

This prevents cases such as `eval-019` ("这两篇论文") from accidentally inheriting prior conversation state or checkpoint context.

Only `eval-020` resumes within the same case:

```json
"resume_within_case": true
```

For `eval-020`, start the case with a fresh thread, then resume `local_deep_search` after the first interrupt and `answer_with_gaps` after the second interrupt in that same thread. Do not create a new thread between resumes.

## Page Numbering

All pages are 1-based physical PDF pages:

- PDF viewer page 1 is page `1`.
- Do not use 0-based Chroma metadata directly.
- Do not infer from printed paper page numbers.
- If retrieval output only exposes a 0-based `metadata_page`, convert it first:

```python
display_page = metadata_page + 1
```

LoRA Table 12 is on physical PDF page 21. Page 20 introduces Table 12, parameter budgets, and typical rank configurations.

## Retrieval Metric

The primary retrieval metric for v1-final is `group_recall`.

Raw page recall may be reported as a diagnostic metric, but it is not the pass/fail metric for these 20 cases. `page_recall` remains in the schema only for future extensions.

`gold_page_groups` define evidence obligations. A group is hit if retrieval returns any one page in that group. For example:

```json
{
  "aspect": "模型结构",
  "pages": [3, 4, 5, 6]
}
```

Retrieving page 3, 4, 5, or 6 marks that group as hit.

Case group recall:

```text
hit aspect groups / all aspect groups
```

For multi-paper questions, calculate group recall separately for each paper, then macro-average across papers. Do not merge all pages from all papers before scoring.

`eval-019` has `retrieval_eval_mode = "none"` and should not trigger paper retrieval.

## Facts And Text Checks

`expected_facts` contains hard facts explicitly requested by the user question. These should be checked with `fact_match_rules`, deterministic extraction, human review, or an LLM judge. Do not check numeric facts by raw substring presence. For example, `attention_heads = 8` is not satisfied merely because the digit `8` appears somewhere; it must be semantically tied to attention heads or `h`.

`supporting_facts` contains optional, stable paper facts. Answers do not fail merely because they omit `supporting_facts`. If an answer uses a supporting fact, the fact must be correct and evidence-supported.

`must_include_any` is a list of concept groups. Each inner list contains synonyms or equivalent phrases; matching any phrase in a group marks that group as hit.

`must_include_policy` controls how many concept groups must be hit:

- `{"mode": "all"}` means all groups.
- `{"mode": "at_least", "min_groups": N}` means at least `N` groups.

Negative facts, citation support, and "the paper does not explicitly state X" judgments still require manual review or an LLM judge. String checks cannot replace semantic review for those cases.

## Aspect Normalization

Normalize aspect names before comparing expected and actual aspects. Do not require raw string-set equality for semantically equivalent names.

```python
ASPECT_ALIASES = {
    "核心问题与动机": "研究问题与动机",
    "研究目标": "研究问题与动机",
    "核心方法与数学公式": "核心方法",
    "训练设置": "训练方式",
    "训练流程": "训练方式",
    "训练过程": "训练方式",
    "主要定量实验结果": "实验结果",
    "实验基础模型、数据集和任务": "实验设置",
    "对比 baseline 和评价指标": "实验设置",
}
```

## Manifest Enforcement

`eval_manifest.json` records:

- `eval_cases_v1.jsonl` SHA256
- PDF SHA256
- PDF page counts
- 1-based page-numbering convention

Before comparing a run to the v1-final baseline, check the manifest:

```bash
python eval/validate_eval_cases.py --check-manifest
```

If the JSONL hash, PDF hash, or PDF page count differs, do not compare the result with the frozen v1-final baseline.

## Deterministic Validation

Run from the project root:

```bash
python -m py_compile eval/validate_eval_cases.py
python -m py_compile eval/verify_pdf_anchors.py
python eval/verify_pdf_anchors.py
python eval/validate_eval_cases.py
python eval/validate_eval_cases.py --check-manifest
```

These commands do not call a real LLM, do not run the Agent, and do not query Chroma.

After v1-final freeze, low scores should be addressed by improving the Agent, retrieval strategy, or evaluation executor. Do not modify `eval_cases_v1.jsonl` to improve scores.
