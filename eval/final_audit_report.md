# Final Audit Report

- dataset_version: v1-final
- dataset_sha256: `b98478afc3defddfd0d7673dbe68836343770fe2fe10a7e29d68dc285c643d9d`
- page_numbering: 1-based physical PDF page
- automatic_validation: passed

## Case Summary

| id | category | intent | papers | retrieval | groups | gold pages | expected_facts | supporting_facts | thread | resume | auto | manual semantic check |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| eval-001 | 单篇完整分析 | paper_analysis | 01 Attention Is All You Need.pdf | group_recall | 5 | `{"01 Attention Is All You Need.pdf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-002 | 单篇完整分析 | paper_analysis | 02 Language Models are Few-Shot Learners.pdf | group_recall | 6 | `{"02 Language Models are Few-Shot Learners.pdf": [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 18, 20, 22, 25, 27, 33, 34, 35, 36, 39, 40, 41]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-003 | 单篇完整分析 | paper_analysis | 03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf | group_recall | 6 | `{"03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf": [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-004 | 单篇完整分析 | paper_analysis | 05 LoRA - Low-Rank Adaptation of Large Language Models.pdf | group_recall | 5 | `{"05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-005 | 方法与公式解释 | method_explain | 01 Attention Is All You Need.pdf | group_recall | 2 | `{"01 Attention Is All You Need.pdf": [4, 5]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-006 | 方法与公式解释 | method_explain | 05 LoRA - Low-Rank Adaptation of Large Language Models.pdf | group_recall | 2 | `{"05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": [4, 5]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-007 | 方法与公式解释 | method_explain | 04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf | group_recall | 3 | `{"04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": [2, 3, 4]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-008 | 方法与公式解释 | method_explain | 06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf | group_recall | 2 | `{"06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": [3, 4]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-009 | 实验结果分析 | experiment_analysis | 05 LoRA - Low-Rank Adaptation of Large Language Models.pdf | group_recall | 2 | `{"05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": [6, 7, 8, 10, 11, 12]}` | `{}` | `{}` | fresh_per_case | False | passed | yes |
| eval-010 | 实验结果分析 | experiment_analysis | 02 Language Models are Few-Shot Learners.pdf | group_recall | 5 | `{"02 Language Models are Few-Shot Learners.pdf": [5, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}` | `{}` | `{}` | fresh_per_case | False | passed | yes |
| eval-011 | 实验结果分析 | experiment_analysis | 03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf | group_recall | 4 | `{"03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf": [2, 3, 4, 11, 12, 13, 14, 17, 19, 20]}` | `{}` | `{}` | fresh_per_case | False | passed | yes |
| eval-012 | 多论文比较 | paper_comparison | 01 Attention Is All You Need.pdf<br>02 Language Models are Few-Shot Learners.pdf | group_recall | 6 | `{"01 Attention Is All You Need.pdf": [2, 3, 4, 5, 7, 8, 10], "02 Language Models are Few-Shot Learners.pdf": [3, 4, 5, 7, 8, 9, 10]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-013 | 多论文比较 | paper_comparison | 05 LoRA - Low-Rank Adaptation of Large Language Models.pdf<br>03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf | group_recall | 5 | `{"05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": [1, 2, 4, 5], "03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf": [1, 2, 3, 6, 7, 8, 9, 19, 20]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-014 | 多论文比较 | paper_comparison | 04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf<br>05 LoRA - Low-Rank Adaptation of Large Language Models.pdf | group_recall | 7 | `{"04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": [1, 2, 3, 4, 8, 17], "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": [1, 2, 4, 5, 17]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-015 | 多论文比较 | paper_comparison | 02 Language Models are Few-Shot Learners.pdf<br>06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf | group_recall | 6 | `{"02 Language Models are Few-Shot Learners.pdf": [4, 5, 7, 10], "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": [3, 4, 5]}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-016 | 精确参数或数字查询 | general_qa | 01 Attention Is All You Need.pdf | group_recall | 1 | `{"01 Attention Is All You Need.pdf": [5, 9]}` | `{"attention_heads": 8, "d_model": 512}` | `{}` | fresh_per_case | False | passed | no |
| eval-017 | 精确参数或数字查询 | reproduction_plan | 05 LoRA - Low-Rank Adaptation of Large Language Models.pdf | group_recall | 3 | `{"05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": [19, 20, 21, 23]}` | `{"optimizer": "AdamW", "batch_size": 128, "epochs": 2, "lora_learning_rate": 0.0002, "typical_rank_configs": ["rq = rv = 1", "rv = 2", "rq = rv = 8", "rq = rk = rv = ro = 2"]}` | `{"weight_decay": 0.1, "warmup_tokens": 250000, "lr_schedule": "Linear"}` | fresh_per_case | False | passed | no |
| eval-018 | 论文未明确说明 | general_qa | 06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf | group_recall | 2 | `{"06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": [5, 15]}` | `{"explicit_token_level_nll": false}` | `{"finetuning_trajectory_count": 3000, "hotpotqa_finetuning_batch_size": 64, "hotpotqa_react_steps_palm_8b": 4000, "hotpotqa_react_steps_palm_62b": 4000}` | fresh_per_case | False | passed | yes |
| eval-019 | 模糊指代与澄清 | clarification | none | none | 0 | `{}` | `{}` | `{}` | fresh_per_case | False | passed | no |
| eval-020 | Human-in-the-loop | paper_comparison | 04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf<br>06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf | group_recall | 7 | `{"04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": [2, 3, 4], "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": [3, 4, 5, 15]}` | `{}` | `{}` | fresh_per_case | True | passed | yes |

## Final Modifications

- eval-004 deleted the unrelated GPT-3 hyperparameter evidence group; eval-017 owns precise LoRA GPT-3 hyperparameter checks.
- eval-005 deleted Q/K/V single-letter regex checks and replaced them with formula concept groups.
- eval-005, eval-006, eval-007, eval-008, and eval-016 now use group_recall.
- eval-011 experiment-result keywords no longer force SFT/RM/PPO and use an at_least policy.
- eval-016 treats pages [5, 9] as one alternative evidence group.
- eval-017 Table 12 is corrected to physical PDF page 21.
- eval-017 removed pages 10 and 24 from gold evidence.
- eval-017 requires at least two typical rank configurations.
- All cases include thread isolation fields.
- Added PDF anchors and SHA256 manifest freeze checks.

## Extra Modifications

No unprompted extra modifications were made beyond the final audit requirements.

## Manifest

```json
{
  "dataset_version": "v1-final",
  "page_numbering": "1-based-physical-pdf-page",
  "dataset_file": "eval/eval_cases_v1.jsonl",
  "dataset_sha256": "b98478afc3defddfd0d7673dbe68836343770fe2fe10a7e29d68dc285c643d9d",
  "case_count": 20,
  "pdfs": {
    "01 Attention Is All You Need.pdf": {
      "page_count": 15,
      "sha256": "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"
    },
    "02 Language Models are Few-Shot Learners.pdf": {
      "page_count": 75,
      "sha256": "97fd272f1fdfc18677462d0292f5fbf26ca86b4d1b485c2dba03269b643a0e83"
    },
    "03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf": {
      "page_count": 68,
      "sha256": "c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98"
    },
    "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": {
      "page_count": 19,
      "sha256": "23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233"
    },
    "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": {
      "page_count": 26,
      "sha256": "e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a"
    },
    "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": {
      "page_count": 33,
      "sha256": "f285b0971ae4a790e402fb93966bed3adde2cf0a04977d08b2b40d6ab0cace69"
    }
  }
}
```
