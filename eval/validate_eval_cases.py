from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from verify_pdf_anchors import EXPECTED_PDFS, PdfInfo, verify_all


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
V1_PATH = EVAL_DIR / "eval_cases_v1.jsonl"
DRAFT_PATH = EVAL_DIR / "eval_cases_draft.jsonl"
MANIFEST_PATH = EVAL_DIR / "eval_manifest.json"
AUDIT_REPORT_PATH = EVAL_DIR / "final_audit_report.md"
PAPER_AGENT_PATH = PROJECT_ROOT / "paper_agent.py"

EXPECTED_TOTAL = 20
EXPECTED_IDS = [f"eval-{index:03d}" for index in range(1, EXPECTED_TOTAL + 1)]

EXPECTED_CATEGORY_COUNTS = {
    "单篇完整分析": 4,
    "方法与公式解释": 4,
    "实验结果分析": 3,
    "多论文比较": 4,
    "精确参数或数字查询": 2,
    "论文未明确说明": 1,
    "模糊指代与澄清": 1,
    "Human-in-the-loop": 1,
}

ALLOWED_HUMAN_DECISIONS = {
    "answer_with_gaps",
    "local_deep_search",
    "revise_question",
    "cancel",
}

ALLOWED_RETRIEVAL_MODES = {"page_recall", "group_recall", "none"}
ALLOWED_REVIEW_STATUSES = {
    "needs_manual_review",
    "needs_manual_fact_review",
    "reviewed",
    "reviewed_with_manual_semantic_check",
}
ALLOWED_FACT_RULE_MODES = {
    "case_insensitive_exact",
    "numeric_exact",
    "numeric_equivalent",
    "numeric_relation",
    "subset",
}

EXPECTED_EVAL_016_FACTS = {"attention_heads": 8, "d_model": 512}

EXPECTED_EVAL_017_FACTS = {
    "optimizer": "AdamW",
    "batch_size": 128,
    "epochs": 2,
    "lora_learning_rate": 0.0002,
    "typical_rank_configs": [
        "rq = rv = 1",
        "rv = 2",
        "rq = rv = 8",
        "rq = rk = rv = ro = 2",
    ],
}

EXPECTED_EVAL_017_SUPPORTING_FACTS = {
    "weight_decay": 0.1,
    "warmup_tokens": 250000,
    "lr_schedule": "Linear",
}

EXPECTED_EVAL_018_SUPPORTING_FACTS = {
    "finetuning_trajectory_count": 3000,
    "hotpotqa_finetuning_batch_size": 64,
    "hotpotqa_react_steps_palm_8b": 4000,
    "hotpotqa_react_steps_palm_62b": 4000,
}

LORA_SOURCE = "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"
TRANSFORMER_SOURCE = "01 Attention Is All You Need.pdf"
REACT_SOURCE = "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_supported_intents() -> set[str]:
    source = PAPER_AGENT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PAPER_AGENT_PATH))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "SUPPORTED_INTENTS":
                value = ast.literal_eval(node.value)
                return {str(item) for item in value}
    raise RuntimeError("Cannot find SUPPORTED_INTENTS in paper_agent.py")


def read_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path.name} line {line_number}: each line must be a JSON object")
        cases.append(item)
    return cases


def sorted_unique(values: list[int]) -> list[int]:
    return sorted(set(values))


def normalized_items(values: list[Any]) -> list[str]:
    return [str(item).strip().casefold() for item in values if str(item).strip()]


def flatten_include_groups(groups: Any) -> list[str]:
    if not isinstance(groups, list):
        return []
    return [item for group in groups if isinstance(group, list) for item in group if isinstance(item, str)]


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_string_list(case_id: str, field_name: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{case_id}: {field_name} must be a list of strings")


def validate_must_include_policy(case: dict[str, Any], errors: list[str]) -> None:
    case_id = str(case.get("id"))
    groups = case.get("must_include_any")
    if not isinstance(groups, list):
        errors.append(f"{case_id}: must_include_any must be a list")
        return
    for index, group in enumerate(groups, 1):
        if not isinstance(group, list) or not all(isinstance(item, str) for item in group):
            errors.append(f"{case_id}: must_include_any group {index} must be a list of strings")
        elif not any(item.strip() for item in group):
            errors.append(f"{case_id}: must_include_any group {index} must contain a non-empty string")

    policy = case.get("must_include_policy", {"mode": "all"})
    if not isinstance(policy, dict):
        errors.append(f"{case_id}: must_include_policy must be an object when present")
        return
    mode = policy.get("mode", "all")
    if mode not in {"all", "at_least"}:
        errors.append(f"{case_id}: must_include_policy.mode must be all or at_least")
    if mode == "at_least":
        min_groups = policy.get("min_groups")
        if not isinstance(min_groups, int) or not (1 <= min_groups <= len(groups)):
            errors.append(f"{case_id}: at_least min_groups must be between 1 and must_include_any group count")


def validate_fact_match_rules(case: dict[str, Any], errors: list[str]) -> None:
    case_id = str(case.get("id"))
    facts = case.get("expected_facts")
    rules = case.get("fact_match_rules", {})
    if not isinstance(facts, dict):
        errors.append(f"{case_id}: expected_facts must be an object")
        return
    if not isinstance(rules, dict):
        errors.append(f"{case_id}: fact_match_rules must be an object when present")
        return
    for key, rule in rules.items():
        if key not in facts:
            errors.append(f"{case_id}: fact_match_rules key {key!r} is not present in expected_facts")
        if not isinstance(rule, dict):
            errors.append(f"{case_id}: fact_match_rules[{key!r}] must be an object")
            continue
        mode = rule.get("mode")
        if mode not in ALLOWED_FACT_RULE_MODES:
            errors.append(f"{case_id}: unsupported fact_match_rules mode {mode!r} for key {key!r}")
        if mode == "numeric_relation":
            if not isinstance(rule.get("aliases"), list) or not rule["aliases"]:
                errors.append(f"{case_id}: numeric_relation rule {key!r} needs aliases")
            if rule.get("expected") != facts.get(key):
                errors.append(f"{case_id}: numeric_relation expected for {key!r} must match expected_facts")
        if mode == "numeric_equivalent":
            tolerance = rule.get("tolerance")
            if not isinstance(tolerance, (int, float)) or tolerance < 0:
                errors.append(f"{case_id}: numeric_equivalent rule {key!r} needs non-negative tolerance")
            if not isinstance(rule.get("accepted_forms"), list) or not rule["accepted_forms"]:
                errors.append(f"{case_id}: numeric_equivalent rule {key!r} needs accepted_forms")
        if mode == "subset":
            if not isinstance(rule.get("min_matches"), int) or rule["min_matches"] < 1:
                errors.append(f"{case_id}: subset rule {key!r} needs min_matches >= 1")


def validate_regexes(case: dict[str, Any], errors: list[str]) -> None:
    case_id = str(case.get("id"))
    regexes = case.get("must_match_regex")
    if regexes is None:
        return
    if not isinstance(regexes, list):
        errors.append(f"{case_id}: must_match_regex must be a list")
        return
    for pattern in regexes:
        if not isinstance(pattern, str) or not pattern:
            errors.append(f"{case_id}: must_match_regex entries must be non-empty strings")
            continue
        try:
            re.compile(pattern)
        except re.error as exc:
            errors.append(f"{case_id}: invalid regex {pattern!r}: {exc}")


def validate_gold_pages(case: dict[str, Any], errors: list[str]) -> None:
    case_id = str(case.get("id"))
    mode = case.get("retrieval_eval_mode")
    groups_by_paper = case.get("gold_page_groups")
    gold_pages = case.get("gold_pages")
    if mode not in ALLOWED_RETRIEVAL_MODES:
        errors.append(f"{case_id}: retrieval_eval_mode must be one of {sorted(ALLOWED_RETRIEVAL_MODES)}")
    if not isinstance(groups_by_paper, dict):
        errors.append(f"{case_id}: gold_page_groups must be an object")
        return
    if not isinstance(gold_pages, dict):
        errors.append(f"{case_id}: gold_pages must be an object")
        return
    if mode == "none":
        if groups_by_paper or gold_pages:
            errors.append(f"{case_id}: retrieval_eval_mode=none requires empty gold_page_groups and gold_pages")
        return

    expected_pages: dict[str, list[int]] = {}
    for paper, groups in groups_by_paper.items():
        if paper not in EXPECTED_PDFS:
            errors.append(f"{case_id}: unknown paper in gold_page_groups: {paper!r}")
            continue
        if not isinstance(groups, list) or not groups:
            errors.append(f"{case_id}: gold_page_groups[{paper!r}] must be a non-empty list")
            continue
        collected: list[int] = []
        for group_index, group in enumerate(groups, 1):
            if not isinstance(group, dict):
                errors.append(f"{case_id}: group {group_index} for {paper!r} must be an object")
                continue
            aspect = group.get("aspect")
            pages = group.get("pages")
            if not isinstance(aspect, str) or not aspect.strip():
                errors.append(f"{case_id}: group {group_index} for {paper!r} needs non-empty aspect")
            if not isinstance(pages, list) or not pages:
                errors.append(f"{case_id}: group {group_index} for {paper!r} needs non-empty pages")
                continue
            for page in pages:
                if not isinstance(page, int) or page <= 0:
                    errors.append(f"{case_id}: page values must be positive integers")
                    continue
                if page > EXPECTED_PDFS[paper]:
                    errors.append(f"{case_id}: page {page} exceeds {paper!r} page count {EXPECTED_PDFS[paper]}")
                collected.append(page)
        expected_pages[paper] = sorted_unique(collected)

    if set(gold_pages) != set(expected_pages):
        errors.append(f"{case_id}: gold_pages keys must match gold_page_groups keys")
    for paper, pages in gold_pages.items():
        if not isinstance(pages, list):
            errors.append(f"{case_id}: gold_pages[{paper!r}] must be a list")
            continue
        if pages != expected_pages.get(paper, []):
            errors.append(
                f"{case_id}: gold_pages[{paper!r}] must equal sorted unique group union; "
                f"expected {expected_pages.get(paper, [])}, found {pages}"
            )


def validate_special_cases(case: dict[str, Any], errors: list[str]) -> None:
    case_id = str(case.get("id"))
    if case_id == "eval-004":
        groups = case.get("gold_page_groups", {}).get(LORA_SOURCE, [])
        require(
            not any(isinstance(group, dict) and group.get("aspect") == "GPT-3 训练超参数" for group in groups),
            errors,
            "eval-004: must not contain GPT-3 训练超参数 group",
        )
        require(case.get("gold_pages", {}).get(LORA_SOURCE) == [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13], errors, "eval-004: gold_pages must exclude GPT-3 appendix hyperparameter pages")

    if case_id == "eval-005":
        require("must_match_regex" not in case, errors, "eval-005: must not contain must_match_regex Q/K/V checks")
        terms = {item.casefold() for item in flatten_include_groups(case.get("must_include_any"))}
        require("qk^t" in terms, errors, "eval-005: must_include_any should include QK^T formula variant")
        require("与 v 相乘" in terms, errors, "eval-005: must_include_any should include value multiplication concept")

    if case_id == "eval-011":
        include_terms = {item.casefold() for item in flatten_include_groups(case.get("must_include_any"))}
        for forbidden in {"sft", "ppo", "reward model", "奖励模型", "监督微调"}:
            if forbidden in include_terms:
                errors.append(f"eval-011: must_include_any must not force method-stage term {forbidden!r}")
        policy = case.get("must_include_policy")
        require(policy == {"mode": "at_least", "min_groups": 4}, errors, "eval-011: must_include_policy must require at least 4 groups")

    if case_id == "eval-016":
        groups = case.get("gold_page_groups", {}).get(TRANSFORMER_SOURCE)
        require(
            groups == [{"aspect": "Transformer base 的 h=8 和 d_model=512", "pages": [5, 9]}],
            errors,
            "eval-016: must have exactly one alternative evidence group with pages [5, 9]",
        )
        require(case.get("expected_facts") == EXPECTED_EVAL_016_FACTS, errors, "eval-016: expected_facts mismatch")

    if case_id == "eval-017":
        facts = case.get("expected_facts", {})
        require("至少两种" in str(case.get("question", "")), errors, "eval-017: question must contain 至少两种")
        require(case.get("gold_pages", {}).get(LORA_SOURCE) == [19, 20, 21, 23], errors, "eval-017: gold_pages must be [19, 20, 21, 23]")
        all_pages = case.get("gold_pages", {}).get(LORA_SOURCE, [])
        require(10 not in all_pages and 24 not in all_pages, errors, "eval-017: must not include pages 10 or 24")
        groups = case.get("gold_page_groups", {}).get(LORA_SOURCE, [])
        table_groups = [group for group in groups if isinstance(group, dict) and group.get("aspect") == "Table 12 的 optimizer、batch、epoch 和 learning rate"]
        require(table_groups and table_groups[0].get("pages") == [21], errors, "eval-017: Table 12 group must be [21]")
        for key, expected in EXPECTED_EVAL_017_FACTS.items():
            if key == "lora_learning_rate":
                if not isinstance(facts.get(key), (int, float)) or not math.isclose(float(facts[key]), 0.0002, abs_tol=1e-12):
                    errors.append("eval-017: lora_learning_rate must be numerically equivalent to 0.0002")
            elif facts.get(key) != expected:
                errors.append(f"eval-017: expected_facts[{key!r}] mismatch")
        for key in {"weight_decay", "warmup_tokens", "lr_schedule"}:
            if key in facts:
                errors.append(f"eval-017: {key} belongs in supporting_facts, not expected_facts")
        support = case.get("supporting_facts")
        require(support == EXPECTED_EVAL_017_SUPPORTING_FACTS, errors, "eval-017: supporting_facts mismatch")
        rank_rule = case.get("fact_match_rules", {}).get("typical_rank_configs", {})
        require(rank_rule.get("min_matches") == 2, errors, "eval-017: rank rule min_matches must be 2")
        require(case.get("must_include_any") == [], errors, "eval-017: must_include_any must be empty")

    if case_id == "eval-018":
        require(case.get("retrieval_eval_mode") == "group_recall", errors, "eval-018: retrieval_eval_mode must be group_recall")
        require(case.get("expected_facts", {}).get("explicit_token_level_nll") is False, errors, "eval-018: explicit_token_level_nll must be false")
        require(case.get("supporting_facts") == EXPECTED_EVAL_018_SUPPORTING_FACTS, errors, "eval-018: supporting_facts mismatch")
        forced = {item.casefold() for item in flatten_include_groups(case.get("must_include_any"))}
        for forbidden in {"3000", "64", "4000", "batch size"}:
            if forbidden in forced:
                errors.append(f"eval-018: must_include_any must not force supporting fact {forbidden!r}")

    if case_id == "eval-019":
        require(case.get("retrieval_eval_mode") == "none", errors, "eval-019: retrieval_eval_mode must be none")
        require(case.get("expected_papers") == [], errors, "eval-019: expected_papers must be empty")
        require(case.get("gold_pages") == {}, errors, "eval-019: gold_pages must be empty")
        require(case.get("gold_page_groups") == {}, errors, "eval-019: gold_page_groups must be empty")
        require(case.get("expected_intent") == "clarification", errors, "eval-019: expected_intent must be clarification")

    if case_id == "eval-020":
        require(case.get("expected_human_review") is True, errors, "eval-020: expected_human_review must be true")
        require(case.get("human_decisions") == ["local_deep_search", "answer_with_gaps"], errors, "eval-020: human_decisions mismatch")
        require(case.get("resume_within_case") is True, errors, "eval-020: resume_within_case must be true")


def validate_cases(cases: list[dict[str, Any]], label: str) -> list[str]:
    errors: list[str] = []
    supported_intents = load_supported_intents()

    if len(cases) != EXPECTED_TOTAL:
        errors.append(f"{label}: expected {EXPECTED_TOTAL} cases, found {len(cases)}")

    ids = [case.get("id") for case in cases]
    if ids != EXPECTED_IDS:
        errors.append(f"{label}: ids must be consecutive eval-001..eval-020 in order")
    if len(ids) != len(set(ids)):
        errors.append(f"{label}: duplicate ids found")

    counts = Counter(case.get("category") for case in cases)
    for category, expected_count in EXPECTED_CATEGORY_COUNTS.items():
        if counts.get(category, 0) != expected_count:
            errors.append(f"{label}: category {category!r} expected {expected_count}, found {counts.get(category, 0)}")

    covered_papers: set[str] = set()
    for case in cases:
        case_id = str(case.get("id"))
        require(isinstance(case.get("question"), str) and bool(case["question"].strip()), errors, f"{case_id}: question must be non-empty")
        require(case.get("expected_intent") in supported_intents, errors, f"{case_id}: expected_intent is not in SUPPORTED_INTENTS")
        require(case.get("thread_mode") == "fresh_per_case", errors, f"{case_id}: thread_mode must be fresh_per_case")
        require(case.get("resume_within_case") is (case_id == "eval-020"), errors, f"{case_id}: resume_within_case must be true only for eval-020")

        if case_id != "eval-019":
            require(case.get("retrieval_eval_mode") == "group_recall", errors, f"{case_id}: retrieval_eval_mode must be group_recall in v1-final")

        expected_papers = case.get("expected_papers")
        if not isinstance(expected_papers, list):
            errors.append(f"{case_id}: expected_papers must be a list")
        else:
            for paper in expected_papers:
                if paper not in EXPECTED_PDFS:
                    errors.append(f"{case_id}: unknown expected_paper {paper!r}")
                else:
                    covered_papers.add(paper)

        validate_string_list(case_id, "expected_aspects", case.get("expected_aspects"), errors)
        require(isinstance(case.get("expected_human_review"), bool), errors, f"{case_id}: expected_human_review must be boolean")
        human_decisions = case.get("human_decisions")
        if not isinstance(human_decisions, list):
            errors.append(f"{case_id}: human_decisions must be a list")
        else:
            invalid = sorted(set(human_decisions) - ALLOWED_HUMAN_DECISIONS)
            if invalid:
                errors.append(f"{case_id}: invalid human_decisions {invalid}")
        require(case.get("review_status") in ALLOWED_REVIEW_STATUSES, errors, f"{case_id}: invalid review_status")

        validate_gold_pages(case, errors)
        validate_must_include_policy(case, errors)
        validate_fact_match_rules(case, errors)
        validate_regexes(case, errors)
        validate_string_list(case_id, "manual_checks", case.get("manual_checks"), errors)
        validate_string_list(case_id, "must_include", case.get("must_include"), errors)
        validate_string_list(case_id, "must_not_include", case.get("must_not_include"), errors)

        support = case.get("supporting_facts")
        if support is not None:
            require(isinstance(support, dict), errors, f"{case_id}: supporting_facts must be an object")
            facts = case.get("expected_facts", {})
            if isinstance(facts, dict):
                overlap = sorted(set(facts) & set(support))
                if overlap:
                    errors.append(f"{case_id}: supporting_facts overlap expected_facts keys {overlap}")

        validate_special_cases(case, errors)

    missing = sorted(set(EXPECTED_PDFS) - covered_papers)
    if missing:
        errors.append(f"{label}: not all PDFs are covered by expected_papers: {missing}")
    return errors


def collect_paper_pages(cases: list[dict[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for case in cases:
        for paper, pages in case.get("gold_pages", {}).items():
            result[paper].update(pages)
    return {paper: sorted(pages) for paper, pages in sorted(result.items())}


def build_manifest(cases: list[dict[str, Any]], pdf_infos: dict[str, PdfInfo]) -> dict[str, Any]:
    return {
        "dataset_version": "v1-final",
        "page_numbering": "1-based-physical-pdf-page",
        "dataset_file": "eval/eval_cases_v1.jsonl",
        "dataset_sha256": sha256_file(V1_PATH),
        "case_count": len(cases),
        "pdfs": {
            filename: {"page_count": pdf_infos[filename].page_count, "sha256": pdf_infos[filename].sha256}
            for filename in sorted(pdf_infos)
        },
    }


def write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_manifest(current_manifest: dict[str, Any]) -> list[str]:
    if not MANIFEST_PATH.exists():
        return ["eval_manifest.json does not exist"]
    try:
        recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"eval_manifest.json is invalid JSON: {exc}"]
    if recorded != current_manifest:
        return ["eval_manifest.json does not match current JSONL/PDF hashes or page counts"]
    return []


def group_count(case: dict[str, Any]) -> int:
    return sum(len(groups) for groups in case.get("gold_page_groups", {}).values())


def write_audit_report(cases: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Final Audit Report",
        "",
        "- dataset_version: v1-final",
        f"- dataset_sha256: `{manifest['dataset_sha256']}`",
        "- page_numbering: 1-based physical PDF page",
        "- automatic_validation: passed",
        "",
        "## Case Summary",
        "",
        "| id | category | intent | papers | retrieval | groups | gold pages | expected_facts | supporting_facts | thread | resume | auto | manual semantic check |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        papers = "<br>".join(case.get("expected_papers", [])) or "none"
        gold_pages = json.dumps(case.get("gold_pages", {}), ensure_ascii=False)
        expected_facts = json.dumps(case.get("expected_facts", {}), ensure_ascii=False)
        supporting_facts = json.dumps(case.get("supporting_facts", {}), ensure_ascii=False)
        manual = "yes" if case.get("review_status") == "reviewed_with_manual_semantic_check" else "no"
        lines.append(
            f"| {case['id']} | {case['category']} | {case['expected_intent']} | {papers} | "
            f"{case['retrieval_eval_mode']} | {group_count(case)} | `{gold_pages}` | "
            f"`{expected_facts}` | `{supporting_facts}` | {case['thread_mode']} | "
            f"{case['resume_within_case']} | passed | {manual} |"
        )

    lines.extend(
        [
            "",
            "## Final Modifications",
            "",
            "- eval-004 deleted the unrelated GPT-3 hyperparameter evidence group; eval-017 owns precise LoRA GPT-3 hyperparameter checks.",
            "- eval-005 deleted Q/K/V single-letter regex checks and replaced them with formula concept groups.",
            "- eval-005, eval-006, eval-007, eval-008, and eval-016 now use group_recall.",
            "- eval-011 experiment-result keywords no longer force SFT/RM/PPO and use an at_least policy.",
            "- eval-016 treats pages [5, 9] as one alternative evidence group.",
            "- eval-017 Table 12 is corrected to physical PDF page 21.",
            "- eval-017 removed pages 10 and 24 from gold evidence.",
            "- eval-017 requires at least two typical rank configurations.",
            "- All cases include thread isolation fields.",
            "- Added PDF anchors and SHA256 manifest freeze checks.",
            "",
            "## Extra Modifications",
            "",
            "No unprompted extra modifications were made beyond the final audit requirements.",
            "",
            "## Manifest",
            "",
            "```json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    AUDIT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(cases: list[dict[str, Any]], manifest: dict[str, Any], pdf_anchor_errors: list[str]) -> None:
    print(f"OK: eval_cases_v1.jsonl: {len(cases)} eval cases validated.")
    print(f"PDF anchors: {'passed' if not pdf_anchor_errors else 'failed'}")
    print(f"dataset_sha256: {manifest['dataset_sha256']}")
    print("Category distribution:")
    for category in EXPECTED_CATEGORY_COUNTS:
        print(f"- {category}: {Counter(case['category'] for case in cases)[category]}")
    print("Retrieval eval modes:")
    for case in cases:
        print(f"- {case['id']}: {case['retrieval_eval_mode']}")
    print("Gold pages by paper:")
    for paper, pages in collect_paper_pages(cases).items():
        print(f"- {paper}: {pages}")
    print("PDF manifest:")
    for filename, info in manifest["pdfs"].items():
        print(f"- {filename}: pages={info['page_count']}, sha256={info['sha256']}")
    semantic = [case["id"] for case in cases if case.get("review_status") == "reviewed_with_manual_semantic_check"]
    print("Manual semantic check cases:")
    for case_id in semantic:
        print(f"- {case_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-manifest", action="store_true")
    args = parser.parse_args()

    all_errors: list[str] = []
    try:
        v1_cases = read_cases(V1_PATH)
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    all_errors.extend(validate_cases(v1_cases, V1_PATH.name))

    if DRAFT_PATH.exists():
        try:
            draft_cases = read_cases(DRAFT_PATH)
            all_errors.extend(validate_cases(draft_cases, DRAFT_PATH.name))
            if draft_cases != v1_cases:
                all_errors.append("eval_cases_draft.jsonl and eval_cases_v1.jsonl must be synchronized")
        except Exception as exc:
            all_errors.append(f"{DRAFT_PATH.name}: {exc}")

    pdf_anchor_errors, pdf_infos = verify_all()
    all_errors.extend(f"PDF anchor: {error}" for error in pdf_anchor_errors)

    if all_errors:
        print("Validation failed:", file=sys.stderr)
        for error in all_errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    manifest = build_manifest(v1_cases, pdf_infos)
    if args.check_manifest:
        manifest_errors = check_manifest(manifest)
        if manifest_errors:
            print("Manifest check failed:", file=sys.stderr)
            for error in manifest_errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("OK: manifest matches current dataset and PDFs.")
        print_summary(v1_cases, manifest, pdf_anchor_errors)
        return 0

    write_manifest(manifest)
    write_audit_report(v1_cases, manifest)
    print_summary(v1_cases, manifest, pdf_anchor_errors)
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {AUDIT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
