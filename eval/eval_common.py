from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = EVAL_DIR / "eval_manifest.json"
DEFAULT_CASES_PATH = EVAL_DIR / "eval_cases_v1.jsonl"
RESULT_STATUSES = {"pass", "fail", "manual_review", "unsupported", "error", "dry_run"}

ASPECT_ALIASES = {
    "核心问题与动机": "研究问题与动机",
    "研究目标": "研究问题与动机",
    "研究问题与贡献": "研究问题与动机",
    "核心方法与数学公式": "核心方法",
    "数学公式": "核心方法",
    "训练设置": "训练方式",
    "训练流程": "训练方式",
    "训练过程": "训练方式",
    "主要定量实验结果": "实验结果",
    "实验基础模型、数据集和任务": "实验设置",
    "对比 baseline 和评价指标": "实验设置",
    "部署与推理开销": "推理与部署",
    "推理合并与部署开销": "推理与部署",
    "优势与局限": "优势与局限性",
}

_PUNCT_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "、": " ",
        "－": "-",
        "–": "-",
        "—": "-",
        "×": "x",
        "✕": "x",
        "−": "-",
    }
)

_SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+",
    }
)


def load_eval_cases(path: Path, case_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    path = resolve_project_path(path)
    wanted = set(case_ids or [])
    cases: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                case = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not wanted or case.get("id") in wanted:
                cases.append(case)
    return cases


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def resolve_project_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    cwd_candidate = Path.cwd() / raw
    project_candidate = PROJECT_ROOT / raw
    if cwd_candidate.exists():
        return cwd_candidate
    if project_candidate.exists():
        return project_candidate
    if raw.parts and raw.parts[0].lower() == "eval":
        return project_candidate
    return cwd_candidate


def prepare_run_outputs(run_dir: Path, *, resume: bool = False) -> None:
    if resume:
        return
    for name in ("cases.jsonl", "errors.jsonl", "summary.json", "report.md", "run_config.json"):
        path = run_dir / name
        if path.exists() and path.is_file():
            path.unlink()
    raw_dir = run_dir / "raw"
    if raw_dir.exists() and raw_dir.is_dir():
        for path in raw_dir.glob("*.json"):
            if path.is_file():
                path.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_case_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def normalize_text(text: str, *, expand_equivalents: bool = True) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = re.sub(r"\\[\(\[]|\\[\)\]]|\$", " ", normalized)
    normalized = re.sub(r"\\(?:mathsf|mathrm|mathbf)\s*\{\s*([^{}]+?)\s*\}", r"\1", normalized)
    normalized = re.sub(r"\\(?:mathsf|mathrm|mathbf)\s+([A-Za-z])", r"\1", normalized)
    normalized = normalized.replace("\\top", "T").replace("ᵀ", "T")
    normalized = normalized.replace("√", "sqrt")
    normalized = normalized.translate(_PUNCT_TRANSLATION).translate(_SUPERSCRIPT_MAP)
    normalized = normalized.casefold()
    normalized = re.sub(r"q\s*k\s*\^\s*\{?\s*t\s*\}?", " qk^t ", normalized)
    normalized = re.sub(r"q\s*k\s*\{\s*t\s*\}", " qk^t ", normalized)
    normalized = re.sub(r"\\?sqrt\s*\{\s*d\s*_\s*k\s*\}", " sqrt(d_k) ", normalized)
    normalized = re.sub(r"sqrt\s*\(?\s*d\s*_\s*k\s*\)?", " sqrt(d_k) ", normalized)
    normalized = normalized.replace("\\", " ")
    normalized = normalized.replace("{", " ").replace("}", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    expansions: list[str] = []
    if re.search(r"(?:multiply|multiplied by|times)\s+v\b", normalized) or re.search(r"\)\s*v\b", normalized):
        expansions.extend(["乘以 v", "与 v 相乘", "value vector", "值向量"])
    if "乘以 v" in normalized or "与 v 相乘" in normalized:
        expansions.extend(["multiply v", "value vector"])
    if "weighted sum of values" in normalized or "values 的加权和" in normalized or "值的加权和" in normalized:
        expansions.extend(["values 的加权和", "值的加权和", "value vector", "值向量"])
    if "value vector" in normalized or "值向量" in normalized:
        expansions.extend(["value vector", "值向量"])
    if expand_equivalents and expansions:
        normalized = f"{normalized} {' '.join(dict.fromkeys(expansions))}"
    return normalized.strip()


def normalize_aspect(aspect: str) -> str:
    cleaned = str(aspect).strip()
    return ASPECT_ALIASES.get(cleaned, cleaned)


def normalize_aspects(aspects: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for aspect in aspects or []:
        normalized = normalize_aspect(str(aspect))
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def set_metrics(expected: Iterable[str], actual: Iterable[str]) -> dict[str, Any]:
    expected_set = set(expected or [])
    actual_set = set(actual or [])
    exact_match = expected_set == actual_set
    if not expected_set and not actual_set:
        return {"exact_match": True, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected_set and actual_set:
        return {"exact_match": False, "precision": 0.0, "recall": 1.0, "f1": 0.0}
    if expected_set and not actual_set:
        return {"exact_match": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    intersection = expected_set & actual_set
    precision = len(intersection) / len(actual_set) if actual_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _sorted_ints(values: Iterable[Any]) -> list[int]:
    return sorted({int(value) for value in values if isinstance(value, int) or str(value).isdigit()})


def score_group_recall(
    gold_page_groups: dict[str, list[dict[str, Any]]] | None,
    retrieved_pages_by_paper: dict[str, list[int]] | None,
) -> dict[str, Any]:
    gold_page_groups = gold_page_groups or {}
    retrieved_pages_by_paper = {
        paper: _sorted_ints(pages) for paper, pages in (retrieved_pages_by_paper or {}).items()
    }
    group_hits_by_paper: dict[str, list[dict[str, Any]]] = {}
    group_recall_by_paper: dict[str, float] = {}

    for paper, groups in gold_page_groups.items():
        retrieved = set(retrieved_pages_by_paper.get(paper, []))
        hits: list[dict[str, Any]] = []
        hit_count = 0
        for group in groups:
            gold_pages = _sorted_ints(group.get("pages", []))
            matched_pages = sorted(retrieved & set(gold_pages))
            hit = bool(matched_pages)
            if hit:
                hit_count += 1
            hits.append(
                {
                    "aspect": group.get("aspect", ""),
                    "gold_pages": gold_pages,
                    "matched_pages": matched_pages,
                    "hit": hit,
                }
            )
        group_hits_by_paper[paper] = hits
        group_recall_by_paper[paper] = hit_count / len(groups) if groups else 1.0

    recalls = list(group_recall_by_paper.values())
    return {
        "group_hits_by_paper": group_hits_by_paper,
        "group_recall_by_paper": group_recall_by_paper,
        "macro_group_recall": sum(recalls) / len(recalls) if recalls else None,
    }


def score_raw_page_recall(
    gold_pages: dict[str, list[int]] | None,
    retrieved_pages_by_paper: dict[str, list[int]] | None,
) -> dict[str, Any]:
    gold_pages = gold_pages or {}
    retrieved_pages_by_paper = {
        paper: _sorted_ints(pages) for paper, pages in (retrieved_pages_by_paper or {}).items()
    }
    recalls: dict[str, float] = {}
    for paper, pages in gold_pages.items():
        gold_set = set(_sorted_ints(pages))
        retrieved_set = set(retrieved_pages_by_paper.get(paper, []))
        recalls[paper] = len(gold_set & retrieved_set) / len(gold_set) if gold_set else 1.0
    values = list(recalls.values())
    return {
        "raw_page_recall_by_paper": recalls,
        "macro_raw_page_recall": sum(values) / len(values) if values else None,
    }


def extract_display_page(chunk: dict[str, Any]) -> int | None:
    if chunk.get("page") is not None:
        try:
            return int(chunk["page"])
        except (TypeError, ValueError):
            return None
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    metadata_page = chunk.get("metadata_page", metadata.get("page"))
    if metadata_page is not None:
        try:
            return int(metadata_page) + 1
        except (TypeError, ValueError):
            return None
    return None


def pages_by_paper_from_chunks(chunks: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    pages: dict[str, set[int]] = defaultdict(set)
    for chunk in chunks or []:
        source = chunk.get("source") or (chunk.get("metadata") or {}).get("source")
        page = extract_display_page(chunk)
        if source and page is not None:
            pages[str(source)].add(page)
    return {source: sorted(values) for source, values in sorted(pages.items())}


def collect_chunks_from_tool_result(result: Any) -> list[dict[str, Any]]:
    if not result:
        return []
    if isinstance(result, list):
        return [dict(item) for item in result if isinstance(item, dict)]
    if not isinstance(result, dict):
        return []
    chunks: list[dict[str, Any]] = []
    for key in ("results", "chunks", "retrieved_docs"):
        value = result.get(key)
        if isinstance(value, list):
            chunks.extend(dict(item) for item in value if isinstance(item, dict))
    for query_result in result.get("query_results", []) or []:
        if isinstance(query_result, dict) and isinstance(query_result.get("results"), list):
            chunks.extend(dict(item) for item in query_result["results"] if isinstance(item, dict))
    return chunks


def summarize_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_id": chunk.get("result_id") or chunk.get("id", ""),
        "source": chunk.get("source") or (chunk.get("metadata") or {}).get("source", ""),
        "page": extract_display_page(chunk),
        "score": chunk.get("score"),
        "score_label": chunk.get("score_label"),
        "content_preview": str(chunk.get("content") or chunk.get("page_content") or "")[:240],
    }


def score_must_include(answer: str, terms: Iterable[str]) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)
    details = []
    for term in terms or []:
        normalized_term = normalize_text(term, expand_equivalents=False)
        matched = bool(normalized_term and normalized_term in normalized_answer)
        details.append({"expected": term, "matched": matched, "matched_form": term if matched else ""})
    return {"passed": all(item["matched"] for item in details), "details": details}


def score_must_include_any(
    answer: str,
    groups: Iterable[Iterable[str]],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)
    details = []
    for group in groups or []:
        options = [str(option) for option in group if str(option).strip()]
        matched_form = ""
        for option in options:
            if normalize_text(option, expand_equivalents=False) in normalized_answer:
                matched_form = option
                break
        details.append({"options": options, "matched": bool(matched_form), "matched_form": matched_form})
    mode = (policy or {}).get("mode", "all")
    hit_count = sum(1 for item in details if item["matched"])
    group_count = len(details)
    required_count = int((policy or {}).get("min_groups", group_count if mode == "all" else 0))
    if mode == "at_least":
        passed = hit_count >= required_count
    else:
        required_count = group_count
        passed = hit_count == group_count
    return {
        "group_count": group_count,
        "hit_count": hit_count,
        "required_count": required_count,
        "passed": passed,
        "details": details,
    }


def score_must_not_include(answer: str, terms: Iterable[str]) -> dict[str, Any]:
    normalized_answer = normalize_text(answer)
    hits = []
    for term in terms or []:
        normalized_term = normalize_text(term, expand_equivalents=False)
        if normalized_term and normalized_term in normalized_answer:
            hits.append(term)
    return {"passed": not hits, "hits": hits}


def _replace_superscript(text: str) -> str:
    return str(text).translate(_SUPERSCRIPT_MAP)


def parse_numeric_value(text: Any) -> float | None:
    raw = normalize_text(_replace_superscript(str(text)))
    raw = raw.replace(" ", "")
    multiplication_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)x10\^?([+-]?\d+)", raw)
    if multiplication_match:
        return float(multiplication_match.group(1)) * (10 ** int(multiplication_match.group(2)))
    match = re.search(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def numeric_values_in_text(text: str) -> list[float]:
    normalized = normalize_text(_replace_superscript(text))
    values: list[float] = []
    for number in re.findall(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", normalized):
        try:
            values.append(float(number))
        except ValueError:
            continue
    for base, exp in re.findall(r"([+-]?\d+(?:\.\d+)?)\s*x\s*10\^?([+-]?\d+)", normalized):
        values.append(float(base) * (10 ** int(exp)))
    return values


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。.!?；;])\s*|\n+", str(text)) if part.strip()]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(normalize_text(alias))
    if normalize_text(alias) == "h":
        return re.compile(r"(?<![a-z0-9])h\s*(?:=|is|为|:)\s*", re.I)
    return re.compile(escaped, re.I)


def numeric_relation_match(answer: str, aliases: Iterable[str], expected_value: Any) -> bool:
    expected_number = parse_numeric_value(expected_value)
    if expected_number is None:
        return False
    for sentence in split_sentences(answer):
        normalized_sentence = normalize_text(sentence)
        for alias in aliases:
            if not _alias_pattern(alias).search(normalized_sentence):
                continue
            for value in numeric_values_in_text(sentence):
                if math.isclose(value, expected_number, rel_tol=1e-9, abs_tol=1e-12):
                    return True
    return False


def numeric_equivalent_match(answer: str, expected_value: Any) -> bool:
    expected_number = parse_numeric_value(expected_value)
    if expected_number is None:
        return False
    return any(
        math.isclose(value, expected_number, rel_tol=1e-6, abs_tol=1e-12)
        for value in numeric_values_in_text(answer)
    )


def normalize_rank_config(value: str) -> str:
    text = normalize_text(value)
    text = text.replace("\\(", "").replace("\\)", "").replace("$", "")
    text = re.sub(r"r\s*_\s*\{\s*([qkvo])\s*\}", r"r\1", text)
    text = re.sub(r"r\s*_\s*([qkvo])", r"r\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("，", ",")
    return text


def rank_subset_check(answer: str, accepted_values: Iterable[str], min_matches: int = 1) -> dict[str, Any]:
    normalized_answer = normalize_rank_config(answer)
    matches = []
    for value in accepted_values or []:
        normalized_value = normalize_rank_config(value)
        if normalized_value and normalized_value in normalized_answer:
            matches.append(value)
    return {
        "status": "pass" if len(matches) >= min_matches else "fail",
        "matches": matches,
        "min_matches": min_matches,
    }


def score_fact_rules(case: dict[str, Any], answer: str) -> dict[str, Any]:
    expected_facts = case.get("expected_facts") or {}
    fact_rules = case.get("fact_match_rules") or {}
    checks: dict[str, Any] = {}

    for key, expected in expected_facts.items():
        rule = fact_rules.get(key, {}) if isinstance(fact_rules, dict) else {}
        mode = rule.get("mode")
        if key == "explicit_token_level_nll" and expected is False:
            checks[key] = {
                "status": "manual_semantic_required",
                "expected": expected,
                "reason": "negative semantic fact requires human or judge review",
            }
            continue
        if mode == "numeric_relation":
            aliases = rule.get("aliases") or [key]
            passed = numeric_relation_match(answer, aliases, expected)
            checks[key] = {"status": "pass" if passed else "fail", "expected": expected, "mode": mode}
        elif mode == "numeric_equivalent":
            passed = numeric_equivalent_match(answer, expected)
            checks[key] = {"status": "pass" if passed else "fail", "expected": expected, "mode": mode}
        elif mode == "numeric_exact":
            passed = numeric_equivalent_match(answer, expected)
            checks[key] = {"status": "pass" if passed else "fail", "expected": expected, "mode": mode}
        elif mode == "subset" or isinstance(expected, list):
            accepted_values = rule.get("accepted_values") or expected
            min_matches = int(rule.get("min_matches", 1))
            checks[key] = {
                **rank_subset_check(answer, accepted_values, min_matches=min_matches),
                "expected": expected,
                "mode": "subset",
            }
        else:
            passed = normalize_text(str(expected)) in normalize_text(answer)
            checks[key] = {
                "status": "pass" if passed else "fail",
                "expected": expected,
                "mode": mode or "case_insensitive_exact",
            }
    return checks


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    include = score_must_include(answer, case.get("must_include") or [])
    include_any = score_must_include_any(
        answer,
        case.get("must_include_any") or [],
        case.get("must_include_any_policy") or {"mode": "all"},
    )
    must_not = score_must_not_include(answer, case.get("must_not_include") or [])
    fact_checks = score_fact_rules(case, answer)
    fact_statuses = [item.get("status") for item in fact_checks.values()]
    deterministic_pass = (
        include["passed"]
        and include_any["passed"]
        and must_not["passed"]
        and not any(status == "fail" for status in fact_statuses)
    )
    manual_review_required = (
        case.get("review_status") == "reviewed_with_manual_semantic_check"
        or any(status == "manual_semantic_required" for status in fact_statuses)
        or bool(case.get("manual_checks"))
    )
    return {
        "must_include": include,
        "must_include_any": include_any,
        "must_not_include": must_not,
        "fact_checks": fact_checks,
        "deterministic_pass": deterministic_pass,
        "manual_review_required": manual_review_required,
    }


def parse_citations(answer: str) -> dict[str, Any]:
    text = str(answer)
    source_split = re.split(r"【资料来源】|\[资料来源\]", text, maxsplit=1)
    body = source_split[0]
    source_text = source_split[1] if len(source_split) > 1 else ""
    inline_reference_ids = [int(value) for value in re.findall(r"\[(\d+)\]", body)]
    source_entries: dict[int, dict[str, Any]] = {}
    cited_pages_by_paper: dict[str, set[int]] = defaultdict(set)

    pattern = re.compile(r"\[(\d+)\]\s*(.+?\.pdf).*?第\s*(\d+)\s*页", re.I)
    for line in source_text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        index = int(match.group(1))
        source = match.group(2).strip()
        page = int(match.group(3))
        source_entries[index] = {"id": index, "source": source, "page": page, "raw": line.strip()}
        cited_pages_by_paper[source].add(page)

    unresolved = sorted({idx for idx in inline_reference_ids if idx not in source_entries})
    return {
        "inline_reference_ids": inline_reference_ids,
        "source_entries": [source_entries[idx] for idx in sorted(source_entries)],
        "source_entries_by_id": source_entries,
        "unresolved_reference_ids": unresolved,
        "cited_pages_by_paper": {paper: sorted(pages) for paper, pages in sorted(cited_pages_by_paper.items())},
    }


def check_citations(case: dict[str, Any], answer: str) -> dict[str, Any]:
    parsed = parse_citations(answer)
    expected_papers = set(case.get("expected_papers") or [])
    source_papers = [entry["source"] for entry in parsed["source_entries"]]
    unexpected = sorted(set(source_papers) - expected_papers) if expected_papers else []
    group_score = score_group_recall(case.get("gold_page_groups") or {}, parsed["cited_pages_by_paper"])
    return {
        **parsed,
        "has_inline_citations": bool(parsed["inline_reference_ids"]),
        "has_source_list": bool(parsed["source_entries"]),
        "all_inline_resolved": not parsed["unresolved_reference_ids"],
        "unexpected_source_papers": unexpected,
        "citation_group_recall": group_score,
        "semantic_support_manual_review": True,
    }


def run_manifest_check(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    skip: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if skip:
        return {"skipped": True, "passed": None, "returncode": None, "stdout": "", "stderr": ""}
    if dry_run:
        return {"skipped": True, "passed": None, "returncode": None, "stdout": "", "stderr": "dry-run"}
    command = [sys.executable, str(EVAL_DIR / "validate_eval_cases.py"), "--check-manifest"]
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=60,
        check=False,
    )
    return {
        "skipped": False,
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    status = run_git(["status", "--porcelain"])
    return {
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
    }


def make_run_id(layer: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + layer


def create_run_dir(output: Path, layer: str) -> Path:
    output = resolve_project_path(output)
    if output.name:
        output.mkdir(parents=True, exist_ok=True)
        return output
    run_dir = output / make_run_id(layer)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def redact_secret(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if re.search(r"api[_-]?key|token|secret", str(key), re.I) else redact_secret(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact_secret(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(api[_-]?key=)[^&\s]+", r"\1<redacted>", value, flags=re.I)
        value = re.sub(r"(token=)[^&\s]+", r"\1<redacted>", value, flags=re.I)
    return value


def completed_case_ids(results_path: Path) -> set[str]:
    completed: set[str] = set()
    if not results_path.exists():
        return completed
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = item.get("overall_status", "pass" if not item.get("error") else "error")
            if item.get("id") and status not in {"error"}:
                completed.add(item["id"])
    return completed


def summarize_statuses(results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(item.get("overall_status", "pass" if not item.get("error") else "error") for item in results)
    latencies = [float(item.get("latency_seconds") or 0) for item in results]
    return {
        "case_count": len(results),
        "status_counts": dict(statuses),
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "median_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "error_count": statuses.get("error", 0),
    }


def write_basic_report(run_dir: Path, title: str, config: dict[str, Any], summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"| {item.get('id')} | {item.get('overall_status', 'pass' if not item.get('error') else 'error')} | {item.get('error', '')} |"
        for item in results
    )
    text = (
        f"# {title}\n\n"
        f"- layer: {config.get('layer')}\n"
        f"- dataset_sha256: `{config.get('dataset_sha256')}`\n"
        f"- git_commit: `{config.get('git_commit')}`\n"
        f"- git_dirty: {config.get('git_dirty')}\n"
        f"- live_api: {config.get('live_api')}\n"
        f"- manifest_check_skipped: {config.get('manifest_check_skipped')}\n"
        f"- manifest_check_passed: {config.get('manifest_check_passed')}\n"
        f"- case_count: {summary.get('case_count')}\n\n"
        "## Summary\n\n"
        f"```json\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n```\n\n"
        "## Cases\n\n"
        "| id | status | error |\n|---|---|---|\n"
        f"{rows}\n"
    )
    (run_dir / "report.md").write_text(text, encoding="utf-8")


def make_run_config(
    *,
    layer: str,
    dataset_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    case_ids: list[str] | None = None,
    live_api: bool = False,
    top_k: int | None = None,
    source_mode: str | None = None,
    tool_mode: str | None = None,
    max_iterations: int | None = None,
    notes: str = "",
    manifest_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    git = git_metadata()
    return {
        "run_id": make_run_id(layer),
        "layer": layer,
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path) if Path(dataset_path).exists() else "",
        "manifest_path": str(manifest_path),
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "chat_model": os.environ.get("CHAT_MODEL"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        "tool_mode": tool_mode or os.environ.get("AGENT_TOOL_MODE"),
        "max_iterations": max_iterations,
        "top_k": top_k,
        "source_mode": source_mode,
        "case_ids": case_ids or [],
        "live_api": live_api,
        "notes": notes,
        "manifest_check_skipped": bool((manifest_check or {}).get("skipped")),
        "manifest_check_passed": (manifest_check or {}).get("passed"),
    }


def failure_tags(result: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if result.get("expected_intent") and result.get("actual_intent") and result["expected_intent"] != result["actual_intent"]:
        tags.append("routing_failure")
    if result.get("paper_exact_match") is False:
        tags.append("paper_selection_failure")
    if result.get("macro_group_recall") is not None and result.get("macro_group_recall", 1) < 0.5:
        tags.append("retrieval_failure")
    if result.get("deterministic_checks_pass") is False:
        tags.append("answer_failure")
    citation = result.get("citation_checks") or {}
    if citation and not citation.get("all_inline_resolved", True):
        tags.append("citation_format_failure")
    if result.get("hitl_sequence_correct") is False:
        tags.append("hitl_failure")
    return tags or ["unclassified"]
