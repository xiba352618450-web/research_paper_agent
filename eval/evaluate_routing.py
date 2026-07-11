from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_agent import (  # noqa: E402
    PAPER_ANALYSIS_REQUIRED_ASPECTS,
    detect_explicit_papers,
    extract_comparison_aspects,
    infer_intent,
)

from eval_common import (  # noqa: E402
    append_jsonl,
    completed_case_ids,
    create_run_dir,
    load_eval_cases,
    make_run_config,
    normalize_aspects,
    parse_case_ids,
    prepare_run_outputs,
    set_metrics,
    summarize_statuses,
    write_basic_report,
    write_json,
)
from eval_models import RoutingCaseResult, dataclass_to_dict  # noqa: E402


def actual_aspects_for_intent(intent: str, question: str) -> list[str]:
    if intent == "paper_comparison":
        return extract_comparison_aspects(question)
    if intent == "paper_analysis":
        return list(PAPER_ANALYSIS_REQUIRED_ASPECTS)
    return []


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    error = ""
    actual_intent = ""
    actual_papers: list[str] = []
    detected_explicit_papers: list[str] = []
    actual_aspects: list[str] = []
    try:
        actual_intent, actual_papers = infer_intent(case["question"], recent_sources=[])
        detected_explicit_papers = detect_explicit_papers(case["question"])
        actual_aspects = actual_aspects_for_intent(actual_intent, case["question"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    expected_papers = list(case.get("expected_papers") or [])
    expected_aspects = normalize_aspects(case.get("expected_aspects") or [])
    normalized_actual_aspects = normalize_aspects(actual_aspects)
    paper_metrics = set_metrics(expected_papers, actual_papers)
    aspect_metrics = set_metrics(expected_aspects, normalized_actual_aspects)
    clarification_correct = None
    if case.get("expected_intent") == "clarification":
        clarification_correct = actual_intent == "clarification" and not actual_papers

    result = RoutingCaseResult(
        id=case["id"],
        question=case["question"],
        expected_intent=case.get("expected_intent", ""),
        actual_intent=actual_intent,
        intent_correct=actual_intent == case.get("expected_intent"),
        expected_papers=expected_papers,
        actual_papers=actual_papers,
        detected_explicit_papers=detected_explicit_papers,
        paper_exact_match=paper_metrics["exact_match"],
        paper_precision=paper_metrics["precision"],
        paper_recall=paper_metrics["recall"],
        paper_f1=paper_metrics["f1"],
        expected_aspects=expected_aspects,
        actual_aspects=normalized_actual_aspects,
        aspect_precision=aspect_metrics["precision"],
        aspect_recall=aspect_metrics["recall"],
        aspect_f1=aspect_metrics["f1"],
        clarification_correct=clarification_correct,
        latency_seconds=time.perf_counter() - started,
        error=error,
    )
    data = dataclass_to_dict(result)
    data["overall_status"] = "error" if error else ("pass" if data["intent_correct"] and data["paper_exact_match"] else "fail")
    return data


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    base = summarize_statuses(results)
    if not results:
        return base
    applicable_paper_items = [item for item in results if bool(item.get("expected_papers"))]
    observable_aspect_items = [
        item for item in results if item.get("actual_intent") in {"paper_analysis", "paper_comparison"}
    ]

    def average(items: list[dict[str, Any]], key: str) -> float | None:
        if not items:
            return None
        return sum(float(item.get(key) or 0) for item in items) / len(items)

    base.update(
        {
            "intent_accuracy": sum(1 for item in results if item.get("intent_correct")) / len(results),
            "paper_exact_match_rate": sum(1 for item in results if item.get("paper_exact_match")) / len(results),
            "paper_macro_precision_all_cases": average(results, "paper_precision"),
            "paper_macro_recall_all_cases": average(results, "paper_recall"),
            "paper_macro_f1_all_cases": average(results, "paper_f1"),
            "paper_macro_precision_applicable_cases": average(applicable_paper_items, "paper_precision"),
            "paper_macro_recall_applicable_cases": average(applicable_paper_items, "paper_recall"),
            "paper_macro_f1_applicable_cases": average(applicable_paper_items, "paper_f1"),
            "paper_applicable_case_count": len(applicable_paper_items),
            "initial_aspect_macro_precision_all_cases": average(results, "aspect_precision"),
            "initial_aspect_macro_recall_all_cases": average(results, "aspect_recall"),
            "initial_aspect_macro_f1_all_cases": average(results, "aspect_f1"),
            "initial_aspect_macro_precision_observable_cases": average(observable_aspect_items, "aspect_precision"),
            "initial_aspect_macro_recall_observable_cases": average(observable_aspect_items, "aspect_recall"),
            "initial_aspect_macro_f1_observable_cases": average(observable_aspect_items, "aspect_f1"),
            "aspect_observable_case_count": len(observable_aspect_items),
            "aspect_observable_coverage": len(observable_aspect_items) / len(results),
        }
    )
    # Backward-compatible aliases now explicitly point to the all-case values.
    base["paper_macro_precision"] = base["paper_macro_precision_all_cases"]
    base["paper_macro_recall"] = base["paper_macro_recall_all_cases"]
    base["paper_macro_f1"] = base["paper_macro_f1_all_cases"]
    base["initial_aspect_macro_precision"] = base["initial_aspect_macro_precision_all_cases"]
    base["initial_aspect_macro_recall"] = base["initial_aspect_macro_recall_all_cases"]
    base["initial_aspect_macro_f1"] = base["initial_aspect_macro_f1_all_cases"]
    clarification_items = [item for item in results if item.get("clarification_correct") is not None]
    base["clarification_accuracy"] = (
        sum(1 for item in clarification_items if item.get("clarification_correct")) / len(clarification_items)
        if clarification_items
        else None
    )
    confusion: dict[str, dict[str, int]] = {}
    for item in results:
        expected = item.get("expected_intent", "")
        actual = item.get("actual_intent", "")
        confusion.setdefault(expected, {})
        confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
    base["confusion_matrix"] = confusion
    return base


def run_routing_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    case_ids = parse_case_ids(args.case_ids)
    cases = load_eval_cases(Path(args.cases), case_ids=case_ids)
    run_dir = create_run_dir(Path(args.output), "routing")
    prepare_run_outputs(run_dir, resume=args.resume)
    results_path = run_dir / "cases.jsonl"
    errors_path = run_dir / "errors.jsonl"
    skipped = completed_case_ids(results_path) if args.resume else set()
    config = make_run_config(
        layer="routing",
        dataset_path=Path(args.cases),
        case_ids=case_ids,
        live_api=False,
        notes=args.notes or "",
        manifest_check={"skipped": bool(args.dry_run), "passed": None},
    )
    write_json(run_dir / "run_config.json", config)

    results: list[dict[str, Any]] = []
    for case in cases:
        if case["id"] in skipped:
            continue
        result = evaluate_case(case)
        append_jsonl(results_path, result)
        if result.get("error"):
            append_jsonl(errors_path, result)
        results.append(result)

    if args.resume and results_path.exists():
        results = load_eval_cases(results_path)
    summary = build_summary(results)
    write_json(run_dir / "summary.json", summary)
    write_basic_report(run_dir, "Routing Evaluation", config, summary, results)
    return {"run_dir": str(run_dir), "summary": summary, "results": results}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deterministic routing without API calls.")
    parser.add_argument("--cases", default=str(Path("eval") / "eval_cases_v1.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--notes", default="")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_routing_evaluation(args)
    print(json.dumps({"run_dir": result["run_dir"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
