from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_common import (  # noqa: E402
    append_jsonl,
    check_citations,
    collect_chunks_from_tool_result,
    completed_case_ids,
    create_run_dir,
    failure_tags,
    load_eval_cases,
    make_run_config,
    normalize_aspects,
    pages_by_paper_from_chunks,
    parse_case_ids,
    prepare_run_outputs,
    run_manifest_check,
    score_answer,
    score_group_recall,
    summarize_statuses,
    write_basic_report,
    write_json,
)
from eval_models import (  # noqa: E402
    AgentCaseResult,
    compute_agent_success_layers,
    dataclass_to_dict,
    summarize_agent_success_layers,
)
from eval_recording import HITLAdapter, RecordingPaperTools  # noqa: E402


SMOKE_CASE_IDS = ["eval-001", "eval-005", "eval-013", "eval-018", "eval-020"]


def select_agent_cases(cases: list[dict[str, Any]], case_ids: list[str], run_all: bool) -> list[dict[str, Any]]:
    if case_ids:
        wanted = set(case_ids)
    elif run_all:
        wanted = {case["id"] for case in cases}
    else:
        wanted = set(SMOKE_CASE_IDS)
    return [case for case in cases if case["id"] in wanted]


def _default_tools_factory() -> Any:
    from agent_tools import PaperTools  # noqa: WPS433

    return PaperTools.from_env()


def _default_agent_factory(tools: Any, args: argparse.Namespace) -> Any:
    from paper_agent import PaperAgent  # noqa: WPS433

    kwargs: dict[str, Any] = {}
    if args.tool_mode:
        kwargs["tool_mode"] = args.tool_mode
    if args.max_iterations is not None:
        kwargs["max_iterations"] = args.max_iterations
    return PaperAgent(tools, **kwargs)


def _actual_papers_from_state(state: dict[str, Any]) -> list[str]:
    for key in ("comparison_papers", "selected_papers"):
        papers = [str(item) for item in state.get(key) or [] if str(item).strip()]
        if papers:
            return list(dict.fromkeys(papers))
    docs = state.get("retrieved_docs") or []
    return list(
        dict.fromkeys(
            [
                str(doc.get("source") or (doc.get("metadata") or {}).get("source"))
                for doc in docs
                if doc.get("source") or (doc.get("metadata") or {}).get("source")
            ]
        )
    )


def _actual_papers_from_tool_calls(records: list[dict[str, Any]]) -> list[str]:
    papers: list[str] = []
    for record in records or []:
        for source in (record.get("source_pages") or {}).keys():
            source_text = str(source).strip()
            if source_text and source_text not in papers:
                papers.append(source_text)
    return papers


def _actual_papers_from_citations(citation_checks: dict[str, Any]) -> list[str]:
    papers: list[str] = []
    for entry in citation_checks.get("source_entries") or []:
        source = str(entry.get("source") or "").strip()
        if source and source not in papers:
            papers.append(source)
    return papers


def _actual_aspects_from_state(state: dict[str, Any]) -> list[str]:
    for key in ("comparison_aspects", "required_aspects", "covered_aspects"):
        aspects = normalize_aspects(state.get(key) or [])
        if aspects:
            return aspects
    return []


def _observed_evidence_status(case: dict[str, Any], state: dict[str, Any]) -> str | None:
    if case.get("retrieval_eval_mode") == "none":
        return "not_applicable"
    for key in ("evidence_status", "human_review_status", "evidence_sufficient_with_gaps"):
        value = state.get(key)
        if value:
            return str(value)
    return None


def _tool_pages(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    merged: dict[str, set[int]] = {}
    for record in records:
        for source, pages in (record.get("source_pages") or {}).items():
            merged.setdefault(source, set()).update(int(page) for page in pages)
    return {source: sorted(pages) for source, pages in sorted(merged.items())}


def evaluate_case_live(
    case: dict[str, Any],
    args: argparse.Namespace,
    *,
    tools_factory: Callable[[], Any],
    agent_factory: Callable[[Any, argparse.Namespace], Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    old_stream = os.environ.get("STREAM_FINAL_ANSWER")
    old_animation = os.environ.get("SHOW_PROGRESS_ANIMATION")
    os.environ["STREAM_FINAL_ANSWER"] = "false"
    os.environ["SHOW_PROGRESS_ANIMATION"] = "false"

    recording_tools: RecordingPaperTools | None = None
    result: dict[str, Any] | None = None
    initial_result: dict[str, Any] | None = None
    state: dict[str, Any] = {}
    answer = ""
    trace: list[str] = []
    error = ""
    hitl_adapter = HITLAdapter()
    unsupported_hitl = False

    try:
        base_tools = tools_factory()
        recording_tools = base_tools if isinstance(base_tools, RecordingPaperTools) else RecordingPaperTools(base_tools=base_tools)
        agent = agent_factory(recording_tools, args)
        initial_result = agent.ask(case["question"], trace_enabled=True)
        result = initial_result
        state = dict(initial_result.get("state") or {})
        if case.get("expected_human_review"):
            if not HITLAdapter.has_interrupt(state):
                error = "expected_interrupt_not_observed"
            elif not hasattr(agent, "resume") or not callable(agent.resume):
                unsupported_hitl = True
                error = "unsupported_hitl_interface"
            else:
                resumed, resume_error = hitl_adapter.resume_with_decisions(
                    agent,
                    list(case.get("human_decisions") or []),
                    initial_state=state,
                    trace_enabled=True,
                )
                if resume_error:
                    error = resume_error
                if resumed is not None:
                    result = resumed
                    state = dict(resumed.get("state") or state)
        answer = str((result or {}).get("answer") or state.get("final_answer") or "")
        trace = list((result or {}).get("trace") or state.get("trace") or [])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if old_stream is None:
            os.environ.pop("STREAM_FINAL_ANSWER", None)
        else:
            os.environ["STREAM_FINAL_ANSWER"] = old_stream
        if old_animation is None:
            os.environ.pop("SHOW_PROGRESS_ANIMATION", None)
        else:
            os.environ["SHOW_PROGRESS_ANIMATION"] = old_animation

    retrieved_docs = state.get("retrieved_docs") or []
    retrieved_pages = pages_by_paper_from_chunks(retrieved_docs)
    group = score_group_recall(case.get("gold_page_groups") or {}, retrieved_pages)
    all_tool_pages = _tool_pages((recording_tools.records if recording_tools else []))
    all_tool_group = score_group_recall(case.get("gold_page_groups") or {}, all_tool_pages)
    answer_checks = score_answer(case, answer)
    citation_checks = check_citations(case, answer)
    initial_state = dict((initial_result or {}).get("state") or {})
    human_review_triggered = HITLAdapter.has_interrupt(initial_state)
    expected_decisions = list(case.get("human_decisions") or [])
    observed_decisions = [record.get("decision") for record in hitl_adapter.records]
    hitl_sequence_correct = None
    if case.get("expected_human_review"):
        hitl_sequence_correct = expected_decisions == observed_decisions and all(
            record.get("same_thread") is True for record in hitl_adapter.records
        )

    actual_intent = state.get("intent")
    tool_call_records = recording_tools.records if recording_tools else []
    actual_papers_from_state = _actual_papers_from_state(state)
    actual_papers_from_tool_calls = _actual_papers_from_tool_calls(tool_call_records)
    actual_papers_from_citations = _actual_papers_from_citations(citation_checks)
    actual_papers = actual_papers_from_tool_calls or actual_papers_from_state
    actual_aspects = _actual_aspects_from_state(state)
    expected_papers = list(case.get("expected_papers") or [])
    observed_evidence_status = _observed_evidence_status(case, state)
    evidence_status_observable = observed_evidence_status is not None
    evidence_status_correct = (
        observed_evidence_status == case.get("expected_evidence_status")
        if evidence_status_observable
        else None
    )
    routing_correct = actual_intent == case.get("expected_intent")
    paper_selection_correct = set(actual_papers) == set(expected_papers)
    retrieval_correct = case.get("retrieval_eval_mode") == "none" or group["macro_group_recall"] == 1.0
    if case.get("expected_human_review"):
        hitl_correct = bool(
            human_review_triggered
            and hitl_sequence_correct is True
            and all(record.get("same_thread") is True for record in hitl_adapter.records)
        )
    else:
        hitl_correct = not human_review_triggered
    answer_correct = bool(answer_checks["deterministic_pass"])
    citation_format_correct = bool(citation_checks["all_inline_resolved"])
    deterministic_pass = bool(answer_correct and citation_format_correct)
    manual_review_required = bool(
        answer_checks["manual_review_required"] or citation_checks["semantic_support_manual_review"]
    )
    user_visible_success, strict_pipeline_success = compute_agent_success_layers(
        {
            "paper_selection_correct": paper_selection_correct,
            "retrieval_correct": retrieval_correct,
            "answer_correct": answer_correct,
            "citation_format_correct": citation_format_correct,
            "hitl_correct": hitl_correct,
            "routing_correct": routing_correct,
            "evidence_status_correct": evidence_status_correct,
        }
    )
    core_failures = [
        routing_correct is False,
        paper_selection_correct is False,
        retrieval_correct is False,
        evidence_status_correct is False,
        hitl_correct is False,
        answer_correct is False,
        citation_format_correct is False,
    ]
    controlled_failures = {"expected_interrupt_not_observed", "next_interrupt_not_observed"}
    if unsupported_hitl or error == "unsupported_hitl_interface":
        overall_status = "unsupported" if unsupported_hitl or error == "unsupported_hitl_interface" else "error"
    elif error and error not in controlled_failures:
        overall_status = "error"
    elif any(core_failures):
        overall_status = "fail"
    elif manual_review_required or evidence_status_correct is None:
        overall_status = "manual_review"
    else:
        overall_status = "pass"

    result_obj = AgentCaseResult(
        id=case["id"],
        question=case["question"],
        answer=answer,
        trace=trace,
        expected_intent=case.get("expected_intent", ""),
        actual_intent=actual_intent,
        expected_papers=expected_papers,
        actual_papers=actual_papers,
        expected_aspects=normalize_aspects(case.get("expected_aspects") or []),
        actual_aspects=actual_aspects,
        retrieved_pages_by_paper=retrieved_pages,
        group_recall_by_paper=group["group_recall_by_paper"],
        macro_group_recall=group["macro_group_recall"],
        evidence_sufficient_raw=state.get("evidence_sufficient"),
        evidence_reason=str(state.get("evidence_reason") or ""),
        observed_evidence_status=observed_evidence_status,
        expected_evidence_status=case.get("expected_evidence_status"),
        human_review_triggered=human_review_triggered,
        expected_human_review=bool(case.get("expected_human_review")),
        human_decisions_expected=expected_decisions,
        human_decisions_observed=observed_decisions,
        hitl_sequence_correct=hitl_sequence_correct,
        answer_checks=answer_checks,
        fact_checks=answer_checks["fact_checks"],
        citation_checks=citation_checks,
        routing_correct=routing_correct,
        paper_selection_correct=paper_selection_correct,
        retrieval_correct=retrieval_correct,
        evidence_status_correct=evidence_status_correct,
        evidence_status_observable=evidence_status_observable,
        hitl_correct=hitl_correct,
        answer_correct=answer_correct,
        citation_format_correct=citation_format_correct,
        deterministic_checks_pass=deterministic_pass,
        manual_review_required=manual_review_required,
        user_visible_success=user_visible_success,
        strict_pipeline_success=strict_pipeline_success,
        overall_status=overall_status,
        iteration=state.get("iteration"),
        tool_calls=tool_call_records,
        latency_seconds=time.perf_counter() - started,
        error=error,
    )
    data = dataclass_to_dict(result_obj)
    data["actual_papers_from_state"] = actual_papers_from_state
    data["actual_papers_from_tool_calls"] = actual_papers_from_tool_calls
    data["actual_papers_from_citations"] = actual_papers_from_citations
    data["actual_papers_scoring_source"] = "tool_calls" if actual_papers_from_tool_calls else "state"
    data["all_tool_calls_retrieved_pages_by_paper"] = all_tool_pages
    data["all_tool_calls_group_recall_by_paper"] = all_tool_group["group_recall_by_paper"]
    data["all_tool_calls_macro_group_recall"] = all_tool_group["macro_group_recall"]
    tags = failure_tags(data) if overall_status in {"fail", "error", "unsupported"} else []
    if user_visible_success and not strict_pipeline_success and "pipeline_diagnostic_mismatch" not in tags:
        tags.append("pipeline_diagnostic_mismatch")
    data["failure_tags"] = tags
    return data


def dry_run_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_intent": case.get("expected_intent"),
        "expected_papers": case.get("expected_papers") or [],
        "expected_human_review": bool(case.get("expected_human_review")),
        "human_decisions_expected": case.get("human_decisions") or [],
        "overall_status": "dry_run",
        "latency_seconds": 0.0,
        "error": "",
    }


def run_agent_evaluation(
    args: argparse.Namespace,
    *,
    tools_factory: Callable[[], Any] | None = None,
    agent_factory: Callable[[Any, argparse.Namespace], Any] | None = None,
) -> dict[str, Any]:
    case_ids = parse_case_ids(args.case_ids)
    cases = load_eval_cases(Path(args.cases))
    cases = select_agent_cases(cases, case_ids, bool(args.all))
    run_dir = create_run_dir(Path(args.output), "agent")
    prepare_run_outputs(run_dir, resume=args.resume)
    results_path = run_dir / "cases.jsonl"
    errors_path = run_dir / "errors.jsonl"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    skipped = completed_case_ids(results_path) if args.resume else set()

    live_api = bool(args.run_live)
    manifest = run_manifest_check(skip=args.skip_manifest_check, dry_run=not live_api)
    if live_api and not manifest.get("skipped") and not manifest.get("passed"):
        config = make_run_config(
            layer="agent",
            dataset_path=Path(args.cases),
            case_ids=[case["id"] for case in cases],
            live_api=live_api,
            tool_mode=args.tool_mode,
            max_iterations=args.max_iterations,
            notes=args.notes or "",
            manifest_check=manifest,
        )
        write_json(run_dir / "run_config.json", config)
        write_json(run_dir / "summary.json", {"error": "manifest_check_failed", "manifest": manifest})
        return {"run_dir": str(run_dir), "summary": {"error": "manifest_check_failed"}, "results": []}

    config = make_run_config(
        layer="agent",
        dataset_path=Path(args.cases),
        case_ids=[case["id"] for case in cases],
        live_api=live_api,
        tool_mode=args.tool_mode,
        max_iterations=args.max_iterations,
        notes=args.notes or "",
        manifest_check=manifest,
    )
    write_json(run_dir / "run_config.json", config)

    if live_api:
        tools_factory = tools_factory or _default_tools_factory
        agent_factory = agent_factory or _default_agent_factory

    results: list[dict[str, Any]] = []
    for case in cases:
        if case["id"] in skipped:
            continue
        if live_api:
            result = evaluate_case_live(
                case,
                args,
                tools_factory=tools_factory or _default_tools_factory,
                agent_factory=agent_factory or _default_agent_factory,
            )
        else:
            result = dry_run_case(case)
        append_jsonl(results_path, result)
        write_json(raw_dir / f"{case['id']}.json", result)
        if result.get("error"):
            append_jsonl(errors_path, result)
        results.append(result)

    if args.resume and results_path.exists():
        results = load_eval_cases(results_path)
    summary = summarize_statuses(results)
    if results:
        summary.update(
            {
                "smoke_case_ids": SMOKE_CASE_IDS,
                "selected_case_ids": [case["id"] for case in cases],
                "deterministic_pass_rate": sum(
                    1 for item in results if item.get("deterministic_checks_pass")
                )
                / len(results)
                if live_api
                else None,
                "manual_review_rate": sum(1 for item in results if item.get("manual_review_required")) / len(results)
                if live_api
                else None,
                "human_review_trigger_accuracy": _human_review_trigger_accuracy(results) if live_api else None,
                **(
                    summarize_agent_success_layers(results) if live_api else {
                    "user_visible_success_rate": None,
                    "strict_pipeline_success_rate": None,
                    "routing_diagnostic_accuracy": None,
                    "evidence_status_diagnostic_accuracy": None,
                    }
                ),
            }
        )
    write_json(run_dir / "summary.json", summary)
    write_basic_report(run_dir, "Agent End-to-End Evaluation", config, summary, results)
    return {"run_dir": str(run_dir), "summary": summary, "results": results}


def _human_review_trigger_accuracy(results: list[dict[str, Any]]) -> float | None:
    relevant = [item for item in results if "human_review_triggered" in item]
    if not relevant:
        return None
    return sum(
        1 for item in relevant if bool(item.get("human_review_triggered")) == bool(item.get("expected_human_review"))
    ) / len(relevant)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end agent evaluation with dry-run default.")
    parser.add_argument("--cases", default=str(Path("eval") / "eval_cases_v1.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--skip-manifest-check", action="store_true")
    parser.add_argument("--tool-mode", default="")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--notes", default="")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_agent_evaluation(args)
    print(json.dumps({"run_dir": result["run_dir"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 1 if result["summary"].get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
