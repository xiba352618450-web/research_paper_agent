from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_agent import infer_intent  # noqa: E402

from eval_common import (  # noqa: E402
    append_jsonl,
    collect_chunks_from_tool_result,
    completed_case_ids,
    create_run_dir,
    failure_tags,
    load_eval_cases,
    make_run_config,
    pages_by_paper_from_chunks,
    parse_case_ids,
    prepare_run_outputs,
    run_manifest_check,
    score_group_recall,
    score_raw_page_recall,
    summarize_chunk,
    summarize_statuses,
    write_basic_report,
    write_json,
)
from eval_models import RetrievalCaseResult, dataclass_to_dict  # noqa: E402


class ProgressReporter:
    def __init__(self, *, enabled: bool = True, stream: Any | None = None):
        self.enabled = enabled
        self.stream = stream or sys.stderr

    def message(self, text: str) -> None:
        if self.enabled:
            print(text, file=self.stream, flush=True)

    def bar(self, current: int, total: int, width: int = 24) -> str:
        if total <= 0:
            return "[" + "-" * width + "]"
        filled = min(width, max(0, round(width * current / total)))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def case_start(self, index: int, total: int, case: dict[str, Any], queried_papers: list[str]) -> None:
        papers = ", ".join(queried_papers) if queried_papers else "(no papers)"
        self.message(f"[retrieval] {index}/{total} start case={case['id']} papers={papers}")

    def source_start(self, case_id: str, index: int, total: int, paper: str) -> None:
        self.message(f"[retrieval]   query {index}/{total} case={case_id} source={paper}")

    def case_done(self, index: int, total: int, result: dict[str, Any]) -> None:
        pages = result.get("retrieved_pages_by_paper") or {}
        page_bits = []
        for source, source_pages in pages.items():
            preview = ",".join(str(page) for page in list(source_pages)[:6])
            suffix = "..." if len(source_pages) > 6 else ""
            page_bits.append(f"{source}:p{preview}{suffix}" if preview else f"{source}:none")
        page_summary = "; ".join(page_bits) if page_bits else "no_pages"
        self.message(
            f"[retrieval] {self.bar(index, total)} {index}/{total} done "
            f"case={result.get('id')} status={result.get('overall_status')} "
            f"chunks={len(result.get('retrieved_chunks') or [])} "
            f"group_recall={result.get('macro_group_recall')} "
            f"elapsed={float(result.get('latency_seconds') or 0):.1f}s "
            f"{page_summary}"
        )


def selected_sources_for_case(case: dict[str, Any], source_mode: str) -> list[str]:
    if case.get("retrieval_eval_mode") == "none":
        return []
    if source_mode == "oracle":
        return list(case.get("expected_papers") or [])
    _intent, papers = infer_intent(case["question"], recent_sources=[])
    return list(papers)


def retrieval_overall_status(error: str, macro_group_recall: float | None) -> str:
    if error:
        return "error"
    if macro_group_recall == 1.0:
        return "pass"
    return "fail"


def evaluate_case_live(
    case: dict[str, Any],
    tools: Any,
    source_mode: str,
    top_k: int,
    neighbor_radius: int,
    *,
    progress: ProgressReporter | None = None,
    queried_papers: list[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    queried_papers = queried_papers if queried_papers is not None else selected_sources_for_case(case, source_mode)
    initial_chunks: list[dict[str, Any]] = []
    neighbor_chunks: list[dict[str, Any]] = []
    query_texts: list[str] = []
    error = ""

    try:
        for paper_index, paper in enumerate(queried_papers, start=1):
            if progress is not None:
                progress.source_start(case["id"], paper_index, len(queried_papers), paper)
            query_texts.append(case["question"])
            result = tools.search_paper_tool(query=case["question"], source=paper, k=top_k)
            current_chunks = collect_chunks_from_tool_result(result)
            initial_chunks.extend(current_chunks)
            if neighbor_radius > 0:
                seen_pairs = {
                    (chunk.get("source"), chunk.get("page"))
                    for chunk in current_chunks
                    if chunk.get("source") and chunk.get("page") is not None
                }
                for source, page in sorted(seen_pairs):
                    neighbor_result = tools.get_neighbor_chunks_tool(source=source, page=int(page), radius=neighbor_radius)
                    neighbor_chunks.extend(collect_chunks_from_tool_result(neighbor_result))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    expanded_chunks = [*initial_chunks, *neighbor_chunks]
    initial_pages = pages_by_paper_from_chunks(initial_chunks)
    initial_group = score_group_recall(case.get("gold_page_groups") or {}, initial_pages)
    initial_raw = score_raw_page_recall(case.get("gold_pages") or {}, initial_pages)
    expanded_pages = pages_by_paper_from_chunks(expanded_chunks)
    expanded_group = score_group_recall(case.get("gold_page_groups") or {}, expanded_pages)
    expanded_raw = score_raw_page_recall(case.get("gold_pages") or {}, expanded_pages)
    expected = set(case.get("expected_papers") or [])
    wrong_source_count = sum(
        1
        for chunk in expanded_chunks
        if (chunk.get("source") or (chunk.get("metadata") or {}).get("source")) not in expected
    )
    overall_status = retrieval_overall_status(error, expanded_group["macro_group_recall"])

    result = RetrievalCaseResult(
        id=case["id"],
        question=case["question"],
        source_mode=source_mode,
        expected_papers=list(case.get("expected_papers") or []),
        queried_papers=queried_papers,
        query_texts=query_texts,
        top_k=top_k,
        retrieved_chunks=[summarize_chunk(chunk) for chunk in expanded_chunks],
        retrieved_pages_by_paper=expanded_pages,
        initial_retrieved_pages_by_paper=initial_pages,
        initial_group_recall_by_paper=initial_group["group_recall_by_paper"],
        initial_macro_group_recall=initial_group["macro_group_recall"],
        initial_raw_page_recall_by_paper=initial_raw["raw_page_recall_by_paper"],
        initial_macro_raw_page_recall=initial_raw["macro_raw_page_recall"],
        expanded_retrieved_pages_by_paper=expanded_pages,
        expanded_group_recall_by_paper=expanded_group["group_recall_by_paper"],
        expanded_macro_group_recall=expanded_group["macro_group_recall"],
        expanded_raw_page_recall_by_paper=expanded_raw["raw_page_recall_by_paper"],
        expanded_macro_raw_page_recall=expanded_raw["macro_raw_page_recall"],
        gold_pages=case.get("gold_pages") or {},
        group_hits_by_paper=expanded_group["group_hits_by_paper"],
        group_recall_by_paper=expanded_group["group_recall_by_paper"],
        macro_group_recall=expanded_group["macro_group_recall"],
        raw_page_recall_by_paper=expanded_raw["raw_page_recall_by_paper"],
        macro_raw_page_recall=expanded_raw["macro_raw_page_recall"],
        wrong_source_count=wrong_source_count,
        latency_seconds=time.perf_counter() - started,
        error=error,
        overall_status=overall_status,
    )
    data = dataclass_to_dict(result)
    data["failure_tags"] = failure_tags(data) if overall_status == "fail" else []
    return data


def dry_run_case(case: dict[str, Any], source_mode: str, top_k: int) -> dict[str, Any]:
    queried_papers = selected_sources_for_case(case, source_mode)
    return {
        "id": case["id"],
        "question": case["question"],
        "source_mode": source_mode,
        "expected_papers": list(case.get("expected_papers") or []),
        "queried_papers": queried_papers,
        "query_texts": [case["question"] for _ in queried_papers],
        "top_k": top_k,
        "retrieved_chunks": [],
        "retrieved_pages_by_paper": {},
        "initial_retrieved_pages_by_paper": {},
        "initial_group_recall_by_paper": {},
        "initial_macro_group_recall": None,
        "initial_raw_page_recall_by_paper": {},
        "initial_macro_raw_page_recall": None,
        "expanded_retrieved_pages_by_paper": {},
        "expanded_group_recall_by_paper": {},
        "expanded_macro_group_recall": None,
        "expanded_raw_page_recall_by_paper": {},
        "expanded_macro_raw_page_recall": None,
        "gold_pages": case.get("gold_pages") or {},
        "group_hits_by_paper": {},
        "group_recall_by_paper": {},
        "macro_group_recall": None,
        "raw_page_recall_by_paper": {},
        "macro_raw_page_recall": None,
        "wrong_source_count": 0,
        "latency_seconds": 0.0,
        "error": "",
        "overall_status": "dry_run",
    }


def run_retrieval_evaluation(
    args: argparse.Namespace,
    *,
    tools_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    case_ids = parse_case_ids(args.case_ids)
    cases = load_eval_cases(Path(args.cases), case_ids=case_ids)
    cases = [case for case in cases if case.get("retrieval_eval_mode") != "none"]
    run_dir = create_run_dir(Path(args.output), "retrieval")
    prepare_run_outputs(run_dir, resume=args.resume)
    results_path = run_dir / "cases.jsonl"
    errors_path = run_dir / "errors.jsonl"
    skipped = completed_case_ids(results_path) if args.resume else set()

    live_api = bool(args.run_live)
    progress = ProgressReporter(enabled=live_api and getattr(args, "progress", True))
    if live_api:
        progress.message("[retrieval] manifest check...")
    manifest = run_manifest_check(skip=args.skip_manifest_check, dry_run=not live_api)
    if live_api:
        progress.message(
            "[retrieval] manifest check "
            + ("skipped" if manifest.get("skipped") else "passed" if manifest.get("passed") else "failed")
        )
    if live_api and not manifest.get("skipped") and not manifest.get("passed"):
        config = make_run_config(
            layer="retrieval",
            dataset_path=Path(args.cases),
            case_ids=case_ids,
            live_api=live_api,
            top_k=args.top_k,
            source_mode=args.source_mode,
            notes=args.notes or "",
            manifest_check=manifest,
        )
        write_json(run_dir / "run_config.json", config)
        write_json(run_dir / "summary.json", {"error": "manifest_check_failed", "manifest": manifest})
        return {"run_dir": str(run_dir), "summary": {"error": "manifest_check_failed"}, "results": []}

    config = make_run_config(
        layer="retrieval",
        dataset_path=Path(args.cases),
        case_ids=case_ids,
        live_api=live_api,
        top_k=args.top_k,
        source_mode=args.source_mode,
        notes=args.notes or "",
        manifest_check=manifest,
    )
    write_json(run_dir / "run_config.json", config)

    tools = None
    if live_api:
        if tools_factory is None:
            from agent_tools import PaperTools  # noqa: WPS433

            tools_factory = PaperTools.from_env
        progress.message("[retrieval] initializing PaperTools / Chroma / embeddings...")
        tools = tools_factory()
        progress.message("[retrieval] tools ready")

    results: list[dict[str, Any]] = []
    cases_to_run = [case for case in cases if case["id"] not in skipped]
    for case_index, case in enumerate(cases_to_run, start=1):
        if live_api:
            queried_papers = selected_sources_for_case(case, args.source_mode)
            progress.case_start(case_index, len(cases_to_run), case, queried_papers)
            result = evaluate_case_live(
                case,
                tools,
                args.source_mode,
                args.top_k,
                args.neighbor_radius,
                progress=progress,
                queried_papers=queried_papers,
            )
            progress.case_done(case_index, len(cases_to_run), result)
        else:
            result = dry_run_case(case, args.source_mode, args.top_k)
        append_jsonl(results_path, result)
        if result.get("error"):
            append_jsonl(errors_path, result)
        results.append(result)

    if args.resume and results_path.exists():
        results = load_eval_cases(results_path)
    summary = summarize_statuses(results)
    if results:
        macro_group_values = [item["macro_group_recall"] for item in results if item.get("macro_group_recall") is not None]
        macro_raw_values = [item["macro_raw_page_recall"] for item in results if item.get("macro_raw_page_recall") is not None]
        initial_group_values = [
            item["initial_macro_group_recall"] for item in results if item.get("initial_macro_group_recall") is not None
        ]
        expanded_group_values = [
            item["expanded_macro_group_recall"] for item in results if item.get("expanded_macro_group_recall") is not None
        ]
        initial_raw_values = [
            item["initial_macro_raw_page_recall"] for item in results if item.get("initial_macro_raw_page_recall") is not None
        ]
        expanded_raw_values = [
            item["expanded_macro_raw_page_recall"] for item in results if item.get("expanded_macro_raw_page_recall") is not None
        ]
        summary.update(
            {
                "source_mode": args.source_mode,
                "top_k": args.top_k,
                "macro_group_recall": sum(macro_group_values) / len(macro_group_values)
                if macro_group_values
                else None,
                "macro_raw_page_recall": sum(macro_raw_values) / len(macro_raw_values)
                if macro_raw_values
                else None,
                "initial_macro_group_recall": sum(initial_group_values) / len(initial_group_values)
                if initial_group_values
                else None,
                "expanded_macro_group_recall": sum(expanded_group_values) / len(expanded_group_values)
                if expanded_group_values
                else None,
                "initial_macro_raw_page_recall": sum(initial_raw_values) / len(initial_raw_values)
                if initial_raw_values
                else None,
                "expanded_macro_raw_page_recall": sum(expanded_raw_values) / len(expanded_raw_values)
                if expanded_raw_values
                else None,
            }
        )
    write_json(run_dir / "summary.json", summary)
    write_basic_report(run_dir, "Retrieval Evaluation", config, summary, results)
    if live_api:
        progress.message(f"[retrieval] finished run_dir={run_dir}")
    return {"run_dir": str(run_dir), "summary": summary, "results": results}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with explicit live-run gating.")
    parser.add_argument("--cases", default=str(Path("eval") / "eval_cases_v1.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-mode", choices=["oracle", "predicted"], default="oracle")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbor-radius", type=int, default=0)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="Disable live retrieval progress output.")
    parser.set_defaults(progress=True)
    parser.add_argument("--skip-manifest-check", action="store_true")
    parser.add_argument("--notes", default="")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_retrieval_evaluation(args)
    print(json.dumps({"run_dir": result["run_dir"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 1 if result["summary"].get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
