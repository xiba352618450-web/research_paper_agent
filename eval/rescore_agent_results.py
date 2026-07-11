from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_common import append_jsonl, load_eval_cases, summarize_statuses, write_json  # noqa: E402
from eval_models import compute_agent_success_layers, summarize_agent_success_layers  # noqa: E402


def rescore_case(record: dict[str, Any]) -> dict[str, Any]:
    """Add current E2E presentation fields without touching captured outputs."""
    rescored = dict(record)
    user_visible_success, strict_pipeline_success = compute_agent_success_layers(rescored)
    rescored["user_visible_success"] = user_visible_success
    rescored["strict_pipeline_success"] = strict_pipeline_success

    tags = list(rescored.get("failure_tags") or [])
    if user_visible_success and not strict_pipeline_success and "pipeline_diagnostic_mismatch" not in tags:
        tags.append("pipeline_diagnostic_mismatch")
    rescored["failure_tags"] = tags
    return rescored


def rescore_agent_results(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到已有评测结果：{source}")
    if destination == source.parent:
        raise ValueError("输出目录必须不同于输入 cases.jsonl 所在目录，避免覆盖原始结果。")

    records = load_eval_cases(source)
    rescored = [rescore_case(record) for record in records]
    destination.mkdir(parents=True, exist_ok=True)
    cases_path = destination / "cases.jsonl"
    if cases_path.exists():
        cases_path.unlink()
    for record in rescored:
        append_jsonl(cases_path, record)

    summary = summarize_statuses(rescored)
    summary.update(summarize_agent_success_layers(rescored))
    summary["rescored_from"] = str(source)
    write_json(destination / "summary.json", summary)
    return {"output_dir": str(destination), "summary": summary, "results": rescored}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rescore existing agent evaluation JSONL without initializing an Agent.")
    parser.add_argument("--input", required=True, help="已有 cases.jsonl 路径")
    parser.add_argument("--output", required=True, help="新的输出目录")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = rescore_agent_results(args.input, args.output)
    print(json.dumps({"output_dir": result["output_dir"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
