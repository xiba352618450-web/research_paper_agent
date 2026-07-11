from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


OverallStatus = Literal["pass", "fail", "manual_review", "unsupported", "error"]


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


def compute_agent_success_layers(result: dict[str, Any]) -> tuple[bool, bool]:
    """Separate user-visible answer quality from strict pipeline diagnostics."""
    user_visible_success = all(
        result.get(field) is True
        for field in (
            "paper_selection_correct",
            "retrieval_correct",
            "answer_correct",
            "citation_format_correct",
            "hitl_correct",
        )
    )
    strict_pipeline_success = bool(
        user_visible_success
        and result.get("routing_correct") is True
        and result.get("evidence_status_correct") is True
    )
    return user_visible_success, strict_pipeline_success


def summarize_agent_success_layers(results: list[dict[str, Any]]) -> dict[str, float | None]:
    if not results:
        return {
            "user_visible_success_rate": None,
            "strict_pipeline_success_rate": None,
            "routing_diagnostic_accuracy": None,
            "evidence_status_diagnostic_accuracy": None,
        }

    evidence_observable = [item for item in results if item.get("evidence_status_correct") is not None]
    return {
        "user_visible_success_rate": sum(bool(item.get("user_visible_success")) for item in results) / len(results),
        "strict_pipeline_success_rate": sum(bool(item.get("strict_pipeline_success")) for item in results) / len(results),
        "routing_diagnostic_accuracy": sum(item.get("routing_correct") is True for item in results) / len(results),
        "evidence_status_diagnostic_accuracy": (
            sum(item.get("evidence_status_correct") is True for item in evidence_observable) / len(evidence_observable)
            if evidence_observable
            else None
        ),
    }


@dataclass
class EvalRunConfig:
    run_id: str
    layer: str
    dataset_path: str
    dataset_sha256: str
    manifest_path: str
    git_commit: str | None
    timestamp_utc: str
    python_version: str
    platform: str
    chat_model: str | None
    embedding_model: str | None
    tool_mode: str | None
    max_iterations: int | None
    top_k: int | None
    source_mode: str | None
    case_ids: list[str]
    live_api: bool
    notes: str = ""
    manifest_check_skipped: bool = False
    manifest_check_passed: bool | None = None
    git_dirty: bool | None = None


@dataclass
class RoutingCaseResult:
    id: str
    question: str
    expected_intent: str
    actual_intent: str
    intent_correct: bool
    expected_papers: list[str]
    actual_papers: list[str]
    detected_explicit_papers: list[str]
    paper_exact_match: bool
    paper_precision: float
    paper_recall: float
    paper_f1: float
    expected_aspects: list[str]
    actual_aspects: list[str]
    aspect_precision: float
    aspect_recall: float
    aspect_f1: float
    clarification_correct: bool | None
    latency_seconds: float
    error: str = ""


@dataclass
class RetrievalCaseResult:
    id: str
    question: str
    source_mode: str
    expected_papers: list[str]
    queried_papers: list[str]
    query_texts: list[str]
    top_k: int
    retrieved_chunks: list[dict[str, Any]]
    retrieved_pages_by_paper: dict[str, list[int]]
    initial_retrieved_pages_by_paper: dict[str, list[int]]
    initial_group_recall_by_paper: dict[str, float]
    initial_macro_group_recall: float | None
    initial_raw_page_recall_by_paper: dict[str, float]
    initial_macro_raw_page_recall: float | None
    expanded_retrieved_pages_by_paper: dict[str, list[int]]
    expanded_group_recall_by_paper: dict[str, float]
    expanded_macro_group_recall: float | None
    expanded_raw_page_recall_by_paper: dict[str, float]
    expanded_macro_raw_page_recall: float | None
    gold_pages: dict[str, list[int]]
    group_hits_by_paper: dict[str, list[dict[str, Any]]]
    group_recall_by_paper: dict[str, float]
    macro_group_recall: float | None
    raw_page_recall_by_paper: dict[str, float]
    macro_raw_page_recall: float | None
    wrong_source_count: int
    latency_seconds: float
    error: str = ""
    overall_status: OverallStatus | str = "pass"


@dataclass
class AgentCaseResult:
    id: str
    question: str
    answer: str
    trace: list[str]
    expected_intent: str
    actual_intent: str | None
    expected_papers: list[str]
    actual_papers: list[str]
    expected_aspects: list[str]
    actual_aspects: list[str]
    retrieved_pages_by_paper: dict[str, list[int]]
    group_recall_by_paper: dict[str, float]
    macro_group_recall: float | None
    evidence_sufficient_raw: bool | None
    evidence_reason: str
    observed_evidence_status: str | None
    expected_evidence_status: str | None
    human_review_triggered: bool
    expected_human_review: bool
    human_decisions_expected: list[str]
    human_decisions_observed: list[str]
    hitl_sequence_correct: bool | None
    answer_checks: dict[str, Any]
    fact_checks: dict[str, Any]
    citation_checks: dict[str, Any]
    routing_correct: bool
    paper_selection_correct: bool
    retrieval_correct: bool
    evidence_status_correct: bool | None
    evidence_status_observable: bool
    hitl_correct: bool
    answer_correct: bool
    citation_format_correct: bool
    deterministic_checks_pass: bool
    manual_review_required: bool
    user_visible_success: bool
    strict_pipeline_success: bool
    overall_status: OverallStatus | str
    iteration: int | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0
    error: str = ""
