from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

@dataclass(frozen=True)
class UISource:
    reference_id: int | None
    paper: str
    page: int | None
    text: str


@dataclass(frozen=True)
class UIPendingHITL:
    reason: str
    missing_aspects: list[str]
    actions: list[str]
    action_labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class UIRunResult:
    answer: str | None
    sources: list[UISource]
    trace: list[str]
    trace_summary: list[str]
    pending_hitl: UIPendingHITL | None
    status: str
    error: str | None
    selected_papers: list[str] = field(default_factory=list)
    unavailable_papers: list[str] = field(default_factory=list)
    available_papers: list[str] = field(default_factory=list)
    debug_error: str | None = None


SOURCE_HEADER_RE = re.compile(r"【资料来源】|\[资料来源\]")
SOURCE_WITH_PAGE_RE = re.compile(r"^\s*\[(\d+)]\s*(.+?)[，,]\s*第\s*(\d+)\s*页\s*$")
SOURCE_RE = re.compile(r"^\s*\[(\d+)]\s*(.+?)\s*$")
WINDOWS_PATH_RE = re.compile(r"(?i)\b[a-z]:\\[^\s，。；;]+")
UNIX_PATH_RE = re.compile(r"(?<!\w)/(?:[^\s/]+/)+[^\s，。；;]*")
API_KEY_ASSIGNMENT_RE = re.compile(r"(?i)(OPENAI_API_KEY\s*[:=]\s*)\S+")
SECRET_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")
CHUNK_ID_RE = re.compile(r"\bchunk-[A-Za-z0-9_-]+\b", re.IGNORECASE)
NUMBERED_PDF_RE = re.compile(r"^\d+\s+")


def paper_display_name(paper: str) -> str:
    name = Path(str(paper or "")).name
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return NUMBERED_PDF_RE.sub("", name).strip()


def load_available_papers(agent: Any) -> list[str]:
    tools = getattr(agent, "tools", None)
    list_papers = getattr(tools, "list_papers_tool", None)
    if not callable(list_papers):
        raise RuntimeError("当前 Agent 未提供论文列表接口。")
    payload = list_papers()
    papers = payload.get("papers") if isinstance(payload, dict) else None
    if not isinstance(papers, list):
        raise RuntimeError("论文列表返回格式不完整。")
    return [str(paper) for paper in papers if str(paper).strip()]


def parse_sources(answer: str) -> list[UISource]:
    parts = SOURCE_HEADER_RE.split(str(answer or ""), maxsplit=1)
    if len(parts) < 2:
        return []
    sources: list[UISource] = []
    for raw_line in parts[1].splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SOURCE_WITH_PAGE_RE.match(line)
        if match:
            sources.append(
                UISource(
                    reference_id=int(match.group(1)),
                    paper=match.group(2).strip(),
                    page=int(match.group(3)),
                    text=line,
                )
            )
            continue
        match = SOURCE_RE.match(line)
        if match:
            sources.append(
                UISource(
                    reference_id=int(match.group(1)),
                    paper=match.group(2).strip(),
                    page=None,
                    text=line,
                )
            )
    return sources


def sanitize_trace(trace: Iterable[Any]) -> list[str]:
    cleaned: list[str] = []
    for item in trace or []:
        line = str(item or "").strip()
        if not line:
            continue
        if line.startswith(("{", "[")) and len(line) > 800:
            continue
        line = API_KEY_ASSIGNMENT_RE.sub(r"\1[已隐藏]", line)
        line = SECRET_TOKEN_RE.sub("[已隐藏密钥]", line)
        line = WINDOWS_PATH_RE.sub("[本地路径]", line)
        line = UNIX_PATH_RE.sub("[本地路径]", line)
        line = CHUNK_ID_RE.sub("[内部片段]", line)
        if "checkpoint" in line.casefold() and len(line) > 500:
            continue
        cleaned.append(line[:800])
    return cleaned


def summarize_trace(trace: Iterable[Any], selected_papers: list[str] | None = None) -> list[str]:
    summary: list[str] = []
    if selected_papers:
        labels = "、".join(paper_display_name(paper).split(" - ", 1)[0] for paper in selected_papers)
        summary.append(f"自动选择论文：{labels}")

    mappings = (
        ("【任务判断】", "任务判断"),
        ("【Planner动作】", "执行动作"),
        ("【观察结果】", "检索结果"),
        ("【证据覆盖】", "证据覆盖"),
        ("【证据状态】", "证据状态"),
        ("【缺失信息】", "缺失信息"),
        ("【缺失细节】", "缺失信息"),
        ("【改写查询】", "查询改写"),
        ("【人工选择】", "人工决策"),
        ("【图状态】", "执行状态"),
        ("【最终回答】", "最终状态"),
    )
    for line in sanitize_trace(trace):
        for marker, label in mappings:
            if line.startswith(marker):
                value = line[len(marker) :].strip()
                rendered = f"{label}：{value}" if value else label
                if rendered not in summary:
                    summary.append(rendered)
                break
    return summary[:12]


def _extract_pending_hitl(state: dict[str, Any]) -> UIPendingHITL | None:
    payload = state.get("last_interrupt_payload")
    if not isinstance(payload, dict):
        payload = {}
    if state.get("awaiting_human") is not True and not payload:
        return None

    options = payload.get("options") if isinstance(payload.get("options"), list) else []
    actions: list[str] = []
    labels: dict[str, str] = {}
    for option in options:
        if not isinstance(option, dict):
            continue
        action = str(option.get("action") or "").strip()
        if not action or action in actions:
            continue
        actions.append(action)
        labels[action] = str(option.get("label") or action).strip()
    missing = payload.get("missing_aspects") or state.get("missing_aspects") or []
    return UIPendingHITL(
        reason=str(payload.get("reason") or state.get("human_review_reason") or "需要人工决定下一步。"),
        missing_aspects=[str(item) for item in missing if str(item).strip()],
        actions=actions,
        action_labels=labels,
    )


def _clean_paper_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(dict.fromkeys(str(item) for item in value if str(item).strip()))


def normalize_agent_result(
    raw_result: dict[str, Any] | None,
    *,
    available_papers: list[str] | None = None,
) -> UIRunResult:
    raw = raw_result if isinstance(raw_result, dict) else {}
    state = raw.get("state") if isinstance(raw.get("state"), dict) else {}
    trace = sanitize_trace(raw.get("trace") or state.get("trace") or [])
    selected = _clean_paper_list(state.get("selected_papers") or state.get("comparison_papers"))
    unavailable = _clean_paper_list(raw.get("unavailable_papers") or state.get("unavailable_papers"))
    papers = list(available_papers or [])

    if unavailable or raw.get("paper_not_found") or state.get("paper_not_found"):
        names = unavailable or _clean_paper_list(state.get("requested_papers"))
        answer = _paper_not_available_message(names)
        return UIRunResult(
            answer=answer,
            sources=[],
            trace=trace,
            trace_summary=summarize_trace(trace, selected),
            pending_hitl=None,
            status="paper_not_available",
            error=None,
            selected_papers=selected,
            unavailable_papers=names,
            available_papers=papers,
        )

    pending = _extract_pending_hitl(state)
    if pending is not None:
        return UIRunResult(
            answer=None,
            sources=[],
            trace=trace,
            trace_summary=summarize_trace(trace, selected),
            pending_hitl=pending,
            status="awaiting_human",
            error=None,
            selected_papers=selected,
            available_papers=papers,
        )

    answer = str(raw.get("answer") or state.get("final_answer") or "").strip()
    state_error = str(state.get("error") or raw.get("error") or "").strip()
    if not answer and str(state.get("evidence_status") or "") in {"insufficient", "sufficient_with_gaps"}:
        answer = "当前论文库中未检索到足够证据。"
    if not answer:
        return UIRunResult(
            answer=None,
            sources=[],
            trace=trace,
            trace_summary=summarize_trace(trace, selected),
            pending_hitl=None,
            status="error",
            error="Agent 未返回可展示的回答。" if not state_error else "Agent 调用失败，请检查模型配置后重试。",
            selected_papers=selected,
            available_papers=papers,
            debug_error=_sanitize_debug_error(state_error) if state_error else None,
        )

    return UIRunResult(
        answer=answer,
        sources=parse_sources(answer),
        trace=trace,
        trace_summary=summarize_trace(trace, selected),
        pending_hitl=None,
        status="completed",
        error=None,
        selected_papers=selected,
        available_papers=papers,
    )


def _paper_not_available_message(unavailable_papers: list[str]) -> str:
    labels = "、".join(unavailable_papers) if unavailable_papers else "指定论文"
    return f"当前论文库中没有找到以下论文：{labels}。\n因此无法基于论文原文回答该问题。"


def _bind_thread(agent: Any, thread_id: str) -> None:
    cleaned = str(thread_id or "").strip()
    if not cleaned:
        raise ValueError("thread_id 丢失，请新建会话后重试。")
    if getattr(agent, "thread_id", None) == cleaned:
        return
    switch_thread = getattr(agent, "switch_thread", None)
    if not callable(switch_thread):
        raise RuntimeError("当前 Agent 不支持切换会话线程。")
    switch_thread(cleaned)


def _sanitize_debug_error(error: Any) -> str:
    text = str(error or "").strip()
    return sanitize_trace([text])[0] if text and sanitize_trace([text]) else ""


def _error_result(message: str, exc: Exception | None = None) -> UIRunResult:
    return UIRunResult(
        answer=None,
        sources=[],
        trace=[],
        trace_summary=[],
        pending_hitl=None,
        status="error",
        error=message,
        debug_error=f"{type(exc).__name__}: {_sanitize_debug_error(exc)}" if exc else None,
    )


def run_question(agent: Any, question: str, thread_id: str) -> UIRunResult:
    original_question = str(question or "")
    if not original_question.strip():
        return _error_result("问题不能为空。")
    try:
        available = load_available_papers(agent)
        _bind_thread(agent, thread_id)
        raw = agent.ask(original_question, trace_enabled=True)
        return normalize_agent_result(raw, available_papers=available)
    except Exception as exc:
        return _error_result("Agent 调用失败，请检查模型配置后重试。", exc)


def resume_question(
    agent: Any,
    thread_id: str,
    action: str,
    *,
    revised_question: str | None = None,
) -> UIRunResult:
    cleaned_action = str(action or "").strip()
    if not cleaned_action:
        return _error_result("人工决策无效，请重试。")
    payload: dict[str, Any] = {"action": cleaned_action}
    if cleaned_action == "revise_question":
        payload["revised_question"] = str(revised_question or "").strip()
    try:
        _bind_thread(agent, thread_id)
        available = load_available_papers(agent)
        raw = agent.resume(payload, trace_enabled=True)
        return normalize_agent_result(raw, available_papers=available)
    except Exception as exc:
        return _error_result("恢复任务失败，请稍后重试。", exc)
