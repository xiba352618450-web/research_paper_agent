from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

from agent_tools import PaperTools
from paper_agent import PaperAgent
from ui_agent_adapter import (
    UIPendingHITL,
    UIRunResult,
    load_available_papers,
    paper_display_name,
    resume_question,
    run_question,
)


st.set_page_config(page_title="Research Paper Agent", page_icon="📚", layout="wide")


ACTION_LABELS = {
    "local_deep_search": "继续深度检索",
    "answer_with_gaps": "带证据缺口回答",
    "cancel": "取消任务",
    "revise_question": "修改问题",
    "retry_current_retrieval": "重试当前检索",
    "continue_with_existing_evidence": "基于已有证据继续",
}


def agent_cache_key() -> tuple[tuple[str, int], ...]:
    root = Path(__file__).resolve().parent
    tracked = (
        "streamlit_app.py",
        "ui_agent_adapter.py",
        "paper_agent.py",
        "agent_tools.py",
    )
    return tuple((name, (root / name).stat().st_mtime_ns) for name in tracked)


@st.cache_resource
def get_agent(cache_key: tuple[tuple[str, int], ...]) -> PaperAgent:
    _ = cache_key
    tools = PaperTools.from_env()
    return PaperAgent(tools=tools)


def initialize_session_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "thread_id": str(uuid.uuid4()),
        "pending_hitl": None,
        "is_running": False,
        "latest_trace": [],
        "last_error": None,
        "paper_catalog": [],
        "paper_catalog_error": None,
        "revision_text": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def new_session() -> None:
    st.session_state.messages = []
    st.session_state.pending_hitl = None
    st.session_state.latest_trace = []
    st.session_state.last_error = None
    st.session_state.revision_text = ""
    st.session_state.is_running = False
    st.session_state.thread_id = str(uuid.uuid4())


def refresh_paper_catalog(agent: PaperAgent) -> None:
    try:
        st.session_state.paper_catalog = load_available_papers(agent)
        st.session_state.paper_catalog_error = None
    except Exception as exc:
        st.session_state.paper_catalog = []
        st.session_state.paper_catalog_error = f"{type(exc).__name__}"


def render_sidebar(agent: PaperAgent | None) -> None:
    with st.sidebar:
        st.header("Research Paper Agent")
        if agent is not None and not st.session_state.paper_catalog and not st.session_state.paper_catalog_error:
            refresh_paper_catalog(agent)

        papers = st.session_state.paper_catalog
        st.write(f"当前论文库：{len(papers)} 篇")
        if papers:
            with st.expander("查看论文列表", expanded=False):
                for paper in papers:
                    st.write(f"- {paper_display_name(paper)}")
        elif st.session_state.paper_catalog_error:
            st.warning("暂时无法读取论文列表。")

        st.divider()
        if st.button("新建会话", use_container_width=True, disabled=st.session_state.is_running):
            new_session()
            st.rerun()
        st.caption(f"会话：{st.session_state.thread_id[:8]}")


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("资料来源", expanded=False):
        for source in sources:
            reference_id = source.get("reference_id")
            paper = paper_display_name(str(source.get("paper") or "未知论文"))
            page = source.get("page")
            prefix = f"[{reference_id}] " if reference_id is not None else ""
            suffix = f"，第 {page} 页" if page is not None else ""
            st.write(f"{prefix}{paper}{suffix}")


def render_trace(trace_summary: list[str], trace: list[str]) -> None:
    if trace_summary:
        with st.expander("查看 Agent 执行过程", expanded=False):
            for line in trace_summary:
                st.write(line)
    if trace:
        with st.expander("查看详细日志", expanded=False):
            for line in trace:
                st.text(line)


def render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources") or [])
                render_trace(message.get("trace_summary") or [], message.get("trace") or [])


def store_result(result: UIRunResult) -> None:
    st.session_state.latest_trace = list(result.trace)
    if result.error:
        st.session_state.last_error = result.error
        return

    st.session_state.last_error = None
    st.session_state.pending_hitl = result.pending_hitl
    if result.answer:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": [asdict(source) for source in result.sources],
                "trace": list(result.trace),
                "trace_summary": list(result.trace_summary),
            }
        )


def execute_resume(agent: PaperAgent, action: str, revised_question: str | None = None) -> None:
    if st.session_state.is_running:
        return
    st.session_state.is_running = True
    try:
        with st.status("Agent 正在恢复任务……", expanded=False):
            result = resume_question(
                agent,
                st.session_state.thread_id,
                action,
                revised_question=revised_question,
            )
        if result.error:
            st.session_state.last_error = result.error
        else:
            store_result(result)
            st.session_state.revision_text = ""
    finally:
        st.session_state.is_running = False
    st.rerun()


def render_pending_hitl(agent: PaperAgent | None) -> None:
    pending = st.session_state.pending_hitl
    if not isinstance(pending, UIPendingHITL):
        return

    st.warning("证据不足，需要人工决定下一步。")
    if pending.reason:
        st.write(pending.reason)
    if pending.missing_aspects:
        st.write("尚缺少：")
        for aspect in pending.missing_aspects:
            st.write(f"- {aspect}")

    if not pending.actions:
        st.error("后端没有返回可用的人工决策选项。")
        return
    if agent is None:
        st.error("Agent 当前不可用，暂时无法恢复任务。")
        return

    if "revise_question" in pending.actions:
        st.session_state.revision_text = st.text_input(
            "修改后的问题",
            value=st.session_state.revision_text,
            placeholder="输入新的问题后点击“修改问题”",
        )

    columns = st.columns(min(4, len(pending.actions)))
    for index, action in enumerate(pending.actions):
        label = pending.action_labels.get(action) or ACTION_LABELS.get(action) or action
        disabled = st.session_state.is_running or (
            action == "revise_question" and not st.session_state.revision_text.strip()
        )
        if columns[index % len(columns)].button(label, key=f"hitl-{action}", disabled=disabled):
            execute_resume(
                agent,
                action,
                revised_question=st.session_state.revision_text if action == "revise_question" else None,
            )


def main() -> None:
    initialize_session_state()

    agent: PaperAgent | None = None
    initialization_error: str | None = None
    try:
        agent = get_agent(agent_cache_key())
    except Exception as exc:
        initialization_error = type(exc).__name__

    render_sidebar(agent)
    st.title("Research Paper Agent")
    st.caption("基于 LangGraph 的多论文检索、证据检查与人工决策 Agent")

    if initialization_error:
        st.error("Agent 初始化失败，请检查本地配置后重试。")
        with st.expander("开发日志", expanded=False):
            st.text(initialization_error)

    render_messages()
    render_pending_hitl(agent)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    if st.session_state.pending_hitl is not None:
        st.info("请先处理当前人工决策，再提交新问题。")

    prompt = st.chat_input(
        "输入论文分析或多论文比较问题",
        disabled=(
            agent is None
            or st.session_state.is_running
            or st.session_state.pending_hitl is not None
        ),
    )
    if not prompt:
        return
    if st.session_state.pending_hitl is not None:
        st.warning("请先处理当前人工决策。")
        return
    if agent is None:
        st.error("Agent 当前不可用。")
        return

    if not prompt.strip():
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.is_running = True
    try:
        with st.status("Agent 正在处理问题……", expanded=False):
            result = run_question(agent, prompt, st.session_state.thread_id)
        store_result(result)
    finally:
        st.session_state.is_running = False
    st.rerun()


if __name__ == "__main__":
    main()
