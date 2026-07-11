from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ui_agent_adapter as adapter


FINAL_ANSWER = (
    "Transformer 使用多头注意力[1]。\n\n"
    "【资料来源】\n"
    "[1] 01 Attention Is All You Need.pdf，第 3 页"
)


class FakeAgent:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = list(results)
        self.thread_id = "initial-thread"
        self.tools = FakeTools()
        self.switch_calls: list[str] = []
        self.ask_calls: list[dict[str, Any]] = []
        self.resume_calls: list[dict[str, Any]] = []

    def switch_thread(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.switch_calls.append(thread_id)

    def ask(
        self,
        question: str,
        *,
        trace_enabled: bool = False,
    ) -> dict[str, Any]:
        self.ask_calls.append(
            {
                "question": question,
                "trace_enabled": trace_enabled,
                "thread_id": self.thread_id,
            }
        )
        return self.results.pop(0)

    def resume(self, payload: dict[str, Any], *, trace_enabled: bool = False) -> dict[str, Any]:
        self.resume_calls.append(
            {"payload": payload, "trace_enabled": trace_enabled, "thread_id": self.thread_id}
        )
        return self.results.pop(0)


class FakeTools:
    def __init__(self) -> None:
        self.papers = [
            "01 Attention Is All You Need.pdf",
            "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf",
            "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf",
        ]

    def list_papers_tool(self) -> dict[str, Any]:
        return {"papers": list(self.papers), "count": len(self.papers)}

def final_result(answer: str = FINAL_ANSWER) -> dict[str, Any]:
    trace = [
        "【任务判断】paper_analysis",
        "【Planner动作】search_multiple_queries",
        "【观察结果】检索到 8 个片段，页码 [1, 2, 3]。",
        "【证据状态】sufficient",
        "【最终回答】已基于检索证据生成回答。",
    ]
    return {
        "answer": answer,
        "trace": trace,
        "state": {
            "final_answer": answer,
            "trace": trace,
            "awaiting_human": False,
            "last_interrupt_payload": {},
        },
    }


def hitl_result() -> dict[str, Any]:
    payload = {
        "type": "evidence_review",
        "reason": "ReAct 微调损失函数缺少明确证据。",
        "missing_aspects": ["ReAct 微调实验的具体损失函数"],
        "options": [
            {"action": "local_deep_search", "label": "继续深度检索"},
            {"action": "answer_with_gaps", "label": "带证据缺口回答"},
            {"action": "cancel", "label": "取消任务"},
        ],
    }
    return {
        "answer": "",
        "trace": ["【证据状态】sufficient_with_gaps", "【图状态】已暂停，等待用户选择"],
        "state": {
            "awaiting_human": True,
            "last_interrupt_payload": payload,
            "trace": ["【证据状态】sufficient_with_gaps", "【图状态】已暂停，等待用户选择"],
        },
    }


def backend_unavailable_result(fake_answer: str = "") -> dict[str, Any]:
    return {
        "answer": fake_answer,
        "trace": ["【任务判断】general_qa"],
        "state": {
            "final_answer": fake_answer,
            "trace": ["【任务判断】general_qa"],
            "awaiting_human": False,
            "paper_not_found": True,
            "unavailable_papers": ["BERT"],
            "requested_papers": ["BERT"],
        },
    }


def test_normalize_final_answer_result() -> None:
    result = adapter.normalize_agent_result(final_result())

    assert result.answer == FINAL_ANSWER
    assert result.pending_hitl is None
    assert result.status == "completed"
    assert result.sources[0].reference_id == 1


def test_normalize_hitl_result() -> None:
    result = adapter.normalize_agent_result(hitl_result())

    assert result.answer is None
    assert result.pending_hitl is not None
    assert result.pending_hitl.reason == "ReAct 微调损失函数缺少明确证据。"
    assert result.pending_hitl.missing_aspects == ["ReAct 微调实验的具体损失函数"]
    assert result.pending_hitl.actions == ["local_deep_search", "answer_with_gaps", "cancel"]


def test_run_question_passes_same_thread_id() -> None:
    agent = FakeAgent([final_result()])

    adapter.run_question(agent, "完整分析 Transformer", "thread-123")

    assert agent.ask_calls[0]["thread_id"] == "thread-123"
    assert agent.thread_id == "thread-123"


def test_resume_uses_original_thread_id() -> None:
    agent = FakeAgent([final_result()])

    adapter.resume_question(agent, "original-thread", "answer_with_gaps")

    assert agent.resume_calls[0]["thread_id"] == "original-thread"
    assert agent.thread_id == "original-thread"


def test_resume_passes_human_action() -> None:
    agent = FakeAgent([final_result()])

    adapter.resume_question(agent, "thread-1", "local_deep_search")

    assert agent.resume_calls[0]["payload"] == {"action": "local_deep_search"}


def test_second_interrupt_is_preserved() -> None:
    agent = FakeAgent([hitl_result()])

    result = adapter.resume_question(agent, "thread-1", "local_deep_search")

    assert result.status == "awaiting_human"
    assert result.answer is None
    assert result.pending_hitl is not None


def test_trace_summary_extracts_key_steps() -> None:
    summary = adapter.summarize_trace(
        [
            "【任务判断】paper_analysis",
            "【Planner动作】search_multiple_queries",
            "【观察结果】检索到 8 个片段，页码 [1, 2, 3]。",
            "【证据覆盖】5/5",
            "【缺失信息】无",
            "【改写查询】训练设置",
            "【最终回答】已生成回答。",
        ]
    )

    assert any("任务判断" in line for line in summary)
    assert any("执行动作" in line for line in summary)
    assert any("检索结果" in line for line in summary)
    assert any("证据覆盖" in line for line in summary)
    assert any("最终状态" in line for line in summary)


def test_trace_summary_ignores_unknown_lines() -> None:
    assert adapter.summarize_trace(["future log format", "", "random detail"]) == []


def test_sensitive_trace_is_sanitized() -> None:
    fake_secret = "sk-" + "secret-value"
    sanitized = adapter.sanitize_trace(
        [
            f"OPENAI_API_KEY={fake_secret}",
            r"数据库路径 F:\paper\大模型项目\research_paper_agent\db",
            "命中 chunk-a1b2c3d4e5",
        ]
    )
    joined = "\n".join(sanitized)

    assert fake_secret not in joined
    assert r"F:\paper" not in joined
    assert "chunk-a1b2c3d4e5" not in joined


def test_source_parser() -> None:
    sources = adapter.parse_sources(
        "正文[1][2]\n\n【资料来源】\n"
        "[1] 01 Attention Is All You Need.pdf，第 3 页\n"
        "[2] 06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf，第 15 页"
    )

    assert [(item.reference_id, item.paper, item.page) for item in sources] == [
        (1, "01 Attention Is All You Need.pdf", 3),
        (2, "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf", 15),
    ]


def test_empty_answer_returns_safe_error() -> None:
    result = adapter.normalize_agent_result(final_result(answer=""))

    assert result.status == "error"
    assert result.error == "Agent 未返回可展示的回答。"


def test_available_papers_are_loaded_dynamically() -> None:
    agent = FakeAgent([])

    papers = adapter.load_available_papers(agent)

    assert papers == agent.tools.papers


def test_sidebar_paper_names_are_cleaned_for_display() -> None:
    assert adapter.paper_display_name("04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf") == (
        "RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    )


def test_selected_papers_are_extracted_from_agent_result() -> None:
    raw = final_result()
    raw["state"]["selected_papers"] = [
        "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf",
        "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf",
    ]

    result = adapter.normalize_agent_result(raw)

    assert result.selected_papers == raw["state"]["selected_papers"]
    assert any("自动选择论文" in line for line in result.trace_summary)


def test_transformer_alias_does_not_trigger_paper_not_available() -> None:
    agent = FakeAgent([final_result()])

    result = adapter.run_question(
        agent,
        "请解释 Transformer 论文中的缩放点积注意力公式。",
        "thread-1",
    )

    assert result.status == "completed"
    assert len(agent.ask_calls) == 1


def test_transformer_no_space_alias_reaches_agent_unchanged() -> None:
    agent = FakeAgent([final_result()])
    question = "请解释 Transformer论文中的缩放点积注意力公式。"

    result = adapter.run_question(agent, question, "thread-1")

    assert result.status == "completed"
    assert agent.ask_calls == [
        {"question": question, "trace_enabled": True, "thread_id": "thread-1"}
    ]


def test_transformer_spaced_alias_reaches_agent_unchanged() -> None:
    agent = FakeAgent([final_result()])
    question = "  请解释 Transformer 论文中的缩放点积注意力公式。  "

    result = adapter.run_question(agent, question, "thread-1")

    assert result.status == "completed"
    assert agent.ask_calls == [
        {"question": question, "trace_enabled": True, "thread_id": "thread-1"}
    ]


def test_transformer_multiple_space_alias_reaches_agent_unchanged() -> None:
    agent = FakeAgent([final_result()])
    question = "请解释   Transformer   论文中的缩放点积注意力公式。"

    result = adapter.run_question(agent, question, "thread-1")

    assert result.status == "completed"
    assert agent.ask_calls == [
        {"question": question, "trace_enabled": True, "thread_id": "thread-1"}
    ]


def test_transformer_alias_spacing_does_not_change_ui_status() -> None:
    no_space_agent = FakeAgent([final_result()])
    spaced_agent = FakeAgent([final_result()])

    no_space = adapter.run_question(no_space_agent, "请解释 Transformer论文。", "thread-1")
    spaced = adapter.run_question(spaced_agent, "请解释 Transformer 论文。", "thread-1")

    assert no_space.status == spaced.status == "completed"


def test_attention_is_all_you_need_runs_agent() -> None:
    agent = FakeAgent([final_result()])

    result = adapter.run_question(agent, "请分析 Attention Is All You Need。", "thread-1")

    assert result.status == "completed"
    assert len(agent.ask_calls) == 1


def test_rag_and_react_aliases_run_agent() -> None:
    agent = FakeAgent([final_result()])

    result = adapter.run_question(agent, "请比较 RAG 和 ReAct。", "thread-1")

    assert result.status == "completed"
    assert len(agent.ask_calls) == 1


def test_topic_question_is_not_treated_as_missing_paper() -> None:
    agent = FakeAgent([final_result()])

    result = adapter.run_question(agent, "这些论文中有没有讨论位置编码？", "thread-1")

    assert result.status == "completed"
    assert len(agent.ask_calls) == 1


def test_ui_does_not_maintain_separate_incomplete_alias_logic() -> None:
    assert not hasattr(adapter, "find_unavailable_explicit_papers")


def test_topic_without_named_paper_runs_normal_retrieval() -> None:
    agent = FakeAgent([final_result()])

    result = adapter.run_question(agent, "参数高效微调有哪些核心方法？", "thread-1")

    assert result.status == "completed"
    assert len(agent.ask_calls) == 1


def test_no_evidence_is_distinguished_from_missing_paper() -> None:
    raw = final_result(answer="")
    raw["state"]["evidence_status"] = "insufficient"
    agent = FakeAgent([raw])

    result = adapter.run_question(agent, "有哪些方法适合长尾误差建模？", "thread-1")

    assert result.status == "completed"
    assert result.unavailable_papers == []
    assert result.answer == "当前论文库中未检索到足够证据。"


def test_paper_not_available_does_not_trigger_hitl() -> None:
    raw = hitl_result()
    raw["state"]["paper_not_found"] = True
    raw["state"]["unavailable_papers"] = ["BERT"]
    agent = FakeAgent([raw])

    result = adapter.run_question(agent, "请分析 BERT 论文", "thread-1")

    assert result.status == "paper_not_available"
    assert result.pending_hitl is None
    assert len(agent.ask_calls) == 1


def test_backend_confirmed_unavailable_paper_is_rendered_safely() -> None:
    agent = FakeAgent([backend_unavailable_result()])

    result = adapter.run_question(agent, "请分析 BERT 论文", "thread-1")

    assert len(agent.ask_calls) == 1
    assert result.status == "paper_not_available"
    assert result.unavailable_papers == ["BERT"]
    assert "当前论文库中没有找到以下论文：BERT" in (result.answer or "")


def test_unavailable_paper_does_not_generate_fake_answer() -> None:
    agent = FakeAgent([backend_unavailable_result("BERT 的内部常识回答")])

    result = adapter.run_question(agent, "请分析 BERT 论文", "thread-1")

    assert len(agent.ask_calls) == 1
    assert result.status == "paper_not_available"
    assert "内部常识回答" not in (result.answer or "")
def test_streamlit_agent_cache_key_tracks_backend_sources() -> None:
    import streamlit_app

    key = streamlit_app.agent_cache_key()
    tracked_files = {name for name, _mtime in key}

    assert {
        "streamlit_app.py",
        "ui_agent_adapter.py",
        "paper_agent.py",
        "agent_tools.py",
    }.issubset(tracked_files)
