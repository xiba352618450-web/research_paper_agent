from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_agent
import eval_common
import evaluate_retrieval
import evaluate_routing
from eval_common import PROJECT_ROOT
from eval_recording import RecordingPaperTools


def write_cases(path: Path, cases: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")


def base_case(case_id: str = "eval-x", question: str = "完整分析 LoRA 的方法") -> dict:
    return {
        "id": case_id,
        "category": "方法与公式解释",
        "question": question,
        "expected_intent": "paper_analysis",
        "expected_papers": ["05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"],
        "expected_aspects": ["核心方法"],
        "expected_evidence_status": "sufficient",
        "expected_human_review": False,
        "human_decisions": [],
        "retrieval_eval_mode": "group_recall",
        "gold_page_groups": {
            "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": [
                {"aspect": "核心方法", "pages": [4, 5]}
            ]
        },
        "gold_pages": {"05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": [4, 5]},
        "must_include": [],
        "must_include_any": [["low-rank", "低秩"]],
        "must_not_include": [],
        "expected_facts": {},
        "review_status": "reviewed",
    }


def retrieval_args(cases: Path, output: Path, **overrides) -> argparse.Namespace:
    data = {
        "cases": str(cases),
        "output": str(output),
        "case_ids": "",
        "resume": False,
        "source_mode": "oracle",
        "top_k": 5,
        "neighbor_radius": 0,
        "run_live": False,
        "skip_manifest_check": True,
        "notes": "",
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def agent_args(cases: Path, output: Path, **overrides) -> argparse.Namespace:
    data = {
        "cases": str(cases),
        "output": str(output),
        "case_ids": "",
        "resume": False,
        "all": False,
        "run_live": False,
        "skip_manifest_check": True,
        "tool_mode": "",
        "max_iterations": None,
        "notes": "",
    }
    data.update(overrides)
    return argparse.Namespace(**data)


def test_routing_runner_uses_real_infer_intent_without_llm(tmp_path):
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-r", "完整分析 LoRA 的方法、公式和实验结果")])
    args = argparse.Namespace(cases=str(cases), output=str(tmp_path / "out"), case_ids="", resume=False, dry_run=True, notes="")
    result = evaluate_routing.run_routing_evaluation(args)
    assert result["results"][0]["actual_intent"] == "paper_analysis"


def test_routing_does_not_override_infer_intent_papers(monkeypatch):
    monkeypatch.setattr(evaluate_routing, "infer_intent", lambda question, recent_sources=None: ("general_qa", []))
    monkeypatch.setattr(evaluate_routing, "detect_explicit_papers", lambda question: ["05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"])
    result = evaluate_routing.evaluate_case(base_case("eval-route", "LoRA 是什么？"))
    assert result["actual_papers"] == []
    assert result["detected_explicit_papers"] == ["05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"]


def test_routing_reports_observable_aspect_metrics():
    results = [
        {
            "expected_intent": "paper_analysis",
            "actual_intent": "paper_analysis",
            "expected_papers": ["a.pdf"],
            "paper_precision": 1.0,
            "paper_recall": 1.0,
            "paper_f1": 1.0,
            "aspect_precision": 1.0,
            "aspect_recall": 0.5,
            "aspect_f1": 2 / 3,
            "paper_exact_match": True,
            "intent_correct": True,
        },
        {
            "expected_intent": "method_explain",
            "actual_intent": "general_qa",
            "expected_papers": ["b.pdf"],
            "paper_precision": 0.0,
            "paper_recall": 0.0,
            "paper_f1": 0.0,
            "aspect_precision": 0.0,
            "aspect_recall": 0.0,
            "aspect_f1": 0.0,
            "paper_exact_match": False,
            "intent_correct": False,
        },
    ]
    summary = evaluate_routing.build_summary(results)
    assert "initial_aspect_macro_f1_all_cases" in summary
    assert summary["aspect_observable_case_count"] == 1
    assert summary["aspect_observable_coverage"] == 0.5
    assert summary["initial_aspect_macro_f1_observable_cases"] == 2 / 3


def test_routing_excludes_clarification_from_applicable_paper_macro():
    results = [
        {
            "expected_intent": "clarification",
            "actual_intent": "clarification",
            "expected_papers": [],
            "paper_precision": 1.0,
            "paper_recall": 1.0,
            "paper_f1": 1.0,
            "paper_exact_match": True,
            "intent_correct": True,
            "aspect_precision": 1.0,
            "aspect_recall": 1.0,
            "aspect_f1": 1.0,
        },
        {
            "expected_intent": "paper_analysis",
            "actual_intent": "paper_analysis",
            "expected_papers": ["a.pdf"],
            "paper_precision": 0.0,
            "paper_recall": 0.0,
            "paper_f1": 0.0,
            "paper_exact_match": False,
            "intent_correct": True,
            "aspect_precision": 0.0,
            "aspect_recall": 0.0,
            "aspect_f1": 0.0,
        },
    ]
    summary = evaluate_routing.build_summary(results)
    assert summary["paper_applicable_case_count"] == 1
    assert summary["paper_macro_f1_all_cases"] == 0.5
    assert summary["paper_macro_f1_applicable_cases"] == 0.0


def test_eval_019_routing_clarification_does_not_select_papers(tmp_path):
    case = base_case("eval-019", "对比一下这两篇论文")
    case["expected_intent"] = "clarification"
    case["expected_papers"] = []
    case["expected_aspects"] = []
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case])
    args = argparse.Namespace(cases=str(cases), output=str(tmp_path / "out"), case_ids="", resume=False, dry_run=True, notes="")
    result = evaluate_routing.run_routing_evaluation(args)
    assert result["results"][0]["actual_intent"] == "clarification"
    assert result["results"][0]["actual_papers"] == []


def test_retrieval_dry_run_does_not_initialize_papertools(tmp_path):
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case()])

    def boom():
        raise AssertionError("should not initialize tools in dry-run")

    result = evaluate_retrieval.run_retrieval_evaluation(
        retrieval_args(cases, tmp_path / "out"),
        tools_factory=boom,
    )
    assert result["results"][0]["overall_status"] == "dry_run"


def test_retrieval_cases_path_resolves_when_run_from_eval_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT / "eval")
    result = evaluate_retrieval.run_retrieval_evaluation(
        retrieval_args(Path("eval/eval_cases_v1.jsonl"), tmp_path / "out"),
        tools_factory=lambda: (_ for _ in ()).throw(AssertionError("dry-run should not initialize tools")),
    )
    assert result["summary"]["case_count"] == 19


class FakeRetrievalTools:
    def __init__(self):
        self.searches = []

    def search_paper_tool(self, query, source=None, k=5):
        self.searches.append({"query": query, "source": source, "k": k})
        return {
            "results": [
                {
                    "result_id": f"{source}:4:0",
                    "source": source,
                    "page": 4,
                    "content": "low-rank evidence",
                }
            ]
        }

    def get_neighbor_chunks_tool(self, source, page, radius=1):
        return {"results": []}


def test_retrieval_oracle_source_uses_expected_papers(tmp_path):
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case()])
    fake = FakeRetrievalTools()
    result = evaluate_retrieval.run_retrieval_evaluation(
        retrieval_args(cases, tmp_path / "out", run_live=True),
        tools_factory=lambda: fake,
    )
    assert fake.searches[0]["source"] == "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"
    assert result["results"][0]["queried_papers"] == ["05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"]


def test_retrieval_query_does_not_include_gold_aspects_or_expected_facts(tmp_path):
    case = base_case()
    case["expected_aspects"] = ["核心方法", "实验结果"]
    case["expected_facts"] = {"optimizer": "AdamW"}
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case])
    fake = FakeRetrievalTools()
    evaluate_retrieval.run_retrieval_evaluation(
        retrieval_args(cases, tmp_path / "out", run_live=True),
        tools_factory=lambda: fake,
    )
    assert fake.searches[0]["query"] == case["question"]
    assert "AdamW" not in fake.searches[0]["query"]


def test_retrieval_case_error_does_not_stop_batch(tmp_path):
    class FlakyTools(FakeRetrievalTools):
        def search_paper_tool(self, query, source=None, k=5):
            if "first" in query:
                raise RuntimeError("boom")
            return super().search_paper_tool(query, source, k)

    first = base_case("eval-first", "first query")
    second = base_case("eval-second", "second query")
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [first, second])
    result = evaluate_retrieval.run_retrieval_evaluation(
        retrieval_args(cases, tmp_path / "out", run_live=True),
        tools_factory=FlakyTools,
    )
    assert len(result["results"]) == 2
    assert result["results"][0]["overall_status"] == "error"
    assert result["results"][1]["overall_status"] in {"pass", "fail"}


def test_retrieval_partial_group_recall_is_fail():
    case = base_case("eval-partial")
    case["gold_page_groups"]["05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"] = [
        {"aspect": "方法", "pages": [4]},
        {"aspect": "实验", "pages": [8]},
    ]
    fake = FakeRetrievalTools()
    result = evaluate_retrieval.evaluate_case_live(case, fake, "oracle", 5, 0)
    assert result["macro_group_recall"] == 0.5
    assert result["overall_status"] == "fail"


def test_retrieval_full_group_recall_is_pass():
    case = base_case("eval-full")
    fake = FakeRetrievalTools()
    result = evaluate_retrieval.evaluate_case_live(case, fake, "oracle", 5, 0)
    assert result["macro_group_recall"] == 1.0
    assert result["overall_status"] == "pass"


def test_retrieval_initial_and_expanded_are_separate():
    class NeighborTools(FakeRetrievalTools):
        def get_neighbor_chunks_tool(self, source, page, radius=1):
            return {
                "results": [
                    {"result_id": f"{source}:5:0", "source": source, "page": 5, "content": "neighbor evidence"}
                ]
            }

    case = base_case("eval-expanded")
    case["gold_page_groups"]["05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"] = [
        {"aspect": "neighbor", "pages": [5]}
    ]
    result = evaluate_retrieval.evaluate_case_live(case, NeighborTools(), "oracle", 5, 1)
    assert result["initial_macro_group_recall"] == 0.0
    assert result["expanded_macro_group_recall"] == 1.0
    assert result["macro_group_recall"] == result["expanded_macro_group_recall"]


def test_retrieval_live_prints_progress_to_stderr(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-progress")])
    evaluate_retrieval.run_retrieval_evaluation(
        retrieval_args(cases, tmp_path / "out", run_live=True, progress=True),
        tools_factory=FakeRetrievalTools,
    )
    captured = capsys.readouterr()
    assert "retrieval" in captured.err
    assert "eval-progress" in captured.err
    assert "1/1" in captured.err
    assert "status=pass" in captured.err


def test_recording_paper_tools_records_search_paper_tool():
    fake = FakeRetrievalTools()
    recording = RecordingPaperTools(base_tools=fake)
    recording.search_paper_tool("q", source="a.pdf", k=2)
    assert recording.records[0]["tool_name"] == "search_paper_tool"
    assert recording.records[0]["result_count"] == 1


def test_recording_paper_tools_records_multi_query():
    class FakeMulti(FakeRetrievalTools):
        def search_multiple_queries_tool(self, **kwargs):
            return {
                "results": [
                    {"result_id": "a:1:0", "source": "a.pdf", "page": 1, "content": "x"}
                ],
                "query_results": [],
            }

    recording = RecordingPaperTools(base_tools=FakeMulti())
    recording.search_multiple_queries_tool(["q"], source="a.pdf")
    assert recording.records[0]["tool_name"] == "search_multiple_queries_tool"
    assert recording.records[0]["source_pages"] == {"a.pdf": [1]}


def test_recorded_tool_arguments_are_json_serializable():
    class FakeMulti(FakeRetrievalTools):
        def search_multiple_queries_tool(self, **kwargs):
            return {"results": [], "query_results": []}

    recording = RecordingPaperTools(base_tools=FakeMulti())
    recording.search_multiple_queries_tool(
        ["q"],
        source="a.pdf",
        seen_result_ids={"b", "a"},
        seen_source_pages={("a.pdf", 1)},
        progress_callback=lambda *_args: None,
    )
    json.dumps(recording.records, ensure_ascii=False)
    args = recording.records[0]["sanitized_arguments"]
    assert args["seen_result_ids"] == ["a", "b"]
    assert args["progress_callback"] == "<callable>"


class FakeAgent:
    created = 0

    def __init__(self, tools, args):
        FakeAgent.created += 1
        self.thread_id = f"thread-{FakeAgent.created}"

    def ask(self, question, trace_enabled=False):
        return {
            "answer": "LoRA uses low-rank updates[1]\n\n【资料来源】\n[1] 05 LoRA - Low-Rank Adaptation of Large Language Models.pdf，第 4 页",
            "trace": [],
            "state": {
                "intent": "paper_analysis",
                "selected_papers": ["05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"],
                "required_aspects": ["核心方法"],
                "retrieved_docs": [
                    {"source": "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf", "page": 4, "content": "x"}
                ],
                "evidence_sufficient": True,
                "evidence_status": "sufficient",
                "iteration": 1,
            },
        }


def test_each_e2e_case_creates_independent_agent(tmp_path):
    FakeAgent.created = 0
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001"), base_case("eval-005")])
    evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: FakeAgent(tools, args),
    )
    assert FakeAgent.created == 2


def test_eval_020_resume_keeps_same_thread(tmp_path):
    class HITLAgent(FakeAgent):
        def __init__(self, tools, args):
            super().__init__(tools, args)
            self.resume_count = 0

        def ask(self, question, trace_enabled=False):
            return {"answer": "", "trace": [], "state": {"awaiting_human": True, "human_review_required": True}}

        def resume(self, payload, trace_enabled=False):
            self.resume_count += 1
            if self.resume_count == 1:
                return {"answer": "", "trace": [], "state": {"awaiting_human": True, "last_interrupt_payload": {"step": 2}}}
            return super().ask("done", trace_enabled=trace_enabled)

    case = base_case("eval-020", "对比 RAG 和 ReAct 的核心方法、训练方式、具体损失函数和推理流程")
    case["expected_human_review"] = True
    case["human_decisions"] = ["local_deep_search", "answer_with_gaps"]
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: HITLAgent(tools, args),
    )
    assert result["results"][0]["human_decisions_observed"] == ["local_deep_search", "answer_with_gaps"]
    assert result["results"][0]["hitl_sequence_correct"] is True


def test_hitl_none_thread_is_not_same_thread():
    class NoThreadAgent:
        thread_id = None

        def resume(self, payload, trace_enabled=False):
            return {"answer": "", "state": {"awaiting_human": False}, "trace": []}

    adapter = evaluate_agent.HITLAdapter()
    adapter.resume_with_decisions(NoThreadAgent(), ["answer_with_gaps"], initial_state={"awaiting_human": True})
    assert adapter.records[0]["same_thread"] is False


def test_hitl_does_not_resume_without_real_interrupt(tmp_path):
    class NoInterruptAgent(FakeAgent):
        def __init__(self, tools, args):
            super().__init__(tools, args)
            self.resume_count = 0

        def ask(self, question, trace_enabled=False):
            result = super().ask(question, trace_enabled)
            result["state"]["human_review_required"] = True
            result["state"]["awaiting_human"] = False
            result["state"]["last_interrupt_payload"] = {}
            return result

        def resume(self, payload, trace_enabled=False):
            self.resume_count += 1
            return super().ask("done", trace_enabled=trace_enabled)

    case = base_case("eval-020")
    case["expected_human_review"] = True
    case["human_decisions"] = ["answer_with_gaps"]
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case])
    agents = []

    def factory(tools, args):
        agent = NoInterruptAgent(tools, args)
        agents.append(agent)
        return agent

    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=factory,
    )
    assert agents[0].resume_count == 0
    assert result["results"][0]["error"] == "expected_interrupt_not_observed"
    assert result["results"][0]["overall_status"] == "fail"


def test_human_review_trigger_not_inferred_from_adapter_records(tmp_path):
    class NoInitialInterruptAgent(FakeAgent):
        def ask(self, question, trace_enabled=False):
            result = super().ask(question, trace_enabled)
            result["state"]["awaiting_human"] = False
            result["state"]["last_interrupt_payload"] = {}
            return result

    case = base_case("eval-020")
    case["expected_human_review"] = True
    case["human_decisions"] = []
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: NoInitialInterruptAgent(tools, args),
    )
    assert result["results"][0]["human_review_triggered"] is False


def test_unsupported_hitl_returns_unsupported_not_fake_pass(tmp_path):
    class NoResumeAgent(FakeAgent):
        resume = None

        def ask(self, question, trace_enabled=False):
            return {"answer": "", "trace": [], "state": {"awaiting_human": True}}

    case = base_case("eval-020")
    case["expected_human_review"] = True
    case["human_decisions"] = ["local_deep_search", "answer_with_gaps"]
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: NoResumeAgent(tools, args),
    )
    assert result["results"][0]["overall_status"] == "unsupported"
    assert result["results"][0]["error"] == "unsupported_hitl_interface"


def test_e2e_wrong_intent_cannot_pass(tmp_path):
    class WrongIntentAgent(FakeAgent):
        def ask(self, question, trace_enabled=False):
            result = super().ask(question, trace_enabled)
            result["state"]["intent"] = "general_qa"
            result["state"]["evidence_status"] = "sufficient"
            return result

    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: WrongIntentAgent(tools, args),
    )
    assert result["results"][0]["routing_correct"] is False
    assert result["results"][0]["overall_status"] == "fail"


def test_e2e_incomplete_retrieval_cannot_pass(tmp_path):
    class IncompleteRetrievalAgent(FakeAgent):
        def ask(self, question, trace_enabled=False):
            result = super().ask(question, trace_enabled)
            result["state"]["retrieved_docs"] = []
            result["state"]["evidence_status"] = "sufficient"
            return result

    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: IncompleteRetrievalAgent(tools, args),
    )
    assert result["results"][0]["retrieval_correct"] is False
    assert result["results"][0]["overall_status"] == "fail"


def test_agent_eval_uses_tool_call_papers_when_state_is_polluted(tmp_path):
    transformer = "01 Attention Is All You Need.pdf"
    rag = "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf"

    class PollutedStateAgent(FakeAgent):
        def __init__(self, tools, args):
            super().__init__(tools, args)
            self.tools = tools

        def ask(self, question, trace_enabled=False):
            self.tools.search_paper_tool("attention formula", source=transformer, k=5)
            return {
                "answer": f"Transformer attention answer[1]\n\n【资料来源】\n[1] {transformer}，第 4 页",
                "trace": [],
                "state": {
                    "intent": "method_explain",
                    "selected_papers": [transformer, rag],
                    "comparison_papers": [],
                    "required_aspects": ["核心方法与数学公式"],
                    "retrieved_docs": [
                        {"source": transformer, "page": 4, "content": "attention formula"},
                    ],
                    "evidence_sufficient": True,
                    "evidence_status": "sufficient",
                    "iteration": 1,
                },
            }

    class ToolCallRecordingFakeTools(FakeRetrievalTools):
        pass

    case = base_case("eval-005", "解释 Transformer 的 scaled dot-product attention 和 multi-head attention 公式")
    case["expected_intent"] = "method_explain"
    case["expected_papers"] = [transformer]
    case["gold_page_groups"] = {transformer: [{"aspect": "scaled dot-product attention", "pages": [4]}]}
    case["gold_pages"] = {transformer: [4]}
    case["must_include_any"] = [["Transformer", "attention"]]
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [case])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=ToolCallRecordingFakeTools,
        agent_factory=lambda tools, args: PollutedStateAgent(tools, args),
    )
    item = result["results"][0]
    assert item["actual_papers"] == [transformer]
    assert item["paper_selection_correct"] is True
    assert rag in item["actual_papers_from_state"]
    assert rag not in item["actual_papers_from_tool_calls"]


def test_resume_skips_successful_case(tmp_path):
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    out = tmp_path / "out"
    out.mkdir()
    (out / "cases.jsonl").write_text(json.dumps({"id": "eval-001", "overall_status": "pass"}) + "\n", encoding="utf-8")
    evaluate_agent.run_agent_evaluation(agent_args(cases, out, resume=True))
    assert len((out / "cases.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_raw_results_do_not_leak_openai_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-key")
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    out = tmp_path / "out"
    evaluate_agent.run_agent_evaluation(agent_args(cases, out))
    assert "super-secret-key" not in (out / "cases.jsonl").read_text(encoding="utf-8")


def test_output_summary_and_cases_jsonl(tmp_path):
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    out = tmp_path / "out"
    evaluate_agent.run_agent_evaluation(agent_args(cases, out))
    assert (out / "summary.json").exists()
    assert (out / "cases.jsonl").exists()


def test_manifest_failure_blocks_live_run(tmp_path, monkeypatch):
    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    monkeypatch.setattr(evaluate_retrieval, "run_manifest_check", lambda **kwargs: {"skipped": False, "passed": False})
    result = evaluate_retrieval.run_retrieval_evaluation(
        retrieval_args(cases, tmp_path / "out", run_live=True, skip_manifest_check=False),
        tools_factory=lambda: (_ for _ in ()).throw(AssertionError("should not initialize")),
    )
    assert result["summary"]["error"] == "manifest_check_failed"


def test_manifest_check_uses_utf8_decoding(monkeypatch):
    calls = {}

    class Proc:
        returncode = 0
        stdout = "OK: 中文输出"
        stderr = ""

    def fake_run(*args, **kwargs):
        calls.update(kwargs)
        return Proc()

    monkeypatch.setattr(eval_common.subprocess, "run", fake_run)
    result = eval_common.run_manifest_check(skip=False, dry_run=False)
    assert result["passed"] is True
    assert calls["encoding"] == "utf-8"
    assert calls["errors"] == "replace"


def test_user_visible_success_can_pass_with_intent_diagnostic_mismatch(tmp_path):
    class WrongIntentAgent(FakeAgent):
        def ask(self, question, trace_enabled=False):
            result = super().ask(question, trace_enabled)
            result["state"]["intent"] = "general_qa"
            return result

    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: WrongIntentAgent(tools, args),
    )
    item = result["results"][0]

    assert item["user_visible_success"] is True
    assert item["strict_pipeline_success"] is False
    assert "pipeline_diagnostic_mismatch" in item["failure_tags"]
    assert item["overall_status"] == "fail"


def test_strict_pipeline_requires_routing_and_evidence_status(tmp_path):
    class WrongEvidenceAgent(FakeAgent):
        def ask(self, question, trace_enabled=False):
            result = super().ask(question, trace_enabled)
            result["state"]["evidence_status"] = "insufficient"
            return result

    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: WrongEvidenceAgent(tools, args),
    )
    item = result["results"][0]

    assert item["user_visible_success"] is True
    assert item["strict_pipeline_success"] is False
    assert item["evidence_status_correct"] is False


def test_wrong_answer_never_counts_as_user_visible_success(tmp_path):
    class WrongAnswerAgent(FakeAgent):
        def ask(self, question, trace_enabled=False):
            result = super().ask(question, trace_enabled)
            result["answer"] = "不相关的回答[1]\n\n【资料来源】\n[1] unrelated.pdf，第 1 页"
            return result

    cases = tmp_path / "cases.jsonl"
    write_cases(cases, [base_case("eval-001")])
    result = evaluate_agent.run_agent_evaluation(
        agent_args(cases, tmp_path / "out", run_live=True, all=True),
        tools_factory=FakeRetrievalTools,
        agent_factory=lambda tools, args: WrongAnswerAgent(tools, args),
    )
    item = result["results"][0]

    assert item["answer_correct"] is False
    assert item["user_visible_success"] is False
    assert item["strict_pipeline_success"] is False


def _rescore_fixture_record() -> dict:
    return {
        "id": "eval-001",
        "answer": "LoRA uses low-rank updates[1]",
        "tool_calls": [{"tool_name": "search_paper_tool", "result_count": 1}],
        "routing_correct": False,
        "paper_selection_correct": True,
        "retrieval_correct": True,
        "evidence_status_correct": True,
        "answer_correct": True,
        "citation_format_correct": True,
        "hitl_correct": True,
        "overall_status": "fail",
        "failure_tags": ["routing_failure"],
        "latency_seconds": 1.25,
    }


def _load_rescore_module():
    return importlib.import_module("rescore_agent_results")


def test_rescore_does_not_initialize_agent(tmp_path, monkeypatch):
    import paper_agent

    module = _load_rescore_module()
    source = tmp_path / "cases.jsonl"
    write_cases(source, [_rescore_fixture_record()])
    monkeypatch.setattr(paper_agent, "PaperAgent", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not initialize agent")))

    module.rescore_agent_results(source, tmp_path / "rescored")

    assert (tmp_path / "rescored" / "cases.jsonl").exists()


def test_rescore_preserves_answers_and_tool_calls(tmp_path):
    module = _load_rescore_module()
    source = tmp_path / "cases.jsonl"
    original = _rescore_fixture_record()
    write_cases(source, [original])

    module.rescore_agent_results(source, tmp_path / "rescored")
    rescored = json.loads((tmp_path / "rescored" / "cases.jsonl").read_text(encoding="utf-8").strip())

    assert rescored["answer"] == original["answer"]
    assert rescored["tool_calls"] == original["tool_calls"]
    assert rescored["user_visible_success"] is True
    assert rescored["strict_pipeline_success"] is False
    assert "pipeline_diagnostic_mismatch" in rescored["failure_tags"]


def test_rescore_writes_new_summary(tmp_path):
    module = _load_rescore_module()
    source = tmp_path / "cases.jsonl"
    write_cases(source, [_rescore_fixture_record()])

    result = module.rescore_agent_results(source, tmp_path / "rescored")
    summary = json.loads((tmp_path / "rescored" / "summary.json").read_text(encoding="utf-8"))

    assert result["summary"] == summary
    assert summary["user_visible_success_rate"] == 1.0
    assert summary["strict_pipeline_success_rate"] == 0.0
