from __future__ import annotations

import builtins
import json
import threading
import warnings
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agent_tools
import paper_agent


LORA_SOURCE = "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf"
RAG_SOURCE = "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf"
REACT_SOURCE = "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf"
TRANSFORMER_SOURCE = "01 Attention Is All You Need.pdf"
CORPUS_DRAG_QUESTION = "\u8fd9\u4e9b\u8bba\u6587\u662f\u5426\u8ba8\u8bba\u8fc7 drag \u76f8\u5173\u5185\u5bb9\uff1f"


@pytest.mark.parametrize(
    "question",
    [
        "请解释 Transformer 论文中的缩放点积注意力。",
        "请解释 Transformer论文中的缩放点积注意力。",
        "请解释 Transformer   论文中的缩放点积注意力。",
        "请解释 Transformer　论文中的缩放点积注意力。",
        "请解释“Transformer”论文中的缩放点积注意力。",
        "请解释 Transformer 的论文中的缩放点积注意力。",
        "Explain the Transformer paper.",
        "请解释 TRANSFORMER 论文。",
    ],
)
def test_transformer_alias_variants_select_transformer_source(question: str) -> None:
    assert paper_agent.detect_explicit_papers(question) == [TRANSFORMER_SOURCE]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Transformer 论文 ", "transformer 论文"),
        ("Transformer   论文", "transformer 论文"),
        ("Transformer　论文", "transformer 论文"),
        ("TRANSFORMER 论文", "transformer 论文"),
        ("Ａｔｔｅｎｔｉｏｎ　Ｉｓ　Ａｌｌ　Ｙｏｕ　Ｎｅｅｄ", "attention is all you need"),
    ],
)
def test_normalize_paper_name_handles_unicode_case_and_whitespace(raw: str, expected: str) -> None:
    assert paper_agent.normalize_paper_name(raw) == expected


@pytest.mark.parametrize(
    "paper_reference",
    [
        "Transformer",
        "Transformer 论文",
        "Transformer   论文",
        "Transformer　论文",
        "“Transformer”论文",
        "《Transformer》论文",
        "Transformer 的论文",
        "Explain the Transformer paper",
        "TRANSFORMER 论文",
    ],
)
def test_runtime_resolver_maps_transformer_variants_to_canonical_pdf(paper_reference: str) -> None:
    assert paper_agent.resolve_paper_reference(
        paper_reference,
        [TRANSFORMER_SOURCE],
    ) == TRANSFORMER_SOURCE


def test_short_aliases_use_ascii_boundaries_without_false_substrings() -> None:
    assert paper_agent.detect_explicit_papers("drag is unrelated") == []
    assert paper_agent.detect_explicit_papers("bragging is unrelated") == []
    assert paper_agent.detect_explicit_papers("请解释 RAG") == [RAG_SOURCE]
    assert paper_agent.detect_explicit_papers("请解释 ReAct") == [REACT_SOURCE]
    assert paper_agent.detect_explicit_papers("请解释 LoRA") == [LORA_SOURCE]
    assert paper_agent.detect_explicit_papers("Attention Is All You Need") == [TRANSFORMER_SOURCE]


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def get(self, where=None, include=None, limit=None, offset=None, **kwargs):
        docs = self._filter_documents(where)
        if offset:
            docs = docs[offset:]
        if limit is not None:
            docs = docs[:limit]
        return {
            "ids": [f"doc-{idx}" for idx, _doc in enumerate(docs)],
            "documents": [doc.page_content for doc in docs],
            "metadatas": [doc.metadata for doc in docs],
        }

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4, **kwargs):
        docs = self._filter_documents(kwargs.get("filter"))
        query_lower = query.lower()
        ranked = []
        for doc in docs:
            text = (doc.page_content + " " + doc.metadata.get("source", "")).lower()
            score = 0.95 if "lora" in query_lower and "lora" in text else 0.65
            ranked.append((doc, score))
        return ranked[:k]

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        docs = self._filter_documents(kwargs.get("filter"))
        return [(doc, 0.1 + idx / 10) for idx, doc in enumerate(docs[:k])]

    def _filter_documents(self, where):
        if not where:
            return list(self.documents)
        source = where.get("source")
        if source:
            return [doc for doc in self.documents if doc.metadata.get("source") == source]
        return list(self.documents)


class FlakyVectorStore(FakeVectorStore):
    def __init__(self, documents: list[Document], failures: list[Exception]) -> None:
        super().__init__(documents)
        self.failures = list(failures)
        self.calls = 0

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return super().similarity_search_with_relevance_scores(query, k=k, **kwargs)


class FlakyScoreVectorStore(FakeVectorStore):
    def __init__(self, documents: list[Document], failures: list[Exception]) -> None:
        super().__init__(documents)
        self.failures = list(failures)
        self.calls = 0

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return super().similarity_search_with_score(query, k=k, **kwargs)


class WarningOnRelevanceVectorStore(FakeVectorStore):
    def similarity_search_with_relevance_scores(self, query: str, k: int = 4, **kwargs):
        warnings.warn("Relevance scores must be between 0 and 1", UserWarning)
        return super().similarity_search_with_relevance_scores(query, k=k, **kwargs)


class AlwaysFailScoreVectorStore(FakeVectorStore):
    def __init__(self, documents: list[Document], exc: Exception | None = None) -> None:
        super().__init__(documents)
        self.exc = exc or RuntimeError("HTTP 429")
        self.calls = 0

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        self.calls += 1
        raise self.exc


class FakeJsonLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, messages):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return AIMessage(content=self.responses[idx])


class FailingNativeLLM(FakeJsonLLM):
    def bind_tools(self, tools):
        raise RuntimeError("tool calling is not supported")


class BindExplodingLLM(FakeJsonLLM):
    def bind_tools(self, tools):
        raise AssertionError("JSON mode must not call bind_tools")


class FakeStreamingLLM(FakeJsonLLM):
    def __init__(self, responses: list[str], stream_chunks: list[str] | None = None) -> None:
        super().__init__(responses)
        self.stream_chunks = stream_chunks or []
        self.stream_called = False

    def stream(self, messages):
        self.stream_called = True
        for chunk in self.stream_chunks:
            yield AIMessageChunk(content=chunk)


class NoStreamLLM(FakeJsonLLM):
    def stream(self, messages):
        raise RuntimeError("streaming not supported")


class ExplodingStreamLLM(FakeJsonLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.stream_called = False

    def stream(self, messages):
        self.stream_called = True
        raise RuntimeError("streaming not supported")
        yield AIMessageChunk(content="")


class RecordingCheckpointer(InMemorySaver):
    def __init__(self) -> None:
        super().__init__()
        self.deleted_threads: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)
        super().delete_thread(thread_id)


def make_chunk(
    result_id: str,
    source: str,
    page: int,
    content: str,
    score_label: str = "distance_score",
) -> agent_tools.PaperChunk:
    return {
        "result_id": result_id,
        "source": source,
        "page": page,
        "metadata_page": page - 1,
        "score": 0.1,
        "score_label": score_label,
        "content": content,
    }


class RecordingPaperTools:
    def __init__(self) -> None:
        self.multiple_calls: list[dict[str, Any]] = []
        self.neighbor_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.inspect_calls: list[str] = []
        self.papers = [TRANSFORMER_SOURCE, RAG_SOURCE, REACT_SOURCE, LORA_SOURCE]

    def list_papers_tool(self) -> dict[str, Any]:
        return {"papers": list(self.papers), "count": len(self.papers)}

    def search_paper_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.search_calls.append(dict(kwargs))
        source = kwargs.get("source") or REACT_SOURCE
        return {
            "results": [make_chunk(f"search-{len(self.search_calls)}", source, 5, "local search result")],
            "score_label": "distance_score",
        }

    def search_multiple_queries_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.multiple_calls.append(dict(kwargs))
        sources = list(kwargs.get("sources") or ([kwargs.get("source")] if kwargs.get("source") else []))
        results = [
            make_chunk(f"multi-{idx}", source, 10 + idx, f"{source} local deep result")
            for idx, source in enumerate(sources, start=1)
        ]
        return {
            "results": results,
            "query_results": [
                {
                    "query": query,
                    "source": source,
                    "covered_aspect": "损失函数",
                    "result_count": 1,
                    "result_ids": [result["result_id"]],
                }
                for query in kwargs.get("queries", [])
                for source, result in zip(sources, results)
            ],
            "score_label": "distance_score",
        }

    def get_neighbor_chunks_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.neighbor_calls.append(dict(kwargs))
        return {
            "results": [
                make_chunk(
                    f"neighbor-{len(self.neighbor_calls)}",
                    str(kwargs["source"]),
                    int(kwargs["page"]),
                    "neighbor local context",
                    score_label="neighbor_context",
                )
            ]
        }

    def inspect_paper_scope_tool(self, source: str) -> dict[str, Any]:
        self.inspect_calls.append(source)
        return {"source": source, "exists": True, "page_range": [1, 20], "chunk_count": 20}


class PartialComparisonTools(RecordingPaperTools):
    def search_multiple_queries_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.multiple_calls.append(dict(kwargs))
        sources = list(kwargs.get("sources") or ([kwargs.get("source")] if kwargs.get("source") else []))
        results: list[agent_tools.PaperChunk] = []
        query_results: list[dict[str, Any]] = []
        for query_index, query in enumerate(kwargs.get("queries", []), start=1):
            for source in sources:
                query_lower = str(query).lower()
                if source == REACT_SOURCE and "loss" in query_lower:
                    content = "The retrieved evidence does not explicitly specify the fine-tuning loss function."
                elif source == RAG_SOURCE and "loss" in query_lower:
                    content = "RAG training minimizes the negative log-likelihood loss for generation."
                else:
                    content = f"{source} evidence for {query}"
                chunk = make_chunk(
                    f"partial-{len(results)}",
                    source,
                    query_index + 1,
                    content,
                )
                results.append(chunk)
                result_ids = [chunk["result_id"]]
                result_count = 1
                inferred_aspects = paper_agent.infer_comparison_aspects_from_query(query)
                covered_aspect = "损失函数" if "loss" in query_lower else inferred_aspects[0]
                query_results.append(
                    {
                        "query": query,
                        "source": source,
                        "covered_aspect": covered_aspect,
                        "result_count": result_count,
                        "result_ids": result_ids,
                    }
                )
        return {"results": results, "query_results": query_results, "score_label": "distance_score"}


@pytest.fixture()
def fake_documents() -> list[Document]:
    return [
        Document(
            page_content="LoRA freezes pretrained weights and trains low-rank adaptation matrices.",
            metadata={"source": LORA_SOURCE, "page": 0},
        ),
        Document(
            page_content="LoRA experiments compare trainable parameters and downstream quality.",
            metadata={"source": LORA_SOURCE, "page": 1},
        ),
        Document(
            page_content="Retrieval-Augmented Generation combines a retriever with a generator.",
            metadata={"source": RAG_SOURCE, "page": 2},
        ),
        Document(
            page_content="ReAct interleaves reasoning traces and actions for language model agents.",
            metadata={"source": REACT_SOURCE, "page": 3},
        ),
    ]


@pytest.fixture()
def tools(fake_documents: list[Document]) -> agent_tools.PaperTools:
    return agent_tools.PaperTools(vector_store=FakeVectorStore(fake_documents))


def test_db_missing_message(tmp_path: Path) -> None:
    missing = tmp_path / "db"

    message = agent_tools.check_db_ready(missing)

    assert "python ingest.py" in message


def test_default_chat_model_is_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHAT_MODEL", raising=False)

    assert paper_agent.get_chat_model_name() == "deepseek-v4-flash"


def test_default_agent_tool_mode_is_json(monkeypatch: pytest.MonkeyPatch, tools: agent_tools.PaperTools) -> None:
    monkeypatch.delenv("AGENT_TOOL_MODE", raising=False)
    llm = FakeJsonLLM(['{"action":"answer","arguments":{},"reason_summary":"无需检索"}'])

    agent = paper_agent.PaperAgent(tools=tools, llm=llm)

    assert agent.tool_mode == "json"


def test_embedding_model_default_remains_text_embedding_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    assert agent_tools.get_embedding_model_name() == "text-embedding-3-small"


def test_complete_lora_analysis_is_paper_analysis(tools: agent_tools.PaperTools) -> None:
    llm = FakeJsonLLM(
        [
            '{"sufficient":true,"reason":"证据覆盖方法与实验","covered_aspects":["核心方法与数学公式","主要定量实验结果"],"missing_aspects":[]}',
            "LoRA 分析回答[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("完整分析 LoRA 的方法、核心公式、训练过程和实验结果")

    assert result["state"]["intent"] == "paper_analysis"
    assert result["state"]["required_aspects"]


def test_comparison_requested_aspects_are_user_requested_four_items(tools: agent_tools.PaperTools) -> None:
    llm = FakeJsonLLM(
        [
            '{"sufficient":true,"reason":"充分","covered_aspects":["研究目标","核心方法","训练方式","推理流程"],"missing_aspects":[]}',
            "比较回答[1][2]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程")

    assert result["state"]["requested_aspects"] == ["研究目标", "核心方法", "训练方式", "推理流程"]
    assert result["state"]["required_aspects"] == result["state"]["requested_aspects"]


def test_related_aspects_derive_only_strongly_related_comparison_details() -> None:
    related = paper_agent.derive_related_aspects(
        "paper_comparison",
        ["核心方法", "训练方式"],
        "对比 RAG 和 ReAct 的核心方法和训练方式",
    )

    assert "核心公式或机制" in related
    assert "方法关键参数" in related
    assert "冻结和更新哪些参数" in related
    assert "GPU 数量" not in related
    assert "优势与局限" not in related


def test_related_aspects_missing_does_not_make_evidence_insufficient(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    state = {
        "question": "对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["研究目标", "核心方法", "训练方式", "推理流程"],
        "related_aspects": ["核心公式或机制", "方法关键参数"],
        "coverage_by_paper": {
            RAG_SOURCE: {"研究目标": "covered", "核心方法": "covered", "训练方式": "covered", "推理流程": "covered"},
            REACT_SOURCE: {"研究目标": "covered", "核心方法": "covered", "训练方式": "covered", "推理流程": "covered"},
        },
        "retrieved_docs": [],
        "covered_aspects": [],
        "missing_aspects": [],
        "trace": [],
        "trace_enabled": False,
        "iteration": 1,
        "max_iterations": 3,
    }

    result = agent.evidence_check_node(state)

    assert result["evidence_sufficient"] is True
    assert result["evidence_status"] == "sufficient"


def test_comparison_partial_core_aspect_is_sufficient_with_gaps(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    state = {
        "question": "对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["研究目标", "核心方法", "训练方式", "推理流程"],
        "coverage_by_paper": {
            RAG_SOURCE: {"研究目标": "covered", "核心方法": "covered", "训练方式": "covered", "推理流程": "covered"},
            REACT_SOURCE: {"研究目标": "covered", "核心方法": "covered", "训练方式": "partial", "推理流程": "covered"},
        },
        "retrieved_docs": [],
        "trace": [],
        "trace_enabled": False,
        "iteration": 1,
        "max_iterations": 3,
    }

    result = agent.evidence_check_node(state)

    assert result["evidence_sufficient"] is False
    assert result["evidence_status"] == "sufficient_with_gaps"
    assert "ReAct" in result["evidence_reason"]


def test_compute_evidence_status_treats_not_applicable_as_sufficient() -> None:
    status = paper_agent.compute_evidence_status(
        coverage_by_paper={
            RAG_SOURCE: {"损失函数": "covered"},
            REACT_SOURCE: {"损失函数": "not_applicable"},
        },
        requested_aspects=["损失函数"],
        target_papers=[RAG_SOURCE, REACT_SOURCE],
    )

    assert status == "sufficient"


def test_compute_evidence_status_partial_overrides_llm_sufficient() -> None:
    status = paper_agent.compute_evidence_status(
        coverage_by_paper={
            RAG_SOURCE: {"损失函数": "covered"},
            REACT_SOURCE: {"损失函数": "partial"},
        },
        requested_aspects=["损失函数"],
        target_papers=[RAG_SOURCE, REACT_SOURCE],
    )

    assert status == "sufficient_with_gaps"


def test_compute_evidence_status_missing_is_insufficient() -> None:
    status = paper_agent.compute_evidence_status(
        coverage_by_paper={
            RAG_SOURCE: {"损失函数": "covered"},
            REACT_SOURCE: {"损失函数": "missing"},
        },
        requested_aspects=["损失函数"],
        target_papers=[RAG_SOURCE, REACT_SOURCE],
    )

    assert status == "insufficient"


def test_query_result_target_does_not_prove_coverage_when_content_says_not_found(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    result_id = "react-loss-not-found"
    initial_coverage = {
        RAG_SOURCE: {"损失函数": "covered"},
        REACT_SOURCE: {"损失函数": "missing"},
    }
    state = {
        "question": "对比 RAG 和 ReAct 的具体损失函数",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["损失函数"],
        "coverage_by_paper": initial_coverage,
        "retrieved_docs": [
            make_chunk(
                result_id,
                REACT_SOURCE,
                8,
                "The retrieved evidence does not explicitly specify the fine-tuning loss function.",
            )
        ],
        "tool_result": {
            "query_results": [
                {
                    "query": "loss objective supervision training details",
                    "source": REACT_SOURCE,
                    "covered_aspect": "损失函数",
                    "result_count": 1,
                    "result_ids": [result_id],
                }
            ],
            "results": [],
        },
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.evidence_check_node(state)

    assert initial_coverage[REACT_SOURCE]["损失函数"] == "missing"
    assert result["coverage_by_paper"][REACT_SOURCE]["损失函数"] == "partial"
    assert result["evidence_status"] == "sufficient_with_gaps"
    assert result["evidence_sufficient"] is False
    assert "ReAct 微调实验的具体损失函数" in result["missing_aspects"]


def test_rag_marginal_nll_formula_covers_general_loss_function(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    result_id = "rag-loss-covered"
    state = {
        "question": "对比 RAG 和 ReAct 的具体损失函数",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["损失函数"],
        "coverage_by_paper": {
            RAG_SOURCE: {"损失函数": "missing"},
            REACT_SOURCE: {"损失函数": "not_applicable"},
        },
        "retrieved_docs": [
            make_chunk(
                result_id,
                RAG_SOURCE,
                6,
                "RAG-Sequence and RAG-Token training optimize the negative marginal log-likelihood of each target answer token.",
            )
        ],
        "tool_result": {
            "query_results": [
                {
                    "query": "marginal log likelihood equation",
                    "source": RAG_SOURCE,
                    "covered_aspect": "损失函数",
                    "result_count": 1,
                    "result_ids": [result_id],
                }
            ],
            "results": [],
        },
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.evidence_check_node(state)

    assert result["coverage_by_paper"][RAG_SOURCE]["损失函数"] == "covered"
    assert result["evidence_status"] == "sufficient"


def test_gradient_details_are_partial_when_user_explicitly_asks_gradient(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    result_id = "rag-loss-no-gradient"
    state = {
        "question": "对比 RAG 和 ReAct 的损失函数和梯度传播细节",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE],
        "requested_aspects": ["损失函数"],
        "coverage_by_paper": {RAG_SOURCE: {"损失函数": "missing"}},
        "retrieved_docs": [
            make_chunk(
                result_id,
                RAG_SOURCE,
                6,
                "RAG training optimizes the negative marginal log-likelihood objective.",
            )
        ],
        "tool_result": {
            "query_results": [
                {
                    "query": "marginal log likelihood equation",
                    "source": RAG_SOURCE,
                    "covered_aspect": "损失函数",
                    "result_count": 1,
                    "result_ids": [result_id],
                }
            ],
            "results": [],
        },
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.evidence_check_node(state)

    assert result["coverage_by_paper"][RAG_SOURCE]["损失函数"] == "partial"
    assert result["evidence_status"] == "sufficient_with_gaps"


def test_not_applicable_requires_direct_no_parameter_update_evidence(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    result_id = "react-prompting-no-update"
    state = {
        "question": "对比 RAG 和 ReAct 的 prompting 损失函数",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["损失函数"],
        "coverage_by_paper": {
            RAG_SOURCE: {"损失函数": "covered"},
            REACT_SOURCE: {"损失函数": "missing"},
        },
        "retrieved_docs": [
            make_chunk(
                result_id,
                REACT_SOURCE,
                7,
                "ReAct uses few-shot prompting at inference time and does not update model parameters; prompting has no training loss.",
            )
        ],
        "tool_result": {
            "query_results": [
                {
                    "query": "loss objective supervision training details",
                    "source": REACT_SOURCE,
                    "covered_aspect": "损失函数",
                    "result_count": 1,
                    "result_ids": [result_id],
                }
            ],
            "results": [],
        },
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.evidence_check_node(state)

    assert result["coverage_by_paper"][REACT_SOURCE]["损失函数"] == "not_applicable"
    assert result["evidence_status"] == "sufficient"
    assert result["evidence_sufficient"] is True


def test_comparison_missing_core_aspect_triggers_rewrite_only_for_missing(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    state = {
        "question": "对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "selected_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["研究目标", "核心方法", "训练方式", "推理流程"],
        "coverage_by_paper": {
            RAG_SOURCE: {"研究目标": "covered", "核心方法": "covered", "训练方式": "covered", "推理流程": "covered"},
            REACT_SOURCE: {"研究目标": "covered", "核心方法": "covered", "训练方式": "missing", "推理流程": "covered"},
        },
        "missing_aspects": ["训练方式"],
        "trace": [],
    }

    next_state = agent.rewrite_query_node(state)
    source_queries = next_state["tool_action"]["arguments"]["source_queries"]
    queries = [query for item in source_queries for query in item["queries"]]
    joined = " ".join(queries).lower()

    assert len(source_queries) == 1
    assert source_queries[0]["source"] == REACT_SOURCE
    assert source_queries[0]["aspect"] == "训练方式"
    assert "training" in joined
    assert "research goal" not in joined


def test_comparison_rewrite_excludes_covered_and_not_applicable_items(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    state = {
        "question": "对比 RAG 和 ReAct 的核心方法、训练方式、具体损失函数和推理流程",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "selected_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["核心方法", "训练方式", "损失函数", "推理流程"],
        "coverage_by_paper": {
            RAG_SOURCE: {"核心方法": "covered", "训练方式": "covered", "损失函数": "covered", "推理流程": "covered"},
            REACT_SOURCE: {"核心方法": "covered", "训练方式": "covered", "损失函数": "partial", "推理流程": "covered"},
        },
        "missing_aspects": ["ReAct 微调实验的具体损失函数"],
        "trace": [],
    }

    next_state = agent.rewrite_query_node(state)
    source_queries = next_state["tool_action"]["arguments"]["source_queries"]

    assert len(source_queries) == 1
    assert source_queries[0]["source"] == REACT_SOURCE
    assert source_queries[0]["aspect"] == "损失函数"
    assert source_queries[0]["status"] == "partial"
    assert "ReAct 微调实验的具体损失函数" in source_queries[0]["missing_detail"]
    assert "sources" not in next_state["tool_action"]["arguments"]


def test_comparison_rewrite_generates_distinct_queries_for_distinct_gaps(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    state = {
        "question": "对比 RAG 和 ReAct 的核心方法、训练方式、具体损失函数和推理流程",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "selected_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["核心方法", "训练方式", "损失函数", "推理流程"],
        "coverage_by_paper": {
            RAG_SOURCE: {"核心方法": "covered", "训练方式": "covered", "损失函数": "partial", "推理流程": "covered"},
            REACT_SOURCE: {"核心方法": "covered", "训练方式": "covered", "损失函数": "partial", "推理流程": "covered"},
        },
        "missing_aspects": ["RAG 的 损失函数（部分细节缺失）", "ReAct 微调实验的具体损失函数"],
        "trace": [],
    }

    next_state = agent.rewrite_query_node(state)
    source_queries = next_state["tool_action"]["arguments"]["source_queries"]
    by_source = {item["source"]: item for item in source_queries}
    rag_queries = " ".join(by_source[RAG_SOURCE]["queries"]).lower()
    react_queries = " ".join(by_source[REACT_SOURCE]["queries"]).lower()

    assert len(source_queries) == 2
    assert "marginal log likelihood" in rag_queries
    assert "rag sequence token" in rag_queries
    assert "trajectory supervision" not in rag_queries
    assert "fine-tuning loss objective" in react_queries
    assert "trajectory supervision" in react_queries


def test_comparison_initial_plan_uses_three_shared_queries(
    tools: agent_tools.PaperTools,
) -> None:
    llm = FakeJsonLLM(['{"action":"answer","arguments":{}}'])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")
    state = {
        "question": "对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "selected_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["研究目标", "核心方法", "训练方式", "推理流程"],
        "missing_aspects": ["研究目标", "核心方法", "训练方式", "推理流程"],
    }

    planned = agent._plan_next_action(state)

    assert planned.action == "search_multiple_queries"
    assert len(planned.arguments["queries"]) == 3
    assert all(len(query.split()) <= 18 for query in planned.arguments["queries"])
    assert planned.arguments["sources"] == [RAG_SOURCE, REACT_SOURCE]


def test_comparison_two_papers_total_retrievals_do_not_exceed_six(
    tools: agent_tools.PaperTools,
) -> None:
    progress: list[tuple[int, int, str]] = []

    result = tools.search_multiple_queries_tool(
        queries=paper_agent.build_comparison_queries(["研究目标", "核心方法", "训练方式", "推理流程"]),
        sources=[RAG_SOURCE, REACT_SOURCE],
        k_per_query=2,
        progress_callback=lambda index, total, query: progress.append((index, total, query)),
    )

    assert len(result["queries"]) == 3
    assert len(result["query_results"]) == 6
    assert progress[-1][1] == 6


def test_comparison_does_not_continue_for_related_gaps(
    tools: agent_tools.PaperTools,
) -> None:
    llm = FakeJsonLLM(
        [
            "RAG 与 ReAct 的比较[1][2]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=3)
    initial_state = {
        "messages": [],
        "question": "对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程",
        "intent": "paper_comparison",
        "selected_papers": [RAG_SOURCE, REACT_SOURCE],
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["研究目标", "核心方法", "训练方式", "推理流程"],
        "related_aspects": ["优势与局限", "GPU 数量"],
        "required_aspects": ["研究目标", "核心方法", "训练方式", "推理流程"],
        "coverage_by_paper": {
            RAG_SOURCE: {"研究目标": "covered", "核心方法": "covered", "训练方式": "covered", "推理流程": "covered"},
            REACT_SOURCE: {"研究目标": "covered", "核心方法": "covered", "训练方式": "covered", "推理流程": "covered"},
        },
        "retrieval_history": [],
        "retrieved_docs": [
            agent_tools.PaperChunk(result_id="r1", source=RAG_SOURCE, page=3, metadata_page=2, score=0.9, score_label="relevance_score", content="RAG evidence."),
            agent_tools.PaperChunk(result_id="r2", source=REACT_SOURCE, page=4, metadata_page=3, score=0.9, score_label="relevance_score", content="ReAct evidence."),
        ],
        "evidence_sufficient": False,
        "evidence_status": "insufficient",
        "trace": [],
        "trace_enabled": False,
        "iteration": 1,
        "max_iterations": 3,
    }

    checked = agent.evidence_check_node(initial_state)

    assert agent.route_after_evidence_check(checked) == "answer"
    assert checked["evidence_sufficient"] is True


def test_reproduction_terms_only_required_when_user_asks() -> None:
    normal_related = paper_agent.derive_related_aspects(
        "paper_comparison",
        ["训练方式"],
        "对比 RAG 和 ReAct 的训练方式",
    )
    reproduction_related = paper_agent.derive_related_aspects(
        "paper_comparison",
        ["训练方式"],
        "对比 RAG 和 ReAct 的训练超参数、学习率和 batch size",
    )

    assert paper_agent.requests_reproduction_details("对比 RAG 和 ReAct 的训练方式", "paper_comparison") is False
    assert "学习率" not in normal_related
    assert paper_agent.requests_reproduction_details("对比 RAG 和 ReAct 的训练超参数", "paper_comparison") is True
    assert "学习率" in reproduction_related
    assert "batch size" in reproduction_related


def test_paper_analysis_generates_independent_subqueries() -> None:
    queries = paper_agent.build_paper_analysis_queries(
        source=LORA_SOURCE,
        required_aspects=paper_agent.PAPER_ANALYSIS_REQUIRED_ASPECTS,
        missing_aspects=paper_agent.PAPER_ANALYSIS_REQUIRED_ASPECTS,
    )

    assert len(queries) >= 3
    assert len(set(queries)) == len(queries)


def test_paper_analysis_queries_are_not_keyword_soup() -> None:
    queries = paper_agent.build_paper_analysis_queries(
        source=LORA_SOURCE,
        required_aspects=paper_agent.PAPER_ANALYSIS_REQUIRED_ASPECTS,
        missing_aspects=paper_agent.PAPER_ANALYSIS_REQUIRED_ASPECTS,
    )

    assert all(len(query.split()) <= 25 for query in queries)
    assert not any("optimizer" in query.lower() and "datasets" in query.lower() and "ablation" in query.lower() for query in queries)


def test_paper_analysis_queries_do_not_invent_forbidden_terms() -> None:
    queries = paper_agent.build_paper_analysis_queries(
        source=LORA_SOURCE,
        required_aspects=paper_agent.PAPER_ANALYSIS_REQUIRED_ASPECTS,
        missing_aspects=paper_agent.PAPER_ANALYSIS_REQUIRED_ASPECTS,
    )
    joined = " ".join(queries).lower()

    for term in ["t5", "bert", "superglue", "mixed precision", "adamw"]:
        assert term not in joined


def test_rewrite_for_paper_analysis_uses_only_missing_aspects(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    state = {
        "question": "完整分析 LoRA 的方法、核心公式、训练过程和实验结果",
        "intent": "paper_analysis",
        "selected_papers": [LORA_SOURCE],
        "required_aspects": paper_agent.PAPER_ANALYSIS_REQUIRED_ASPECTS,
        "covered_aspects": ["核心方法与数学公式"],
        "missing_aspects": ["主要定量实验结果", "rank 与目标矩阵消融"],
        "trace": [],
    }

    next_state = agent.rewrite_query_node(state)
    queries = next_state["tool_action"]["arguments"]["queries"]
    joined = " ".join(queries)

    assert "quantitative" in joined.lower() or "results" in joined.lower()
    assert "rank" in joined.lower() or "target matrices" in joined.lower()
    assert "equation initialization scaling" not in joined.lower()


def test_list_papers_tool_returns_sorted_papers(tools: agent_tools.PaperTools) -> None:
    result = tools.list_papers_tool()

    assert result["papers"] == [RAG_SOURCE, LORA_SOURCE, REACT_SOURCE]


def test_search_paper_tool_can_retrieve_lora(tools: agent_tools.PaperTools) -> None:
    result = tools.search_paper_tool("LoRA low-rank adaptation", k=5)

    assert result["results"]
    assert result["results"][0]["source"] == LORA_SOURCE
    assert result["score_label"] == "distance_score"
    assert result["score_meaning"] == "距离数值越小表示越相关。"


def test_search_paper_tool_does_not_call_relevance_scores(fake_documents: list[Document]) -> None:
    tools = agent_tools.PaperTools(vector_store=WarningOnRelevanceVectorStore(fake_documents))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = tools.search_paper_tool("RAG method", source=RAG_SOURCE, k=1)

    assert result["score_label"] == "distance_score"
    assert not any("Relevance scores must be between 0 and 1" in str(item.message) for item in caught)


def test_search_paper_tool_source_filter_limits_results(tools: agent_tools.PaperTools) -> None:
    result = tools.search_paper_tool("method", source=RAG_SOURCE, k=5)

    assert result["results"]
    assert {item["source"] for item in result["results"]} == {RAG_SOURCE}


def test_search_paper_tool_clamps_k(tools: agent_tools.PaperTools) -> None:
    high = tools.search_paper_tool("LoRA", k=99)
    low = tools.search_paper_tool("LoRA", k=0)

    assert high["k"] == 10
    assert low["k"] == 1


def test_neighbor_chunks_tool_returns_nearby_pages(tools: agent_tools.PaperTools) -> None:
    result = tools.get_neighbor_chunks_tool(source=LORA_SOURCE, page=1, radius=1)

    assert result["results"]
    assert {item["page"] for item in result["results"]} == {1, 2}


def test_search_multiple_queries_deduplicates_repeated_result_ids(tools: agent_tools.PaperTools) -> None:
    result = tools.search_multiple_queries_tool(
        queries=["LoRA low-rank equation", "LoRA low-rank equation"],
        source=LORA_SOURCE,
        k_per_query=4,
    )
    result_ids = [item["result_id"] for item in result["results"]]

    assert len(result_ids) == len(set(result_ids))


def test_search_multiple_queries_can_cover_method_and_experiment(tools: agent_tools.PaperTools) -> None:
    result = tools.search_multiple_queries_tool(
        queries=["LoRA low-rank equation", "LoRA experiments results"],
        source=LORA_SOURCE,
        k_per_query=4,
    )

    covered = {item["covered_aspect"] for item in result["query_results"]}
    assert "核心方法与数学公式" in covered
    assert "主要定量实验结果" in covered
    assert result["score_label"] == "distance_score"


def test_search_multiple_queries_accepts_two_sources(tools: agent_tools.PaperTools) -> None:
    progress: list[tuple[int, int, str]] = []

    result = tools.search_multiple_queries_tool(
        queries=["research goal motivation", "inference execution flow"],
        sources=[RAG_SOURCE, REACT_SOURCE],
        k_per_query=2,
        progress_callback=lambda index, total, query: progress.append((index, total, query)),
    )

    assert result["source_filters"] == [RAG_SOURCE, REACT_SOURCE]
    assert {item["source"] for item in result["results"]} == {RAG_SOURCE, REACT_SOURCE}
    assert {(item["source"], item["query"]) for item in result["query_results"]} == {
        (RAG_SOURCE, "research goal motivation"),
        (RAG_SOURCE, "inference execution flow"),
        (REACT_SOURCE, "research goal motivation"),
        (REACT_SOURCE, "inference execution flow"),
    }
    assert all(item["covered_aspect"] for item in result["query_results"])
    assert [item[1] for item in progress] == [4, 4, 4, 4]


def test_search_multiple_queries_retrieves_both_comparison_papers(
    tools: agent_tools.PaperTools,
) -> None:
    result = tools.search_multiple_queries_tool(
        queries=["core method architecture"],
        sources=[RAG_SOURCE, REACT_SOURCE],
        k_per_query=2,
    )

    assert {item["source"] for item in result["results"]} == {RAG_SOURCE, REACT_SOURCE}


def test_validate_search_multiple_queries_inherits_comparison_papers(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    planned = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_multiple_queries",
            intent="paper_comparison",
            arguments={
                "queries": ["research goal motivation", "core method architecture"],
                "k_per_query": 4,
            },
        ),
        {
            "question": "对比 RAG 和 ReAct",
            "intent": "paper_comparison",
            "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
            "selected_papers": [],
        },
    )

    assert not planned.error
    assert planned.arguments["sources"] == [RAG_SOURCE, REACT_SOURCE]
    assert "source" not in planned.arguments


def test_validate_search_multiple_queries_canonicalizes_selected_paper_alias() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
    )
    planned = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_multiple_queries",
            intent="method_explain",
            arguments={"queries": ["scaled dot product attention equation"]},
        ),
        {
            "question": "请解释 Transformer 论文中的缩放点积注意力公式。",
            "intent": "method_explain",
            "comparison_papers": [],
            "selected_papers": ["Transformer"],
        },
    )

    assert not planned.error
    assert planned.arguments["source"] == TRANSFORMER_SOURCE


def test_validate_search_multiple_queries_rejects_illegal_source(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))
    planned = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_multiple_queries",
            intent="paper_comparison",
            arguments={
                "queries": ["research goal motivation"],
                "sources": [RAG_SOURCE, "../secret.pdf"],
                "k_per_query": 4,
            },
        ),
        {
            "question": "对比 RAG 和非法论文",
            "intent": "paper_comparison",
            "comparison_papers": [RAG_SOURCE, "../secret.pdf"],
            "selected_papers": [],
        },
    )

    assert planned.action == "answer"
    assert "source 不在当前数据库" in planned.error


def test_rag_react_comparison_without_explicit_sources_does_not_require_source(
    tools: agent_tools.PaperTools,
) -> None:
    llm = FakeJsonLLM(
        [
            '{"action":"search_multiple_queries","intent":"paper_comparison","arguments":{"queries":["research goal motivation","core method architecture"],"k_per_query":2},"reason_summary":"按比较维度检索"}',
            '{"sufficient":true,"reason":"证据充分","covered_aspects":["研究目标","核心方法"],"missing_aspects":[]}',
            "RAG 与 ReAct 可从研究目标和核心方法比较[1][2]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程")

    assert "source 是必填参数" not in result["answer"]
    assert "工具执行失败" not in result["answer"]
    assert result["state"].get("error") in (None, "")
    assert result["state"]["tool_result"]["source_filters"] == [RAG_SOURCE, REACT_SOURCE]
    assert {item["source"] for item in result["state"]["tool_result"]["results"]} == {
        RAG_SOURCE,
        REACT_SOURCE,
    }


def test_agent_stops_at_max_iterations(tools: agent_tools.PaperTools) -> None:
    llm = FakeJsonLLM(
        [
            '{"action":"search_paper","arguments":{"query":"LoRA method","k":5},"reason_summary":"检索 LoRA 方法"}',
            '{"sufficient":false,"reason":"缺少实验","missing_aspects":["实验"],"suggested_query":"LoRA experiments"}',
            '{"query":"LoRA experiments results"}',
            '{"sufficient":false,"reason":"仍缺少实验","missing_aspects":["实验"],"suggested_query":"LoRA results"}',
            "LoRA 只训练低秩增量矩阵[1]。\n\n【资料来源】\n[1] 05 LoRA - Low-Rank Adaptation of Large Language Models.pdf，第 1 页",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("LoRA 的方法是什么？")

    assert result["state"]["iteration"] == 1
    assert result["answer"]


def test_clarification_without_context_for_pronoun(tools: agent_tools.PaperTools) -> None:
    llm = FakeJsonLLM(['{"action":"list_papers","arguments":{}}'])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    result = agent.ask("这篇论文的方法是什么？")

    assert "请明确" in result["answer"]
    assert result["state"]["intent"] == "clarification"


def test_citation_numbers_match_unique_sources() -> None:
    docs = [
        agent_tools.PaperChunk(
            result_id="r1",
            source=LORA_SOURCE,
            page=1,
            metadata_page=0,
            score=0.9,
            score_label="relevance_score",
            content="LoRA freezes weights.",
        ),
        agent_tools.PaperChunk(
            result_id="r2",
            source=LORA_SOURCE,
            page=1,
            metadata_page=0,
            score=0.88,
            score_label="relevance_score",
            content="Duplicate source.",
        ),
    ]

    evidence, sources = paper_agent.build_evidence_block(docs)

    assert evidence.count("[1]") >= 1
    assert len(sources) == 1


def test_unused_sources_are_removed_from_final_source_list() -> None:
    answer = "正文只引用第一条[1]，没有引用第二条。"
    sources = [
        {"source": RAG_SOURCE, "page": 3},
        {"source": REACT_SOURCE, "page": 4},
    ]

    body, used_sources = paper_agent.compact_used_citations(answer, sources)

    assert body == "正文只引用第一条[1]，没有引用第二条。"
    assert used_sources == [{"source": RAG_SOURCE, "page": 3}]
    assert REACT_SOURCE not in paper_agent.format_source_list(used_sources)


def test_citations_are_renumbered_continuously() -> None:
    answer = "先引用第三条[3]，再引用第一条[1][3]。"
    sources = [
        {"source": RAG_SOURCE, "page": 3},
        {"source": LORA_SOURCE, "page": 1},
        {"source": REACT_SOURCE, "page": 4},
    ]

    body, used_sources = paper_agent.compact_used_citations(answer, sources)

    assert body == "先引用第三条[1]，再引用第一条[2][1]。"
    assert used_sources == [
        {"source": REACT_SOURCE, "page": 4},
        {"source": RAG_SOURCE, "page": 3},
    ]
    assert "[3]" not in body


def test_citation_renumbering_does_not_cascade_replace() -> None:
    sources = [
        {"source": "old-1.pdf", "page": 1},
        {"source": "old-2.pdf", "page": 2},
        {"source": "old-3.pdf", "page": 3},
    ]

    body, used_sources = paper_agent.compact_used_citations("A[3] B[1] C[2]", sources)

    assert body == "A[1] B[2] C[3]"
    assert used_sources == [sources[2], sources[0], sources[1]]


def test_non_contiguous_citations_keep_original_source_order_by_body() -> None:
    sources = [{"source": f"source-{idx}.pdf", "page": idx} for idx in range(1, 11)]

    body, used_sources = paper_agent.compact_used_citations("A[1] B[6] C[10]", sources)

    assert body == "A[1] B[2] C[3]"
    assert used_sources == [sources[0], sources[5], sources[9]]


def test_repeated_old_citation_maps_to_same_new_number() -> None:
    sources = [
        {"source": "old-1.pdf", "page": 1},
        {"source": "old-2.pdf", "page": 2},
        {"source": "old-3.pdf", "page": 3},
    ]

    body, used_sources = paper_agent.compact_used_citations("A[3] B[3] C[1]", sources)

    assert body == "A[1] B[1] C[2]"
    assert used_sources == [sources[2], sources[0]]


def test_out_of_range_citations_are_deleted_during_compaction() -> None:
    sources = [{"source": f"source-{idx}.pdf", "page": idx} for idx in range(1, 14)]

    body, used_sources = paper_agent.compact_used_citations("A[1] B[16]", sources)

    assert body == "A[1] B"
    assert used_sources == [sources[0]]
    assert max(int(item) for item in paper_agent.CITATION_RE.findall(body)) <= len(used_sources)


def test_compacted_source_ids_match_body_ids_exactly() -> None:
    sources = [{"source": f"source-{idx}.pdf", "page": idx} for idx in range(1, 7)]

    body, used_sources = paper_agent.compact_used_citations("A[1] B[6] C[3] D[6]", sources)
    body, used_sources = paper_agent.validate_final_citations(body, used_sources)

    body_ids = {int(item) for item in paper_agent.CITATION_RE.findall(body)}
    source_ids = set(range(1, len(used_sources) + 1))

    assert body_ids == source_ids
    assert all(f"[{idx}]" in body for idx in source_ids)


def test_compaction_uses_body_order_not_page_order() -> None:
    sources = [
        {"source": "page-10.pdf", "page": 10},
        {"source": "page-1.pdf", "page": 1},
        {"source": "page-5.pdf", "page": 5},
    ]

    body, used_sources = paper_agent.compact_used_citations("First[1] Second[3] Third[2]", sources)

    assert body == "First[1] Second[2] Third[3]"
    assert used_sources == [sources[0], sources[2], sources[1]]


def test_compaction_does_not_merge_distinct_old_ids_with_same_source_page() -> None:
    sources = [
        {"source": RAG_SOURCE, "page": 3},
        {"source": RAG_SOURCE, "page": 3},
        {"source": REACT_SOURCE, "page": 4},
    ]

    body, used_sources = paper_agent.compact_used_citations("A[1] B[2] C[3]", sources)

    assert body == "A[1] B[2] C[3]"
    assert used_sources == [sources[0], sources[1], sources[2]]


def test_model_generated_source_list_is_stripped_and_regenerated() -> None:
    sources = [
        {"source": RAG_SOURCE, "page": 3},
        {"source": REACT_SOURCE, "page": 4},
    ]
    answer = (
        "正文引用第二条[2]。\n\n"
        "【资料来源】\n"
        "[1] 模型编的来源.pdf，第 99 页"
    )

    body, used_sources = paper_agent.compact_used_citations(answer, sources)
    final = f"{body}\n\n{paper_agent.format_source_list(used_sources)}"

    assert "模型编的来源" not in final
    assert body == "正文引用第二条[1]。"
    assert used_sources == [sources[1]]


def test_answer_citations_and_sources_match_exactly() -> None:
    answer = "A[2] B[4] C[2] D[99]。"
    sources = [
        {"source": RAG_SOURCE, "page": 3},
        {"source": LORA_SOURCE, "page": 1},
        {"source": REACT_SOURCE, "page": 4},
        {"source": "extra.pdf", "page": 9},
    ]

    body = paper_agent.remove_invalid_citations(answer, len(sources))
    body, used_sources = paper_agent.compact_used_citations(body, sources)
    final = f"{body}\n\n{paper_agent.format_source_list(used_sources)}"

    body_refs = set(int(item) for item in paper_agent.CITATION_RE.findall(body))
    source_refs = set(range(1, len(used_sources) + 1))

    assert body_refs == source_refs
    assert "99" not in final


def test_duplicate_source_citation_is_listed_once() -> None:
    answer = "同一来源多次引用[1][1]。"
    sources = [{"source": RAG_SOURCE, "page": 3}]

    body, used_sources = paper_agent.compact_used_citations(answer, sources)

    assert body == "同一来源多次引用[1][1]。"
    assert used_sources == [{"source": RAG_SOURCE, "page": 3}]


def test_invalid_citation_numbers_are_removed() -> None:
    answer = "合法[1]，非法[5]。"

    cleaned = paper_agent.remove_invalid_citations(answer, 1)

    assert cleaned == "合法[1]，非法。"


def test_answer_node_does_not_append_unused_sources(tools: agent_tools.PaperTools) -> None:
    llm = FakeJsonLLM(["只引用 ReAct 证据[2]。"])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")
    state = {
        "question": "对比 RAG 和 ReAct",
        "intent": "paper_comparison",
        "retrieved_docs": [
            agent_tools.PaperChunk(result_id="r1", source=RAG_SOURCE, page=3, metadata_page=2, score=0.9, score_label="relevance_score", content="RAG evidence."),
            agent_tools.PaperChunk(result_id="r2", source=REACT_SOURCE, page=4, metadata_page=3, score=0.9, score_label="relevance_score", content="ReAct evidence."),
        ],
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.answer_node(state)

    assert "只引用 ReAct 证据[1]。" in result["final_answer"]
    assert REACT_SOURCE in result["final_answer"]
    assert RAG_SOURCE not in result["final_answer"]


def test_final_answer_does_not_ask_to_continue(tools: agent_tools.PaperTools) -> None:
    llm = FakeJsonLLM(["结论内容[1]。如果你还想继续分析，我可以继续。"])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")
    state = {
        "question": "对比 RAG 和 ReAct",
        "intent": "paper_comparison",
        "retrieved_docs": [
            agent_tools.PaperChunk(result_id="r1", source=RAG_SOURCE, page=3, metadata_page=2, score=0.9, score_label="relevance_score", content="RAG evidence."),
        ],
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.answer_node(state)

    assert "继续分析" not in result["final_answer"]
    assert "还想" not in result["final_answer"]


def test_temporary_429_retries_with_backoff(monkeypatch: pytest.MonkeyPatch, fake_documents: list[Document]) -> None:
    sleeps: list[int] = []
    monkeypatch.setattr(agent_tools.time, "sleep", lambda seconds: sleeps.append(seconds))
    tools = agent_tools.PaperTools(
        vector_store=FlakyScoreVectorStore(
            fake_documents,
            [RuntimeError("HTTP 429"), RuntimeError("RateLimitError"), RuntimeError("负载已饱和")],
        )
    )

    result = tools.search_paper_tool("RAG method", source=RAG_SOURCE, k=1)

    assert result["results"]
    assert sleeps == [1, 2, 4]


def test_non_temporary_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch, fake_documents: list[Document]) -> None:
    sleeps: list[int] = []
    monkeypatch.setattr(agent_tools.time, "sleep", lambda seconds: sleeps.append(seconds))
    tools = agent_tools.PaperTools(
        vector_store=FlakyScoreVectorStore(fake_documents, [ValueError("bad query")])
    )

    with pytest.raises(ValueError):
        tools.search_paper_tool("RAG method", source=RAG_SOURCE, k=1)

    assert sleeps == []
    assert tools.vector_store.calls == 1


def test_multi_query_keeps_partial_results_after_temporary_failures(
    monkeypatch: pytest.MonkeyPatch,
    fake_documents: list[Document],
) -> None:
    monkeypatch.setattr(agent_tools.time, "sleep", lambda _seconds: None)
    tools = agent_tools.PaperTools(
        vector_store=FlakyScoreVectorStore(
            fake_documents,
            [
                RuntimeError("HTTP 429"),
                RuntimeError("HTTP 429"),
                RuntimeError("HTTP 429"),
                RuntimeError("HTTP 429"),
            ],
        )
    )

    result = tools.search_multiple_queries_tool(
        queries=["research goal motivation", "core method architecture"],
        sources=[RAG_SOURCE],
        k_per_query=1,
    )

    assert result["results"]
    assert any(item.get("error") for item in result["query_results"])


def test_search_multiple_queries_marks_all_429_failures_as_failed(
    monkeypatch: pytest.MonkeyPatch,
    fake_documents: list[Document],
) -> None:
    monkeypatch.setattr(agent_tools.time, "sleep", lambda _seconds: None)
    tools = agent_tools.PaperTools(
        vector_store=AlwaysFailScoreVectorStore(fake_documents, RuntimeError("HTTP 429"))
    )

    result = tools.search_multiple_queries_tool(
        queries=["research goal motivation", "core method architecture"],
        sources=[RAG_SOURCE],
        k_per_query=1,
    )

    assert result["retrieval_status"] == "failed"
    assert result["successful_query_count"] == 0
    assert result["failed_query_count"] == 2
    assert all(item["success"] is False for item in result["query_results"])
    assert all(item["error_type"] == "RuntimeError" for item in result["query_results"])


def test_search_multiple_queries_partial_failure_keeps_successful_results(
    monkeypatch: pytest.MonkeyPatch,
    fake_documents: list[Document],
) -> None:
    monkeypatch.setattr(agent_tools.time, "sleep", lambda _seconds: None)
    tools = agent_tools.PaperTools(
        vector_store=FlakyScoreVectorStore(
            fake_documents,
            [
                RuntimeError("HTTP 429"),
                RuntimeError("HTTP 429"),
                RuntimeError("HTTP 429"),
                RuntimeError("HTTP 429"),
            ],
        )
    )

    result = tools.search_multiple_queries_tool(
        queries=["research goal motivation", "core method architecture"],
        sources=[RAG_SOURCE],
        k_per_query=1,
    )

    assert result["retrieval_status"] == "partial_failure"
    assert result["successful_query_count"] == 1
    assert result["failed_query_count"] == 1
    assert result["results"]
    assert [item["success"] for item in result["query_results"]] == [False, True]


def test_search_multiple_queries_circuit_breaks_after_two_final_failures(
    monkeypatch: pytest.MonkeyPatch,
    fake_documents: list[Document],
) -> None:
    monkeypatch.setattr(agent_tools.time, "sleep", lambda _seconds: None)
    tools = agent_tools.PaperTools(
        vector_store=AlwaysFailScoreVectorStore(fake_documents, RuntimeError("HTTP 429"))
    )

    result = tools.search_multiple_queries_tool(
        queries=["research goal motivation", "core method architecture", "training objective"],
        sources=[RAG_SOURCE],
        k_per_query=1,
    )

    assert result["retrieval_status"] == "failed"
    assert result["circuit_breaker_triggered"] is True
    assert result["failed_query_count"] == agent_tools.MAX_CONSECUTIVE_QUERY_FAILURES
    assert len(result["query_results"]) == agent_tools.MAX_CONSECUTIVE_QUERY_FAILURES
    assert tools.vector_store.calls == agent_tools.MAX_CONSECUTIVE_QUERY_FAILURES * (
        len(agent_tools.RETRY_DELAYS_SECONDS) + 1
    )


def test_json_planner_rejects_unknown_action() -> None:
    action = paper_agent.parse_json_action('{"action":"delete_db","arguments":{}}')

    assert action.action == "answer"
    assert "未知 action" in action.error


def test_json_planner_rejects_non_dict_arguments() -> None:
    action = paper_agent.parse_json_action('{"action":"search_paper","arguments":["bad"]}')

    assert action.action == "answer"
    assert "参数必须是对象" in action.error


def test_json_code_block_can_be_parsed() -> None:
    action = paper_agent.parse_json_action(
        '```json\n{"action":"list_papers","arguments":{},"reason_summary":"列出论文"}\n```'
    )

    assert action.action == "list_papers"
    assert action.reason_summary == "列出论文"


def test_json_with_surrounding_text_can_be_extracted() -> None:
    action = paper_agent.parse_json_action(
        '好的，下一步如下：{"action":"answer","arguments":{},"reason_summary":"证据充分"}'
    )

    assert action.action == "answer"


def test_invalid_json_returns_clear_error() -> None:
    action = paper_agent.parse_json_action("not json at all")

    assert action.action == "answer"
    assert "JSON Planner 解析失败" in action.error


def test_json_planner_does_not_use_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_eval(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("eval must not be called")

    monkeypatch.setattr(builtins, "eval", fail_eval)

    action = paper_agent.parse_json_action('{"action":"list_papers","arguments":{}}')

    assert action.action == "list_papers"


def test_json_mode_does_not_call_bind_tools(tools: agent_tools.PaperTools) -> None:
    llm = BindExplodingLLM(
        [
            '{"action":"search_paper","arguments":{"query":"LoRA method","k":5},"reason_summary":"检索 LoRA"}',
            '{"sufficient":true,"reason":"证据充分","missing_aspects":[]}',
            "LoRA 方法基于低秩增量[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    result = agent.ask("LoRA 的方法是什么？")

    assert result["answer"]


def test_valid_search_json_executes_tool(tools: agent_tools.PaperTools) -> None:
    llm = FakeJsonLLM(
        [
            '{"action":"search_paper","arguments":{"query":"LoRA low-rank","source":"05 LoRA - Low-Rank Adaptation of Large Language Models.pdf","k":5},"reason_summary":"检索 LoRA"}',
            '{"sufficient":true,"reason":"证据充分","missing_aspects":[]}',
            "LoRA 训练低秩增量[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    result = agent.ask("LoRA 的方法是什么？")

    assert result["state"]["tool_result"]["source_filter"] == LORA_SOURCE
    assert result["state"]["tool_result"]["k"] == 5


def test_planner_argument_validation_clamps_k_and_radius(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))

    search = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_paper",
            arguments={"query": "LoRA", "source": LORA_SOURCE, "k": 99},
        )
    )
    neighbor = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="get_neighbor_chunks",
            arguments={"source": LORA_SOURCE, "page": 1, "radius": 99},
        )
    )

    assert search.arguments["k"] == 10
    assert neighbor.arguments["radius"] == 2


def test_planner_rejects_source_outside_current_database(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))

    planned = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_paper",
            arguments={"query": "LoRA", "source": "../secret.pdf", "k": 5},
        )
    )

    assert planned.action == "answer"
    assert "source 不在当前数据库" in planned.error


@pytest.mark.parametrize(
    "planner_source",
    ["Transformer 论文", "Transformer   论文"],
)
def test_planner_source_alias_is_canonicalized_before_search(
    planner_source: str,
) -> None:
    tools = RecordingPaperTools()
    llm = FakeJsonLLM(
        [
            json.dumps(
                {
                    "action": "search_paper",
                    "intent": "general_qa",
                    "arguments": {
                        "query": "scaled dot product attention equation",
                        "source": planner_source,
                        "k": 4,
                    },
                }
            ),
            '{"sufficient":true,"reason":"证据充分","covered_aspects":[],"missing_aspects":[]}',
            "Transformer 回答[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("请解释 Transformer 论文中的缩放点积注意力公式。")

    assert result["state"]["selected_papers"] == [TRANSFORMER_SOURCE]
    assert tools.search_calls[0]["source"] == TRANSFORMER_SOURCE
    assert result["state"].get("paper_not_found") is not True
    assert result["state"].get("unavailable_papers", []) == []


def test_drag_has_no_explicit_rag_alias() -> None:
    assert paper_agent.detect_explicit_papers(CORPUS_DRAG_QUESTION) == []


def test_dragging_has_no_explicit_rag_alias() -> None:
    assert paper_agent.detect_explicit_papers("these papers discuss dragging or drag-and-drop") == []


def test_standalone_rag_is_explicitly_matched() -> None:
    assert paper_agent.detect_explicit_papers("对比 RAG 和 ReAct") == [RAG_SOURCE, REACT_SOURCE]


def test_corpus_drag_question_has_no_single_source() -> None:
    tools = RecordingPaperTools()
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))

    planned = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_paper",
            intent="general_qa",
            arguments={"query": "drag dragging drag-and-drop", "source": RAG_SOURCE, "k": 5},
        ),
        {
            "question": CORPUS_DRAG_QUESTION,
            "intent": "general_qa",
            "selected_papers": [],
            "comparison_papers": [],
        },
    )

    assert not planned.error
    assert planned.arguments["source"] is None


def test_corpus_drag_question_uses_full_catalog() -> None:
    tools = RecordingPaperTools()
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))

    planned = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_multiple_queries",
            intent="general_qa",
            arguments={
                "queries": ["drag dragging drag-and-drop"],
                "source": RAG_SOURCE,
                "k_per_query": 4,
            },
        ),
        {
            "question": CORPUS_DRAG_QUESTION,
            "intent": "general_qa",
            "selected_papers": [],
            "comparison_papers": [],
        },
    )

    assert not planned.error
    assert "source" not in planned.arguments
    assert planned.arguments["sources"] == tools.papers


def test_planner_spurious_rag_source_is_removed_for_corpus_scope() -> None:
    tools = RecordingPaperTools()
    llm = FakeJsonLLM(
        [
            json.dumps(
                {
                    "action": "search_paper",
                    "intent": "general_qa",
                    "arguments": {
                        "query": "drag dragging drag-and-drop",
                        "source": RAG_SOURCE,
                        "k": 5,
                    },
                },
                ensure_ascii=False,
            ),
            '{"sufficient":true,"reason":"已检索全库","covered_aspects":[],"missing_aspects":[]}',
            "未发现 drag 相关证据[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask(CORPUS_DRAG_QUESTION)

    assert result["state"]["intent"] == "general_qa"
    assert tools.search_calls[0]["query"] == "drag dragging drag-and-drop"
    assert tools.search_calls[0]["source"] is None


def test_explicit_rag_question_keeps_rag_source() -> None:
    tools = RecordingPaperTools()
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))

    planned = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_paper",
            intent="general_qa",
            arguments={"query": "retriever generator", "source": RAG_SOURCE, "k": 5},
        ),
        {
            "question": "RAG 论文是否讨论 retriever？",
            "intent": "general_qa",
            "selected_papers": [RAG_SOURCE],
            "comparison_papers": [],
        },
    )

    assert not planned.error
    assert planned.arguments["source"] == RAG_SOURCE


def test_corpus_scope_with_rag_and_react_keeps_two_explicit_papers() -> None:
    tools = RecordingPaperTools()
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']))

    planned = agent.validate_planned_action(
        paper_agent.PlannedAction(
            action="search_multiple_queries",
            intent="paper_comparison",
            arguments={
                "queries": ["core method architecture"],
                "source": RAG_SOURCE,
                "k_per_query": 4,
            },
        ),
        {
            "question": "在这些论文中比较 RAG 和 ReAct",
            "intent": "paper_comparison",
            "selected_papers": [RAG_SOURCE, REACT_SOURCE],
            "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        },
    )

    assert not planned.error
    assert planned.arguments["sources"] == [RAG_SOURCE, REACT_SOURCE]


def test_code_does_not_contain_eval_or_exec() -> None:
    source = Path("paper_agent.py").read_text(encoding="utf-8")

    assert "eval(" not in source
    assert "exec(" not in source


def test_auto_mode_falls_back_to_json_planner(tools: agent_tools.PaperTools) -> None:
    llm = FailingNativeLLM(
        [
            '{"action":"list_papers","arguments":{},"reason_summary":"列出论文"}',
            '{"sufficient":true,"reason":"已列出论文","missing_aspects":[]}',
            "当前数据库包含论文[1]。\n\n【资料来源】\n[1] 05 LoRA - Low-Rank Adaptation of Large Language Models.pdf，第 1 页",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="auto")

    result = agent.ask("有哪些论文？")

    assert any("JSON Planner" in event for event in result["trace"])


def test_native_tool_schema_hides_internal_retry_callbacks(tools: agent_tools.PaperTools) -> None:
    schemas = {
        tool.name: tool.args
        for tool in tools.as_langchain_tools()
        if tool.name in {"search_paper_tool", "search_multiple_queries_tool"}
    }

    assert "retry_callback" not in schemas["search_paper_tool"]
    assert "retry_callback" not in schemas["search_multiple_queries_tool"]


def test_emit_progress_uses_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_print(*args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(builtins, "print", fake_print)

    paper_agent.emit_progress("正在分析问题……", trace_enabled=False, category="当前状态")

    assert calls
    assert calls[0]["flush"] is True


def test_trace_off_only_shows_basic_status(capsys: pytest.CaptureFixture[str]) -> None:
    paper_agent.emit_progress("正在分析问题……", trace_enabled=False, category="当前状态")
    paper_agent.emit_progress("paper_analysis", trace_enabled=False, category="任务判断")

    output = capsys.readouterr().out

    assert "正在分析问题" in output
    assert "paper_analysis" not in output


def test_trace_on_shows_detailed_progress(capsys: pytest.CaptureFixture[str]) -> None:
    paper_agent.emit_progress("paper_analysis", trace_enabled=True, category="任务判断")

    output = capsys.readouterr().out

    assert "【任务判断】paper_analysis" in output


def test_planner_emits_progress_before_llm_call(monkeypatch: pytest.MonkeyPatch, tools: agent_tools.PaperTools) -> None:
    events: list[tuple[str, str]] = []

    def fake_emit(message: str, *, trace_enabled: bool, category: str = "状态") -> None:
        events.append((category, message))

    monkeypatch.setattr(paper_agent, "emit_progress", fake_emit)
    llm = FakeJsonLLM(
        [
            '{"action":"search_paper","arguments":{"query":"LoRA method","k":5},"reason_summary":"检索"}',
            '{"sufficient":true,"reason":"充分","missing_aspects":[]}',
            "答案[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    agent.ask("LoRA 的方法是什么？")

    assert events[0] == ("当前状态", "正在分析问题……")
    assert ("当前状态", "正在生成行动计划……") in events


def test_retrieval_emits_progress_before_tool_call(monkeypatch: pytest.MonkeyPatch, tools: agent_tools.PaperTools) -> None:
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        paper_agent,
        "emit_progress",
        lambda message, *, trace_enabled, category="状态": events.append((category, message)),
    )
    llm = FakeJsonLLM(
        [
            '{"action":"search_paper","arguments":{"query":"LoRA method","k":5},"reason_summary":"检索"}',
            '{"sufficient":true,"reason":"充分","missing_aspects":[]}',
            "答案[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    agent.ask("LoRA 的方法是什么？")

    assert ("当前状态", "正在检索论文……") in events


def test_evidence_check_emits_progress_before_llm_call(monkeypatch: pytest.MonkeyPatch, tools: agent_tools.PaperTools) -> None:
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        paper_agent,
        "emit_progress",
        lambda message, *, trace_enabled, category="状态": events.append((category, message)),
    )
    llm = FakeJsonLLM(
        [
            '{"action":"search_paper","arguments":{"query":"LoRA method","k":5},"reason_summary":"检索"}',
            '{"sufficient":true,"reason":"充分","missing_aspects":[]}',
            "答案[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    agent.ask("LoRA 的方法是什么？")

    assert ("当前状态", "正在检查证据覆盖……") in events


def test_answer_node_streams_tokens_and_keeps_final_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tools: agent_tools.PaperTools,
) -> None:
    monkeypatch.setenv("STREAM_FINAL_ANSWER", "true")
    llm = FakeStreamingLLM([], stream_chunks=["LoRA", " 答案", "[1]。"])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")
    state = {
        "question": "LoRA 的方法是什么？",
        "intent": "method_explain",
        "retrieved_docs": [
            agent_tools.PaperChunk(
                result_id="r1",
                source=LORA_SOURCE,
                page=1,
                metadata_page=0,
                score=0.9,
                score_label="relevance_score",
                content="LoRA freezes weights.",
            )
        ],
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.answer_node(state)

    assert result["final_answer_streamed"] is True
    assert "LoRA 答案[1]。" in result["final_answer"]
    assert capsys.readouterr().out.count(result["final_answer"]) == 1


def test_stream_true_does_not_print_raw_chunks_before_citation_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tools: agent_tools.PaperTools,
) -> None:
    monkeypatch.setenv("STREAM_FINAL_ANSWER", "true")
    llm = FakeStreamingLLM(
        [],
        stream_chunks=[
            "RAG 内容[2]，非法引用[16]。",
            "\n\n【资料来源】\n[1] 模型自带来源.pdf，第 99 页",
        ],
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")
    state = {
        "question": "对比 RAG 和 ReAct",
        "intent": "paper_comparison",
        "retrieved_docs": [
            agent_tools.PaperChunk(result_id="r1", source=RAG_SOURCE, page=3, metadata_page=2, score=0.1, score_label="distance_score", content="RAG evidence."),
            agent_tools.PaperChunk(result_id="r2", source=REACT_SOURCE, page=4, metadata_page=3, score=0.2, score_label="distance_score", content="ReAct evidence."),
        ],
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.answer_node(state)
    output = capsys.readouterr().out

    assert result["final_answer_streamed"] is True
    assert output.count(result["final_answer"]) == 1
    assert output.strip().endswith(result["final_answer"])
    assert "[16]" not in output
    assert "模型自带来源" not in output
    assert "RAG 内容[1]，非法引用。" in output
    assert result["final_answer"] in output


def test_streamed_answer_is_not_printed_twice(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tools: agent_tools.PaperTools,
) -> None:
    monkeypatch.setenv("STREAM_FINAL_ANSWER", "true")
    llm = FakeStreamingLLM(
        [
            '{"action":"search_paper","arguments":{"query":"LoRA method","k":5},"reason_summary":"检索"}',
            '{"sufficient":true,"reason":"充分","missing_aspects":[]}',
        ],
        stream_chunks=["一次", "输出[1]。"],
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    result = agent.ask("LoRA 的方法是什么？")
    output = capsys.readouterr().out

    assert result["state"]["final_answer_streamed"] is True
    assert result["answer"].count("一次输出") == 1
    assert output.count(result["answer"]) == 1


def test_streaming_falls_back_to_invoke(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tools: agent_tools.PaperTools) -> None:
    monkeypatch.setenv("STREAM_FINAL_ANSWER", "true")
    llm = NoStreamLLM(["完整回答[1]。"])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")
    state = {
        "question": "LoRA 的方法是什么？",
        "intent": "method_explain",
        "retrieved_docs": [
            agent_tools.PaperChunk(
                result_id="r1",
                source=LORA_SOURCE,
                page=1,
                metadata_page=0,
                score=0.9,
                score_label="relevance_score",
                content="LoRA freezes weights.",
            )
        ],
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.answer_node(state)
    output = capsys.readouterr().out

    assert result["final_answer_streamed"] is True
    assert "当前 API 不支持流式输出" in output
    assert "完整回答[1]。" in result["final_answer"]
    assert output.count(result["final_answer"]) == 1


def test_stream_fallback_invoke_prints_cleaned_final_answer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tools: agent_tools.PaperTools,
) -> None:
    monkeypatch.setenv("STREAM_FINAL_ANSWER", "true")
    llm = ExplodingStreamLLM(["fallback 引用第二条[2]，非法[16]。\n\n【资料来源】\n[1] raw.pdf，第 1 页"])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")
    state = {
        "question": "对比 RAG 和 ReAct",
        "intent": "paper_comparison",
        "retrieved_docs": [
            agent_tools.PaperChunk(result_id="r1", source=RAG_SOURCE, page=3, metadata_page=2, score=0.1, score_label="distance_score", content="RAG evidence."),
            agent_tools.PaperChunk(result_id="r2", source=REACT_SOURCE, page=4, metadata_page=3, score=0.2, score_label="distance_score", content="ReAct evidence."),
        ],
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.answer_node(state)
    output = capsys.readouterr().out

    assert llm.stream_called is True
    assert result["final_answer_streamed"] is True
    assert "[16]" not in output
    assert "raw.pdf" not in output
    assert "fallback 引用第二条[1]，非法。" in result["final_answer"]
    assert output.count(result["final_answer"]) == 1


def test_stream_false_returns_cleaned_answer_without_printing_final(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tools: agent_tools.PaperTools,
) -> None:
    monkeypatch.setenv("STREAM_FINAL_ANSWER", "false")
    llm = FakeJsonLLM(["false 模式引用第二条[2]，非法[16]。\n\n【资料来源】\n[1] raw.pdf，第 1 页"])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")
    state = {
        "question": "对比 RAG 和 ReAct",
        "intent": "paper_comparison",
        "retrieved_docs": [
            agent_tools.PaperChunk(result_id="r1", source=RAG_SOURCE, page=3, metadata_page=2, score=0.1, score_label="distance_score", content="RAG evidence."),
            agent_tools.PaperChunk(result_id="r2", source=REACT_SOURCE, page=4, metadata_page=3, score=0.2, score_label="distance_score", content="ReAct evidence."),
        ],
        "trace": [],
        "trace_enabled": False,
    }

    result = agent.answer_node(state)
    output = capsys.readouterr().out

    assert result["final_answer_streamed"] is False
    assert "[16]" not in result["final_answer"]
    assert "raw.pdf" not in result["final_answer"]
    assert "false 模式引用第二条[1]，非法。" in result["final_answer"]
    assert result["final_answer"] not in output


def test_answer_prompt_tells_model_not_to_output_source_list(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM(["答案[1]。"]), tool_mode="json")

    prompt = agent._answer_prompt(
        {"question": "LoRA 的方法是什么？", "intent": "method_explain"},
        "[1] evidence",
        [{"source": LORA_SOURCE, "page": 1}],
    )

    assert "不要输出【资料来源】列表" in prompt
    assert "来源列表由程序生成" in prompt


def test_planner_json_is_not_streamed(monkeypatch: pytest.MonkeyPatch, tools: agent_tools.PaperTools) -> None:
    llm = FakeStreamingLLM(
        [
            '{"action":"search_paper","arguments":{"query":"LoRA method","k":5},"reason_summary":"检索"}',
            '{"sufficient":true,"reason":"充分","missing_aspects":[]}',
        ],
        stream_chunks=["答案[1]。"],
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    agent.ask("LoRA 的方法是什么？")

    assert llm.stream_called is True
    assert llm.calls == 2


def test_progress_indicator_stops_normally() -> None:
    indicator = paper_agent.ProgressIndicator("测试", enabled=True, dynamic=False)

    with indicator:
        pass

    assert not indicator.is_running


def test_progress_indicator_stops_on_exception() -> None:
    indicator = paper_agent.ProgressIndicator("测试", enabled=True, dynamic=False)

    with pytest.raises(RuntimeError):
        with indicator:
            raise RuntimeError("boom")

    assert not indicator.is_running


def test_progress_indicator_ctrl_c_stops() -> None:
    indicator = paper_agent.ProgressIndicator("测试", enabled=True, dynamic=False)

    try:
        with indicator:
            raise KeyboardInterrupt()
    except KeyboardInterrupt:
        pass

    assert not indicator.is_running
    assert not any(thread.name.startswith("ProgressIndicator") and thread.is_alive() for thread in threading.enumerate())


def test_trace_sanitizes_api_key_and_reasoning_content(capsys: pytest.CaptureFixture[str]) -> None:
    paper_agent.emit_progress("sk-secret reasoning_content hidden", trace_enabled=True, category="Planner动作")

    output = capsys.readouterr().out

    assert "sk-secret" not in output
    assert "reasoning_content" not in output


def test_graph_stream_failure_falls_back_to_invoke(monkeypatch: pytest.MonkeyPatch, tools: agent_tools.PaperTools) -> None:
    llm = FakeJsonLLM(['{"action":"answer","arguments":{},"reason_summary":"无需检索"}'])
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json")

    def fail_stream(*args: Any, **kwargs: Any):
        raise TypeError("stream unsupported")

    monkeypatch.setattr(agent.graph, "stream", fail_stream)

    result = agent.ask("普通问题")

    assert result["state"]["graph_stream_fallback"] is True


def test_infer_intent_recognizes_rag_react_comparison() -> None:
    intent, papers = paper_agent.infer_intent(
        "对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程"
    )

    assert intent == "paper_comparison"
    assert papers == [
        RAG_SOURCE,
        "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf",
    ]


@pytest.mark.parametrize(
    "question",
    [
        "比较 LoRA 与 RAG 使用的方法",
        "RAG 和 ReAct 有什么区别？",
        "RAG vs ReAct",
    ],
)
def test_comparison_phrases_are_rule_routed(question: str) -> None:
    intent, papers = paper_agent.infer_intent(question)

    assert intent == "paper_comparison"
    assert len(papers) >= 2


def test_comparison_without_two_papers_requires_clarification() -> None:
    intent, papers = paper_agent.infer_intent("对比一下这两篇论文")

    assert intent == "clarification"
    assert papers == []


def test_comparison_can_inherit_recent_context() -> None:
    react_source = "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf"
    intent, papers = paper_agent.infer_intent(
        "对比一下这两篇论文",
        recent_sources=[RAG_SOURCE, react_source],
    )

    assert intent == "paper_comparison"
    assert papers == [RAG_SOURCE, react_source]


def test_infer_intent_no_longer_depends_on_undefined_llm_classifier() -> None:
    assert not hasattr(paper_agent, "infer_intent_with_llm")
    intent, _papers = paper_agent.infer_intent("RAG 的检索器怎么训练？")
    assert intent == "general_qa"


def test_agent_preserves_rule_resolved_comparison_intent(
    tools: agent_tools.PaperTools,
) -> None:
    react_source = "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf"
    # Planner deliberately returns general_qa; PaperAgent must retain the
    # high-confidence rule-resolved paper_comparison intent.
    llm = FakeJsonLLM(
        [
            '{"action":"search_paper","intent":"general_qa","arguments":{"query":"RAG ReAct comparison","k":5},"reason_summary":"检索两篇论文"}',
            '{"sufficient":true,"reason":"证据充分","covered_aspects":["研究目标","核心方法","训练方式","推理流程"],"missing_aspects":[]}',
            "RAG 与 ReAct 的比较[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("对比 RAG 和 ReAct 的研究目标、核心方法、训练方式和推理流程")

    assert result["state"]["intent"] == "paper_comparison"
    assert result["state"]["comparison_papers"] == [RAG_SOURCE, react_source]
    assert result["state"]["comparison_aspects"] == ["研究目标", "核心方法", "训练方式", "推理流程"]


def test_explicit_transformer_analysis_does_not_clarify() -> None:
    tools = RecordingPaperTools()
    llm = FakeJsonLLM(
        [
            '{"action":"clarify","arguments":{"question":"请说明你要完整分析哪一篇论文。"}}',
            '{"sufficient":true,"reason":"证据充分","covered_aspects":["模型结构"],"missing_aspects":[]}',
            "Transformer 分析答案[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("完整分析 Transformer 论文的研究问题、核心方法、训练流程和主要实验结论。")

    assert result["state"]["selected_papers"] == [TRANSFORMER_SOURCE]
    assert result["state"]["next_action"] != "clarify"
    assert tools.multiple_calls or tools.search_calls


def test_list_papers_does_not_pollute_selected_papers() -> None:
    tools = RecordingPaperTools()
    llm = FakeJsonLLM(
        [
            '{"action":"list_papers","arguments":{}}',
            '{"sufficient":true,"reason":"候选列表不是论文证据","covered_aspects":[],"missing_aspects":[]}',
            "候选列表回答。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("有哪些论文？")

    assert result["state"]["selected_papers"] == []
    assert result["state"]["comparison_papers"] == []


def test_method_explain_transformer_does_not_select_rag() -> None:
    tools = RecordingPaperTools()
    llm = FakeJsonLLM(
        [
            '{"action":"search_paper","intent":"method_explain","arguments":{"query":"scaled dot-product attention multi-head attention formula","k":5}}',
            '{"sufficient":true,"reason":"证据充分","covered_aspects":["核心方法与数学公式"],"missing_aspects":[]}',
            "Attention 公式答案[1]。",
        ]
    )
    agent = paper_agent.PaperAgent(tools=tools, llm=llm, tool_mode="json", max_iterations=1)

    result = agent.ask("解释 Transformer 的 scaled dot-product attention 和 multi-head attention 公式，各个符号分别代表什么？")

    assert [call.get("source") for call in tools.search_calls] == [TRANSFORMER_SOURCE]
    assert RAG_SOURCE not in result["state"]["selected_papers"]
    assert result["state"]["comparison_papers"] == []


def test_react_paper_analysis_queries_do_not_use_lora_terms() -> None:
    queries = paper_agent.build_paper_analysis_queries(
        source=REACT_SOURCE,
        required_aspects=paper_agent.get_paper_analysis_aspects(REACT_SOURCE),
        missing_aspects=paper_agent.get_paper_analysis_aspects(REACT_SOURCE),
    )
    joined = " ".join(queries).lower()
    assert "low-rank" not in joined
    assert "rank ablation" not in joined
    assert "target matrices" not in joined
    assert "merge inference deployment" not in joined


def test_transformer_paper_analysis_uses_attention_training_experiment_terms() -> None:
    queries = paper_agent.build_paper_analysis_queries(
        source=TRANSFORMER_SOURCE,
        required_aspects=paper_agent.get_paper_analysis_aspects(TRANSFORMER_SOURCE),
        missing_aspects=paper_agent.get_paper_analysis_aspects(TRANSFORMER_SOURCE),
    )
    joined = " ".join(queries).lower()
    assert "attention" in joined
    assert "training" in joined
    assert "experiment" in joined or "results" in joined


def test_lora_paper_analysis_keeps_lora_specific_terms() -> None:
    queries = paper_agent.build_paper_analysis_queries(
        source=LORA_SOURCE,
        required_aspects=paper_agent.get_paper_analysis_aspects(LORA_SOURCE),
        missing_aspects=paper_agent.get_paper_analysis_aspects(LORA_SOURCE),
    )
    joined = " ".join(queries).lower()
    assert "low-rank" in joined
    assert "rank" in joined
    assert "merge" in joined


def test_negative_react_nll_question_answers_without_hitl() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(["论文未明确说明 token-level NLL；它只说明了 fine-tuning/微调设置[1]。"]),
        tool_mode="json",
        enable_human_review=True,
    )
    state = {
        "question": "ReAct 论文是否明确说明了微调实验使用 token-level NLL 作为具体损失函数？请不要用常识补全。",
        "intent": "general_qa",
        "selected_papers": [REACT_SOURCE],
        "comparison_papers": [],
        "retrieved_docs": [
            make_chunk("react-ft", REACT_SOURCE, 5, "ReAct uses fine-tuning/bootstrapping trajectories but does not explicitly specify token-level NLL.")
        ],
        "required_aspects": ["损失函数", "训练方式"],
        "requested_aspects": ["损失函数", "训练方式"],
        "covered_aspects": ["训练方式"],
        "missing_aspects": ["损失函数"],
        "evidence_sufficient": False,
        "evidence_status": "insufficient",
        "evidence_reason": "当前证据仅说明论文做了 finetuning/bootstrapping 微调并给出数据来源与设置，但没有明确说明 token-level NLL 或具体损失函数。",
        "trace": [],
        "iteration": 1,
        "max_iterations": 3,
        "auto_search_rounds": 3,
        "max_auto_search_rounds": 3,
    }

    checked = agent.evidence_check_node(state)
    assert checked["human_review_required"] is False
    assert checked["evidence_status"] == "sufficient"
    assert agent.route_after_evidence_check(checked) == "answer"
    answered = agent.answer_node(checked)
    assert answered["final_answer"]
    assert "论文未明确说明" in answered["final_answer"] or "没有明确说明" in answered["final_answer"]
    assert "fine-tuning" in answered["final_answer"] or "微调" in answered["final_answer"]
    assert "明确采用 token-level NLL" not in answered["final_answer"]


def test_negative_evidence_reason_with_unprovided_routes_to_answer(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM([]), enable_human_review=True)
    state = {
        "question": "ReAct 论文是否明确说明了微调实验使用 token-level NLL？请不要用常识补全。",
        "intent": "general_qa",
        "retrieved_docs": [make_chunk("react-loss", REACT_SOURCE, 15, "fine-tuning batch size and steps")],
        "evidence_reason": "当前证据未提供任何具体训练目标或损失函数定义。",
        "evidence_status": "insufficient",
        "evidence_sufficient": False,
        "missing_aspects": ["损失函数"],
        "requested_aspects": ["损失函数"],
        "auto_search_rounds": 2,
        "max_auto_search_rounds": 2,
    }

    assert paper_agent.supports_negative_evidence_answer(
        state["question"], state["evidence_reason"], state["retrieved_docs"]
    ) is True


def test_negative_evidence_reason_with_cannot_determine_explicitly_routes_to_answer(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM([]), enable_human_review=True)
    state = {
        "question": "ReAct 论文有没有明确说明微调损失函数？",
        "intent": "general_qa",
        "retrieved_docs": [make_chunk("react-loss", REACT_SOURCE, 5, "3,000 trajectories for fine-tuning")],
        "evidence_reason": "因此无法判断论文是否明确说明使用 token-level NLL。",
        "evidence_status": "insufficient",
        "evidence_sufficient": False,
        "missing_aspects": ["损失函数"],
        "requested_aspects": ["损失函数"],
        "auto_search_rounds": 2,
        "max_auto_search_rounds": 2,
    }

    assert paper_agent.supports_negative_evidence_answer(
        state["question"], state["evidence_reason"], state["retrieved_docs"]
    ) is True


def test_react_nll_question_is_general_qa() -> None:
    intent, papers = paper_agent.infer_intent(
        "ReAct 论文是否明确说明了微调实验使用 token-level NLL 作为具体损失函数？请不要用常识补全。"
    )

    assert intent == "general_qa"
    assert papers == [REACT_SOURCE]


def test_negative_evidence_answer_is_nonempty_without_hitl() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(
            [
                '{"sufficient":false,"reason":"当前证据未提供具体损失函数定义，无法判断论文是否明确说明 token-level NLL。","covered_aspects":["训练方式"],"missing_aspects":["损失函数"]}',
                "论文未明确说明微调实验使用 token-level NLL；已有证据仅给出微调轨迹、batch size 和训练 steps [1]。",
            ]
        ),
        tool_mode="json",
        enable_human_review=True,
    )
    state = {
        "question": "ReAct 论文是否明确说明了微调实验使用 token-level NLL？请不要用常识补全。",
        "intent": "general_qa",
        "selected_papers": [REACT_SOURCE],
        "retrieved_docs": [make_chunk("react-loss", REACT_SOURCE, 15, "fine-tuning batch size and training steps")],
        "required_aspects": ["损失函数", "训练方式"],
        "requested_aspects": ["损失函数", "训练方式"],
        "covered_aspects": ["训练方式"],
        "missing_aspects": ["损失函数"],
        "trace": [],
        "iteration": 2,
        "max_iterations": 3,
        "auto_search_rounds": 2,
        "max_auto_search_rounds": 2,
    }

    checked = agent.evidence_check_node(state)
    answered = agent.answer_node(checked)

    assert checked["evidence_status"] == "sufficient"
    assert checked["human_review_required"] is False
    assert agent.route_after_evidence_check(checked) == "answer"
    assert answered["final_answer"]
    assert "token-level NLL" in answered["final_answer"]
    assert "明确采用 token-level NLL" not in answered["final_answer"]


def test_paper_analysis_rewrite_keeps_multiple_missing_aspects(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM([]))
    required_aspects = paper_agent.get_paper_analysis_aspects(TRANSFORMER_SOURCE)
    state = {
        "question": "完整分析 Transformer 论文的研究问题、核心方法、训练流程和主要实验结论。",
        "intent": "paper_analysis",
        "selected_papers": [TRANSFORMER_SOURCE],
        "required_aspects": required_aspects,
        "missing_aspects": ["研究问题与贡献", "训练设置", "实验结果与结论"],
        "trace": [],
    }

    rewritten = agent.rewrite_query_node(state)
    queries = rewritten["tool_action"]["arguments"]["queries"]

    assert len(queries) == 3
    assert any("motivation" in query for query in queries)
    assert any("optimizer" in query for query in queries)
    assert any("BLEU" in query for query in queries)


def test_transformer_missing_training_generates_training_query() -> None:
    queries = paper_agent.build_paper_analysis_queries(
        source=TRANSFORMER_SOURCE,
        required_aspects=paper_agent.get_paper_analysis_aspects(TRANSFORMER_SOURCE),
        missing_aspects=["训练设置"],
    )

    assert queries == ["Attention Is All optimizer adam learning rate warmup label smoothing batch size training steps"]


def test_partial_paper_analysis_answers_with_gaps_instead_of_empty_hitl(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM([]), enable_human_review=True, max_auto_search_rounds=2)

    route = agent.route_after_evidence_check(
        {
            "intent": "paper_analysis",
            "evidence_status": "insufficient",
            "evidence_sufficient": False,
            "retrieved_docs": [make_chunk("transformer-method", TRANSFORMER_SOURCE, 4, "multi-head attention")],
            "covered_aspects": ["模型结构"],
            "missing_aspects": ["训练设置"],
            "requested_aspects": ["模型结构", "训练设置"],
            "auto_search_rounds": 2,
            "max_auto_search_rounds": 2,
        }
    )

    assert route == "answer"


def test_eval020_expected_hitl_path_remains_unchanged(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(tools=tools, llm=FakeJsonLLM([]), enable_human_review=True, max_auto_search_rounds=2)

    route = agent.route_after_evidence_check(
        {
            "intent": "paper_comparison",
            "evidence_status": "sufficient_with_gaps",
            "evidence_sufficient": False,
            "retrieved_docs": [make_chunk("react-loss", REACT_SOURCE, 5, "fine-tuning trajectories")],
            "covered_aspects": ["核心方法", "训练方式", "推理流程"],
            "missing_aspects": ["ReAct 微调实验的具体损失函数"],
            "requested_aspects": ["核心方法", "训练方式", "损失函数", "推理流程"],
            "auto_search_rounds": 2,
            "max_auto_search_rounds": 2,
        }
    )

    assert route == "human_review"


def test_human_loop_graph_uses_checkpointer_and_thread_id(tools: agent_tools.PaperTools) -> None:
    saver = InMemorySaver()
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=saver,
        thread_id="thread-a",
    )

    assert agent.checkpointer is saver
    assert agent.thread_id == "thread-a"
    assert agent._graph_config()["configurable"]["thread_id"] == "thread-a"


def test_evidence_routes_sufficient_directly_to_answer(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        enable_human_review=True,
    )

    route = agent.route_after_evidence_check(
        {
            "evidence_status": "sufficient",
            "evidence_sufficient": True,
            "missing_aspects": [],
            "requested_aspects": ["核心方法"],
        }
    )

    assert route == "answer"


def test_first_insufficient_round_rewrites_before_interrupt(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        enable_human_review=True,
        max_auto_search_rounds=2,
    )

    route = agent.route_after_evidence_check(
        {
            "evidence_status": "insufficient",
            "evidence_sufficient": False,
            "auto_search_rounds": 1,
            "max_auto_search_rounds": 2,
            "missing_aspects": ["损失函数"],
            "requested_aspects": ["损失函数"],
        }
    )

    assert route == "rewrite"


def test_insufficient_after_auto_rounds_routes_to_human_review(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        enable_human_review=True,
        max_auto_search_rounds=2,
    )

    route = agent.route_after_evidence_check(
        {
            "evidence_status": "insufficient",
            "evidence_sufficient": False,
            "auto_search_rounds": 2,
            "max_auto_search_rounds": 2,
            "missing_aspects": ["损失函数"],
            "requested_aspects": ["损失函数"],
        }
    )

    assert route == "human_review"


def test_sufficient_with_gaps_rewrites_before_auto_rounds_exhausted(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        enable_human_review=True,
        max_auto_search_rounds=2,
    )

    route = agent.route_after_evidence_check(
        {
            "evidence_status": "sufficient_with_gaps",
            "evidence_sufficient": False,
            "auto_search_rounds": 1,
            "max_auto_search_rounds": 2,
            "missing_aspects": ["ReAct 微调实验的具体损失函数"],
            "requested_aspects": ["损失函数"],
        }
    )

    assert route == "rewrite"


def test_sufficient_with_gaps_interrupts_after_auto_rounds_exhausted(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        enable_human_review=True,
        max_auto_search_rounds=2,
    )

    route = agent.route_after_evidence_check(
        {
            "evidence_status": "sufficient_with_gaps",
            "evidence_sufficient": False,
            "auto_search_rounds": 2,
            "max_auto_search_rounds": 2,
            "missing_aspects": ["ReAct 微调实验的具体损失函数"],
            "requested_aspects": ["损失函数"],
        }
    )

    assert route == "human_review"


def test_failed_retrieval_routes_to_retrieval_failure_not_rewrite(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        enable_human_review=True,
    )
    state = {
        "question": "对比 RAG 和 ReAct",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "selected_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["核心方法", "损失函数"],
        "coverage_by_paper": paper_agent.initial_coverage_by_paper(
            [RAG_SOURCE, REACT_SOURCE],
            ["核心方法", "损失函数"],
        ),
        "tool_result": {
            "retrieval_status": "failed",
            "retrieval_error_reason": "RuntimeError: HTTP 429",
            "successful_query_count": 0,
            "failed_query_count": 2,
            "consecutive_query_failures": 2,
            "query_results": [
                {"query": "q1", "source": RAG_SOURCE, "success": False, "result_count": 0},
                {"query": "q2", "source": REACT_SOURCE, "success": False, "result_count": 0},
            ],
            "results": [],
        },
        "retrieval_status": "failed",
        "retrieval_error_reason": "RuntimeError: HTTP 429",
        "successful_query_count": 0,
        "failed_query_count": 2,
        "consecutive_query_failures": 2,
        "trace": [],
        "trace_enabled": False,
    }

    checked = agent.evidence_check_node(state)

    assert checked["evidence_status"] == "retrieval_failed"
    assert checked["missing_aspects"] == []
    assert checked["coverage_by_paper"] == state["coverage_by_paper"]
    assert agent.route_after_evidence_check(checked) == "retrieval_failure"


def test_partial_failure_does_not_mark_failed_query_as_missing(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
    )
    rag_result_id = "rag-loss"
    state = {
        "question": "对比 RAG 和 ReAct 的具体损失函数",
        "intent": "paper_comparison",
        "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["损失函数"],
        "coverage_by_paper": {
            RAG_SOURCE: {"损失函数": "missing"},
            REACT_SOURCE: {"损失函数": "missing"},
        },
        "retrieved_docs": [
            make_chunk(
                rag_result_id,
                RAG_SOURCE,
                6,
                "RAG training optimizes the negative marginal log-likelihood objective.",
            )
        ],
        "tool_result": {
            "retrieval_status": "partial_failure",
            "successful_query_count": 1,
            "failed_query_count": 1,
            "query_results": [
                {
                    "query": "marginal log likelihood equation",
                    "source": RAG_SOURCE,
                    "covered_aspect": "损失函数",
                    "success": True,
                    "result_count": 1,
                    "result_ids": [rag_result_id],
                },
                {
                    "query": "fine-tuning loss objective",
                    "source": REACT_SOURCE,
                    "covered_aspect": "损失函数",
                    "success": False,
                    "error_type": "RuntimeError",
                    "error_message": "HTTP 429",
                    "result_count": 0,
                    "result_ids": [],
                },
            ],
            "results": [],
        },
        "retrieval_status": "partial_failure",
        "successful_query_count": 1,
        "failed_query_count": 1,
        "trace": [],
        "trace_enabled": False,
    }

    checked = agent.evidence_check_node(state)

    assert checked["coverage_by_paper"][RAG_SOURCE]["损失函数"] == "covered"
    assert checked["coverage_by_paper"][REACT_SOURCE]["损失函数"] == "partial"
    assert checked["evidence_status"] == "sufficient_with_gaps"
    assert "检索部分失败" in "\n".join(checked["trace"])


def test_human_review_menu_prints_paper_level_coverage(capsys: pytest.CaptureFixture[str]) -> None:
    paper_agent.print_human_review_menu(
        {
            "type": "evidence_review",
            "question": "对比 RAG 和 ReAct 的损失函数",
            "evidence_status": "sufficient_with_gaps",
            "requested_aspects": ["核心方法", "训练方式", "损失函数", "推理流程"],
            "coverage_by_paper": {
                RAG_SOURCE: {"核心方法": "covered", "训练方式": "covered", "损失函数": "covered", "推理流程": "covered"},
                REACT_SOURCE: {"核心方法": "covered", "训练方式": "covered", "损失函数": "partial", "推理流程": "covered"},
            },
            "options": [{"action": "answer_with_gaps", "label": "基于现有证据回答"}],
        }
    )

    output = capsys.readouterr().out

    assert "已完整覆盖" in output
    assert "部分覆盖" in output
    assert "损失函数：RAG covered，ReAct partial" in output
    assert "✓ 损失函数" not in output


def test_human_review_disabled_does_not_route_to_interrupt_for_gaps(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        enable_human_review=False,
        max_iterations=1,
    )

    route = agent.route_after_evidence_check(
        {
            "evidence_status": "sufficient_with_gaps",
            "evidence_sufficient": False,
            "iteration": 1,
            "max_iterations": 1,
            "missing_aspects": ["ReAct 微调实验的具体损失函数"],
            "requested_aspects": ["损失函数"],
        }
    )

    assert route == "answer"


def test_related_aspects_missing_does_not_trigger_human_review(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        enable_human_review=True,
    )

    route = agent.route_after_evidence_check(
        {
            "evidence_status": "sufficient_with_gaps",
            "evidence_sufficient": True,
            "auto_search_rounds": 2,
            "max_auto_search_rounds": 2,
            "missing_aspects": ["相关补充"],
            "requested_aspects": ["核心方法"],
            "related_aspects": ["相关补充"],
        }
    )

    assert route == "answer"


def test_human_review_payload_is_json_serializable_and_omits_deep_after_limit(
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        max_deep_search_rounds=1,
    )
    state: paper_agent.AgentState = {
        "question": "对比 RAG 和 ReAct 的损失函数",
        "selected_papers": [RAG_SOURCE, REACT_SOURCE],
        "requested_aspects": ["损失函数"],
        "covered_aspects": ["核心方法"],
        "missing_aspects": ["ReAct 微调实验的具体损失函数"],
        "coverage_by_paper": {REACT_SOURCE: {"损失函数": "missing"}},
        "evidence_status": "sufficient_with_gaps",
        "evidence_reason": "仍缺少 ReAct 损失函数",
        "deep_search_rounds": 1,
        "max_deep_search_rounds": 1,
    }

    payload = agent.build_human_review_payload(state)

    json.dumps(payload, ensure_ascii=False)
    assert payload["type"] == "evidence_review"
    assert payload["evidence_status"] == "sufficient_with_gaps"
    assert payload["missing_aspects"] == ["ReAct 微调实验的具体损失函数"]
    assert "local_deep_search" not in [item["action"] for item in payload["options"]]


def test_graph_interrupts_and_resumes_answer_with_gaps_for_comparison() -> None:
    tools = PartialComparisonTools()
    agent = paper_agent.PaperAgent(
        tools=tools,  # type: ignore[arg-type]
        llm=FakeJsonLLM(["RAG 与 ReAct 可比较，但 ReAct 损失函数缺少明确证据[1]。"]),
        checkpointer=InMemorySaver(),
        thread_id="comparison-thread",
        enable_human_review=True,
        max_auto_search_rounds=1,
        max_deep_search_rounds=1,
    )

    result = agent.ask("对比 RAG 和 ReAct 的核心方法、训练方式、具体损失函数和推理流程")

    assert result["state"]["awaiting_human"] is True
    payload = result["state"]["last_interrupt_payload"]
    assert payload["evidence_status"] == "sufficient_with_gaps"
    assert "ReAct 微调实验的具体损失函数" in payload["missing_aspects"]
    assert "local_deep_search" in [item["action"] for item in payload["options"]]

    resumed = agent.resume({"action": "answer_with_gaps"})

    assert resumed["state"]["awaiting_human"] is False
    assert resumed["state"]["allow_answer_with_gaps"] is True
    assert "当前论文证据中未找到" in resumed["answer"]
    assert agent.thread_id == "comparison-thread"


def test_local_deep_search_uses_missing_aspects_expanded_k_and_neighbors() -> None:
    tools = RecordingPaperTools()
    agent = paper_agent.PaperAgent(
        tools=tools,  # type: ignore[arg-type]
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
    )
    existing = make_chunk("old", REACT_SOURCE, 3, "existing evidence")

    new_state = agent.local_deep_search_node(
        {
            "trace_enabled": False,
            "intent": "paper_comparison",
            "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
            "selected_papers": [RAG_SOURCE, REACT_SOURCE],
            "requested_aspects": ["核心方法", "损失函数"],
            "related_aspects": ["局限性"],
            "coverage_by_paper": {
                RAG_SOURCE: {"核心方法": "covered", "损失函数": "covered"},
                REACT_SOURCE: {"核心方法": "covered", "损失函数": "partial"},
            },
            "missing_aspects": ["ReAct 微调实验的具体损失函数"],
            "retrieved_docs": [existing],
            "retrieval_history": [],
            "seen_result_ids": ["old"],
            "seen_source_pages": [],
            "deep_search_rounds": 0,
            "max_deep_search_rounds": 1,
            "trace": [],
        }
    )

    assert tools.multiple_calls
    call = tools.multiple_calls[0]
    assert call["source"] == REACT_SOURCE
    assert call["k_per_query"] == min(8, agent_tools.MAX_K)
    assert all(len(query.split()) <= 18 for query in call["queries"])
    assert all("局限性" not in query for query in call["queries"])
    assert tools.neighbor_calls
    assert all(call["radius"] == 2 for call in tools.neighbor_calls)
    assert existing in new_state["retrieved_docs"]
    assert new_state["deep_search_rounds"] == 1


def test_local_deep_search_uses_source_specific_queries_for_different_gaps() -> None:
    tools = RecordingPaperTools()
    agent = paper_agent.PaperAgent(
        tools=tools,  # type: ignore[arg-type]
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
    )

    agent.local_deep_search_node(
        {
            "trace_enabled": False,
            "intent": "paper_comparison",
            "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
            "selected_papers": [RAG_SOURCE, REACT_SOURCE],
            "requested_aspects": ["损失函数"],
            "coverage_by_paper": {
                RAG_SOURCE: {"损失函数": "partial"},
                REACT_SOURCE: {"损失函数": "partial"},
            },
            "missing_aspects": ["RAG 的 损失函数（部分细节缺失）", "ReAct 微调实验的具体损失函数"],
            "retrieved_docs": [],
            "retrieval_history": [],
            "seen_result_ids": [],
            "seen_source_pages": [],
            "deep_search_rounds": 0,
            "max_deep_search_rounds": 1,
            "trace": [],
        }
    )

    calls_by_source = {call["source"]: call for call in tools.multiple_calls}
    rag_queries = " ".join(calls_by_source[RAG_SOURCE]["queries"]).lower()
    react_queries = " ".join(calls_by_source[REACT_SOURCE]["queries"]).lower()

    assert "marginal log likelihood" in rag_queries
    assert "rag sequence token" in rag_queries
    assert "trajectory supervision" not in rag_queries
    assert "fine-tuning loss objective" in react_queries
    assert "trajectory supervision" in react_queries
    assert "marginal log likelihood" not in react_queries


def test_revise_question_reuses_docs_when_target_papers_unchanged(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
    )
    existing = make_chunk("old", RAG_SOURCE, 2, "RAG evidence")

    command = agent.revise_question_node(
        {
            "question": "对比 RAG 和 ReAct 的核心方法",
            "revised_question": "对比 RAG 和 ReAct 的核心方法和损失函数",
            "comparison_papers": [RAG_SOURCE, REACT_SOURCE],
            "selected_papers": [RAG_SOURCE, REACT_SOURCE],
            "retrieved_docs": [existing],
            "retrieval_history": [{"query": "old"}],
            "seen_result_ids": ["old"],
            "seen_source_pages": ["page"],
            "trace": [],
        }
    )

    assert isinstance(command, Command)
    assert command.goto == "evidence_check_node"
    assert command.update["question"] == "对比 RAG 和 ReAct 的核心方法和损失函数"
    assert command.update["retrieved_docs"] == [existing]


def test_revise_question_clears_docs_when_target_papers_change(tools: agent_tools.PaperTools) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
    )
    command = agent.revise_question_node(
        {
            "question": "分析 RAG",
            "revised_question": "完整分析 LoRA 的方法和实验结果",
            "selected_papers": [RAG_SOURCE],
            "retrieved_docs": [make_chunk("old", RAG_SOURCE, 2, "RAG evidence")],
            "retrieval_history": [{"query": "old"}],
            "seen_result_ids": ["old"],
            "seen_source_pages": ["page"],
            "trace": [],
        }
    )

    assert isinstance(command, Command)
    assert command.goto == "planner_node"
    assert command.update["selected_papers"] == [LORA_SOURCE]
    assert command.update["retrieved_docs"] == []


def test_cancel_node_does_not_call_llm_or_retrieval() -> None:
    tools = RecordingPaperTools()
    llm = BindExplodingLLM(['{"action":"answer","arguments":{}}'])
    agent = paper_agent.PaperAgent(
        tools=tools,  # type: ignore[arg-type]
        llm=llm,
        checkpointer=InMemorySaver(),
    )

    state = agent.cancel_node({"trace": []})

    assert state["cancelled"] is True
    assert state["final_answer"] == "当前任务已取消。"
    assert tools.multiple_calls == []
    assert tools.search_calls == []


def test_clear_deletes_current_thread_and_new_session_changes_thread_id(tools: agent_tools.PaperTools) -> None:
    saver = RecordingCheckpointer()
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=saver,
        thread_id="old-thread",
    )

    agent.clear()

    assert saver.deleted_threads == ["old-thread"]
    assert agent.thread_id != "old-thread"


def test_resume_uses_same_thread_id_without_new_initial_state(
    monkeypatch: pytest.MonkeyPatch,
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        thread_id="resume-thread",
    )
    captured: dict[str, Any] = {}

    def fake_stream(input_value: Any, **kwargs: Any):
        captured["input"] = input_value
        captured["config"] = kwargs["config"]
        yield ("updates", {"answer_node": {"final_answer": "ok", "final_answer_streamed": False}})

    monkeypatch.setattr(agent.graph, "stream", fake_stream)

    result = agent.resume({"action": "answer_with_gaps"})

    assert isinstance(captured["input"], Command)
    assert captured["input"].resume == {"action": "answer_with_gaps"}
    assert captured["config"]["configurable"]["thread_id"] == "resume-thread"
    assert result["answer"] == "ok"


def test_retry_current_retrieval_resume_uses_same_thread_id(
    monkeypatch: pytest.MonkeyPatch,
    tools: agent_tools.PaperTools,
) -> None:
    agent = paper_agent.PaperAgent(
        tools=tools,
        llm=FakeJsonLLM(['{"action":"answer","arguments":{}}']),
        checkpointer=InMemorySaver(),
        thread_id="retry-thread",
    )
    captured: dict[str, Any] = {}

    def fake_stream(input_value: Any, **kwargs: Any):
        captured["input"] = input_value
        captured["config"] = kwargs["config"]
        yield ("updates", {"tool_node": {"retrieval_status": "success", "final_answer": "retried"}})

    monkeypatch.setattr(agent.graph, "stream", fake_stream)

    agent.resume({"action": "retry_current_retrieval"})

    assert isinstance(captured["input"], Command)
    assert captured["input"].resume == {"action": "retry_current_retrieval"}
    assert captured["config"]["configurable"]["thread_id"] == "retry-thread"


def _transformer_complete_evidence() -> list[agent_tools.PaperChunk]:
    return [
        make_chunk(
            "transformer-complete",
            TRANSFORMER_SOURCE,
            8,
            "Transformer removes recurrence and convolution for parallelization. "
            "The encoder decoder architecture uses multi-head attention and positional encoding. "
            "Training uses Adam, learning rate warmup, label smoothing, and batch training steps. "
            "WMT English German BLEU base big results are reported in Table 2.",
        )
    ]


def test_final_answer_removes_stale_missing_aspects() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(["Transformer complete analysis[1]."]),
        tool_mode="json",
    )
    state = {
        "question": "完整分析 Transformer 论文。",
        "intent": "paper_analysis",
        "selected_papers": [TRANSFORMER_SOURCE],
        "required_aspects": paper_agent.get_paper_analysis_aspects(TRANSFORMER_SOURCE),
        "covered_aspects": ["注意力与位置编码公式", "训练设置", "实验结果与结论"],
        "missing_aspects": ["研究问题与贡献", "模型结构"],
        "evidence_status": "insufficient",
        "evidence_sufficient": False,
        "retrieved_docs": _transformer_complete_evidence(),
        "trace": [],
    }

    answered = agent.answer_node(state)

    assert answered["missing_aspects"] == []
    assert answered["evidence_status"] == "sufficient"
    assert "【未检索到的信息】" not in answered["final_answer"]


def test_final_answer_omits_missing_section_when_all_aspects_covered() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(["Transformer answer[1]."]),
        tool_mode="json",
    )
    state = {
        "question": "完整分析 Transformer 论文。",
        "intent": "paper_analysis",
        "selected_papers": [TRANSFORMER_SOURCE],
        "required_aspects": paper_agent.get_paper_analysis_aspects(TRANSFORMER_SOURCE),
        "covered_aspects": ["研究问题与动机", "核心方法与数学公式", "主要定量实验结果"],
        "missing_aspects": ["训练设置"],
        "retrieved_docs": _transformer_complete_evidence(),
        "trace": [],
    }

    answered = agent.answer_node(state)

    assert answered["missing_aspects"] == []
    assert "【未检索到的信息】" not in answered["final_answer"]
    assert "【证据缺口】" not in answered["final_answer"]


def test_final_answer_keeps_only_remaining_missing_aspects() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(["Transformer answer[1]."]),
        tool_mode="json",
    )
    docs = _transformer_complete_evidence()
    docs[0]["content"] = docs[0]["content"].replace(
        "Training uses Adam, learning rate warmup, label smoothing, and batch training steps. ", ""
    )
    state = {
        "question": "完整分析 Transformer 论文。",
        "intent": "paper_analysis",
        "selected_papers": [TRANSFORMER_SOURCE],
        "required_aspects": paper_agent.get_paper_analysis_aspects(TRANSFORMER_SOURCE),
        "covered_aspects": ["注意力与位置编码公式", "实验结果与结论"],
        "missing_aspects": ["研究问题与贡献", "模型结构", "训练设置"],
        "retrieved_docs": docs,
        "trace": [],
    }

    answered = agent.answer_node(state)

    assert answered["missing_aspects"] == ["训练设置"]
    assert "【未检索到的信息】\n训练设置" in answered["final_answer"]
    assert "研究问题与贡献" not in answered["final_answer"]
    assert "模型结构" not in answered["final_answer"]


def test_negative_evidence_success_sets_sufficient_status() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(
            ['{"sufficient":false,"reason":"论文未提供 token-level NLL 的明确说明。","covered_aspects":[],"missing_aspects":["损失函数"]}']
        ),
        tool_mode="json",
        enable_human_review=True,
    )
    state = {
        "question": "ReAct 论文是否明确说明微调使用 token-level NLL？请不要用常识补全。",
        "intent": "general_qa",
        "retrieved_docs": [make_chunk("react-loss", REACT_SOURCE, 15, "fine-tuning batch size and steps")],
        "required_aspects": ["损失函数"],
        "requested_aspects": ["损失函数"],
        "missing_aspects": ["损失函数"],
        "trace": [],
    }

    checked = agent.evidence_check_node(state)

    assert checked["evidence_status"] == "sufficient"
    assert checked["evidence_sufficient"] is True


def test_negative_evidence_success_clears_awaiting_human() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(
            ['{"sufficient":false,"reason":"没有证据表明论文明确指定 token-level NLL。","covered_aspects":[],"missing_aspects":["损失函数"]}']
        ),
        tool_mode="json",
        enable_human_review=True,
    )
    checked = agent.evidence_check_node(
        {
            "question": "ReAct 论文有没有给出 token-level NLL？请不要用常识补全。",
            "intent": "general_qa",
            "awaiting_human": True,
            "human_review_required": True,
            "retrieved_docs": [make_chunk("react-loss", REACT_SOURCE, 5, "fine-tuning trajectories")],
            "required_aspects": ["损失函数"],
            "requested_aspects": ["损失函数"],
            "missing_aspects": ["损失函数"],
            "trace": [],
        }
    )

    assert checked["awaiting_human"] is False
    assert checked["human_review_required"] is False


def test_negative_evidence_success_does_not_leave_insufficient_state() -> None:
    agent = paper_agent.PaperAgent(
        tools=RecordingPaperTools(),
        llm=FakeJsonLLM(
            ['{"sufficient":false,"reason":"无法判断论文是否明确说明 token-level NLL。","covered_aspects":[],"missing_aspects":["损失函数"]}']
        ),
        tool_mode="json",
    )
    checked = agent.evidence_check_node(
        {
            "question": "ReAct 论文是否写明 token-level NLL？请不要用常识补全。",
            "intent": "general_qa",
            "evidence_status": "insufficient",
            "evidence_sufficient": False,
            "retrieved_docs": [make_chunk("react-loss", REACT_SOURCE, 5, "fine-tuning trajectories")],
            "required_aspects": ["损失函数"],
            "requested_aspects": ["损失函数"],
            "missing_aspects": ["损失函数"],
            "trace": [],
        }
    )

    assert checked["evidence_status"] != "insufficient"
    assert checked["missing_aspects"] == []


def test_specific_explicitness_question_routes_general_qa() -> None:
    intent, _papers = paper_agent.infer_intent("ReAct 论文是否采用 token-level NLL 作为微调损失函数？")

    assert intent == "general_qa"


def test_full_paper_analysis_still_routes_paper_analysis() -> None:
    intent, _papers = paper_agent.infer_intent("完整分析 ReAct 论文，并说明它是否采用 token-level NLL。")

    assert intent == "paper_analysis"
