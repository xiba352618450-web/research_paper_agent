from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, TypedDict

from dotenv import load_dotenv
from langgraph.config import get_stream_writer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

import agent_tools
from agent_tools import PaperChunk, PaperTools


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
DEFAULT_CHECKPOINT_PATH = RUNTIME_DIR / "langgraph_checkpoints.sqlite"

DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
CHAT_MODEL = DEFAULT_CHAT_MODEL
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MAX_AUTO_SEARCH_ROUNDS = 2
DEFAULT_MAX_DEEP_SEARCH_ROUNDS = 1
MAX_INPUT_CHARS = 2000
TOOL_MODE_ENV = "AGENT_TOOL_MODE"
MAX_COMPARISON_QUERIES = 3
CITATION_RE = re.compile(r"\[(\d+)\]")

ACTION_ALIASES = {
    "list_papers_tool": "list_papers",
    "list_papers": "list_papers",
    "search_paper_tool": "search_paper",
    "search_paper": "search_paper",
    "search_multiple_queries_tool": "search_multiple_queries",
    "search_multiple_queries": "search_multiple_queries",
    "get_neighbor_chunks_tool": "get_neighbor_chunks",
    "get_neighbor_chunks": "get_neighbor_chunks",
    "inspect_paper_scope_tool": "inspect_paper_scope",
    "inspect_paper_scope": "inspect_paper_scope",
    "answer": "answer",
    "clarify": "clarify",
    "clarification": "clarify",
}
TOOL_ACTIONS = {"list_papers", "search_paper", "search_multiple_queries", "get_neighbor_chunks", "inspect_paper_scope"}
SUPPORTED_INTENTS = {
    "paper_analysis",
    "paper_summary",
    "method_explain",
    "experiment_analysis",
    "paper_comparison",
    "reproduction_plan",
    "research_inspiration",
    "general_qa",
    "clarification",
}

PAPER_ANALYSIS_REQUIRED_ASPECTS = [
    "核心问题与动机",
    "核心方法与数学公式",
    "参数冻结、初始化与缩放",
    "注入位置与可训练参数",
    "推理合并与部署开销",
    "实验基础模型、数据集和任务",
    "对比 baseline 和评价指标",
    "主要定量实验结果",
    "rank 与目标矩阵消融",
    "低秩更新矩阵分析与局限性",
]

PAPER_ANALYSIS_ASPECT_TERMS = {
    "核心问题与动机": ["problem", "motivation", "adaptation"],
    "核心方法与数学公式": ["low-rank", "parameterization", "equation"],
    "参数冻结、初始化与缩放": ["parameterization", "initialization", "scaling"],
    "注入位置与可训练参数": ["injection", "trainable", "parameters"],
    "推理合并与部署开销": ["merge", "inference", "deployment", "latency"],
    "实验基础模型、数据集和任务": ["experiments", "base models", "datasets", "tasks"],
    "对比 baseline 和评价指标": ["baselines", "evaluation", "metrics"],
    "主要定量实验结果": ["quantitative", "results", "scores"],
    "rank 与目标矩阵消融": ["rank", "target matrices", "ablation"],
    "低秩更新矩阵分析与局限性": ["low-rank", "update", "singular values", "subspace", "limitations"],
}

GENERIC_PAPER_ANALYSIS_ASPECTS = [
    "研究问题与动机",
    "核心方法",
    "训练设置",
    "实验设置",
    "实验结果",
    "优势与局限",
]

PAPER_TYPE_BY_SOURCE = {
    "01 Attention Is All You Need.pdf": "transformer_architecture",
    "02 Language Models are Few-Shot Learners.pdf": "few_shot_scaling",
    "03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf": "rlhf_alignment",
    "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": "retrieval_augmented_generation",
    "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": "lora_parameter_efficient",
    "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": "react_reasoning_action",
}

PAPER_TYPE_ANALYSIS_ASPECTS = {
    "transformer_architecture": [
        "研究问题与贡献",
        "模型结构",
        "注意力与位置编码公式",
        "训练设置",
        "实验结果与结论",
    ],
    "few_shot_scaling": [
        "研究问题与 in-context learning 动机",
        "zero/one/few-shot 方法",
        "模型、训练数据与训练过程",
        "评测设置",
        "主要实验结果",
        "局限性与风险",
    ],
    "rlhf_alignment": [
        "研究问题与总体结论",
        "SFT、奖励模型与 PPO 训练流程",
        "训练数据与实验设置",
        "评价指标",
        "实验结果",
        "优势与局限",
    ],
    "retrieval_augmented_generation": [
        "研究问题与动机",
        "retriever 和 generator 结构",
        "RAG-Sequence 与 RAG-Token 公式",
        "训练目标与解码",
        "实验设置与结果",
        "优势与局限",
    ],
    "react_reasoning_action": [
        "研究问题与动机",
        "Thought Action Observation 交错流程",
        "prompting 与微调实验",
        "环境交互与推理流程",
        "实验设置与结果",
        "优势与局限",
    ],
}

PAPER_ANALYSIS_ASPECT_TERMS.update(
    {
        "研究问题与动机": ["problem", "motivation", "contribution"],
        "核心方法": ["core", "method", "architecture"],
        "训练设置": ["training", "objective", "settings"],
        "实验设置": ["experiments", "datasets", "tasks"],
        "实验结果": ["experiments", "results", "metrics"],
        "优势与局限": ["advantages", "limitations", "discussion"],
        "研究问题与贡献": ["transformer", "recurrence", "convolution", "parallelization", "motivation", "contribution"],
        "模型结构": ["transformer", "architecture", "encoder", "decoder", "attention"],
        "注意力与位置编码公式": ["positional", "encoding", "sine", "cosine", "formula", "attention", "equation"],
        "训练设置": ["optimizer", "adam", "learning", "rate", "warmup", "label", "smoothing", "batch", "size", "training", "steps"],
        "实验结果与结论": ["WMT", "English", "German", "BLEU", "base", "big", "results", "Table", "2", "conclusion"],
        "研究问题与 in-context learning 动机": ["in-context", "learning", "motivation"],
        "zero/one/few-shot 方法": ["zero-shot", "one-shot", "few-shot", "method"],
        "模型、训练数据与训练过程": ["model", "training", "data", "process"],
        "评测设置": ["evaluation", "tasks", "setting"],
        "主要实验结果": ["main", "experiments", "results"],
        "局限性与风险": ["limitations", "risks", "discussion"],
        "研究问题与总体结论": ["problem", "motivation", "conclusion"],
        "SFT、奖励模型与 PPO 训练流程": ["SFT", "reward", "model", "PPO", "training"],
        "训练数据与实验设置": ["training", "data", "experiments", "setting"],
        "评价指标": ["evaluation", "metrics", "human", "preference"],
        "retriever 和 generator 结构": ["retriever", "generator", "architecture"],
        "RAG-Sequence 与 RAG-Token 公式": ["RAG-Sequence", "RAG-Token", "equations"],
        "训练目标与解码": ["training", "objective", "negative", "log-likelihood", "decoding"],
        "实验设置与结果": ["experiments", "settings", "results"],
        "Thought Action Observation 交错流程": ["Thought", "Action", "Observation", "trajectory"],
        "prompting 与微调实验": ["prompting", "fine-tuning", "bootstrapping", "experiments"],
        "环境交互与推理流程": ["environment", "interaction", "reasoning", "acting", "procedure"],
    }
)

FORBIDDEN_AUTO_QUERY_TERMS = {"t5", "bert", "superglue", "mixed precision", "adamw"}

COMPARISON_KEYWORDS = (
    "对比",
    "比较",
    "区别",
    "差异",
    "异同",
    "相比",
    "分别有什么不同",
    "哪个好",
    "哪个更",
    "versus",
    " vs ",
    "compare",
    "comparison",
    "difference",
)

DEFAULT_COMPARISON_ASPECTS = [
    "研究目标",
    "核心方法",
    "训练方式",
    "推理流程",
]

COMPARISON_RELATED_ASPECTS = {
    "研究目标": [
        "问题设定",
        "研究动机",
        "适用任务",
    ],
    "核心方法": [
        "关键技术组件",
        "核心公式或机制",
        "方法关键参数",
        "模块之间的数据流",
    ],
    "训练方式": [
        "训练目标或损失函数",
        "训练数据或示例构造",
        "冻结和更新哪些参数",
        "初始化与缩放方式",
        "与方法直接相关的训练配置",
    ],
    "推理流程": [
        "外部知识或环境如何接入",
        "执行循环与停止条件",
        "推理开销与部署特点",
    ],
}

REPRODUCTION_DETAIL_ASPECTS = [
    "学习率",
    "batch size",
    "optimizer",
    "GPU 数量",
    "epoch",
    "mixed precision",
    "训练时长",
]

COMPARISON_QUERY_GROUPS: list[tuple[set[str], str]] = [
    ({"研究目标"}, "research goal motivation problem setting"),
    ({"核心方法"}, "core method architecture mechanism equations key parameters"),
    ({"训练方式", "推理流程"}, "training objective trainable frozen parameters inference procedure"),
    ({"损失函数"}, "loss objective supervision training details"),
    ({"训练超参数"}, "training hyperparameters learning rate batch size optimizer"),
]

COMPARISON_ASPECT_ALIASES: dict[str, tuple[str, ...]] = {
    "研究目标": ("研究目标", "目标", "研究动机", "解决什么问题"),
    "核心方法": ("核心方法", "方法", "核心机制", "模型架构"),
    "训练方式": ("训练方式", "训练", "微调", "优化方式"),
    "损失函数": ("损失函数", "损失", "loss", "目标函数"),
    "推理流程": ("推理流程", "推理", "执行流程", "工作流程"),
    "训练超参数": ("训练超参数", "超参数", "参数设置", "训练配置", "学习率", "batch size", "optimizer"),
    "实验结果": ("实验结果", "实验", "结果", "性能"),
    "优势与局限": ("优势", "局限", "优缺点"),
}
PAPER_ALIASES: dict[str, tuple[str, ...]] = {
    "01 Attention Is All You Need.pdf": (
        "attention is all you need",
        "transformer",
    ),
    "02 Language Models are Few-Shot Learners.pdf": (
        "language models are few-shot learners",
        "gpt-3",
        "gpt3",
    ),
    "03 InstructGPT RLHF - Training Language Models to Follow Instructions with Human Feedback.pdf": (
        "instructgpt",
        "rlhf论文",
        "rlhf 论文",
    ),
    "04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf": (
        "rag",
        "retrieval-augmented generation",
        "retrieval augmented generation",
    ),
    "05 LoRA - Low-Rank Adaptation of Large Language Models.pdf": (
        "lora",
        "low-rank adaptation",
        "low rank adaptation",
    ),
    "06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf": (
        "react",
        "synergizing reasoning and acting",
    ),
}
def normalize_paper_name(value: str) -> str:
    """Normalize paper aliases without discarding meaningful inner spaces."""
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def normalize_question(text: str) -> str:
    """统一大小写与空白，方便关键词匹配。"""
    return normalize_paper_name(text)


def _alias_matches(normalized_question: str, alias: str) -> bool:
    """Match one normalized alias while protecting short ASCII abbreviations."""
    normalized_alias = normalize_paper_name(alias)
    if not normalized_alias:
        return False

    # Keep words in multi-word titles distinct, while accepting ordinary and
    # full-width whitespace in user input after NFKC normalization.
    alias_pattern = re.escape(normalized_alias).replace(r"\ ", r"\s+")
    if re.fullmatch(r"[a-z0-9][a-z0-9\- ]*", normalized_alias):
        alias_pattern = rf"(?<![a-z0-9]){alias_pattern}(?![a-z0-9])"
    return re.search(alias_pattern, normalized_question) is not None


def resolve_paper_reference(reference: str, available_papers: Iterable[str]) -> str | None:
    """Resolve an alias or paper mention to an available canonical PDF name."""
    available_by_name = {
        normalize_paper_name(paper): str(paper)
        for paper in available_papers
        if normalize_paper_name(paper)
    }
    normalized_reference = normalize_paper_name(reference)
    if not normalized_reference:
        return None

    direct_match = available_by_name.get(normalized_reference)
    if direct_match:
        return direct_match

    alias_to_source = {
        normalize_paper_name(alias): source
        for source, aliases in PAPER_ALIASES.items()
        for alias in aliases
    }
    alias_source = alias_to_source.get(normalized_reference)
    if alias_source:
        return available_by_name.get(normalize_paper_name(alias_source))

    for source in detect_explicit_papers(str(reference)):
        canonical = available_by_name.get(normalize_paper_name(source))
        if canonical:
            return canonical
    return None


def detect_explicit_papers(question: str) -> list[str]:
    """识别用户问题中明确提到的论文。"""
    normalized = normalize_question(question)
    matched: list[str] = []

    for source, aliases in PAPER_ALIASES.items():
        for alias in aliases:
            if _alias_matches(normalized, alias):
                matched.append(source)
                break

    return list(dict.fromkeys(matched))


CORPUS_SCOPE_PHRASES: tuple[str, ...] = (
    "这些论文",
    "这些文章",
    "所有论文",
    "当前论文库",
    "哪些论文",
    "these papers",
    "all papers",
    "across the papers",
)


def resolve_explicit_papers(question: str, available_papers: Iterable[str]) -> dict[str, list[str]]:
    """Return matched aliases and canonical available PDFs for a user question."""
    normalized = normalize_question(question)
    mentioned_aliases: list[str] = []
    canonical_papers: list[str] = []

    for source, aliases in PAPER_ALIASES.items():
        for alias in aliases:
            if _alias_matches(normalized, alias):
                mentioned_aliases.append(alias)
                canonical = resolve_paper_reference(source, available_papers)
                if canonical:
                    canonical_papers.append(canonical)
                break

    return {
        "mentioned_aliases": list(dict.fromkeys(mentioned_aliases)),
        "canonical_papers": list(dict.fromkeys(canonical_papers)),
    }


def has_corpus_scope_request(question: str) -> bool:
    """Detect user language that asks across the current paper catalog."""
    normalized = normalize_question(question)
    return any(normalize_paper_name(phrase) in normalized for phrase in CORPUS_SCOPE_PHRASES)


def has_unbounded_corpus_scope(question: str, available_papers: Iterable[str]) -> bool:
    """True when corpus-wide language is present and no paper alias is explicit."""
    resolved = resolve_explicit_papers(question, available_papers)
    return has_corpus_scope_request(question) and not resolved["canonical_papers"]


def extract_comparison_aspects(question: str) -> list[str]:
    """Extract user-requested comparison dimensions; use stable defaults when absent."""
    normalized = normalize_question(question)
    aspects: list[str] = []
    for aspect, aliases in COMPARISON_ASPECT_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            aspects.append(aspect)
    return list(dict.fromkeys(aspects)) or list(DEFAULT_COMPARISON_ASPECTS)


def requests_reproduction_details(question: str, intent: str) -> bool:
    normalized = normalize_question(question)
    if intent == "reproduction_plan":
        return True
    return any(
        term in normalized
        for term in (
            "复现",
            "超参数",
            "参数设置",
            "训练配置",
            "学习率",
            "batch size",
            "batchsize",
            "optimizer",
        )
    )


def derive_related_aspects(
    intent: str,
    requested_aspects: list[str],
    question: str,
) -> list[str]:
    related: list[str] = []
    if intent == "paper_comparison":
        for aspect in requested_aspects:
            related.extend(COMPARISON_RELATED_ASPECTS.get(aspect, []))
        if requests_reproduction_details(question, intent):
            related.extend(REPRODUCTION_DETAIL_ASPECTS)
    return list(dict.fromkeys(related))


def build_comparison_queries(
    requested_aspects: list[str],
    missing_aspects: list[str] | None = None,
    max_queries: int = MAX_COMPARISON_QUERIES,
) -> list[str]:
    targets = [aspect for aspect in (missing_aspects or []) if aspect in requested_aspects]
    if not targets:
        targets = list(requested_aspects or DEFAULT_COMPARISON_ASPECTS)

    queries: list[str] = []
    target_set = set(targets)
    for aspects, query in COMPARISON_QUERY_GROUPS:
        if aspects & target_set and query not in queries:
            queries.append(compact_query_terms(query.split(), max_words=18))

    if not queries:
        queries = [COMPARISON_QUERY_GROUPS[0][1]]
    return queries[:max_queries]


def infer_comparison_aspects_from_query(query: str) -> list[str]:
    lowered = query.lower()
    aspects: list[str] = []
    if any(term in lowered for term in ("research", "goal", "motivation", "problem setting")):
        aspects.append("研究目标")
    if any(term in lowered for term in ("core method", "architecture", "mechanism", "equation", "key parameters")):
        aspects.append("核心方法")
    if any(term in lowered for term in ("training", "objective", "trainable", "frozen", "parameters")):
        aspects.append("训练方式")
    if any(term in lowered for term in ("loss", "supervision", "objective function")):
        aspects.append("损失函数")
    if any(term in lowered for term in ("inference", "procedure", "execution", "flow")):
        aspects.append("推理流程")
    if any(term in lowered for term in ("hyperparameter", "learning rate", "batch size", "optimizer")):
        aspects.append("训练超参数")
    return list(dict.fromkeys(aspects))


def short_source_label(source: str) -> str:
    stem = Path(source).stem
    stem = re.sub(r"^\d+\s*", "", stem)
    return stem.split(" - ", 1)[0].strip() or stem


def initial_coverage_by_paper(
    sources: list[str],
    requested_aspects: list[str],
) -> dict[str, dict[str, str]]:
    return {
        source: {aspect: "missing" for aspect in requested_aspects}
        for source in sources
    }


def merge_comparison_coverage(
    coverage: dict[str, dict[str, str]],
    query_results: list[dict[str, Any]],
    requested_aspects: list[str],
    retrieved_docs: list[PaperChunk] | None = None,
    question: str = "",
) -> dict[str, dict[str, str]]:
    merged = {
        source: dict(aspects)
        for source, aspects in coverage.items()
    }
    requested_set = set(requested_aspects)
    docs_by_id = {
        str(doc.get("result_id")): doc
        for doc in (retrieved_docs or [])
        if doc.get("result_id")
    }
    for item in query_results:
        source = str(item.get("source") or "")
        if not source:
            continue
        merged.setdefault(source, {aspect: "missing" for aspect in requested_aspects})
        result_ids = [str(result_id) for result_id in item.get("result_ids") or []]
        chunks = [docs_by_id[result_id] for result_id in result_ids if result_id in docs_by_id]
        query_aspects = []
        covered_aspect = str(item.get("covered_aspect") or "")
        if covered_aspect in requested_set:
            query_aspects.append(covered_aspect)
        query_aspects.extend(infer_comparison_aspects_from_query(str(item.get("query") or "")))
        for aspect in list(dict.fromkeys(query_aspects)):
            if aspect not in requested_set:
                continue
            if item.get("success") is False:
                status = "partial"
            else:
                status = classify_aspect_coverage_from_chunks(
                    aspect=aspect,
                    source=source,
                    chunks=chunks,
                    result_count=int(item.get("result_count") or 0),
                    question=question,
                )
            current = merged[source].get(aspect, "missing")
            if should_replace_coverage(current, status):
                merged[source][aspect] = status
    return merged


def normalize_coverage_status(status: str) -> str:
    cleaned = str(status or "").strip().lower()
    return cleaned if cleaned in {"covered", "not_applicable", "partial", "missing"} else "missing"


def should_replace_coverage(current: str, candidate: str) -> bool:
    rank = {"missing": 0, "partial": 1, "not_applicable": 2, "covered": 3}
    return rank[normalize_coverage_status(candidate)] > rank[normalize_coverage_status(current)]


def has_not_found_language(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in (
            "not explicitly",
            "does not explicitly",
            "not specify",
            "not specified",
            "not provided",
            "not given",
            "unclear",
            "没有明确",
            "未明确",
            "没有给出",
            "未给出",
            "没有检索到",
            "未找到",
            "证据中没有",
        )
    )


def has_direct_not_applicable_evidence(text: str, question: str) -> bool:
    lowered = text.lower()
    question_lower = question.lower()
    prompting_terms = ("prompting", "few-shot", "few shot", "in-context", "in context", "inference time", "推理时")
    no_update_terms = (
        "does not update",
        "no parameter update",
        "without updating",
        "no training loss",
        "不更新参数",
        "没有训练损失",
        "无训练损失",
    )
    if not any(term in lowered for term in prompting_terms):
        return False
    if not any(term in lowered for term in no_update_terms):
        return False
    if "prompting" in question_lower or "few-shot" in question_lower or "prompt" in question_lower:
        return True
    fine_tuning_terms = ("fine-tuning", "finetuning", "fine tuning", "微调", "supervised")
    return not any(term in lowered for term in fine_tuning_terms)


def has_concrete_loss_evidence(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in (
            "negative marginal log-likelihood",
            "negative marginal log likelihood",
            "marginal log-likelihood",
            "marginal log likelihood",
            "negative log-likelihood",
            "negative log likelihood",
            "answer nll",
            "cross entropy",
            "cross-entropy",
            "nll",
            "log-likelihood loss",
            "loss function is",
            "objective function is",
            "minimize the loss",
            "最大似然",
            "交叉熵",
            "负对数似然",
            "损失函数为",
            "目标函数为",
        )
    )


def question_requests_optimization_details(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered
        for term in (
            "gradient",
            "backpropagation",
            "back-propagation",
            "optimization derivation",
            "梯度",
            "梯度传播",
            "反向传播",
            "优化推导",
            "参数梯度",
        )
    )


def has_optimization_detail_evidence(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in (
            "gradient",
            "backpropagation",
            "back-propagation",
            "differentiate",
            "derivative",
            "梯度",
            "反向传播",
            "求导",
        )
    )


def classify_aspect_coverage_from_chunks(
    *,
    aspect: str,
    source: str,
    chunks: list[PaperChunk],
    result_count: int,
    question: str = "",
) -> str:
    if result_count <= 0:
        return "missing"
    combined = "\n".join(str(chunk.get("content") or "") for chunk in chunks).strip()
    if not combined:
        return "partial"

    if aspect == "损失函数":
        if has_not_found_language(combined):
            return "partial"
        if has_direct_not_applicable_evidence(combined, question):
            return "not_applicable"
        if has_concrete_loss_evidence(combined):
            if question_requests_optimization_details(question) and not has_optimization_detail_evidence(combined):
                return "partial"
            return "covered"
        if any(term in combined.lower() for term in ("loss", "objective", "损失", "目标函数")):
            return "partial"
        return "partial"

    if has_not_found_language(combined):
        return "partial"
    return "covered"


def format_coverage_gap(source: str, aspect: str, status: str) -> str:
    label = short_source_label(source)
    if aspect == "损失函数" and label.lower() == "react":
        return "ReAct 微调实验的具体损失函数"
    if status == "partial":
        return f"{label} 的 {aspect}（部分细节缺失）"
    return f"{label} 的 {aspect}"


def compute_evidence_status(
    coverage_by_paper: dict[str, dict[str, str]],
    requested_aspects: list[str],
    target_papers: list[str],
) -> str:
    saw_partial = False
    for source in target_papers:
        coverage = coverage_by_paper.get(source, {})
        for aspect in requested_aspects:
            status = normalize_coverage_status(coverage.get(aspect, "missing"))
            if status == "missing":
                return "insufficient"
            if status == "partial":
                saw_partial = True
    return "sufficient_with_gaps" if saw_partial else "sufficient"


def comparison_gap_items(
    coverage_by_paper: dict[str, dict[str, str]],
    requested_aspects: list[str],
    target_papers: list[str],
) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    for source in target_papers:
        coverage = coverage_by_paper.get(source, {})
        for aspect in requested_aspects:
            status = normalize_coverage_status(coverage.get(aspect, "missing"))
            if status not in {"partial", "missing"}:
                continue
            gaps.append(
                {
                    "source": source,
                    "aspect": aspect,
                    "status": status,
                    "missing_detail": format_coverage_gap(source, aspect, status),
                }
            )
    return gaps


def build_gap_queries(source: str, aspect: str, missing_detail: str, *, deep: bool = False) -> list[str]:
    label = short_source_label(source).lower()
    if aspect == "损失函数":
        if label == "rag":
            queries = [
                "marginal log likelihood equation",
                "RAG sequence token training objective",
                "answer NLL training loss",
            ]
        elif label == "react":
            queries = [
                "fine-tuning loss objective",
                "trajectory supervision training objective",
                "appendix fine-tuning details",
            ]
        else:
            queries = ["loss objective training details", "appendix implementation details"]
    elif aspect == "训练方式":
        queries = ["training procedure objective", "task adaptation training details"]
    elif aspect == "核心方法":
        queries = ["core method architecture mechanism", "key equation method"]
    elif aspect == "推理流程":
        queries = ["inference execution flow", "reasoning procedure"]
    else:
        words = re.findall(r"[A-Za-z][A-Za-z\-]*", missing_detail)
        queries = [" ".join(words[:8])] if words else [str(aspect)]

    max_queries = 3 if deep else 2
    cleaned: list[str] = []
    for query in queries:
        concise = compact_query_terms(query.split(), max_words=18)
        if concise and concise not in cleaned:
            cleaned.append(concise)
    return cleaned[:max_queries]


def build_source_query_specs(
    coverage_by_paper: dict[str, dict[str, str]],
    requested_aspects: list[str],
    target_papers: list[str],
    *,
    deep: bool = False,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for gap in comparison_gap_items(coverage_by_paper, requested_aspects, target_papers):
        specs.append(
            {
                **gap,
                "queries": build_gap_queries(
                    gap["source"],
                    gap["aspect"],
                    gap["missing_detail"],
                    deep=deep,
                ),
            }
        )
    return specs


def evaluate_comparison_coverage(
    coverage_by_paper: dict[str, dict[str, str]],
    requested_aspects: list[str],
    target_papers: list[str] | None = None,
) -> tuple[str, bool, list[str], list[str], str]:
    missing: list[str] = []
    partial: list[str] = []
    covered: list[str] = []
    sources = list(target_papers or coverage_by_paper.keys())

    for source in sources:
        coverage = coverage_by_paper.get(source, {})
        for aspect in requested_aspects:
            status = normalize_coverage_status(coverage.get(aspect, "missing"))
            if status == "missing":
                missing.append(format_coverage_gap(source, aspect, status))
            elif status == "partial":
                partial.append(format_coverage_gap(source, aspect, status))
            elif status in {"covered", "not_applicable"}:
                covered.append(aspect)

    evidence_status = compute_evidence_status(coverage_by_paper, requested_aspects, sources)
    if evidence_status == "insufficient":
        gaps = list(dict.fromkeys([*missing, *partial]))
        return evidence_status, False, list(dict.fromkeys(covered)), gaps, "；".join(gaps)
    if evidence_status == "sufficient_with_gaps":
        gaps = list(dict.fromkeys(partial))
        return evidence_status, False, list(dict.fromkeys(covered)), gaps, "；".join(gaps)
    return "sufficient", True, list(dict.fromkeys(requested_aspects)), [], "所有用户明确要求的比较维度均已覆盖。"


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    question: str
    intent: str
    selected_papers: list[str]
    comparison_papers: list[str]
    comparison_aspects: list[str]
    plan: str
    next_action: str
    tool_action: dict[str, Any]
    current_query: str
    retrieval_history: list[dict[str, Any]]
    retrieved_docs: list[PaperChunk]
    tool_result: dict[str, Any]
    retrieval_status: str
    retrieval_error_reason: str
    successful_query_count: int
    failed_query_count: int
    consecutive_query_failures: int
    observation: str
    evidence_sufficient: bool
    evidence_reason: str
    requested_aspects: list[str]
    related_aspects: list[str]
    required_aspects: list[str]
    covered_aspects: list[str]
    missing_aspects: list[str]
    coverage_by_paper: dict[str, dict[str, str]]
    evidence_status: str
    seen_result_ids: list[str]
    seen_source_pages: list[str]
    iteration: int
    max_iterations: int
    human_review_required: bool
    awaiting_human: bool
    human_review_reason: str
    human_decision: str
    allow_answer_with_gaps: bool
    auto_search_rounds: int
    max_auto_search_rounds: int
    deep_search_rounds: int
    max_deep_search_rounds: int
    deep_search_queries: list[str]
    deep_search_target_papers: list[str]
    revised_question: str
    cancelled: bool
    last_interrupt_payload: dict[str, Any]
    final_answer: str
    final_answer_streamed: bool
    trace_enabled: bool
    graph_stream_fallback: bool
    error: str
    trace: list[str]


@dataclass
class PlannedAction:
    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    intent: str = "general_qa"
    plan: str = ""
    reason_summary: str = ""
    error: str = ""


def normalize_action_name(action: str) -> str | None:
    return ACTION_ALIASES.get(str(action).strip())


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("没有找到 JSON 对象。")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(cleaned)):
        char = cleaned[idx]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : idx + 1])

    raise ValueError("JSON 对象不完整。")


def parse_json_action(text: str) -> PlannedAction:
    """Parse strict JSON planner output without eval or arbitrary execution."""
    try:
        payload = extract_json_object(text)
    except Exception as exc:
        return PlannedAction(action="answer", error=f"JSON Planner 解析失败：{exc}")

    action = normalize_action_name(str(payload.get("action", "")))
    if action is None:
        return PlannedAction(action="answer", error=f"未知 action：{payload.get('action')}")

    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return PlannedAction(action="answer", error="JSON Planner 参数必须是对象。")

    intent = str(payload.get("intent") or "general_qa")
    if intent not in SUPPORTED_INTENTS:
        intent = "general_qa"

    return PlannedAction(
        action=action,
        arguments=arguments,
        intent=intent,
        plan=str(payload.get("plan") or payload.get("reason_summary") or ""),
        reason_summary=str(payload.get("reason_summary") or payload.get("plan") or ""),
    )


def parse_json_evidence(text: str) -> dict[str, Any]:
    try:
        payload = extract_json_object(text)
    except Exception:
        return {
            "sufficient": False,
            "reason": "证据判断 JSON 解析失败，继续检索以补充证据。",
            "covered_aspects": [],
            "missing_aspects": ["更明确的证据"],
            "suggested_query": "",
        }
    return {
        "sufficient": bool(payload.get("sufficient")),
        "reason": str(payload.get("reason") or ""),
        "covered_aspects": list(payload.get("covered_aspects") or []),
        "missing_aspects": list(payload.get("missing_aspects") or []),
        "suggested_query": str(payload.get("suggested_query") or ""),
    }


def get_chat_model_name() -> str:
    return (os.getenv("CHAT_MODEL") or DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL


def get_tool_mode() -> str:
    tool_mode = (os.getenv(TOOL_MODE_ENV) or "json").strip().lower()
    if tool_mode not in {"json", "native", "auto"}:
        return "json"
    return tool_mode


def get_max_iterations() -> int:
    raw_value = (os.getenv("MAX_AGENT_ITERATIONS") or str(DEFAULT_MAX_ITERATIONS)).strip()
    try:
        return max(1, min(10, int(raw_value)))
    except ValueError:
        return DEFAULT_MAX_ITERATIONS


def get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_checkpoint_path() -> Path:
    raw_value = (os.getenv("LANGGRAPH_CHECKPOINT_PATH") or "").strip()
    if not raw_value:
        return DEFAULT_CHECKPOINT_PATH
    path = Path(raw_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_max_auto_search_rounds() -> int:
    raw_value = (os.getenv("MAX_AUTO_SEARCH_ROUNDS") or str(DEFAULT_MAX_AUTO_SEARCH_ROUNDS)).strip()
    try:
        return max(0, min(10, int(raw_value)))
    except ValueError:
        return DEFAULT_MAX_AUTO_SEARCH_ROUNDS


def get_max_deep_search_rounds() -> int:
    raw_value = (os.getenv("MAX_LOCAL_DEEP_SEARCH_ROUNDS") or str(DEFAULT_MAX_DEEP_SEARCH_ROUNDS)).strip()
    try:
        return max(0, min(5, int(raw_value)))
    except ValueError:
        return DEFAULT_MAX_DEEP_SEARCH_ROUNDS


def create_sqlite_checkpointer(checkpoint_path: Path | None = None) -> tuple[Any, sqlite3.Connection]:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError as exc:
        raise agent_tools.ToolConfigError(
            "缺少 langgraph-checkpoint-sqlite，请运行\npip install langgraph-checkpoint-sqlite"
        ) from exc

    path = checkpoint_path or get_checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection), connection


def sanitize_progress_message(message: str, max_chars: int = 160) -> str:
    safe = re.sub(r"sk-[A-Za-z0-9_\-]+", "[API_KEY]", str(message))
    safe = safe.replace("reasoning_content", "[hidden]")
    safe = safe.replace("api_key", "[api_key]")
    safe = re.sub(r"\s+", " ", safe).strip()
    if len(safe) > max_chars:
        return safe[:max_chars].rstrip() + "..."
    return safe


def extract_chunk_text(chunk: Any) -> str:
    return str(getattr(chunk, "content", "") or "")


def emit_progress(
    message: str,
    *,
    trace_enabled: bool,
    category: str = "状态",
) -> None:
    if not trace_enabled and category != "当前状态":
        return
    print(f"【{category}】{sanitize_progress_message(message)}", flush=True)


class ProgressIndicator:
    """Small Windows-safe progress animation for blocking API calls."""

    def __init__(self, message: str, *, enabled: bool = True, dynamic: bool | None = None) -> None:
        self.message = sanitize_progress_message(message)
        self.enabled = enabled
        self.dynamic = sys.stdout.isatty() if dynamic is None else dynamic
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.is_running = False

    def __enter__(self) -> "ProgressIndicator":
        if not self.enabled:
            return self
        self.is_running = True
        if not self.dynamic:
            return self
        self._thread = threading.Thread(
            target=self._run,
            name="ProgressIndicatorThread",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.dynamic and self.enabled:
            print("\r" + " " * (len(self.message) + 8) + "\r", end="", flush=True)
        self.is_running = False

    def _run(self) -> None:
        dots = 0
        while not self._stop_event.is_set():
            dots = (dots % 3) + 1
            print(f"\r{self.message}{'.' * dots}", end="", flush=True)
            time.sleep(0.35)


def build_chat_model() -> ChatOpenAI:
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if api_key in agent_tools.PLACEHOLDER_API_KEYS:
        raise agent_tools.ToolConfigError(f"未配置有效的 OPENAI_API_KEY，请检查：{ENV_PATH}")
    if base_url in agent_tools.PLACEHOLDER_BASE_URLS:
        raise agent_tools.ToolConfigError("OPENAI_BASE_URL 仍是示例值，请改成实际地址。")

    kwargs: dict[str, Any] = {
        "model": get_chat_model_name(),
        "temperature": 0,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def build_evidence_block(docs: list[PaperChunk]) -> tuple[str, list[dict[str, Any]]]:
    source_index: dict[tuple[str, int | None], int] = {}
    sources: list[dict[str, Any]] = []
    lines: list[str] = []

    for doc in docs:
        key = (doc["source"], doc.get("page"))
        if key not in source_index:
            source_index[key] = len(sources) + 1
            sources.append({"source": doc["source"], "page": doc.get("page")})
        ref = source_index[key]
        lines.append(
            f"[{ref}] {doc['source']}，第 {doc.get('page') or 'unknown'} 页：{doc['content']}"
        )
    return "\n".join(lines), sources


def merge_unique_chunks(existing: list[PaperChunk], incoming: list[PaperChunk]) -> list[PaperChunk]:
    merged = list(existing)
    seen_ids = {chunk["result_id"] for chunk in merged}
    seen_fingerprints = {
        (chunk["source"], chunk.get("page"), chunk["content"][:500].lower())
        for chunk in merged
    }
    for chunk in incoming:
        fingerprint = (chunk["source"], chunk.get("page"), chunk["content"][:500].lower())
        if chunk["result_id"] in seen_ids or fingerprint in seen_fingerprints:
            continue
        merged.append(chunk)
        seen_ids.add(chunk["result_id"])
        seen_fingerprints.add(fingerprint)
    return merged


def format_source_list(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    lines = ["【资料来源】"]
    for idx, source in enumerate(sources, start=1):
        page = source.get("page") or "unknown"
        lines.append(f"[{idx}] {source.get('source', 'unknown')}，第 {page} 页")
    return "\n".join(lines)


def strip_existing_source_list(answer: str) -> str:
    return answer.split("【资料来源】", 1)[0].strip()


def remove_invalid_citations(answer: str, source_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return match.group(0) if 1 <= number <= source_count else ""

    return CITATION_RE.sub(replace, answer)


def compact_used_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    body = strip_existing_source_list(answer)
    old_numbers: list[int] = []
    for match in CITATION_RE.finditer(body):
        number = int(match.group(1))
        if 1 <= number <= len(sources) and number not in old_numbers:
            old_numbers.append(number)

    if not old_numbers:
        return body, []

    old_to_new = {
        old_number: new_number
        for new_number, old_number in enumerate(old_numbers, start=1)
    }
    used_sources = [dict(sources[old_number - 1]) for old_number in old_numbers]

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number not in old_to_new:
            return ""
        return f"[{old_to_new[number]}]"

    return CITATION_RE.sub(replace, body), used_sources


def validate_final_citations(
    body: str,
    sources: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    cleaned_body = strip_existing_source_list(body)
    cleaned_body = remove_invalid_citations(cleaned_body, len(sources))
    compacted_body, compacted_sources = compact_used_citations(cleaned_body, sources)

    citation_ids = [int(item) for item in CITATION_RE.findall(compacted_body)]
    if not citation_ids:
        return compacted_body, []

    source_count = len(compacted_sources)
    if any(number < 1 or number > source_count for number in citation_ids):
        compacted_body = remove_invalid_citations(compacted_body, source_count)
        compacted_body, compacted_sources = compact_used_citations(compacted_body, compacted_sources)
        citation_ids = [int(item) for item in CITATION_RE.findall(compacted_body)]

    expected_ids = set(range(1, len(compacted_sources) + 1))
    actual_ids = set(citation_ids)
    if actual_ids != expected_ids:
        compacted_body, compacted_sources = compact_used_citations(compacted_body, compacted_sources)

    return compacted_body, compacted_sources


def remove_proactive_followup(answer: str) -> str:
    cleaned = answer
    patterns = [
        r"如果你还想[^。！？\n]*[。！？]?",
        r"如果你想[^。！？\n]*[。！？]?",
        r"如果需要[^。！？\n]*[。！？]?",
        r"我可以继续[^。！？\n]*[。！？]?",
        r"还可以继续[^。！？\n]*[。！？]?",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def ensure_answer_has_citation(answer: str, source_count: int) -> str:
    if source_count <= 0 or re.search(r"\[(\d+)\]", answer):
        return answer
    return f"{answer} [1]"


NEGATIVE_EVIDENCE_QUESTION_MARKERS = (
    "是否明确说明",
    "有没有明确说明",
    "是否明确写出",
    "是否写明",
    "有没有写清楚",
    "有没有给出",
    "是否采用",
    "是否存在",
    "是否明确给出",
    "有没有明确给出",
    "请不要用常识补全",
    "不要用常识补全",
    "not explicitly",
    "not specified",
    "explicitly specify",
    "explicitly state",
)

NEGATIVE_EVIDENCE_TEXT_MARKERS = (
    "没有明确",
    "未明确",
    "未提供",
    "没有提供",
    "没有给出",
    "未给出",
    "未发现",
    "没有发现",
    "没有证据表明",
    "无法判断论文是否明确",
    "not explicitly",
    "not specify",
    "not specified",
    "not provided",
    "no evidence",
    "does not explicitly",
)


def is_negative_evidence_question(question: str) -> bool:
    lowered = str(question or "").casefold()
    return any(marker.casefold() in lowered for marker in NEGATIVE_EVIDENCE_QUESTION_MARKERS)


def is_whole_paper_analysis_request(question: str) -> bool:
    normalized = normalize_question(question)
    return any(
        phrase in normalized
        for phrase in ("完整分析", "全面分析", "全面总结", "详细分析", "详细总结", "系统介绍")
    )


def evidence_text_indicates_not_explicit(*texts: str) -> bool:
    lowered = " ".join(str(text or "") for text in texts).casefold()
    return any(marker.casefold() in lowered for marker in NEGATIVE_EVIDENCE_TEXT_MARKERS)


def supports_negative_evidence_answer(
    question: str,
    evidence_reason: str,
    retrieved_docs: list[dict[str, Any]],
) -> bool:
    """Recognize evidence-backed answers that a paper does not state a detail."""
    return (
        bool(retrieved_docs)
        and is_negative_evidence_question(question)
        and evidence_text_indicates_not_explicit(evidence_reason)
    )


def contains_unresolved_pronoun(question: str) -> bool:
    patterns = ["这篇论文", "这篇", "该论文", "它的", "它们的", "这项工作"]
    return any(pattern in question for pattern in patterns)


def infer_intent(
    question: str,
    recent_sources: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Deterministically recognize high-confidence intents before the LLM planner.

    This function intentionally does not call an extra intent-classification LLM:
    the LangGraph planner already performs LLM planning. Keeping this layer
    rule-based avoids duplicate API calls and prevents an undefined
    ``infer_intent_with_llm`` dependency.
    """
    normalized = normalize_question(question)
    recent_sources = recent_sources or []
    explicit_papers = detect_explicit_papers(question)

    is_comparison = any(keyword in normalized for keyword in COMPARISON_KEYWORDS)

    # Highest priority: explicit comparison with two or more named papers.
    if is_comparison and len(explicit_papers) >= 2:
        return "paper_comparison", explicit_papers

    # Comparison request with incomplete names: inherit recent context if possible.
    if is_comparison:
        context_papers = list(dict.fromkeys([*explicit_papers, *recent_sources]))
        if len(context_papers) >= 2:
            return "paper_comparison", context_papers[:2]
        return "clarification", context_papers

    # Explicitness checks ask for a bounded fact, rather than an end-to-end
    # paper analysis. Keep them out of profile-driven multi-aspect retrieval.
    if is_negative_evidence_question(question) and not is_whole_paper_analysis_request(question):
        return "general_qa", explicit_papers

    # High-confidence whole-paper analysis request.
    if any(
        phrase in normalized
        for phrase in (
            "完整分析",
            "全面分析",
            "全面总结",
            "详细分析",
            "详细总结",
            "方法和实验",
            "方法与实验",
            "训练过程和实验结果",
        )
    ):
        return "paper_analysis", explicit_papers

    # Strong task-specific rules. Ambiguous requests stay general_qa and are
    # further planned by the existing JSON/native planner.
    if any(term in normalized for term in ("实验结果", "消融实验", "实验设置", "评价指标")):
        return "experiment_analysis", explicit_papers
    if any(term in normalized for term in ("复现", "实现计划", "代码计划")):
        return "reproduction_plan", explicit_papers
    if any(term in normalized for term in ("有什么启发", "研究启发", "对我的研究")):
        return "research_inspiration", explicit_papers
    if any(term in normalized for term in ("方法是什么", "解释方法", "核心公式", "公式什么意思", "模型结构")):
        return "method_explain", explicit_papers
    if any(term in normalized for term in ("总结这篇论文", "总结论文", "论文总结")):
        return "paper_summary", explicit_papers

    return "general_qa", explicit_papers


def get_paper_analysis_aspects(source: str | None) -> list[str]:
    paper_type = PAPER_TYPE_BY_SOURCE.get(str(source or ""))
    if paper_type == "lora_parameter_efficient":
        return list(PAPER_ANALYSIS_REQUIRED_ASPECTS)
    return list(PAPER_TYPE_ANALYSIS_ASPECTS.get(str(paper_type), GENERIC_PAPER_ANALYSIS_ASPECTS))


def get_combined_paper_analysis_aspects(sources: list[str]) -> list[str]:
    if not sources:
        return list(GENERIC_PAPER_ANALYSIS_ASPECTS)
    aspects: list[str] = []
    for source in sources:
        for aspect in get_paper_analysis_aspects(source):
            if aspect not in aspects:
                aspects.append(aspect)
    return aspects


PAPER_ANALYSIS_COVERAGE_ALIASES = {
    "研究问题与贡献": {"研究问题与贡献", "研究问题与动机", "核心问题与动机"},
    "模型结构": {"模型结构", "核心方法", "核心方法与数学公式"},
    "注意力与位置编码公式": {"注意力与位置编码公式", "核心方法与数学公式"},
    "训练设置": {"训练设置", "训练方式", "训练过程"},
    "实验结果与结论": {"实验结果与结论", "实验结果", "主要定量实验结果"},
}


def paper_analysis_missing_aspects(
    source: str | None,
    required_aspects: list[str],
    missing_aspects: list[str],
    covered_aspects: list[str],
) -> list[str]:
    """Keep every explicit gap, then fill in profile aspects not yet covered."""
    explicit_missing = [aspect for aspect in missing_aspects if aspect in required_aspects]
    covered = {str(aspect) for aspect in covered_aspects}

    inferred_missing: list[str] = []
    if covered:
        for aspect in required_aspects:
            aliases = PAPER_ANALYSIS_COVERAGE_ALIASES.get(aspect, {aspect})
            if not covered.intersection(aliases):
                inferred_missing.append(aspect)
    elif not explicit_missing:
        inferred_missing = list(required_aspects)

    return list(dict.fromkeys([*explicit_missing, *inferred_missing]))


def _covered_aspect_matches(aspect: str, covered_aspects: list[str]) -> bool:
    aliases = PAPER_ANALYSIS_COVERAGE_ALIASES.get(aspect, {aspect})
    return bool(aliases.intersection({str(item) for item in covered_aspects}))


def _docs_support_paper_analysis_aspect(aspect: str, docs: list[PaperChunk]) -> bool:
    text = "\n".join(str(doc.get("content") or "") for doc in docs).casefold()
    if not text:
        return False
    terms = [str(term).casefold() for term in PAPER_ANALYSIS_ASPECT_TERMS.get(aspect, []) if str(term).strip()]
    if not terms:
        return False
    matches = sum(1 for term in terms if term in text)
    return matches >= min(2, len(terms))


def refresh_final_answer_evidence_state(state: AgentState) -> AgentState:
    """Recompute terminal gaps from cumulative evidence before composing an answer."""
    updated = dict(state)
    docs = list(updated.get("retrieved_docs") or [])
    if not docs:
        return updated

    if updated.get("intent") == "paper_comparison":
        sources = list(updated.get("comparison_papers") or updated.get("selected_papers") or [])
        requested = list(updated.get("requested_aspects") or updated.get("required_aspects") or [])
        coverage = {
            source: dict(aspects)
            for source, aspects in dict(updated.get("coverage_by_paper") or {}).items()
        }
        for source in sources:
            coverage.setdefault(source, {aspect: "missing" for aspect in requested})
            source_docs = [doc for doc in docs if doc.get("source") == source]
            for aspect in requested:
                current = coverage[source].get(aspect, "missing")
                candidate = classify_aspect_coverage_from_chunks(
                    aspect=aspect,
                    source=source,
                    chunks=source_docs,
                    result_count=len(source_docs),
                    question=str(updated.get("question") or ""),
                )
                if should_replace_coverage(current, candidate):
                    coverage[source][aspect] = candidate
        status, sufficient, covered, missing, reason = evaluate_comparison_coverage(coverage, requested, sources)
        return {
            **updated,
            "coverage_by_paper": coverage,
            "covered_aspects": covered,
            "missing_aspects": missing,
            "evidence_status": status,
            "evidence_sufficient": sufficient,
            "evidence_reason": reason,
        }

    required = list(updated.get("required_aspects") or updated.get("requested_aspects") or [])
    if not required:
        return updated
    covered = list(updated.get("covered_aspects") or [])
    for aspect in required:
        if not _covered_aspect_matches(aspect, covered) and _docs_support_paper_analysis_aspect(aspect, docs):
            covered.append(aspect)
    remaining = [aspect for aspect in required if not _covered_aspect_matches(aspect, covered)]
    if not remaining:
        return {
            **updated,
            "covered_aspects": list(dict.fromkeys(covered)),
            "missing_aspects": [],
            "evidence_status": "sufficient",
            "evidence_sufficient": True,
            "evidence_reason": "当前累计检索证据已覆盖所需方面。",
        }
    return {
        **updated,
        "covered_aspects": list(dict.fromkeys(covered)),
        "missing_aspects": remaining,
        "evidence_reason": f"当前累计检索仍缺少：{'、'.join(remaining)}。",
    }


def _analysis_priority_for_source(source: str, required_aspects: list[str]) -> list[str]:
    paper_type = PAPER_TYPE_BY_SOURCE.get(str(source or ""))
    if paper_type == "lora_parameter_efficient":
        base = list(PAPER_ANALYSIS_REQUIRED_ASPECTS)
        priority = [base[index] for index in (1, 2, 4, 5, 7, 8, 6, 3, 9, 0) if index < len(base)]
        ordered = [aspect for aspect in priority if aspect in required_aspects]
        ordered.extend(aspect for aspect in required_aspects if aspect not in ordered)
        return ordered
    return list(required_aspects)


def title_terms_from_source(source: str) -> list[str]:
    stem = Path(source).stem
    stem = re.sub(r"^\d+\s*", "", stem)
    tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9]+", stem)
        if token and token.lower() not in {"the", "and", "for", "with", "large", "language", "models"}
    ]
    if not tokens:
        return []
    first = tokens[0]
    extras = [token for token in tokens[1:] if token.lower() not in {"low", "rank", "adaptation"}][:2]
    return [first, *extras]


def compact_query_terms(terms: list[str], max_words: int = 18) -> str:
    words: list[str] = []
    for term in terms:
        for word in str(term).split():
            lowered = word.lower()
            if any(lowered == forbidden for forbidden in FORBIDDEN_AUTO_QUERY_TERMS):
                continue
            if word not in words:
                words.append(word)
            if len(words) >= max_words:
                return " ".join(words)
    return " ".join(words)


def validate_concise_query(query: str) -> str:
    cleaned = re.sub(r"\s+", " ", query).strip()
    lowered = cleaned.lower()
    for forbidden in FORBIDDEN_AUTO_QUERY_TERMS:
        lowered = lowered.replace(forbidden, " ")
    cleaned = re.sub(r"\s+", " ", lowered).strip()
    if not cleaned:
        raise ValueError("query 不能为空。")
    words = cleaned.split()
    if len(words) > 25:
        raise ValueError("query 过长，请按单一信息点拆分。")
    broad_targets = [
        any(term in cleaned for term in ["equation", "formula", "method"]),
        any(term in cleaned for term in ["optimizer", "training hyperparameter"]),
        any(term in cleaned for term in ["dataset", "datasets", "task", "tasks"]),
        any(term in cleaned for term in ["metric", "metrics", "baseline", "baselines"]),
        any(term in cleaned for term in ["ablation", "rank"]),
    ]
    if sum(1 for item in broad_targets if item) >= 4:
        raise ValueError("query 同时包含过多目标，请拆分为 aspect 子查询。")
    return cleaned


def build_paper_analysis_queries(
    source: str,
    required_aspects: list[str],
    missing_aspects: list[str],
    max_queries: int = 6,
) -> list[str]:
    title_terms = title_terms_from_source(source)
    aspects = [aspect for aspect in missing_aspects if aspect in required_aspects] or required_aspects
    if set(aspects) == set(required_aspects):
        priority = _analysis_priority_for_source(source, list(required_aspects))
        ordered = [aspect for aspect in priority if aspect in aspects]
        ordered.extend(aspect for aspect in aspects if aspect not in ordered)
    else:
        # Rewrite callers order explicit missing aspects before inferred gaps.
        ordered = list(dict.fromkeys(aspects))

    queries: list[str] = []
    for aspect in ordered[:max_queries]:
        aspect_terms = PAPER_ANALYSIS_ASPECT_TERMS.get(aspect, [aspect])
        query = compact_query_terms([*title_terms, *aspect_terms])
        if query and query not in queries:
            queries.append(query)
    return queries[:max_queries]


def normalize_missing_aspects(
    required: list[str],
    covered: list[str],
    missing: list[str],
) -> list[str]:
    if not required:
        return missing
    covered_set = set(covered)
    if missing:
        return [aspect for aspect in missing if aspect in required and aspect not in covered_set]
    return [aspect for aspect in required if aspect not in covered_set]


def build_deep_search_queries(missing_aspects: list[str], max_queries: int = 4) -> list[str]:
    """Build precise local-only queries for missing requested aspects."""
    text = " ".join(str(item) for item in missing_aspects).lower()
    queries: list[str] = []
    if any(term in text for term in ("loss", "损失", "objective", "目标函数", "监督")):
        queries.extend(
            [
                "fine-tuning loss objective",
                "supervised fine-tuning training objective",
                "trajectory supervision loss",
                "appendix training details objective supervision loss",
            ]
        )
    if any(term in text for term in ("training", "训练", "fine-tuning", "微调")):
        queries.extend(["implementation details training procedure", "training details appendix"])
    if any(term in text for term in ("inference", "推理", "流程")):
        queries.extend(["inference execution flow", "reasoning acting procedure"])
    if any(term in text for term in ("result", "实验", "指标", "baseline", "消融")):
        queries.extend(["experiments baselines evaluation metrics results", "ablation results analysis"])
    if not queries:
        for aspect in missing_aspects:
            words = re.findall(r"[A-Za-z][A-Za-z\-]*", str(aspect))
            if words:
                queries.append(" ".join(words[:8]))
        if not queries:
            queries.append("appendix implementation details")

    cleaned: list[str] = []
    for query in queries:
        concise = compact_query_terms(query.split(), max_words=18)
        if concise and concise not in cleaned:
            cleaned.append(concise)
    return cleaned[:max_queries]


class PaperAgent:
    """LangGraph-based research-paper agent with planning, tools, observation, and retry."""

    def __init__(
        self,
        tools: PaperTools,
        llm: Any | None = None,
        tool_mode: str | None = None,
        max_iterations: int | None = None,
        thread_id: str | None = None,
        checkpointer: Any | None = None,
        checkpoint_path: Path | None = None,
        enable_human_review: bool | None = None,
        max_auto_search_rounds: int | None = None,
        max_deep_search_rounds: int | None = None,
    ) -> None:
        if llm is None:
            load_dotenv(dotenv_path=ENV_PATH)
        self.tools = tools
        self.thread_id = thread_id or str(uuid.uuid4())
        self._checkpoint_connection: sqlite3.Connection | None = None
        if checkpointer is not None:
            self.checkpointer = checkpointer
        elif llm is None:
            self.checkpointer, self._checkpoint_connection = create_sqlite_checkpointer(
                checkpoint_path or get_checkpoint_path()
            )
        else:
            from langgraph.checkpoint.memory import InMemorySaver

            self.checkpointer = InMemorySaver()

        self.llm = llm or build_chat_model()
        self.chat_model_name = get_chat_model_name()
        self.tool_mode = (tool_mode or get_tool_mode()).strip().lower()
        if self.tool_mode not in {"auto", "native", "json"}:
            self.tool_mode = "json"
        self.max_iterations = max_iterations if max_iterations is not None else get_max_iterations()
        self.enable_human_review = (
            get_bool_env("ENABLE_HUMAN_REVIEW", True) if llm is None else False
            if enable_human_review is None
            else bool(enable_human_review)
        )
        self.max_auto_search_rounds = (
            max_auto_search_rounds
            if max_auto_search_rounds is not None
            else get_max_auto_search_rounds()
        )
        self.max_deep_search_rounds = (
            max_deep_search_rounds
            if max_deep_search_rounds is not None
            else get_max_deep_search_rounds()
        )
        self.messages: list[dict[str, str]] = []
        self.last_sources: list[str] = []
        self.graph = self._build_graph()

    def ask(self, question: str, *, trace_enabled: bool = False) -> dict[str, Any]:
        emit_progress("正在分析问题……", trace_enabled=trace_enabled, category="当前状态")
        available_papers = self._available_papers()
        explicit_papers = [
            source
            for source in (
                resolve_paper_reference(source, available_papers)
                for source in detect_explicit_papers(question)
            )
            if source
        ]
        selected_from_context = [] if explicit_papers else self._resolve_sources_from_context(question)
        inferred_intent, inferred_papers = infer_intent(
            question,
            recent_sources=self.last_sources,
        )
        selected_candidates = explicit_papers or [*inferred_papers, *selected_from_context]
        selected = list(
            dict.fromkeys(
                source
                for source in (
                    resolve_paper_reference(candidate, available_papers)
                    for candidate in selected_candidates
                )
                if source
            )
        )
        comparison_aspects = (
            extract_comparison_aspects(question)
            if inferred_intent == "paper_comparison"
            else []
        )
        if inferred_intent == "paper_analysis":
            requested_aspects = get_combined_paper_analysis_aspects(selected)
        elif inferred_intent == "paper_comparison":
            requested_aspects = list(comparison_aspects)
        else:
            requested_aspects = []
        required_aspects = list(requested_aspects)
        related_aspects = derive_related_aspects(inferred_intent, requested_aspects, question)
        comparison_papers = list(selected) if inferred_intent == "paper_comparison" else []

        initial_state: AgentState = {
            "messages": list(self.messages),
            "question": question,
            "intent": inferred_intent,
            "selected_papers": selected,
            "comparison_papers": comparison_papers,
            "comparison_aspects": comparison_aspects,
            "plan": "",
            "next_action": "",
            "tool_action": {},
            "current_query": question,
            "retrieval_history": [],
            "retrieved_docs": [],
            "retrieval_status": "success",
            "retrieval_error_reason": "",
            "successful_query_count": 0,
            "failed_query_count": 0,
            "consecutive_query_failures": 0,
            "evidence_sufficient": False,
            "evidence_reason": "",
            "requested_aspects": requested_aspects,
            "related_aspects": related_aspects,
            "required_aspects": required_aspects,
            "covered_aspects": [],
            "missing_aspects": list(required_aspects),
            "coverage_by_paper": initial_coverage_by_paper(comparison_papers, requested_aspects)
            if inferred_intent == "paper_comparison"
            else {},
            "evidence_status": "insufficient",
            "seen_result_ids": [],
            "seen_source_pages": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "human_review_required": False,
            "awaiting_human": False,
            "human_review_reason": "",
            "human_decision": "",
            "allow_answer_with_gaps": False,
            "auto_search_rounds": 0,
            "max_auto_search_rounds": self.max_auto_search_rounds,
            "deep_search_rounds": 0,
            "max_deep_search_rounds": self.max_deep_search_rounds,
            "deep_search_queries": [],
            "deep_search_target_papers": [],
            "revised_question": "",
            "cancelled": False,
            "last_interrupt_payload": {},
            "final_answer": "",
            "final_answer_streamed": False,
            "trace_enabled": trace_enabled,
            "graph_stream_fallback": False,
            "error": "",
            "trace": [f"【会话线程】{self.thread_id}"] if trace_enabled else [],
        }

        final_state = self._run_graph(initial_state)
        answer = "" if final_state.get("awaiting_human") else final_state.get("final_answer") or "未能生成回答。"

        if not final_state.get("awaiting_human"):
            self.messages.append({"role": "user", "content": question})
            self.messages.append({"role": "assistant", "content": answer})
            self._remember_sources(final_state.get("retrieved_docs", []), selected)

        return {"answer": answer, "trace": final_state.get("trace", []), "state": final_state}

    def clear(self) -> None:
        self.delete_current_thread()
        self.new_session()

    def new_session(self) -> str:
        self.thread_id = str(uuid.uuid4())
        self.messages.clear()
        self.last_sources.clear()
        return self.thread_id

    def switch_thread(self, thread_id: str) -> None:
        cleaned = str(thread_id).strip()
        if not cleaned:
            raise ValueError("thread_id 不能为空。")
        self.thread_id = cleaned
        self.messages.clear()
        self.last_sources.clear()

    def delete_current_thread(self) -> None:
        delete_thread = getattr(self.checkpointer, "delete_thread", None)
        if callable(delete_thread):
            delete_thread(self.thread_id)

    def close(self) -> None:
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

    def _graph_config(self) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.max_iterations * 8 + 20,
        }

    def resume(self, resume_payload: dict[str, Any], *, trace_enabled: bool = False) -> dict[str, Any]:
        final_state = self._resume_graph(Command(resume=resume_payload), trace_enabled=trace_enabled)
        answer = "" if final_state.get("awaiting_human") else final_state.get("final_answer") or "未能生成回答。"
        if not final_state.get("awaiting_human") and answer:
            question = final_state.get("question", "")
            if question:
                self.messages.append({"role": "user", "content": question})
            self.messages.append({"role": "assistant", "content": answer})
            self._remember_sources(final_state.get("retrieved_docs", []), final_state.get("selected_papers", []))
        return {"answer": answer, "trace": final_state.get("trace", []), "state": final_state}

    def _run_graph(self, initial_state: AgentState) -> AgentState:
        try:
            return self._run_graph_stream(initial_state)
        except Exception as exc:
            fallback_state = self._run_graph_fallback(initial_state)
            fallback_state["graph_stream_fallback"] = True
            fallback_state["trace"] = fallback_state.get("trace", []) + [f"【异常与重试】graph.stream 不可用，已回退：{type(exc).__name__}"]
            return fallback_state

    def _run_graph_stream(self, initial_state: AgentState) -> AgentState:
        final_state: AgentState = dict(initial_state)
        stream_kwargs: dict[str, Any] = {
            "stream_mode": ["updates", "custom"],
            "config": self._graph_config(),
        }
        try:
            stream = self.graph.stream(initial_state, **stream_kwargs)
        except TypeError:
            stream = self.graph.stream(initial_state, version="v2", **stream_kwargs)

        for event in stream:
            mode, payload = self._normalize_stream_event(event)
            if mode == "updates" and isinstance(payload, dict):
                interrupt_payload = self._extract_interrupt_payload(payload)
                if interrupt_payload is not None:
                    return self._build_interrupted_state(final_state, interrupt_payload)
                for update in payload.values():
                    if isinstance(update, dict):
                        final_state.update(update)
            elif mode == "custom" and isinstance(payload, dict):
                continue

        return final_state

    def _resume_graph(self, command: Command, *, trace_enabled: bool = False) -> AgentState:
        final_state: AgentState = dict(self._checkpoint_values())
        if trace_enabled:
            final_state["trace_enabled"] = True
            final_state["trace"] = final_state.get("trace", []) + [
                "【图状态】从 checkpoint 恢复",
                f"【恢复线程】{self.thread_id}",
            ]
        stream_kwargs: dict[str, Any] = {
            "stream_mode": ["updates", "custom"],
            "config": self._graph_config(),
        }
        try:
            stream = self.graph.stream(command, **stream_kwargs)
        except TypeError:
            stream = self.graph.stream(command, version="v2", **stream_kwargs)

        for event in stream:
            mode, payload = self._normalize_stream_event(event)
            if mode == "updates" and isinstance(payload, dict):
                interrupt_payload = self._extract_interrupt_payload(payload)
                if interrupt_payload is not None:
                    return self._build_interrupted_state(final_state, interrupt_payload)
                for update in payload.values():
                    if isinstance(update, dict):
                        final_state.update(update)
            elif mode == "custom" and isinstance(payload, dict):
                continue

        checkpoint_values = self._checkpoint_values()
        if checkpoint_values:
            checkpoint_values.update(final_state)
            final_state = checkpoint_values
        return final_state

    def _checkpoint_values(self) -> AgentState:
        try:
            snapshot = self.graph.get_state(self._graph_config())
        except Exception:
            return {}
        values = getattr(snapshot, "values", None)
        return dict(values or {})

    def _extract_interrupt_payload(self, payload: Any) -> dict[str, Any] | None:
        interrupt_items: Any = None
        if isinstance(payload, dict):
            interrupt_items = payload.get("__interrupt__")
        if not interrupt_items:
            return None
        if not isinstance(interrupt_items, (list, tuple)):
            interrupt_items = [interrupt_items]
        for item in interrupt_items:
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
            if isinstance(item, dict):
                return item
        return None

    def _build_interrupted_state(self, state: AgentState, payload: dict[str, Any]) -> AgentState:
        checkpoint_state = self._checkpoint_values()
        merged_state: AgentState = {**state, **checkpoint_state}
        trace = merged_state.get("trace", []) + [
            f"【人工干预原因】{payload.get('reason') or merged_state.get('evidence_reason', '')}",
            "【图状态】已暂停，等待用户选择",
        ]
        return {
            **merged_state,
            "awaiting_human": True,
            "human_review_required": True,
            "human_review_reason": str(payload.get("reason") or merged_state.get("evidence_reason", "")),
            "last_interrupt_payload": payload,
            "final_answer": "",
            "final_answer_streamed": False,
            "trace": trace,
        }

    def get_interrupt_payload(self, thread_id: str | None = None) -> dict[str, Any] | None:
        previous_thread_id = self.thread_id
        if thread_id is not None:
            self.thread_id = thread_id
        try:
            snapshot = self.graph.get_state(self._graph_config())
        except Exception:
            return None
        finally:
            if thread_id is not None:
                self.thread_id = previous_thread_id

        interrupts = list(getattr(snapshot, "interrupts", ()) or ())
        if not interrupts:
            for task in getattr(snapshot, "tasks", ()) or ():
                interrupts.extend(getattr(task, "interrupts", ()) or ())
        for item in interrupts:
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                return value
        return None

    def is_interrupted(self, thread_id: str | None = None) -> bool:
        return self.get_interrupt_payload(thread_id) is not None

    def thread_exists(self, thread_id: str | None = None) -> bool:
        previous_thread_id = self.thread_id
        if thread_id is not None:
            self.thread_id = thread_id
        try:
            snapshot = self.graph.get_state(self._graph_config())
        except Exception:
            return False
        finally:
            if thread_id is not None:
                self.thread_id = previous_thread_id
        return bool(
            getattr(snapshot, "values", None)
            or getattr(snapshot, "next", None)
            or getattr(snapshot, "tasks", None)
        )

    def _run_graph_fallback(self, initial_state: AgentState) -> AgentState:
        state: AgentState = self.planner_node(initial_state)
        if self.route_after_planner(state) == "answer":
            return self.answer_node(state)

        while True:
            state = self.tool_node(state)
            state = self.observation_node(state)
            state = self.evidence_check_node(state)
            route = self.route_after_evidence_check(state)
            if route == "answer":
                return self.answer_node(state)
            if route == "retrieval_failure":
                return {
                    **state,
                    "final_answer": "【检索服务暂时不可用】\n本轮查询全部失败，可能由上游限流或服务负载导致。请稍后重试当前检索。",
                    "final_answer_streamed": False,
                    "trace": state.get("trace", []) + ["【图状态】stream 不可用，无法暂停，返回检索失败提示。"],
                }
            if route == "human_review":
                return self.answer_node(
                    {
                        **state,
                        "allow_answer_with_gaps": True,
                        "trace": state.get("trace", []) + ["【图状态】stream 不可用，无法暂停，改为带缺口回答。"],
                    }
                )
            state = self.rewrite_query_node(state)

    def _normalize_stream_event(self, event: Any) -> tuple[str, Any]:
        if isinstance(event, tuple) and len(event) == 2:
            return str(event[0]), event[1]
        if isinstance(event, dict):
            return "updates", event
        return "unknown", event

    def _emit_progress(
        self,
        state: AgentState,
        message: str,
        *,
        category: str = "状态",
        event_type: str = "status",
    ) -> None:
        trace_enabled = bool(state.get("trace_enabled", False))
        emit_progress(message, trace_enabled=trace_enabled, category=category)
        try:
            writer = get_stream_writer()
            writer({"event_type": event_type, "category": category, "message": sanitize_progress_message(message)})
        except Exception:
            pass

    def _progress_indicator(self, state: AgentState, message: str) -> ProgressIndicator:
        enabled = get_bool_env("SHOW_PROGRESS_ANIMATION", True) and not bool(state.get("trace_enabled", False))
        return ProgressIndicator(message, enabled=enabled)

    def validate_planned_action(
        self,
        planned: PlannedAction,
        state: AgentState | None = None,
    ) -> PlannedAction:
        """Validate and normalize planner output before any tool is executed."""
        if planned.error:
            return planned

        action = normalize_action_name(planned.action)
        if action is None:
            return PlannedAction(
                action="answer",
                arguments={},
                intent=planned.intent,
                reason_summary=planned.reason_summary,
                error=f"未知 action：{planned.action}",
            )

        if action in {"answer", "clarify", "list_papers"}:
            planned.action = action
            return planned

        arguments = dict(planned.arguments)
        available_papers = self._available_papers()
        papers = set(available_papers)
        question = str((state or {}).get("question") or "")
        corpus_scope = has_unbounded_corpus_scope(question, available_papers) if question else False

        def validate_source(required: bool) -> str | None:
            raw_source = arguments.get("source")
            if raw_source is None or raw_source == "":
                if required:
                    raise ValueError("source 是必填参数。")
                return None
            source = resolve_paper_reference(str(raw_source), papers)
            if source is None:
                raise ValueError(f"source 不在当前数据库论文列表中：{raw_source}")
            return source

        try:
            if action == "search_paper":
                if corpus_scope:
                    arguments["source"] = None
                source = validate_source(required=False)
                query = validate_concise_query(str(arguments.get("query") or ""))
                arguments = {
                    "query": query,
                    "source": source,
                    "k": agent_tools.clamp_int(int(arguments.get("k", agent_tools.DEFAULT_K)), 1, agent_tools.MAX_K),
                }
            elif action == "search_multiple_queries":
                raw_queries = arguments.get("queries") or []

                if not isinstance(raw_queries, list):
                    raise ValueError(
                        "search_multiple_queries 的 queries 必须是列表。"
                    )

                queries = [
                    validate_concise_query(str(query))
                    for query in raw_queries
                    if str(query).strip()
                ]
                query_limit = (
                    MAX_COMPARISON_QUERIES
                    if (planned.intent == "paper_comparison" or (state or {}).get("intent") == "paper_comparison")
                    else agent_tools.MAX_MULTI_QUERIES
                )
                queries = queries[:query_limit]

                if not queries:
                    raise ValueError(
                        "search_multiple_queries 至少需要 1 个 query。"
                    )

                raw_source = arguments.get("source")
                raw_sources = arguments.get("sources")

                if raw_sources is not None and not isinstance(raw_sources, list):
                    raise ValueError("sources 必须是列表。")

                selected_sources: list[str] = []

                if state is not None:
                    state_sources = list(state.get("comparison_papers") or state.get("selected_papers") or [])
                    resolved_state_sources = [
                        resolve_paper_reference(str(source), papers) or str(source)
                        for source in state_sources
                        if str(source).strip()
                    ]
                    if len(resolved_state_sources) >= 2:
                        selected_sources = list(dict.fromkeys(resolved_state_sources))

                if corpus_scope:
                    selected_sources = list(available_papers)
                elif not selected_sources and raw_source:
                    selected_sources.append(
                        resolve_paper_reference(str(raw_source), papers) or str(raw_source)
                    )

                if not selected_sources:
                    selected_sources.extend(
                        resolve_paper_reference(str(source), papers) or str(source)
                        for source in (raw_sources or [])
                        if str(source).strip()
                    )

                selected_sources = list(dict.fromkeys(selected_sources))

                # Planner 没有显式返回 source/sources 时，使用当前状态中已经识别出的目标论文。
                if not selected_sources:
                    fallback_sources = []
                    if state is not None:
                        fallback_sources = list(state.get("comparison_papers") or state.get("selected_papers") or [])
                    selected_sources = list(
                        dict.fromkeys(
                            resolve_paper_reference(str(source), papers) or str(source)
                            for source in fallback_sources
                            if str(source).strip()
                        )
                    )

                if not selected_sources:
                    raise ValueError(
                        "source 或 sources 至少需要提供一个。"
                    )

                invalid_sources = [
                    source
                    for source in selected_sources
                    if source not in papers
                ]

                if invalid_sources:
                    raise ValueError(
                        "以下 source 不在当前数据库论文列表中："
                        + "、".join(invalid_sources)
                    )

                arguments = {
                    "queries": queries,
                    "k_per_query": agent_tools.clamp_int(
                        int(arguments.get("k_per_query", 4)),
                        1,
                        agent_tools.MAX_K,
                    ),
                }
                if len(selected_sources) == 1:
                    arguments["source"] = selected_sources[0]
                else:
                    arguments["sources"] = selected_sources
            elif action == "get_neighbor_chunks":
                source = validate_source(required=True)
                arguments = {
                    "source": source,
                    "page": max(1, int(arguments.get("page", 1))),
                    "radius": agent_tools.clamp_int(
                        int(arguments.get("radius", 1)),
                        0,
                        agent_tools.MAX_NEIGHBOR_RADIUS,
                    ),
                }
            elif action == "inspect_paper_scope":
                source = validate_source(required=True)
                arguments = {"source": source}
            else:
                raise ValueError(f"未知 action：{action}")
        except Exception as exc:
            return PlannedAction(
                action="answer",
                arguments={},
                intent=planned.intent,
                plan=planned.plan,
                reason_summary=planned.reason_summary,
                error=f"JSON Planner 参数校验失败：{exc}",
            )

        return PlannedAction(
            action=action,
            arguments=arguments,
            intent=planned.intent,
            plan=planned.plan,
            reason_summary=planned.reason_summary,
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("planner_node", self.planner_node)
        graph.add_node("tool_node", self.tool_node)
        graph.add_node("observation_node", self.observation_node)
        graph.add_node("evidence_check_node", self.evidence_check_node)
        graph.add_node("rewrite_query_node", self.rewrite_query_node)
        graph.add_node("human_review_node", self.human_review_node)
        graph.add_node("retrieval_failure_node", self.retrieval_failure_node)
        graph.add_node("local_deep_search_node", self.local_deep_search_node)
        graph.add_node("revise_question_node", self.revise_question_node)
        graph.add_node("cancel_node", self.cancel_node)
        graph.add_node("answer_node", self.answer_node)

        graph.set_entry_point("planner_node")
        graph.add_conditional_edges(
            "planner_node",
            self.route_after_planner,
            {"tool": "tool_node", "answer": "answer_node"},
        )
        graph.add_edge("tool_node", "observation_node")
        graph.add_edge("observation_node", "evidence_check_node")
        graph.add_conditional_edges(
            "evidence_check_node",
            self.route_after_evidence_check,
            {
                "rewrite": "rewrite_query_node",
                "answer": "answer_node",
                "human_review": "human_review_node",
                "retrieval_failure": "retrieval_failure_node",
            },
        )
        graph.add_edge("rewrite_query_node", "tool_node")
        graph.add_edge("local_deep_search_node", "evidence_check_node")
        graph.add_edge("cancel_node", END)
        graph.add_edge("answer_node", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _force_retrieval_for_explicit_papers(self, state: AgentState, planned: PlannedAction) -> PlannedAction:
        selected = list(state.get("selected_papers") or detect_explicit_papers(state.get("question", "")))
        if state.get("intent") == "paper_analysis":
            return self._plan_paper_analysis_action(state)
        if state.get("intent") == "paper_comparison" and len(selected) >= 2:
            return self._plan_paper_comparison_action(state)
        if len(selected) == 1:
            return PlannedAction(
                action="search_paper",
                intent=str(state.get("intent") or planned.intent or "general_qa"),
                arguments={"query": state.get("question", ""), "source": selected[0], "k": agent_tools.DEFAULT_K},
                reason_summary="Explicit paper mention overrides planner clarification.",
            )
        if len(selected) > 1:
            return PlannedAction(
                action="search_multiple_queries",
                intent=str(state.get("intent") or planned.intent or "general_qa"),
                arguments={"queries": [state.get("question", "")], "sources": selected, "k_per_query": 4},
                reason_summary="Explicit paper mentions override planner clarification.",
            )
        return planned

    def planner_node(self, state: AgentState) -> AgentState:
        if contains_unresolved_pronoun(state["question"]) and not state.get("selected_papers"):
            return {
                **state,
                "intent": "clarification",
                "next_action": "answer",
                "final_answer": "请明确你说的“这篇论文”具体是哪一篇。可以先输入 papers 查看当前数据库中的论文列表。",
                "trace": state.get("trace", []) + ["【任务判断】问题中有指代词，但当前对话没有明确论文，需要澄清。"],
            }

        self._emit_progress(state, "正在生成行动计划……", category="当前状态", event_type="planning_started")
        planned = self._plan_next_action(state)

        planned = self.validate_planned_action(planned, state)

        # Do not let a later LLM planner downgrade a high-confidence,
        # rule-resolved intent such as paper_comparison to general_qa.
        locked_intent = state.get("intent")
        if locked_intent in {"paper_comparison", "paper_analysis", "clarification"}:
            planned.intent = str(locked_intent)

        if not planned.error and planned.action == "clarify" and detect_explicit_papers(state["question"]):
            planned = self._force_retrieval_for_explicit_papers(state, planned)
            planned = self.validate_planned_action(planned, state)

        self._emit_progress(state, planned.intent, category="任务判断", event_type="planning_finished")
        if planned.intent == "paper_comparison":
            self._emit_progress(
                state,
                "、".join(state.get("comparison_papers") or state.get("selected_papers") or []),
                category="目标论文",
            )
            self._emit_progress(
                state,
                "、".join(state.get("comparison_aspects") or DEFAULT_COMPARISON_ASPECTS),
                category="对比维度",
            )
        if state.get("required_aspects"):
            self._emit_progress(state, f"共 {len(state.get('required_aspects', []))} 项", category="信息方面")
        trace = state.get("trace", []) + [
            f"【任务判断】{planned.intent}",
            f"【当前模型】{self.chat_model_name}",
            f"【Planner模式】{self.tool_mode.upper() if self.tool_mode != 'json' else 'JSON'}",
            f"【Planner动作】{planned.action}",
            f"【行动说明】{planned.reason_summary or planned.plan or '根据问题选择下一步工具或回答。'}",
        ]
        if planned.error:
            trace.append(f"【计划错误】{planned.error}")
            return {
                **state,
                "intent": "general_qa",
                "next_action": "answer",
                "final_answer": planned.error,
                "error": planned.error,
                "trace": trace,
            }

        if planned.action in {"answer", "clarify"}:
            return {
                **state,
                "intent": "clarification" if planned.action == "clarify" else planned.intent,
                "next_action": "answer",
                "final_answer": (
                    planned.arguments.get("question")
                    or planned.arguments.get("message")
                    or "请补充更明确的问题信息。"
                )
                if planned.action == "clarify"
                else "",
                "trace": trace,
            }

        current_query = str(planned.arguments.get("query") or state["question"])
        return {
            **state,
            "intent": planned.intent,
            "plan": planned.plan,
            "next_action": planned.action,
            "tool_action": {"action": planned.action, "arguments": planned.arguments},
            "current_query": current_query,
            "trace": trace + [format_tool_arguments(planned.arguments), f"【调用工具】{planned.action}_tool"],
        }

    def tool_node(self, state: AgentState) -> AgentState:
        action = state.get("tool_action", {}).get("action")
        arguments = dict(state.get("tool_action", {}).get("arguments") or {})
        trace = list(state.get("trace", []))
        self._emit_progress(state, "正在检索论文……", category="当前状态", event_type="retrieval_started")

        try:
            if action == "list_papers":
                result = self.tools.list_papers_tool()
            elif action == "search_paper":
                result = self.tools.search_paper_tool(
                    query=str(arguments.get("query") or state.get("current_query") or state["question"]),
                    source=arguments.get("source") or self._single_selected_source(state),
                    k=int(arguments.get("k") or agent_tools.DEFAULT_K),
                    retry_callback=lambda attempt, total, delay: self._emit_progress(
                        state,
                        f"第 {attempt}/{total} 次，等待 {delay} 秒",
                        category="检索重试",
                        event_type="retrieval_retry",
                    ),
                )
            elif action == "search_multiple_queries":
                source_queries = list(arguments.get("source_queries") or [])
                if source_queries:
                    result = {
                        "queries": [],
                        "source_queries": source_queries,
                        "source_filters": [],
                        "k_per_query": int(arguments.get("k_per_query") or 4),
                        "score_label": "distance_score",
                        "retrieval_status": "success",
                        "retrieval_error_reason": "",
                        "successful_query_count": 0,
                        "failed_query_count": 0,
                        "consecutive_query_failures": 0,
                        "circuit_breaker_triggered": False,
                        "query_results": [],
                        "results": [],
                    }
                    for item in source_queries:
                        item_queries = list(item.get("queries") or [])
                        current_source = str(item.get("source") or "")
                        if not item_queries or not current_source:
                            continue
                        partial = self.tools.search_multiple_queries_tool(
                            queries=item_queries,
                            source=current_source,
                            k_per_query=int(arguments.get("k_per_query") or 4),
                            seen_result_ids=list(state.get("seen_result_ids", [])),
                            seen_source_pages=list(state.get("seen_source_pages", [])),
                            progress_callback=lambda index, total, query: self._emit_progress(
                                state,
                                f"{index}/{total} {query}",
                                category="检索进度",
                                event_type="retrieval_query_started",
                            ),
                            retry_callback=lambda attempt, total, delay: self._emit_progress(
                                state,
                                f"第 {attempt}/{total} 次，等待 {delay} 秒",
                                category="检索重试",
                                event_type="retrieval_retry",
                            ),
                        )
                        result["queries"].extend(item_queries)
                        if current_source not in result["source_filters"]:
                            result["source_filters"].append(current_source)
                        result["results"].extend(partial.get("results") or [])
                        result["score_label"] = partial.get("score_label") or result["score_label"]
                        result["successful_query_count"] += int(partial.get("successful_query_count") or 0)
                        result["failed_query_count"] += int(partial.get("failed_query_count") or 0)
                        result["consecutive_query_failures"] = max(
                            int(result.get("consecutive_query_failures") or 0),
                            int(partial.get("consecutive_query_failures") or 0),
                        )
                        if partial.get("retrieval_error_reason"):
                            result["retrieval_error_reason"] = partial.get("retrieval_error_reason")
                        if partial.get("circuit_breaker_triggered"):
                            result["circuit_breaker_triggered"] = True
                        for query_result in partial.get("query_results") or []:
                            enriched = dict(query_result)
                            enriched["source"] = current_source
                            enriched["covered_aspect"] = item.get("aspect") or enriched.get("covered_aspect")
                            enriched["coverage_status"] = item.get("status")
                            enriched["missing_detail"] = item.get("missing_detail")
                            result["query_results"].append(enriched)
                    if result["successful_query_count"] and result["failed_query_count"]:
                        result["retrieval_status"] = "partial_failure"
                    elif result["failed_query_count"] and not result["successful_query_count"]:
                        result["retrieval_status"] = "failed"
                else:
                    queries = list(arguments.get("queries") or [])

                    result = self.tools.search_multiple_queries_tool(
                        queries=queries,
                        source=arguments.get("source"),
                        sources=arguments.get("sources"),
                        k_per_query=int(
                            arguments.get("k_per_query") or 4
                        ),
                        seen_result_ids=list(
                            state.get("seen_result_ids", [])
                        ),
                        seen_source_pages=list(
                            state.get("seen_source_pages", [])
                        ),
                        progress_callback=lambda index, total, query: (
                            self._emit_progress(
                                state,
                                f"{index}/{total} {query}",
                                category="检索进度",
                                event_type="retrieval_query_started",
                            )
                        ),
                        retry_callback=lambda attempt, total, delay: self._emit_progress(
                            state,
                            f"第 {attempt}/{total} 次，等待 {delay} 秒",
                            category="检索重试",
                            event_type="retrieval_retry",
                        ),
                    )
            elif action == "get_neighbor_chunks":
                result = self.tools.get_neighbor_chunks_tool(
                    source=str(arguments["source"]),
                    page=int(arguments["page"]),
                    radius=int(arguments.get("radius", 1)),
                )
            elif action == "inspect_paper_scope":
                result = self.tools.inspect_paper_scope_tool(source=str(arguments["source"]))
            else:
                raise ValueError(f"不允许执行未知 action：{action}")
        except Exception as exc:
            message = f"工具执行失败：{type(exc).__name__}: {exc}"
            return {**state, "error": message, "tool_result": {}, "trace": trace + [f"【工具错误】{message}"]}

        docs = list(state.get("retrieved_docs", []))
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            docs = merge_unique_chunks(docs, result["results"])

        if isinstance(result, dict):
            retrieval_status = str(result.get("retrieval_status") or "success")
            retrieval_error_reason = str(result.get("retrieval_error_reason") or "")
            successful_query_count = int(result.get("successful_query_count") or (1 if result.get("results") else 0))
            failed_query_count = int(result.get("failed_query_count") or 0)
            consecutive_query_failures = int(result.get("consecutive_query_failures") or 0)
        else:
            retrieval_status = "success"
            retrieval_error_reason = ""
            successful_query_count = 0
            failed_query_count = 0
            consecutive_query_failures = 0

        seen_result_ids = list(dict.fromkeys([*state.get("seen_result_ids", []), *[doc["result_id"] for doc in docs]]))
        seen_source_pages = list(
            dict.fromkeys(
                [
                    *state.get("seen_source_pages", []),
                    *[f"{doc['source']}|{doc.get('page')}" for doc in docs if doc.get("source")],
                ]
            )
        )

        iteration = int(state.get("iteration", 0))
        auto_search_rounds = int(state.get("auto_search_rounds", 0))
        if action in {"search_paper", "search_multiple_queries", "get_neighbor_chunks", "inspect_paper_scope"}:
            iteration += 1
        if action in {"search_paper", "search_multiple_queries"}:
            auto_search_rounds += 1

        history = list(state.get("retrieval_history", []))
        history.append(
            {
                "action": action,
                "query": arguments.get("query") or arguments.get("queries") or state.get("current_query"),
                "summary": summarize_tool_result(result),
                "retrieval_status": retrieval_status,
                "successful_query_count": successful_query_count,
                "failed_query_count": failed_query_count,
                "result_ids": [item["result_id"] for item in result.get("results", [])] if isinstance(result, dict) else [],
            }
        )

        return {
            **state,
            "tool_result": result,
            "retrieval_status": retrieval_status,
            "retrieval_error_reason": retrieval_error_reason,
            "successful_query_count": successful_query_count,
            "failed_query_count": failed_query_count,
            "consecutive_query_failures": consecutive_query_failures,
            "retrieved_docs": docs,
            "retrieval_history": history,
            "seen_result_ids": seen_result_ids,
            "seen_source_pages": seen_source_pages,
            "iteration": iteration,
            "auto_search_rounds": auto_search_rounds,
        }

    def observation_node(self, state: AgentState) -> AgentState:
        observation = summarize_tool_result(state.get("tool_result", {}))
        return {
            **state,
            "observation": observation,
            "trace": state.get("trace", []) + [f"【观察结果】{observation}"],
        }

    def evidence_check_node(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return {
                **state,
                "evidence_sufficient": True,
                "evidence_reason": state["error"],
                "trace": state.get("trace", []) + [f"【证据判断】{state['error']}"],
            }

        if state.get("retrieval_status") == "failed":
            reason = state.get("retrieval_error_reason") or "本轮检索查询全部失败。"
            trace = state.get("trace", []) + [
                f"【检索状态】failed，成功 {int(state.get('successful_query_count', 0))}，失败 {int(state.get('failed_query_count', 0))}",
                f"【检索失败原因】{reason}",
            ]
            if (state.get("tool_result") or {}).get("circuit_breaker_triggered"):
                trace.append(
                    f"【检索熔断】连续 {agent_tools.MAX_CONSECUTIVE_QUERY_FAILURES} 个查询失败，已停止本批剩余请求"
                )
            return {
                **state,
                "evidence_status": "retrieval_failed",
                "evidence_sufficient": False,
                "evidence_reason": reason,
                "missing_aspects": [],
                "covered_aspects": list(state.get("covered_aspects", [])),
                "trace": trace,
            }

        if state.get("intent") == "paper_comparison":
            requested_aspects = list(state.get("requested_aspects") or state.get("required_aspects") or DEFAULT_COMPARISON_ASPECTS)
            sources = list(state.get("comparison_papers") or state.get("selected_papers") or [])
            coverage = state.get("coverage_by_paper") or initial_coverage_by_paper(sources, requested_aspects)
            tool_result = state.get("tool_result", {})
            if isinstance(tool_result, dict):
                coverage = merge_comparison_coverage(
                    coverage,
                    list(tool_result.get("query_results") or []),
                    requested_aspects,
                    retrieved_docs=list(state.get("retrieved_docs") or []),
                    question=state.get("question", ""),
                )
            retrieval_trace: list[str] = []
            if state.get("retrieval_status") == "partial_failure":
                retrieval_trace.append(
                    f"【检索状态】检索部分失败，成功 {int(state.get('successful_query_count', 0))}，失败 {int(state.get('failed_query_count', 0))}"
                )

            evidence_status, evidence_sufficient, covered_aspects, missing_aspects, reason = evaluate_comparison_coverage(
                coverage,
                requested_aspects,
                sources,
            )
            if evidence_status == "sufficient_with_gaps" and state.get("allow_answer_with_gaps"):
                evidence_sufficient = True
            coverage_lines = ["【证据覆盖】"]
            for source in sources:
                coverage_lines.append(f"{short_source_label(source)}：")
                source_coverage = coverage.get(source, {})
                for aspect in requested_aspects:
                    coverage_lines.append(f"- {aspect}：{source_coverage.get(aspect, 'missing')}")
            coverage_lines.append(f"【证据状态】{evidence_status}")
            if reason and evidence_status != "sufficient":
                coverage_lines.append(f"【缺失细节】{reason}")
            coverage_lines.append(
                "【处理方式】证据足以回答，不再触发额外检索"
                if evidence_sufficient
                else "【处理方式】仅针对缺失的用户必答项继续检索"
            )

            self._emit_progress(state, evidence_status, category="证据状态", event_type="evidence_check_finished")
            if missing_aspects:
                self._emit_progress(state, "、".join(missing_aspects), category="缺失信息")
            self._emit_progress(
                state,
                f"{int(state.get('iteration', 0))}/{int(state.get('max_iterations', DEFAULT_MAX_ITERATIONS))}",
                category="当前轮次",
            )

            return {
                **state,
                "coverage_by_paper": coverage,
                "evidence_status": evidence_status,
                "evidence_sufficient": evidence_sufficient,
                "evidence_reason": reason,
                "covered_aspects": covered_aspects,
                "missing_aspects": missing_aspects,
                "trace": state.get("trace", []) + retrieval_trace + coverage_lines,
            }

        prompt = self._evidence_prompt(state)
        self._emit_progress(state, "正在检查证据覆盖……", category="当前状态", event_type="evidence_check_started")
        try:
            with self._progress_indicator(state, "正在检查证据覆盖"):
                response = self.llm.invoke([SystemMessage(content=prompt)])
            payload = parse_json_evidence(str(response.content))
        except Exception as exc:
            payload = {
                "sufficient": bool(state.get("retrieved_docs")),
                "reason": f"证据判断调用失败：{type(exc).__name__}: {exc}",
                "missing_aspects": [],
                "suggested_query": state.get("current_query", ""),
            }

        covered_aspects = list(payload.get("covered_aspects") or state.get("covered_aspects", []))
        missing_aspects = normalize_missing_aspects(
            required=state.get("required_aspects", []),
            covered=covered_aspects,
            missing=list(payload.get("missing_aspects") or []),
        )
        evidence_sufficient = bool(payload["sufficient"])
        negative_evidence_sufficient = supports_negative_evidence_answer(
            state.get("question", ""),
            "\n".join(
                [
                    str(payload.get("reason") or ""),
                    str(state.get("evidence_reason") or ""),
                ]
            ),
            list(state.get("retrieved_docs") or []),
        )
        if negative_evidence_sufficient:
            evidence_sufficient = True
            missing_aspects = []
            covered_aspects = list(dict.fromkeys([*covered_aspects, *list(state.get("requested_aspects") or [])]))
        if state.get("intent") == "paper_analysis" and missing_aspects:
            evidence_sufficient = False
        evidence_status = "sufficient" if evidence_sufficient else "insufficient"

        if state.get("required_aspects"):
            self._emit_progress(
                state,
                f"{len(covered_aspects)}/{len(state.get('required_aspects', []))}",
                category="证据覆盖",
                event_type="evidence_check_finished",
            )
            self._emit_progress(state, "、".join(missing_aspects) or "无", category="缺失信息")
        self._emit_progress(state, f"{evidence_sufficient}", category="证据判断")
        self._emit_progress(
            state,
            f"{int(state.get('iteration', 0))}/{int(state.get('max_iterations', DEFAULT_MAX_ITERATIONS))}",
            category="当前轮次",
        )

        return {
            **state,
            "evidence_sufficient": evidence_sufficient,
            "evidence_status": evidence_status,
            "evidence_reason": (
                "已检索到支持“论文未明确说明”的证据；按用户要求不使用常识补全。"
                if negative_evidence_sufficient
                else str(payload["reason"])
            ),
            "covered_aspects": covered_aspects,
            "missing_aspects": missing_aspects,
            "awaiting_human": False if negative_evidence_sufficient else state.get("awaiting_human", False),
            "human_review_required": False if negative_evidence_sufficient else state.get("human_review_required", False),
            "current_query": str(payload.get("suggested_query") or state.get("current_query") or state["question"]),
            "trace": state.get("trace", []) + [
                f"【证据判断】{'已检索到支持“论文未明确说明”的证据；按用户要求不使用常识补全。' if negative_evidence_sufficient else payload['reason']}"
            ],
        }

    def rewrite_query_node(self, state: AgentState) -> AgentState:
        self._emit_progress(state, "正在为缺失信息生成补充查询……", category="当前状态", event_type="rewrite_started")
        if state.get("intent") == "paper_analysis":
            source = self._single_selected_source(state)
            required_aspects = state.get("required_aspects") or get_paper_analysis_aspects(source)
            missing_aspects = paper_analysis_missing_aspects(
                source,
                list(required_aspects),
                list(state.get("missing_aspects") or []),
                list(state.get("covered_aspects") or []),
            )
            queries = build_paper_analysis_queries(
                source=source or "",
                required_aspects=required_aspects,
                missing_aspects=missing_aspects or required_aspects,
                max_queries=3,
            )
            tool_action = {
                "action": "search_multiple_queries",
                "intent": "paper_analysis",
                "arguments": {
                    "queries": queries,
                    "source": source,
                    "k_per_query": 4,
                },
            }
            return {
                **state,
                "current_query": "; ".join(queries),
                "next_action": "search_multiple_queries",
                "tool_action": tool_action,
                "trace": state.get("trace", [])
                + [f"【改写查询】仅针对缺失方面：{', '.join(missing_aspects)}", format_tool_arguments(tool_action["arguments"]), "【调用工具】search_multiple_queries_tool"],
            }

        if state.get("intent") == "paper_comparison":
            requested_aspects = list(state.get("requested_aspects") or state.get("required_aspects") or DEFAULT_COMPARISON_ASPECTS)
            sources = list(state.get("comparison_papers") or state.get("selected_papers") or [])
            source_queries = build_source_query_specs(
                state.get("coverage_by_paper") or initial_coverage_by_paper(sources, requested_aspects),
                requested_aspects,
                sources,
                deep=False,
            )
            if not source_queries:
                missing_aspects = [aspect for aspect in state.get("missing_aspects", []) if aspect in requested_aspects]
                queries = build_comparison_queries(
                    requested_aspects=requested_aspects,
                    missing_aspects=missing_aspects,
                )
                source_queries = [
                    {
                        "source": source,
                        "aspect": aspect,
                        "status": "missing",
                        "missing_detail": format_coverage_gap(source, aspect, "missing"),
                        "queries": queries,
                    }
                    for source in sources
                    for aspect in (missing_aspects or requested_aspects)
                ][:4]
            tool_action = {
                "action": "search_multiple_queries",
                "intent": "paper_comparison",
                "arguments": {
                    "source_queries": source_queries,
                    "k_per_query": 4,
                },
            }
            flat_queries = [query for item in source_queries for query in item.get("queries", [])]
            return {
                **state,
                "current_query": "; ".join(flat_queries),
                "next_action": "search_multiple_queries",
                "tool_action": tool_action,
                "trace": state.get("trace", [])
                + [
                    "【改写查询】仅针对 partial/missing 的论文×方面补检索",
                    format_tool_arguments(tool_action["arguments"]),
                    "【调用工具】search_multiple_queries_tool",
                ],
            }

        prompt = (
            "请基于缺失证据改写一个更适合向量检索的英文或中英混合查询。"
            "只输出 JSON：{\"query\":\"...\"}。\n"
            f"用户问题：{state['question']}\n"
            f"当前查询：{state.get('current_query')}\n"
            f"缺失方面：{state.get('missing_aspects', [])}"
        )
        try:
            with self._progress_indicator(state, "正在为缺失信息生成补充查询"):
                response = self.llm.invoke([SystemMessage(content=prompt)])
            payload = extract_json_object(str(response.content))
            query = str(payload.get("query") or state.get("current_query") or state["question"])
        except Exception:
            missing = " ".join(str(item) for item in state.get("missing_aspects", []))
            query = f"{state['question']} {missing}".strip()

        source = self._single_selected_source(state)
        tool_action = {
            "action": "search_paper",
            "arguments": {"query": query, "source": source, "k": agent_tools.DEFAULT_K},
        }
        return {
            **state,
            "current_query": query,
            "next_action": "search_paper",
            "tool_action": tool_action,
            "trace": state.get("trace", []) + [f"【改写查询】{query}", "【调用工具】search_paper_tool"],
        }

    def _requested_missing_aspects(self, state: AgentState) -> list[str]:
        requested = list(state.get("requested_aspects") or state.get("required_aspects") or [])
        if not requested:
            return list(state.get("missing_aspects", []))
        missing: list[str] = []
        for aspect in state.get("missing_aspects", []):
            aspect_text = str(aspect)
            if aspect in requested or any(str(item) in aspect_text for item in requested):
                missing.append(aspect)
        return missing

    def _target_sources_for_missing_aspects(self, state: AgentState, missing_aspects: list[str]) -> list[str]:
        sources = list(dict.fromkeys(state.get("comparison_papers") or state.get("selected_papers") or []))
        if not sources:
            return []
        missing_text = " ".join(str(item).lower() for item in missing_aspects)
        matched: list[str] = []
        for source in sources:
            label = short_source_label(source).lower()
            aliases = [label, Path(source).stem.lower()]
            aliases.extend(alias.lower() for alias in PAPER_ALIASES.get(source, ()))
            if any(alias and alias in missing_text for alias in aliases):
                matched.append(source)
        return matched or sources

    def build_human_review_payload(self, state: AgentState) -> dict[str, Any]:
        options = [
            {
                "action": "answer_with_gaps",
                "label": "基于现有证据回答，并明确标注证据缺口",
            }
        ]
        if int(state.get("deep_search_rounds", 0)) < int(
            state.get("max_deep_search_rounds", self.max_deep_search_rounds)
        ):
            options.append(
                {
                    "action": "local_deep_search",
                    "label": "继续在当前论文和本地论文库中深度检索",
                }
            )
        options.extend(
            [
                {"action": "revise_question", "label": "修改当前问题"},
                {"action": "cancel", "label": "取消当前任务"},
            ]
        )
        return {
            "type": "evidence_review",
            "evidence_status": state.get("evidence_status", "insufficient"),
            "question": state.get("question", ""),
            "selected_papers": list(state.get("selected_papers") or []),
            "requested_aspects": list(state.get("requested_aspects") or []),
            "covered_aspects": list(state.get("covered_aspects") or []),
            "missing_aspects": list(state.get("missing_aspects") or []),
            "coverage_by_paper": dict(state.get("coverage_by_paper") or {}),
            "reason": state.get("evidence_reason", ""),
            "deep_search_rounds": int(state.get("deep_search_rounds", 0)),
            "max_deep_search_rounds": int(state.get("max_deep_search_rounds", self.max_deep_search_rounds)),
            "options": options,
        }

    def human_review_node(self, state: AgentState) -> Command:
        payload = self.build_human_review_payload(state)
        decision = interrupt(payload)
        if not isinstance(decision, dict):
            decision = {"action": "answer_with_gaps"}

        action = str(decision.get("action") or "").strip()
        allowed = {item["action"] for item in payload["options"]}
        if action not in allowed:
            action = "answer_with_gaps"

        trace = state.get("trace", []) + [f"【人工选择】{action}"]
        base_update: dict[str, Any] = {
            "human_decision": action,
            "awaiting_human": False,
            "human_review_required": False,
            "trace": trace,
        }
        if action == "answer_with_gaps":
            return Command(
                update={**base_update, "allow_answer_with_gaps": True},
                goto="answer_node",
            )
        if action == "local_deep_search":
            return Command(update=base_update, goto="local_deep_search_node")
        if action == "revise_question":
            return Command(
                update={
                    **base_update,
                    "revised_question": str(decision.get("revised_question") or "").strip(),
                },
                goto="revise_question_node",
            )
        return Command(
            update={
                **base_update,
                "cancelled": True,
                "final_answer": "当前任务已取消。",
            },
            goto="cancel_node",
        )

    def build_retrieval_failure_payload(self, state: AgentState) -> dict[str, Any]:
        options = [
            {"action": "retry_current_retrieval", "label": "等待后重试当前检索"},
        ]
        if state.get("retrieved_docs"):
            options.append({"action": "continue_with_existing_evidence", "label": "基于当前已有证据继续"})
        options.append({"action": "cancel", "label": "取消当前任务"})
        return {
            "type": "retrieval_failure",
            "question": state.get("question", ""),
            "retrieval_status": state.get("retrieval_status", "failed"),
            "retrieval_error_reason": state.get("retrieval_error_reason", ""),
            "successful_query_count": int(state.get("successful_query_count", 0)),
            "failed_query_count": int(state.get("failed_query_count", 0)),
            "consecutive_query_failures": int(state.get("consecutive_query_failures", 0)),
            "has_existing_evidence": bool(state.get("retrieved_docs")),
            "options": options,
        }

    def retrieval_failure_node(self, state: AgentState) -> Command:
        payload = self.build_retrieval_failure_payload(state)
        decision = interrupt(payload)
        if not isinstance(decision, dict):
            decision = {"action": "cancel"}

        action = str(decision.get("action") or "").strip()
        allowed = {item["action"] for item in payload["options"]}
        if action not in allowed:
            action = "cancel"

        trace = state.get("trace", []) + [f"【人工选择】{action}"]
        if action == "retry_current_retrieval":
            return Command(
                update={
                    "retrieval_status": "success",
                    "retrieval_error_reason": "",
                    "successful_query_count": 0,
                    "failed_query_count": 0,
                    "consecutive_query_failures": 0,
                    "trace": trace + ["【图状态】从检索失败处重试当前检索"],
                },
                goto="tool_node",
            )
        if action == "continue_with_existing_evidence":
            return Command(
                update={
                    "retrieval_status": "partial_failure",
                    "retrieval_error_reason": "用户选择基于当前已有证据继续。",
                    "trace": trace,
                },
                goto="evidence_check_node",
            )
        return Command(
            update={
                "human_decision": "cancel",
                "cancelled": True,
                "final_answer": "当前任务已取消。",
                "awaiting_human": False,
                "trace": trace,
            },
            goto="cancel_node",
        )

    def local_deep_search_node(self, state: AgentState) -> AgentState:
        source_query_specs: list[dict[str, Any]] = []
        if state.get("intent") == "paper_comparison":
            requested_aspects = list(state.get("requested_aspects") or state.get("required_aspects") or DEFAULT_COMPARISON_ASPECTS)
            target_sources_for_comparison = list(state.get("comparison_papers") or state.get("selected_papers") or [])
            source_query_specs = build_source_query_specs(
                state.get("coverage_by_paper") or initial_coverage_by_paper(target_sources_for_comparison, requested_aspects),
                requested_aspects,
                target_sources_for_comparison,
                deep=True,
            )
        if source_query_specs:
            queries = [query for item in source_query_specs for query in item.get("queries", [])]
            target_sources = list(dict.fromkeys(str(item.get("source")) for item in source_query_specs if item.get("source")))
        else:
            missing_requested = self._requested_missing_aspects(state)
            queries = build_deep_search_queries(missing_requested)
            target_sources = self._target_sources_for_missing_aspects(state, missing_requested)
        self._emit_progress(
            state,
            "当前目标论文与本地 Chroma",
            category="深度检索范围",
            event_type="local_deep_search_started",
        )
        self._emit_progress(
            state,
            f"{int(state.get('deep_search_rounds', 0)) + 1}/{int(state.get('max_deep_search_rounds', self.max_deep_search_rounds))}",
            category="深度检索轮次",
        )
        self._emit_progress(state, "；".join(queries), category="深度检索查询")

        if source_query_specs:
            result = {
                "queries": queries,
                "source_queries": source_query_specs,
                "source_filters": target_sources,
                "k_per_query": min(8, agent_tools.MAX_K),
                "score_label": "distance_score",
                "retrieval_status": "success",
                "retrieval_error_reason": "",
                "successful_query_count": 0,
                "failed_query_count": 0,
                "consecutive_query_failures": 0,
                "circuit_breaker_triggered": False,
                "query_results": [],
                "results": [],
            }
            for item in source_query_specs:
                partial = self.tools.search_multiple_queries_tool(
                    queries=list(item.get("queries") or []),
                    source=str(item.get("source")),
                    k_per_query=min(8, agent_tools.MAX_K),
                    seen_result_ids=list(state.get("seen_result_ids", [])),
                    seen_source_pages=list(state.get("seen_source_pages", [])),
                    progress_callback=lambda index, total, query: self._emit_progress(
                        state,
                        f"{index}/{total} {query}",
                        category="检索进度",
                        event_type="retrieval_query_started",
                    ),
                    retry_callback=lambda attempt, total, delay: self._emit_progress(
                        state,
                        f"第 {attempt}/{total} 次，等待 {delay} 秒",
                        category="检索重试",
                        event_type="retrieval_retry",
                    ),
                )
                result["results"].extend(partial.get("results") or [])
                result["score_label"] = partial.get("score_label") or result["score_label"]
                result["successful_query_count"] += int(partial.get("successful_query_count") or 0)
                result["failed_query_count"] += int(partial.get("failed_query_count") or 0)
                result["consecutive_query_failures"] = max(
                    int(result.get("consecutive_query_failures") or 0),
                    int(partial.get("consecutive_query_failures") or 0),
                )
                if partial.get("retrieval_error_reason"):
                    result["retrieval_error_reason"] = partial.get("retrieval_error_reason")
                if partial.get("circuit_breaker_triggered"):
                    result["circuit_breaker_triggered"] = True
                for query_result in partial.get("query_results") or []:
                    enriched = dict(query_result)
                    enriched["source"] = item.get("source")
                    enriched["covered_aspect"] = item.get("aspect") or enriched.get("covered_aspect")
                    enriched["coverage_status"] = item.get("status")
                    enriched["missing_detail"] = item.get("missing_detail")
                    result["query_results"].append(enriched)
            if result["successful_query_count"] and result["failed_query_count"]:
                result["retrieval_status"] = "partial_failure"
            elif result["failed_query_count"] and not result["successful_query_count"]:
                result["retrieval_status"] = "failed"
        else:
            result = self.tools.search_multiple_queries_tool(
                queries=queries,
                sources=target_sources,
                k_per_query=min(8, agent_tools.MAX_K),
                seen_result_ids=list(state.get("seen_result_ids", [])),
                seen_source_pages=list(state.get("seen_source_pages", [])),
                progress_callback=lambda index, total, query: self._emit_progress(
                    state,
                    f"{index}/{total} {query}",
                    category="检索进度",
                    event_type="retrieval_query_started",
                ),
                retry_callback=lambda attempt, total, delay: self._emit_progress(
                    state,
                    f"第 {attempt}/{total} 次，等待 {delay} 秒",
                    category="检索重试",
                    event_type="retrieval_retry",
                ),
            )

        docs = merge_unique_chunks(list(state.get("retrieved_docs", [])), list(result.get("results") or []))
        top_by_source: dict[str, list[PaperChunk]] = {}
        for chunk in result.get("results") or []:
            top_by_source.setdefault(chunk["source"], []).append(chunk)

        neighbor_results: list[PaperChunk] = []
        for source, chunks in top_by_source.items():
            for chunk in chunks[:3]:
                page = chunk.get("page")
                if page is None:
                    continue
                neighbor_result = self.tools.get_neighbor_chunks_tool(
                    source=source,
                    page=int(page),
                    radius=2,
                )
                neighbor_results.extend(list(neighbor_result.get("results") or []))
        if neighbor_results:
            docs = merge_unique_chunks(docs, neighbor_results)

        retrieval_status = str(result.get("retrieval_status") or "success")
        retrieval_error_reason = str(result.get("retrieval_error_reason") or "")
        successful_query_count = int(result.get("successful_query_count") or 0)
        failed_query_count = int(result.get("failed_query_count") or 0)
        consecutive_query_failures = int(result.get("consecutive_query_failures") or 0)

        seen_result_ids = list(
            dict.fromkeys([*state.get("seen_result_ids", []), *[doc["result_id"] for doc in docs]])
        )
        seen_source_pages = list(
            dict.fromkeys(
                [
                    *state.get("seen_source_pages", []),
                    *[f"{doc['source']}|{doc.get('page')}" for doc in docs if doc.get("source")],
                ]
            )
        )
        history = list(state.get("retrieval_history", []))
        history.append(
            {
                "action": "local_deep_search",
                "query": queries,
                "sources": target_sources,
                "summary": summarize_tool_result(result),
                "result_ids": [item["result_id"] for item in result.get("results", [])],
            }
        )
        return {
            **state,
            "tool_result": result,
            "retrieval_status": retrieval_status,
            "retrieval_error_reason": retrieval_error_reason,
            "successful_query_count": successful_query_count,
            "failed_query_count": failed_query_count,
            "consecutive_query_failures": consecutive_query_failures,
            "retrieved_docs": docs,
            "retrieval_history": history,
            "seen_result_ids": seen_result_ids,
            "seen_source_pages": seen_source_pages,
            "deep_search_rounds": int(state.get("deep_search_rounds", 0)) + 1,
            "deep_search_queries": queries,
            "deep_search_target_papers": target_sources,
            "trace": state.get("trace", [])
            + [
                "【深度检索范围】当前目标论文与本地 Chroma",
                f"【深度检索轮次】{int(state.get('deep_search_rounds', 0)) + 1}/{int(state.get('max_deep_search_rounds', self.max_deep_search_rounds))}",
                f"【深度检索查询】{queries}",
            ],
        }

    def revise_question_node(self, state: AgentState) -> Command:
        revised_question = str(state.get("revised_question") or "").strip()
        if not revised_question:
            return Command(
                update={
                    "human_decision": "answer_with_gaps",
                    "allow_answer_with_gaps": True,
                    "trace": state.get("trace", []) + ["【修改问题】输入为空，改为基于现有证据回答。"],
                },
                goto="answer_node",
            )

        old_sources = list(state.get("comparison_papers") or state.get("selected_papers") or [])
        inferred_intent, inferred_papers = infer_intent(
            revised_question,
            recent_sources=old_sources or self.last_sources,
        )
        explicit_papers = detect_explicit_papers(revised_question)
        selected_from_context = [] if explicit_papers else self._resolve_sources_from_context(revised_question)
        selected = list(dict.fromkeys(explicit_papers or [*inferred_papers, *selected_from_context]))
        if not selected and contains_unresolved_pronoun(revised_question):
            selected = old_sources

        if inferred_intent == "paper_analysis":
            requested_aspects = get_combined_paper_analysis_aspects(selected or old_sources)
        elif inferred_intent == "paper_comparison":
            requested_aspects = extract_comparison_aspects(revised_question)
        else:
            requested_aspects = []
        related_aspects = derive_related_aspects(inferred_intent, requested_aspects, revised_question)
        comparison_papers = list(selected) if inferred_intent == "paper_comparison" else []
        target_unchanged = set(selected or old_sources) == set(old_sources)

        update: dict[str, Any] = {
            "question": revised_question,
            "current_query": revised_question,
            "intent": inferred_intent,
            "selected_papers": selected or old_sources,
            "comparison_papers": comparison_papers if selected else (old_sources if inferred_intent == "paper_comparison" else []),
            "comparison_aspects": requested_aspects if inferred_intent == "paper_comparison" else [],
            "requested_aspects": requested_aspects,
            "related_aspects": related_aspects,
            "required_aspects": list(requested_aspects),
            "covered_aspects": [],
            "missing_aspects": list(requested_aspects),
            "allow_answer_with_gaps": False,
            "deep_search_rounds": 0,
            "cancelled": False,
            "trace": state.get("trace", []) + [f"【修改问题】{revised_question}"],
        }
        if target_unchanged:
            update["retrieved_docs"] = list(state.get("retrieved_docs", []))
            update["retrieval_history"] = list(state.get("retrieval_history", []))
            update["coverage_by_paper"] = initial_coverage_by_paper(
                list(update.get("comparison_papers") or update.get("selected_papers") or []),
                requested_aspects,
            ) if inferred_intent == "paper_comparison" else {}
            return Command(update=update, goto="evidence_check_node")

        update.update(
            {
                "retrieved_docs": [],
                "retrieval_history": [],
                "seen_result_ids": [],
                "seen_source_pages": [],
                "coverage_by_paper": initial_coverage_by_paper(comparison_papers, requested_aspects)
                if inferred_intent == "paper_comparison"
                else {},
                "tool_result": {},
                "observation": "",
                "auto_search_rounds": 0,
            }
        )
        return Command(update=update, goto="planner_node")

    def cancel_node(self, state: AgentState) -> AgentState:
        return {
            **state,
            "cancelled": True,
            "final_answer": "当前任务已取消。",
            "final_answer_streamed": False,
            "trace": state.get("trace", []) + ["【图状态】当前任务已取消。"],
        }

    def answer_node(self, state: AgentState) -> AgentState:
        if state.get("final_answer"):
            return state
        if state.get("error"):
            return {**state, "final_answer": state["error"], "final_answer_streamed": False}

        state = refresh_final_answer_evidence_state(state)

        evidence_block, sources = build_evidence_block(state.get("retrieved_docs", []))
        prompt = self._answer_prompt(state, evidence_block, sources)
        self._emit_progress(state, "正在生成最终回答……", category="当前状态", event_type="answer_started")
        try:
            answer, streamed = self._generate_answer_text(state, prompt)
        except Exception as exc:
            answer = self._fallback_answer(state, evidence_block, exc)
            streamed = False

        answer_body = strip_existing_source_list(answer)
        answer_body = remove_proactive_followup(answer_body)
        answer_body = ensure_answer_has_citation(answer_body, len(sources)).strip()
        answer_body, used_sources = validate_final_citations(answer_body, sources)
        answer_body = answer_body.strip()
        if state.get("allow_answer_with_gaps") and state.get("missing_aspects"):
            missing_text = "、".join(str(item) for item in state.get("missing_aspects", []))
            gap_sentence = f"当前论文证据中未找到对以下细节的明确说明：{missing_text}。"
            if gap_sentence not in answer_body:
                answer_body = f"{answer_body}\n\n【证据缺口】\n{gap_sentence}".strip()
        if state.get("intent") == "paper_analysis" and state.get("missing_aspects"):
            missing_text = "、".join(state.get("missing_aspects", []))
            if "未检索到的信息" not in answer_body:
                answer_body = f"{answer_body}\n\n【未检索到的信息】\n{missing_text}"
        source_list = format_source_list(used_sources)
        final = f"{answer_body}\n\n{source_list}".strip() if source_list else answer_body
        if streamed:
            print(final, flush=True)
        return {
            **state,
            "final_answer": final,
            "final_answer_streamed": streamed,
            "trace": state.get("trace", []) + ["【最终回答】已基于检索证据生成回答。"],
        }

    def _generate_answer_text(self, state: AgentState, prompt: str) -> tuple[str, bool]:
        if not get_bool_env("STREAM_FINAL_ANSWER", True):
            with self._progress_indicator(state, "正在生成最终回答"):
                response = self.llm.invoke([SystemMessage(content=prompt)])
            return str(response.content).strip(), False

        chunks: list[str] = []
        try:
            for chunk in self.llm.stream([SystemMessage(content=prompt)]):
                text = extract_chunk_text(chunk)
                if not text:
                    continue
                chunks.append(text)
            if chunks:
                return "".join(chunks).strip(), True
        except KeyboardInterrupt:
            raise
        except Exception:
            print("当前 API 不支持流式输出，已切换为完整回答模式。", flush=True)

        with self._progress_indicator(state, "正在生成最终回答"):
            response = self.llm.invoke([SystemMessage(content=prompt)])
        return str(response.content).strip(), True

    def route_after_planner(self, state: AgentState) -> str:
        return "answer" if state.get("next_action") == "answer" else "tool"

    def route_after_evidence_check(self, state: AgentState) -> str:
        if state.get("evidence_status") == "retrieval_failed" or state.get("retrieval_status") == "failed":
            return "retrieval_failure"
        if state.get("allow_answer_with_gaps"):
            return "answer"
        evidence_status = str(
            state.get("evidence_status")
            or ("sufficient" if state.get("evidence_sufficient") else "insufficient")
        )
        if evidence_status == "sufficient":
            return "answer"
        requested = set(state.get("requested_aspects") or state.get("required_aspects") or [])
        missing_requested = [
            aspect
            for aspect in state.get("missing_aspects", [])
            if not requested or aspect in requested or any(str(item) in str(aspect) for item in requested)
        ]
        if evidence_status == "sufficient_with_gaps" and not missing_requested:
            return "answer"
        if not self.enable_human_review:
            if int(state.get("iteration", 0)) >= int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS)):
                return "answer"
            return "rewrite"

        auto_rounds = int(state.get("auto_search_rounds", state.get("iteration", 0)))
        max_auto_rounds = int(state.get("max_auto_search_rounds", self.max_auto_search_rounds))
        if evidence_status == "insufficient" and auto_rounds < max_auto_rounds:
            return "rewrite"
        if evidence_status == "sufficient_with_gaps" and missing_requested and auto_rounds < max_auto_rounds:
            return "rewrite"
        if (
            state.get("intent") == "paper_analysis"
            and bool(state.get("retrieved_docs"))
            and bool(state.get("covered_aspects"))
            and not state.get("human_review_required")
        ):
            return "answer"
        return "human_review"

    def _plan_next_action(self, state: AgentState) -> PlannedAction:
        if state.get("intent") == "paper_analysis":
            return self._plan_paper_analysis_action(state)
        if state.get("intent") == "paper_comparison":
            return self._plan_paper_comparison_action(state)

        if self.tool_mode == "json":
            return self._plan_with_json(state)

        try:
            return self._plan_with_native_tools(state)
        except Exception:
            if self.tool_mode in {"auto", "native"}:
                state["trace"] = state.get("trace", []) + ["原生工具调用不可用，已切换到 JSON Planner 模式。"]
                return self._plan_with_json(state)
            raise

    def _plan_paper_analysis_action(self, state: AgentState) -> PlannedAction:
        source = self._single_selected_source(state)
        if not source:
            return PlannedAction(
                action="clarify",
                intent="clarification",
                arguments={"question": "请说明你要完整分析哪一篇论文。"},
                reason_summary="完整分析需要先确定具体论文",
            )

        required_aspects = state.get("required_aspects") or get_paper_analysis_aspects(source)
        missing_aspects = state.get("missing_aspects") or required_aspects
        queries = build_paper_analysis_queries(
            source=source,
            required_aspects=required_aspects,
            missing_aspects=missing_aspects,
        )
        return PlannedAction(
            action="search_multiple_queries",
            intent="paper_analysis",
            arguments={
                "queries": queries,
                "source": source,
                "k_per_query": 4,
            },
            reason_summary="按缺失信息点拆分为多个短查询",
        )

    def _plan_paper_comparison_action(self, state: AgentState) -> PlannedAction:
        sources = list(state.get("comparison_papers") or state.get("selected_papers") or [])
        if len(sources) < 2:
            return PlannedAction(
                action="clarify",
                intent="clarification",
                arguments={"question": "请说明你要比较哪两篇论文。"},
                reason_summary="论文对比需要至少两篇明确论文",
            )

        requested_aspects = state.get("requested_aspects") or state.get("required_aspects") or DEFAULT_COMPARISON_ASPECTS
        missing_aspects = state.get("missing_aspects") or requested_aspects
        queries = build_comparison_queries(
            requested_aspects=list(requested_aspects),
            missing_aspects=list(missing_aspects),
        )
        return PlannedAction(
            action="search_multiple_queries",
            intent="paper_comparison",
            arguments={
                "queries": queries,
                "sources": sources,
                "k_per_query": 4,
            },
            reason_summary="按用户要求的比较维度使用共享短查询检索两篇论文",
        )

    def _plan_with_native_tools(self, state: AgentState) -> PlannedAction:
        llm_with_tools = self.llm.bind_tools(self.tools.as_langchain_tools())
        with self._progress_indicator(state, "正在生成行动计划"):
            response = llm_with_tools.invoke(
                [
                    SystemMessage(content=self._planner_prompt(state, native=True)),
                    HumanMessage(content=state["question"]),
                ]
            )
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            return PlannedAction(
                action=normalize_action_name(call.get("name", "")) or "answer",
                arguments=dict(call.get("args") or {}),
                intent="general_qa",
                plan="原生工具调用选择下一步工具。",
                reason_summary="原生工具调用选择下一步工具。",
            )
        return parse_json_action(str(getattr(response, "content", "")))

    def _plan_with_json(self, state: AgentState) -> PlannedAction:
        with self._progress_indicator(state, "正在生成行动计划"):
            response = self.llm.invoke(
                [
                    SystemMessage(content=self._planner_prompt(state, native=False)),
                    HumanMessage(content=state["question"]),
                ]
            )
        return parse_json_action(str(response.content))

    def _planner_prompt(self, state: AgentState, native: bool) -> str:
        selected = state.get("selected_papers", [])
        history = "\n".join(f"{item['role']}: {item['content']}" for item in state.get("messages", [])[-6:])
        mode_note = "可以调用绑定工具。" if native else "只允许输出严格 JSON，不要输出 Markdown、代码块或解释性文字。"
        return f"""
你是一个论文精读 Agent，需要执行“规划-工具调用-观察-再决策”的闭环。
不要输出隐藏推理，只输出可展示的简短行动说明。
{mode_note}

可选 intent：{sorted(SUPPORTED_INTENTS)}
可选 action：
- list_papers
- search_paper，arguments: {{"query": "...", "source": null 或 PDF 文件名, "k": 5}}
- search_multiple_queries，arguments: {{"queries": ["短查询1", "短查询2"], "source": 单篇 PDF 文件名, "sources": 多篇 PDF 文件名列表, "k_per_query": 4}}
- get_neighbor_chunks，arguments: {{"source": PDF 文件名, "page": 1, "radius": 1}}
- inspect_paper_scope，arguments: {{"source": PDF 文件名}}
- clarify，arguments: {{"question": "需要用户补充的信息"}}
- answer，arguments: {{}}

如果问题复杂，应先选择 search_paper，并可在后续证据不足时改写查询再检索。
如果用户说“这篇论文/它”但没有明确 source，应选择 clarify。
当前规则识别 intent：{state.get("intent", "general_qa")}
当前明确论文：{selected}
当前对比维度：{state.get("comparison_aspects", [])}
用户必答项 requested_aspects：{state.get("requested_aspects", [])}
强关联补充 related_aspects：{state.get("related_aspects", [])}
若当前规则识别 intent 为 paper_comparison，不得降级为 general_qa。
最近对话：
{history}

JSON Planner 输出格式：
{{"action":"search_paper","arguments":{{"query":"LoRA low-rank adaptation mechanism","source":null,"k":5}},"reason_summary":"检索方法定义和关键机制"}}
paper_comparison 示例（最多 3 个共享查询）：
{{"action":"search_multiple_queries","intent":"paper_comparison","arguments":{{"queries":["research goal motivation problem setting","core method architecture mechanism equations key parameters","training objective trainable frozen parameters inference procedure"],"sources":["04 RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf","06 ReAct - Synergizing Reasoning and Acting in Language Models.pdf"],"k_per_query":4}},"reason_summary":"分别检索两篇论文在用户要求维度上的证据"}}
""".strip()

    def _evidence_prompt(self, state: AgentState) -> str:
        evidence_block, _sources = build_evidence_block(state.get("retrieved_docs", [])[-10:])
        return f"""
你是证据覆盖度检查器。不要输出推理过程，只输出严格 JSON。
根据用户问题判断当前检索证据是否覆盖了回答所需信息，不能只根据 chunk 数量判断。
覆盖状态允许：covered、not_applicable、partial、missing。
not_applicable 必须有直接证据支持，例如方法只是 inference-time prompting、不更新模型参数，因此 prompting 阶段没有训练损失。
“当前没有检索到”“论文未明确说明”“证据中没有给出”不能标记为 not_applicable，必须标记为 partial 或 missing。
如果用户询问“具体损失函数”，不能用领域常识推测 token-level NLL、cross entropy 等细节；只有证据明确给出时才算 covered。

用户问题：{state['question']}
任务类型：{state.get('intent')}
required_aspects：{state.get('required_aspects', [])}
requested_aspects：{state.get('requested_aspects', [])}
related_aspects（不是硬条件）：{state.get('related_aspects', [])}
covered_aspects：{state.get('covered_aspects', [])}
missing_aspects：{state.get('missing_aspects', [])}
当前证据：
{evidence_block or "无"}

输出格式：
{{"sufficient":true,"reason":"已经覆盖方法定义和关键机制","covered_aspects":["核心方法与数学公式"],"missing_aspects":[],"suggested_query":""}}
或：
{{"sufficient":false,"reason":"缺少实验结果","covered_aspects":["核心方法与数学公式"],"missing_aspects":["主要定量实验结果"],"suggested_query":""}}
""".strip()

    def _answer_prompt(
        self,
        state: AgentState,
        evidence_block: str,
        sources: list[dict[str, Any]],
    ) -> str:
        intent = state.get("intent", "general_qa")
        structure = ""
        if intent == "paper_analysis":
            structure = (
                "使用固定结构：【研究问题与动机】【核心方法】【数学公式】【训练过程】"
                "【推理与部署】【实验设置】【主要实验结果】【消融与机制分析】"
                "【优势】【局限性】【结论】。"
                f"如果证据不足，必须在相应小节写明当前检索证据没有明确说明。"
                f"当前仍缺少的方面：{state.get('missing_aspects', [])}。"
            )
        elif intent == "paper_summary":
            structure = "使用结构：【研究背景】【核心问题】【主要方法】【关键创新】【实验设计】【实验结果】【局限性】【结论】。"
        elif intent == "method_explain":
            structure = "使用结构：【核心思想】【输入与输出】【执行流程】【关键模块】【公式或机制解释】【与相关方法的区别】。"
        elif intent == "paper_comparison":
            requested = state.get("requested_aspects") or state.get("required_aspects") or DEFAULT_COMPARISON_ASPECTS
            related = state.get("related_aspects") or []
            structure = (
                f"先用 Markdown 表格覆盖用户明确要求的维度：{requested}。"
                "这些维度必须排在最前面。可以在单元格内补充已经检索到的强关联信息，"
                f"例如：{related[:8]}。主体回答后最多增加一个【关联补充】，且最多两个方面。"
                "直接相关的关键公式、参数设置、训练目标、部署特点、优势与局限。"
                "关联补充只能来自证据，缺失时写当前检索证据未明确给出，不要因此扩大回答。"
                "不要在结尾主动询问用户是否继续。"
                "对于“训练方式”，表格列名优先写为“训练 / 任务适配方式”。"
                "描述时必须区分以下概念："
                "（1）预训练；"
                "（2）参数微调或监督训练；"
                "（3）few-shot prompting 或 in-context learning；"
                "（4）论文为验证方法而进行的补充微调实验。"

                "必须根据检索证据判断哪一种属于论文方法的核心或默认使用方式，"
                "哪一种只是附加实验、消融实验或扩展验证。"
                "不得把 prompting 直接表述为参数训练；"
                "不得因为论文进行了微调实验，就推断该方法必须微调后才能使用。"

                "描述每种方式时，尽量说明是否更新模型参数。"
                "若证据没有明确说明其地位、训练目标或参数设置，"
                "必须写明“当前检索证据中没有明确说明”，不得自行补全。"

                "关联补充不得喧宾夺主，正文首先回答用户明确要求的问题。"
            )
        elif intent == "research_inspiration":
            structure = (
                "使用结构：【论文中可直接借鉴的机制】【与时间序列插补的对应关系】"
                "【可能有价值的研究假设】【不建议强行迁移的部分】【需要验证的实验】。"
                "明确区分原文支持、合理推导和尚未验证假设。"
                "用户背景：控制科学与工程研一；研究多变量时间序列插补；关注扩散模型、"
                "频率建模、残差生成、长尾误差和异常检测；正在学习 RAG、LoRA、Agent 和大模型应用。"
            )
        if state.get("allow_answer_with_gaps") and state.get("missing_aspects"):
            structure += (
                f"用户已选择基于现有证据回答，但必须明确标注证据缺口：{state.get('missing_aspects', [])}。"
                "不要用常识补全损失函数、超参数或实现细节。"
            )

        return f"""
你是严谨的论文精读助手。最终回答只能基于给定检索证据，不允许编造。
资料没有明确依据时写：“当前检索证据中没有明确说明……”
正文必须使用 [1] 这样的资料编号引用，编号只能来自证据列表。
正文中使用 [n] 引用，但不要输出【资料来源】列表，来源列表由程序生成。
除非论文原文明确称为 default setting，否则不要写“默认方式”或“默认设置”；
使用“论文主要采用”“主要任务适配方式”或“论文重点研究”等表述。
不要输出隐藏推理过程。{structure}

用户问题：{state['question']}
资料编号范围：1 到 {len(sources)}
检索证据：
{evidence_block or "无"}
""".strip()

    def _fallback_answer(self, state: AgentState, evidence_block: str, exc: Exception) -> str:
        if not evidence_block:
            return f"生成回答失败，且当前检索证据中没有明确说明可回答的问题。错误：{type(exc).__name__}: {exc}"
        return (
            f"生成回答时出现异常：{type(exc).__name__}: {exc}\n"
            "以下是当前可用证据摘要：\n"
            f"{evidence_block[:1500]}"
        )

    def _resolve_sources_from_context(self, question: str) -> list[str]:
        papers = self._available_papers()

        matched = []
        normalized_question = normalize_question(question)
        question_tokens = set(re.split(r"[^a-z0-9]+", normalized_question))
        for paper in papers:
            stem_tokens = [
                token
                for token in re.split(r"[^a-zA-Z0-9]+", str(paper).lower())
                if len(token) >= 3 and token not in {"pdf", "the", "and", "with"}
            ]
            if normalize_paper_name(str(paper)) in normalized_question or any(
                token in question_tokens for token in stem_tokens
            ):
                matched.append(str(paper))

        if matched:
            return sorted(set(matched))
        if contains_unresolved_pronoun(question):
            return list(self.last_sources)
        return []

    def _available_papers(self) -> list[str]:
        try:
            papers = self.tools.list_papers_tool().get("papers", [])
            return [str(paper) for paper in papers]
        except Exception:
            return []

    def _single_selected_source(self, state: AgentState) -> str | None:
        selected = state.get("selected_papers") or []
        return selected[0] if len(selected) == 1 else None

    def _remember_sources(self, docs: list[PaperChunk], selected: list[str]) -> None:
        sources = [doc["source"] for doc in docs if doc.get("source")]
        if not sources:
            sources = selected
        if sources:
            self.last_sources = list(dict.fromkeys(sources))[:3]


def summarize_tool_result(result: dict[str, Any]) -> str:
    if not result:
        return "工具没有返回结果。"
    if "papers" in result:
        return f"当前数据库包含 {result.get('count', len(result.get('papers', [])))} 篇论文。"
    if "results" in result:
        items = result.get("results") or []
        if not items:
            return "没有检索到相关片段。"
        sources = sorted({item.get("source", "unknown") for item in items})
        pages = sorted({item.get("page") for item in items if item.get("page") is not None})
        if result.get("query_results"):
            aspects = [item.get("covered_aspect") for item in result.get("query_results", []) if item.get("covered_aspect")]
            return f"检索到 {len(items)} 个相关片段，覆盖 {list(dict.fromkeys(aspects))}，主要来自 {', '.join(sources)}，页码 {pages[:8]}。"
        return f"检索到 {len(items)} 个相关片段，主要来自 {', '.join(sources)}，页码 {pages[:6]}。"
    if result.get("exists") is not None:
        return f"{result.get('source')} chunk 数：{result.get('chunk_count')}，页码范围：{result.get('page_range')}。"
    return "工具返回了结构化结果。"


def format_tool_arguments(arguments: dict[str, Any]) -> str:
    if not arguments:
        return "【工具参数】{}"
    safe_keys = ["query", "queries", "source_queries", "source", "sources", "k", "k_per_query", "page", "radius"]
    lines = ["【工具参数】"]
    for key in safe_keys:
        if key in arguments and arguments[key] is not None:
            lines.append(f"{key}: {arguments[key]}")
    return "\n".join(lines)


def tool_mode_label(tool_mode: str) -> str:
    if tool_mode == "json":
        return "JSON Planner"
    if tool_mode == "native":
        return "Native Function Calling"
    return "Auto"


def print_human_review_menu(payload: dict[str, Any]) -> None:
    if payload.get("type") == "retrieval_failure":
        print("\n【检索服务暂时不可用】")
        print("\n本轮查询全部失败，可能由上游限流或服务负载导致。")
        reason = payload.get("retrieval_error_reason")
        if reason:
            print(f"\n失败原因：{reason}")
        print("\n请选择：")
        for index, option in enumerate(payload.get("options") or [], start=1):
            print(f"{index}. {option.get('label')}")
        return

    print("\n【需要人工确认】")
    print("\n当前问题：")
    print(payload.get("question") or "")
    print("\n当前证据状态：")
    print(payload.get("evidence_status") or "insufficient")
    coverage_by_paper = payload.get("coverage_by_paper") or {}
    requested_aspects = payload.get("requested_aspects") or []
    if coverage_by_paper and requested_aspects:
        fully_covered: list[str] = []
        partial_lines: list[str] = []
        for aspect in requested_aspects:
            statuses = {
                short_source_label(source): normalize_coverage_status(coverage.get(aspect, "missing"))
                for source, coverage in coverage_by_paper.items()
            }
            if statuses and all(status in {"covered", "not_applicable"} for status in statuses.values()):
                fully_covered.append(str(aspect))
            elif statuses:
                partial_lines.append(
                    f"{aspect}：" + "，".join(f"{source} {status}" for source, status in statuses.items())
                )
        if fully_covered:
            print("\n已完整覆盖：")
            for item in fully_covered:
                print(f"- {item}")
        if partial_lines:
            print("\n部分覆盖：")
            for item in partial_lines:
                print(f"- {item}")
    else:
        covered = payload.get("covered_aspects") or []
        if covered:
            print("\n已覆盖：")
            for item in covered:
                print(f"✓ {item}")
        missing = payload.get("missing_aspects") or []
        if missing:
            print("\n尚未找到明确证据：")
            for item in missing:
                print(f"△ {item}")
    print("\n请选择：")
    for index, option in enumerate(payload.get("options") or [], start=1):
        print(f"{index}. {option.get('label')}")


def prompt_human_decision(payload: dict[str, Any]) -> dict[str, Any]:
    options = list(payload.get("options") or [])
    action_by_number = {str(index): str(option.get("action")) for index, option in enumerate(options, start=1)}
    while True:
        choice = input("请输入选项编号：").strip()
        action = action_by_number.get(choice)
        if not action:
            print("输入无效，请输入菜单中的编号。")
            continue
        if action == "revise_question":
            while True:
                revised_question = input("请输入修改后的问题：").strip()
                if revised_question:
                    return {"action": action, "revised_question": revised_question}
                print("修改后的问题不能为空。")
        return {"action": action}


def handle_human_review(agent: PaperAgent, state: AgentState, *, trace_enabled: bool) -> None:
    payload = state.get("last_interrupt_payload") or agent.get_interrupt_payload()
    while payload:
        print_human_review_menu(payload)
        decision = prompt_human_decision(payload)
        try:
            result = agent.resume(decision, trace_enabled=trace_enabled)
        except Exception as exc:
            print(f"恢复执行失败：{type(exc).__name__}: {exc}")
            return
        state = result["state"]
        if state.get("awaiting_human"):
            payload = state.get("last_interrupt_payload") or agent.get_interrupt_payload()
            continue
        if not state.get("final_answer_streamed"):
            print(result["answer"], flush=True)
        return
    print("该线程当前没有待恢复的人工决策。")


def print_startup(agent: PaperAgent) -> None:
    try:
        paper_count = agent.tools.list_papers_tool().get("count", 0)
    except Exception:
        paper_count = 0
    print("Research Paper Agent")
    print(f"Chat model: {agent.chat_model_name}")
    print(f"Tool mode: {tool_mode_label(agent.tool_mode)}")
    print(f"当前论文数量：{paper_count}")
    print("输入问题开始分析，输入 exit / quit / q 退出。")
    print("输入 papers 查看论文列表。")
    print("输入 session 查看当前 thread_id。")
    print("输入 new session 开始新的会话线程。")
    print("输入 resume <thread_id> 恢复等待人工决策的线程。")
    print("输入 clear 清空当前会话上下文。")
    print("输入 trace on 开启执行轨迹。")
    print("输入 trace off 关闭执行轨迹。")


def main() -> int:
    if sys.version_info < (3, 10):
        print("当前脚本要求 Python 3.10 或更高版本。")
        print(f"当前版本：{sys.version.split()[0]}")
        return 1

    try:
        tools = PaperTools.from_env()
        agent = PaperAgent(tools=tools)
    except Exception as exc:
        print("Agent 初始化失败。")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    trace_enabled = False
    print_startup(agent)

    try:
        while True:
            try:
                question = input("\n请输入问题：").strip()
            except KeyboardInterrupt:
                print("\n已退出。")
                return 0
            except EOFError:
                print("\n已退出。")
                return 0

            if not question:
                continue
            command = question.lower()
            if command in {"exit", "quit", "q"}:
                print("已退出。")
                return 0
            if command == "papers":
                try:
                    for paper in tools.list_papers_tool().get("papers", []):
                        print(f"- {paper}")
                except Exception as exc:
                    print(f"读取论文列表失败：{type(exc).__name__}: {exc}")
                continue
            if command == "session":
                print(f"当前 thread_id：{agent.thread_id}")
                continue
            if command == "new session":
                new_thread_id = agent.new_session()
                print(f"已创建新会话 thread_id：{new_thread_id}")
                continue
            if command.startswith("resume "):
                target_thread_id = question.split(maxsplit=1)[1].strip()
                previous_thread_id = agent.thread_id
                if not target_thread_id:
                    print("请提供要恢复的 thread_id。")
                    continue
                if not agent.thread_exists(target_thread_id):
                    print("未找到该 thread_id 的 checkpoint。")
                    agent.switch_thread(previous_thread_id)
                    continue
                agent.switch_thread(target_thread_id)
                payload = agent.get_interrupt_payload()
                if not payload:
                    print("该线程当前没有待恢复的人工决策。")
                    continue
                handle_human_review(agent, {"last_interrupt_payload": payload}, trace_enabled=trace_enabled)
                continue
            if command == "clear":
                old_thread_id = agent.thread_id
                agent.clear()
                print(f"已清空当前会话上下文，并删除线程 checkpoint：{old_thread_id}")
                print(f"新的 thread_id：{agent.thread_id}")
                continue
            if command == "trace on":
                trace_enabled = True
                print("执行轨迹已开启。")
                continue
            if command == "trace off":
                trace_enabled = False
                print("执行轨迹已关闭。")
                continue
            if len(question) > MAX_INPUT_CHARS:
                print(f"输入过长，请控制在 {MAX_INPUT_CHARS} 个字符以内。")
                continue

            try:
                result = agent.ask(question, trace_enabled=trace_enabled)
            except Exception as exc:
                print(f"处理失败：{type(exc).__name__}: {exc}")
                continue

            if result["state"].get("awaiting_human"):
                handle_human_review(agent, result["state"], trace_enabled=trace_enabled)
                continue
            if not result["state"].get("final_answer_streamed"):
                print(result["answer"], flush=True)
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
