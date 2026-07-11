from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / "db"
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
COLLECTION_NAME = "research_papers"
DEFAULT_K = 5
MAX_K = 10
MAX_NEIGHBOR_RADIUS = 2
MAX_MULTI_QUERIES = 6
MAX_CONSECUTIVE_QUERY_FAILURES = 2
RETRY_DELAYS_SECONDS = [1, 2, 4]

PLACEHOLDER_API_KEYS = {
    "",
    "your_api_key_here",
    "your-openai-api-key",
    "your_openai_api_key",
}
PLACEHOLDER_BASE_URLS = {
    "https://your-openai-compatible-api/v1",
    "your_base_url_here",
    "your-openai-compatible-api-base-url",
}


class ToolConfigError(RuntimeError):
    """Raised when the local vector database or API configuration is invalid."""


class PaperChunk(TypedDict):
    result_id: str
    source: str
    page: int | None
    metadata_page: int | None
    score: float
    score_label: str
    content: str


class VectorStoreLike(Protocol):
    def get(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = DEFAULT_K,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        ...

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_K,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        ...


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    base_url: str | None


def check_db_ready(db_dir: Path = DB_DIR) -> str | None:
    """Return an error message when the persisted Chroma directory is missing."""
    if not db_dir.exists():
        return f"未找到 Chroma 数据库目录：{db_dir}。请先运行：python ingest.py"
    return None


def load_openai_settings(env_path: Path = ENV_PATH) -> OpenAISettings:
    """Load OpenAI-compatible API settings from .env without printing secrets."""
    load_dotenv(dotenv_path=env_path)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()

    if api_key in PLACEHOLDER_API_KEYS:
        raise ToolConfigError(f"未配置有效的 OPENAI_API_KEY，请检查：{env_path}")
    if base_url in PLACEHOLDER_BASE_URLS:
        raise ToolConfigError("OPENAI_BASE_URL 仍是示例值，请改成实际的 OpenAI-compatible API 地址。")

    return OpenAISettings(api_key=api_key, base_url=base_url or None)


def build_embeddings(settings: OpenAISettings) -> OpenAIEmbeddings:
    """Create the embedding client used by the existing Chroma collection."""
    kwargs: dict[str, Any] = {
        "model": get_embedding_model_name(),
        "api_key": settings.api_key,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return OpenAIEmbeddings(**kwargs)


def get_embedding_model_name() -> str:
    """Return the embedding model name; default must match ingest.py."""
    return (os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def load_vector_store(
    embeddings: OpenAIEmbeddings,
    db_dir: Path = DB_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Load the persisted Chroma collection created by ingest.py."""
    message = check_db_ready(db_dir)
    if message:
        raise ToolConfigError(message)

    try:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(db_dir),
            create_collection_if_not_exists=False,
        )
    except Exception as exc:
        raise ToolConfigError(
            "加载 Chroma 向量数据库失败。请确认已先运行 python ingest.py，且 "
            f"collection_name={collection_name!r} 与 ingest.py 保持一致。"
        ) from exc


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, int(value)))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def display_page(metadata_page: Any) -> int | None:
    if metadata_page is None or metadata_page == "":
        return None
    try:
        return int(metadata_page) + 1
    except (TypeError, ValueError):
        return None


def should_fallback_from_relevance_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, (AttributeError, NotImplementedError)) or "not support" in message


def is_temporary_retrieval_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in message
        for marker in (
            "http 429",
            "429",
            "http 503",
            "503",
            "ratelimiterror",
            "rate limit",
            "负载已饱和",
            "temporarily unavailable",
        )
    )


def infer_query_aspect(query: str) -> str:
    """Map a concise retrieval query to the paper-analysis aspect it targets."""
    lowered = query.lower()
    if any(term in lowered for term in ["equation", "parameterization", "initialization", "scaling"]):
        return "核心方法与数学公式"
    if any(term in lowered for term in ["frozen", "freeze", "trainable", "parameters"]):
        return "注入位置与可训练参数"
    if any(term in lowered for term in ["merge", "inference", "deployment", "latency"]):
        return "推理合并与部署开销"
    if any(term in lowered for term in ["dataset", "datasets", "tasks", "base models"]):
        return "实验基础模型、数据集和任务"
    if any(term in lowered for term in ["baseline", "baselines", "metrics", "evaluation"]):
        return "对比 baseline 和评价指标"
    if any(term in lowered for term in ["quantitative", "results", "score", "accuracy", "bleu"]):
        return "主要定量实验结果"
    if any(term in lowered for term in ["rank", "target matrices", "ablation"]):
        return "rank 与目标矩阵消融"
    if any(term in lowered for term in ["singular", "subspace", "intrinsic", "analysis"]):
        return "低秩更新矩阵分析与局限性"
    if any(term in lowered for term in ["motivation", "problem"]):
        return "核心问题与动机"
    return "核心方法与数学公式"


class PaperTools:
    """Real tools for reading and retrieving from the local paper Chroma database."""

    def __init__(self, vector_store: VectorStoreLike) -> None:
        self.vector_store = vector_store

    @classmethod
    def from_env(cls) -> "PaperTools":
        settings = load_openai_settings()
        embeddings = build_embeddings(settings)
        vector_store = load_vector_store(embeddings)
        return cls(vector_store=vector_store)

    def list_papers_tool(self) -> dict[str, Any]:
        """Return unique PDF names currently present in Chroma metadata."""
        records = self._get_records()
        papers = sorted(
            {
                str(metadata.get("source"))
                for metadata in records["metadatas"]
                if metadata and metadata.get("source")
            }
        )
        return {"papers": papers, "count": len(papers)}

    def search_paper_tool(
        self,
        query: str,
        source: str | None = None,
        k: int = DEFAULT_K,
        retry_callback: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Search the persisted Chroma database, optionally within one PDF source."""
        cleaned_query = normalize_text(query)
        safe_k = clamp_int(k, 1, MAX_K)
        search_filter = {"source": source} if source else None

        results = self._run_with_retries(
            lambda: self.vector_store.similarity_search_with_score(
                cleaned_query,
                k=safe_k,
                filter=search_filter,
            ),
            retry_callback=retry_callback,
        )
        score_label = "distance_score"
        score_meaning = "距离数值越小表示越相关。"

        chunks = self._format_search_results(results, score_label=score_label)
        return {
            "query": cleaned_query,
            "source_filter": source,
            "k": safe_k,
            "score_label": score_label,
            "score_meaning": score_meaning,
            "results": chunks,
        }

    def search_multiple_queries_tool(
        self,
        queries: list[str],
        source: str | None = None,
        sources: list[str] | None = None,
        k_per_query: int = 4,
        seen_result_ids: list[str] | None = None,
        seen_source_pages: list[str] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        retry_callback: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Run aspect-scoped searches across one or more source PDFs."""
        safe_queries = [
            normalize_text(str(query))
            for query in queries
            if normalize_text(str(query))
        ]
        safe_queries = safe_queries[:MAX_MULTI_QUERIES]
        if not safe_queries:
            raise ValueError("queries 至少需要包含一个有效查询。")

        selected_sources: list[str] = []
        if source:
            selected_sources.append(str(source).strip())
        if sources is not None:
            if not isinstance(sources, list):
                raise ValueError("sources 必须是列表。")
            selected_sources.extend(str(item).strip() for item in sources if str(item).strip())
        selected_sources = list(dict.fromkeys(item for item in selected_sources if item))
        if not selected_sources:
            raise ValueError("source 或 sources 至少需要提供一个。")

        safe_k = clamp_int(k_per_query, 1, MAX_K)
        seen_ids = set(seen_result_ids or [])
        seen_pages = set(seen_source_pages or [])

        query_results: list[dict[str, Any]] = []
        candidates: list[tuple[bool, PaperChunk]] = []
        content_fingerprints: set[str] = set()
        score_label = "distance_score"
        successful_query_count = 0
        failed_query_count = 0
        consecutive_query_failures = 0
        circuit_breaker_triggered = False
        retrieval_error_reason = ""

        total_tasks = len(selected_sources) * len(safe_queries)
        task_index = 0
        for current_source in selected_sources:
            for query in safe_queries:
                if consecutive_query_failures >= MAX_CONSECUTIVE_QUERY_FAILURES:
                    circuit_breaker_triggered = True
                    break
                task_index += 1
                if progress_callback is not None:
                    progress_callback(task_index, total_tasks, f"{Path(current_source).stem}: {query}")
                try:
                    result = self.search_paper_tool(
                        query=query,
                        source=current_source,
                        k=safe_k,
                        retry_callback=retry_callback,
                    )
                except Exception as exc:
                    query_results.append(
                        {
                            "query": query,
                            "source": current_source,
                            "covered_aspect": infer_query_aspect(query),
                            "success": False,
                            "error_type": type(exc).__name__,
                            "error_message": normalize_text(str(exc))[:300],
                            "result_count": 0,
                            "result_ids": [],
                            "error": f"{type(exc).__name__}: {normalize_text(str(exc))[:160]}",
                        }
                    )
                    failed_query_count += 1
                    consecutive_query_failures += 1
                    retrieval_error_reason = f"{type(exc).__name__}: {normalize_text(str(exc))[:160]}"
                    continue
                successful_query_count += 1
                consecutive_query_failures = 0
                score_label = str(result.get("score_label") or score_label)
                query_chunks: list[PaperChunk] = []
                for chunk in result["results"]:
                    if chunk["result_id"] in seen_ids:
                        continue
                    fingerprint = self._chunk_fingerprint(chunk)
                    if fingerprint in content_fingerprints:
                        continue
                    content_fingerprints.add(fingerprint)
                    query_chunks.append(chunk)
                    page_key = self._source_page_key(chunk)
                    candidates.append((page_key in seen_pages, chunk))

                query_results.append(
                    {
                        "query": query,
                        "source": current_source,
                        "covered_aspect": infer_query_aspect(query),
                        "success": True,
                        "error_type": "",
                        "error_message": "",
                        "result_count": len(query_chunks),
                        "result_ids": [chunk["result_id"] for chunk in query_chunks],
                    }
                )
            if circuit_breaker_triggered:
                break

        candidates.sort(key=lambda item: (item[0], item[1]["source"], item[1]["page"] or 0, item[1]["result_id"]))
        if successful_query_count and failed_query_count:
            retrieval_status = "partial_failure"
        elif failed_query_count and not successful_query_count:
            retrieval_status = "failed"
        else:
            retrieval_status = "success"
        return {
            "queries": safe_queries,
            "source_filter": selected_sources[0] if len(selected_sources) == 1 else source,
            "source_filters": selected_sources,
            "k_per_query": safe_k,
            "score_label": score_label,
            "retrieval_status": retrieval_status,
            "retrieval_error_reason": retrieval_error_reason,
            "successful_query_count": successful_query_count,
            "failed_query_count": failed_query_count,
            "consecutive_query_failures": consecutive_query_failures,
            "circuit_breaker_triggered": circuit_breaker_triggered,
            "query_results": query_results,
            "results": [chunk for _seen_page, chunk in candidates],
        }

    def _run_with_retries(
        self,
        operation: Callable[[], Any],
        *,
        retry_callback: Callable[[int, int, int], None] | None = None,
    ) -> Any:
        for attempt_index in range(len(RETRY_DELAYS_SECONDS) + 1):
            try:
                return operation()
            except Exception as exc:
                if not is_temporary_retrieval_error(exc) or attempt_index >= len(RETRY_DELAYS_SECONDS):
                    raise
                delay = RETRY_DELAYS_SECONDS[attempt_index]
                if retry_callback is not None:
                    retry_callback(attempt_index + 1, len(RETRY_DELAYS_SECONDS), delay)
                time.sleep(delay)

    def get_neighbor_chunks_tool(
        self,
        source: str,
        page: int,
        radius: int = 1,
    ) -> dict[str, Any]:
        """Return chunks from the user-visible page and nearby pages in one PDF."""
        safe_radius = clamp_int(radius, 0, MAX_NEIGHBOR_RADIUS)
        target_metadata_page = max(0, int(page) - 1)
        low = target_metadata_page - safe_radius
        high = target_metadata_page + safe_radius

        records = self._get_records(where={"source": source})
        selected: list[PaperChunk] = []
        for idx, (content, metadata) in enumerate(zip(records["documents"], records["metadatas"])):
            metadata_page = metadata.get("page") if metadata else None
            try:
                raw_page = int(metadata_page)
            except (TypeError, ValueError):
                continue
            if low <= raw_page <= high:
                selected.append(
                    self._make_chunk(
                        document=Document(page_content=content or "", metadata=metadata or {}),
                        score=0.0,
                        score_label="neighbor_context",
                        fallback_index=idx,
                    )
                )

        selected.sort(key=lambda item: (item["metadata_page"] if item["metadata_page"] is not None else -1, item["result_id"]))
        return {
            "source": source,
            "page": int(page),
            "radius": safe_radius,
            "results": selected,
        }

    def inspect_paper_scope_tool(self, source: str) -> dict[str, Any]:
        """Return available page range, chunk count, and metadata fields for one PDF."""
        records = self._get_records(where={"source": source})
        pages = sorted(
            {
                int(metadata["page"])
                for metadata in records["metadatas"]
                if metadata and metadata.get("page") is not None
            }
        )
        metadata_keys = sorted(
            {
                key
                for metadata in records["metadatas"]
                if metadata
                for key in metadata.keys()
            }
        )
        return {
            "source": source,
            "exists": bool(records["documents"]),
            "chunk_count": len(records["documents"]),
            "metadata_fields": metadata_keys,
            "metadata_page_range": [pages[0], pages[-1]] if pages else None,
            "page_range": [pages[0] + 1, pages[-1] + 1] if pages else None,
        }

    def as_langchain_tools(self) -> list[StructuredTool]:
        """Expose the real paper database functions as LangChain tools."""
        def search_paper_public(query: str, source: str | None = None, k: int = DEFAULT_K) -> dict[str, Any]:
            return self.search_paper_tool(query=query, source=source, k=k)

        def search_multiple_public(
            queries: list[str],
            source: str | None = None,
            sources: list[str] | None = None,
            k_per_query: int = 4,
        ) -> dict[str, Any]:
            return self.search_multiple_queries_tool(
                queries=queries,
                source=source,
                sources=sources,
                k_per_query=k_per_query,
            )

        return [
            StructuredTool.from_function(
                func=self.list_papers_tool,
                name="list_papers_tool",
                description="List PDF file names present in the local Chroma paper database.",
            ),
            StructuredTool.from_function(
                func=search_paper_public,
                name="search_paper_tool",
                description="Vector-search paper chunks by query, optionally filtered by source PDF.",
            ),
            StructuredTool.from_function(
                func=search_multiple_public,
                name="search_multiple_queries_tool",
                description="Run multiple aspect-scoped vector searches in one or more source PDFs and merge results.",
            ),
            StructuredTool.from_function(
                func=self.get_neighbor_chunks_tool,
                name="get_neighbor_chunks_tool",
                description="Fetch chunks from a source PDF around a user-visible page number.",
            ),
            StructuredTool.from_function(
                func=self.inspect_paper_scope_tool,
                name="inspect_paper_scope_tool",
                description="Inspect available page range and chunk count for one source PDF.",
            ),
        ]

    def _get_records(self, where: dict[str, Any] | None = None) -> dict[str, list[Any]]:
        records = self.vector_store.get(where=where, include=["documents", "metadatas"])
        return {
            "ids": list(records.get("ids") or []),
            "documents": list(records.get("documents") or []),
            "metadatas": list(records.get("metadatas") or []),
        }

    def _format_search_results(
        self,
        results: list[tuple[Document, float]],
        score_label: str,
    ) -> list[PaperChunk]:
        seen: set[str] = set()
        chunks: list[PaperChunk] = []
        for idx, (document, score) in enumerate(results):
            content_key = self._content_fingerprint(document)
            if content_key in seen:
                continue
            seen.add(content_key)
            chunks.append(
                self._make_chunk(
                    document=document,
                    score=float(score),
                    score_label=score_label,
                    fallback_index=idx,
                )
            )
        return chunks

    def _make_chunk(
        self,
        document: Document,
        score: float,
        score_label: str,
        fallback_index: int,
    ) -> PaperChunk:
        metadata = document.metadata or {}
        metadata_page = metadata.get("page")
        try:
            raw_page = int(metadata_page) if metadata_page is not None else None
        except (TypeError, ValueError):
            raw_page = None

        source = str(metadata.get("source", "unknown"))
        digest = hashlib.sha1(
            f"{source}|{metadata_page}|{normalize_text(document.page_content)[:500]}".encode("utf-8")
        ).hexdigest()[:10]
        return PaperChunk(
            result_id=f"chunk-{digest}",
            source=source,
            page=display_page(raw_page),
            metadata_page=raw_page,
            score=score,
            score_label=score_label,
            content=normalize_text(document.page_content),
        )

    def _content_fingerprint(self, document: Document) -> str:
        metadata = document.metadata or {}
        normalized = normalize_text(document.page_content).lower()
        return f"{metadata.get('source')}|{metadata.get('page')}|{normalized[:400]}"

    def _chunk_fingerprint(self, chunk: PaperChunk) -> str:
        digest = hashlib.sha1(chunk["content"][:500].lower().encode("utf-8")).hexdigest()
        return f"{chunk['source']}|{chunk.get('metadata_page')}|{digest}"

    def _source_page_key(self, chunk: PaperChunk) -> str:
        return f"{chunk['source']}|{chunk.get('page')}"


_DEFAULT_TOOLS: PaperTools | None = None


def get_default_tools() -> PaperTools:
    global _DEFAULT_TOOLS
    if _DEFAULT_TOOLS is None:
        _DEFAULT_TOOLS = PaperTools.from_env()
    return _DEFAULT_TOOLS


def list_papers_tool() -> dict[str, Any]:
    """LangChain-compatible tool wrapper for listing available paper sources."""
    return get_default_tools().list_papers_tool()


def search_paper_tool(query: str, source: str | None = None, k: int = DEFAULT_K) -> dict[str, Any]:
    """LangChain-compatible tool wrapper for vector search over the paper database."""
    return get_default_tools().search_paper_tool(query=query, source=source, k=k)


def search_multiple_queries_tool(
    queries: list[str],
    source: str | None = None,
    sources: list[str] | None = None,
    k_per_query: int = 4,
    seen_result_ids: list[str] | None = None,
    seen_source_pages: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    retry_callback: Callable[[int, int, int], None] | None = None,
) -> dict[str, Any]:
    """LangChain-compatible wrapper for multi-query vector search."""
    return get_default_tools().search_multiple_queries_tool(
        queries=queries,
        source=source,
        sources=sources,
        k_per_query=k_per_query,
        seen_result_ids=seen_result_ids,
        seen_source_pages=seen_source_pages,
        progress_callback=progress_callback,
        retry_callback=retry_callback,
    )

def get_neighbor_chunks_tool(source: str, page: int, radius: int = 1) -> dict[str, Any]:
    """LangChain-compatible tool wrapper for retrieving page-neighbor chunks."""
    return get_default_tools().get_neighbor_chunks_tool(source=source, page=page, radius=radius)


def inspect_paper_scope_tool(source: str) -> dict[str, Any]:
    """LangChain-compatible tool wrapper for inspecting one paper's indexed scope."""
    return get_default_tools().inspect_paper_scope_tool(source=source)
