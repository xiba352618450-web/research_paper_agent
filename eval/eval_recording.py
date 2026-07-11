from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from agent_tools import DEFAULT_K, PaperTools
from eval_common import collect_chunks_from_tool_result, pages_by_paper_from_chunks, redact_secret


@dataclass
class ToolCallRecord:
    sequence: int
    tool_name: str
    sanitized_arguments: dict[str, Any]
    started_at: str
    elapsed_seconds: float
    result_count: int
    result_ids: list[str]
    source_pages: dict[str, list[int]]
    error: str = ""


def to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((to_json_safe(item) for item in value), key=lambda item: repr(item))
    if callable(value):
        return "<callable>"
    return repr(value)


class RecordingPaperTools(PaperTools):
    """PaperTools wrapper that records tool calls without changing behavior."""

    def __init__(self, vector_store: Any | None = None, base_tools: PaperTools | None = None):
        self.base_tools = base_tools
        super().__init__(vector_store if vector_store is not None else getattr(base_tools, "vector_store", None))
        self.records: list[dict[str, Any]] = []
        self._sequence = 0

    def _target(self) -> PaperTools:
        return self.base_tools or super()

    def _record_call(self, tool_name: str, func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        self._sequence += 1
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        error = ""
        result: Any = None
        try:
            result = func(**kwargs)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - started
            chunks = collect_chunks_from_tool_result(result)
            result_ids = [
                str(chunk.get("result_id") or chunk.get("id"))
                for chunk in chunks
                if chunk.get("result_id") or chunk.get("id")
            ]
            result_count = len(chunks)
            if isinstance(result, dict):
                if "papers" in result and isinstance(result["papers"], list):
                    result_count = len(result["papers"])
                elif "result_count" in result:
                    try:
                        result_count = int(result["result_count"])
                    except (TypeError, ValueError):
                        pass
            record = ToolCallRecord(
                sequence=self._sequence,
                tool_name=tool_name,
                sanitized_arguments=redact_secret(to_json_safe(dict(kwargs))),
                started_at=started_at,
                elapsed_seconds=elapsed,
                result_count=result_count,
                result_ids=result_ids,
                source_pages=pages_by_paper_from_chunks(chunks),
                error=error,
            )
            self.records.append(record.__dict__)

    def list_papers_tool(self) -> dict[str, Any]:
        target = self.base_tools.list_papers_tool if self.base_tools else super().list_papers_tool
        return self._record_call("list_papers_tool", target, {})

    def search_paper_tool(
        self,
        query: str,
        source: str | None = None,
        k: int = 5,
        retry_callback: Any | None = None,
    ) -> dict[str, Any]:
        target = self.base_tools.search_paper_tool if self.base_tools else super().search_paper_tool
        kwargs = {"query": query, "source": source, "k": k}
        if retry_callback is not None:
            kwargs["retry_callback"] = retry_callback
        return self._record_call("search_paper_tool", target, kwargs)

    def search_multiple_queries_tool(
        self,
        queries: list[str],
        source: str | None = None,
        sources: list[str] | None = None,
        k_per_query: int = 4,
        retry_callback: Any | None = None,
        progress_callback: Any | None = None,
        seen_result_ids: set[str] | None = None,
        seen_source_pages: set[tuple[str, int]] | None = None,
    ) -> dict[str, Any]:
        target = self.base_tools.search_multiple_queries_tool if self.base_tools else super().search_multiple_queries_tool
        kwargs = {
            "queries": queries,
            "source": source,
            "sources": sources,
            "k_per_query": k_per_query,
            "seen_result_ids": seen_result_ids,
            "seen_source_pages": seen_source_pages,
        }
        if retry_callback is not None:
            kwargs["retry_callback"] = retry_callback
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        return self._record_call("search_multiple_queries_tool", target, kwargs)

    def get_neighbor_chunks_tool(self, source: str, page: int, radius: int = 1) -> dict[str, Any]:
        target = self.base_tools.get_neighbor_chunks_tool if self.base_tools else super().get_neighbor_chunks_tool
        return self._record_call(
            "get_neighbor_chunks_tool",
            target,
            {"source": source, "page": page, "radius": radius},
        )

    def inspect_paper_scope_tool(self, source: str) -> dict[str, Any]:
        target = self.base_tools.inspect_paper_scope_tool if self.base_tools else super().inspect_paper_scope_tool
        return self._record_call("inspect_paper_scope_tool", target, {"source": source})

    def as_langchain_tools(self) -> list[Any]:
        # Keep the production schema shape, but bind execution to this wrapper.
        try:
            from langchain_core.tools import StructuredTool
        except Exception as exc:
            raise RuntimeError("RecordingPaperTools failed to create LangChain tools") from exc

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


@dataclass
class HITLAdapter:
    records: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def has_interrupt(state: dict[str, Any] | None) -> bool:
        state = state or {}
        return bool(state.get("awaiting_human") is True or state.get("last_interrupt_payload"))

    def resume_with_decisions(
        self,
        agent: Any,
        decisions: list[str],
        *,
        initial_state: dict[str, Any] | None = None,
        trace_enabled: bool = True,
    ) -> tuple[dict[str, Any] | None, str]:
        if not hasattr(agent, "resume") or not callable(agent.resume):
            return None, "unsupported_hitl_interface"
        final_result: dict[str, Any] | None = None
        initial_thread_id = getattr(agent, "thread_id", None)
        current_state = initial_state or {}
        for decision in decisions:
            if not self.has_interrupt(current_state):
                error = "expected_interrupt_not_observed" if not self.records else "next_interrupt_not_observed"
                return final_result, error
            before_thread_id = getattr(agent, "thread_id", None)
            try:
                final_result = agent.resume({"action": decision}, trace_enabled=trace_enabled)
            except Exception as exc:
                self.records.append(
                    {
                        "decision": decision,
                        "thread_id_before": before_thread_id,
                        "thread_id_after": getattr(agent, "thread_id", None),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return final_result, f"{type(exc).__name__}: {exc}"
            after_thread_id = getattr(agent, "thread_id", None)
            current_state = dict((final_result or {}).get("state") or {})
            self.records.append(
                {
                    "decision": decision,
                    "thread_id_before": before_thread_id,
                    "thread_id_after": after_thread_id,
                    "same_thread": (
                        initial_thread_id is not None
                        and str(initial_thread_id).strip() != ""
                        and before_thread_id == after_thread_id == initial_thread_id
                    ),
                    "awaiting_human": bool(current_state.get("awaiting_human")),
                }
            )
        return final_result, ""
