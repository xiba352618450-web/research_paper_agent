from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / "db"
ENV_PATH = PROJECT_ROOT / ".env"

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "research_papers"
TOP_K = 5
EXIT_COMMANDS = {"exit", "quit", "q"}

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


def is_placeholder_value(value: str, placeholders: set[str]) -> bool:
    return value.strip() in placeholders


def get_openai_config() -> tuple[str | None, str | None]:
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()

    if is_placeholder_value(api_key, PLACEHOLDER_API_KEYS):
        print("未配置有效的 OPENAI_API_KEY。")
        print(f"请在项目根目录的 .env 中配置：{ENV_PATH}")
        return None, None

    if is_placeholder_value(base_url, PLACEHOLDER_BASE_URLS):
        print("OPENAI_BASE_URL 仍是示例值，请在 .env 中改成实际的 OpenAI-compatible API 地址。")
        return None, None

    if not base_url:
        print("未配置 OPENAI_BASE_URL，将使用 OpenAI 默认 API 地址。")

    return api_key, base_url or None


def build_embeddings(api_key: str, base_url: str | None) -> OpenAIEmbeddings:
    kwargs: dict[str, Any] = {
        "model": EMBEDDING_MODEL,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIEmbeddings(**kwargs)


def load_vector_store(embeddings: OpenAIEmbeddings) -> Chroma | None:
    if not DB_DIR.exists():
        print(f"未找到 Chroma 数据库目录：{DB_DIR}")
        print("请先运行：python ingest.py")
        return None

    try:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(DB_DIR),
            create_collection_if_not_exists=False,
        )
    except Exception as exc:
        print("加载 Chroma 向量数据库失败。")
        print(f"数据库位置：{DB_DIR}")
        print(f"collection_name：{COLLECTION_NAME}")
        print("请确认已先运行 python ingest.py，且检索脚本中的 collection_name 与 ingest.py 保持一致。")
        print(f"{type(exc).__name__}: {exc}")
        return None


def should_fallback_from_relevance_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, (AttributeError, NotImplementedError)) or "not support" in message


def retrieve_chunks(
    vector_store: Chroma,
    question: str,
    k: int = TOP_K,
) -> tuple[list[tuple[Document, float]], str]:
    try:
        results = vector_store.similarity_search_with_relevance_scores(question, k=k)
        return results, "relevance_score"
    except Exception as exc:
        if not should_fallback_from_relevance_error(exc):
            raise

    results = vector_store.similarity_search_with_score(question, k=k)
    return results, "distance_score"


def build_preview(text: str, max_chars: int = 500) -> str:
    preview = re.sub(r"\s+", " ", text).strip()
    if len(preview) <= max_chars:
        return preview
    return preview[:max_chars].rstrip() + "..."


def format_page(metadata: dict[str, Any]) -> str:
    page = metadata.get("page")
    if page is None or page == "":
        return "unknown"
    return str(page)


def print_results(results: list[tuple[Document, float]], score_label: str) -> None:
    if not results:
        print("没有检索到相关 chunk。")
        return

    if score_label == "relevance_score":
        print("分数含义：relevance_score，范围通常为 0 到 1，越高越相似。")
    else:
        print("当前 Chroma 版本不支持 similarity_search_with_relevance_scores，已使用 similarity_search_with_score。")
        print("分数含义：distance_score，距离分数越小越相似。")

    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata or {}
        source = metadata.get("source", "unknown")
        page = format_page(metadata)
        preview = build_preview(document.page_content, max_chars=500)

        print(f"\n[{rank}]")
        print(f"source: {source}")
        print(f"page: {page}")
        print(f"{score_label}: {score:.6f}")
        print(f"preview: {preview}")


def run_cli(vector_store: Chroma) -> None:
    print("向量检索测试已启动。输入 exit、quit 或 q 退出。")
    print(f"Chroma 数据库：{DB_DIR}")
    print(f"collection_name：{COLLECTION_NAME}")

    while True:
        try:
            question = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("已退出。")
            break

        try:
            results, score_label = retrieve_chunks(vector_store, question, k=TOP_K)
        except Exception as exc:
            print("检索失败。")
            print(f"{type(exc).__name__}: {exc}")
            continue

        print_results(results, score_label)


def main() -> int:
    if sys.version_info < (3, 10):
        print("当前脚本要求 Python 3.10 或更高版本。")
        print(f"当前版本：{sys.version.split()[0]}")
        return 1

    api_key, base_url = get_openai_config()
    if not api_key:
        return 1

    try:
        embeddings = build_embeddings(api_key=api_key, base_url=base_url)
    except Exception as exc:
        print("初始化 Embedding 模型失败。")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    vector_store = load_vector_store(embeddings)
    if vector_store is None:
        return 1

    run_cli(vector_store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
