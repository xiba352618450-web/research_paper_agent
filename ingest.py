from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

warnings.filterwarnings(
    "ignore",
    message=r"`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"
ENV_PATH = PROJECT_ROOT / ".env"

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "research_papers"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

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


def find_pdf_files(data_dir: Path) -> list[Path]:
    """Return PDF files directly under data_dir, sorted for repeatable output."""
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.glob("*.pdf") if path.is_file())


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    """Load one PDF and normalize metadata for vector storage."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    for page in pages:
        metadata = {"source": pdf_path.name}
        if "page" in page.metadata:
            metadata["page"] = page.metadata["page"]
        page.metadata = metadata

    return pages


def get_openai_config() -> tuple[str | None, str | None]:
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()

    if api_key in PLACEHOLDER_API_KEYS:
        print("未配置有效的 OPENAI_API_KEY。")
        print(f"请在项目根目录的 .env 中配置：{ENV_PATH}")
        print("示例：OPENAI_API_KEY=your_api_key_here")
        return None, None

    if base_url in PLACEHOLDER_BASE_URLS:
        print("OPENAI_BASE_URL 仍是示例值，请在 .env 中改成实际的 OpenAI-compatible API 地址。")
        return None, None

    if not base_url:
        print("未配置 OPENAI_BASE_URL，将使用 OpenAI 默认 API 地址。")

    return api_key, base_url or None


def build_embeddings(api_key: str, base_url: str | None) -> OpenAIEmbeddings:
    kwargs = {
        "model": EMBEDDING_MODEL,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIEmbeddings(**kwargs)


def main() -> int:
    if sys.version_info < (3, 10):
        print("当前脚本要求 Python 3.10 或更高版本。")
        print(f"当前版本：{sys.version.split()[0]}")
        return 1

    if not DATA_DIR.exists():
        print(f"未找到 data/ 文件夹：{DATA_DIR}")
        print("请在项目根目录创建 data/ 文件夹，并放入公开论文 PDF 后再运行。")
        return 1

    pdf_files = find_pdf_files(DATA_DIR)
    if not pdf_files:
        print(f"data/ 文件夹中没有找到 PDF 文件：{DATA_DIR}")
        print("请将论文 PDF 放入 data/ 后再运行。")
        return 1

    print(f"找到 {len(pdf_files)} 个 PDF。")

    api_key, base_url = get_openai_config()
    if not api_key:
        return 1

    all_pages: list[Document] = []
    for pdf_path in pdf_files:
        try:
            pages = load_pdf_pages(pdf_path)
        except Exception as exc:
            print(f"加载 PDF 失败：{pdf_path.name}")
            print(f"{type(exc).__name__}: {exc}")
            return 1

        print(f"{pdf_path.name}: 加载 {len(pages)} 页")
        all_pages.extend(pages)

    if not all_pages:
        print("没有从 PDF 中加载到任何页面，请检查 PDF 文件是否可读取。")
        return 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_pages)
    print(f"总共生成 {len(chunks)} 个 chunk。")

    if not chunks:
        print("切分后没有生成任何 chunk，请检查 PDF 是否包含可提取文本。")
        return 1

    DB_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = build_embeddings(api_key=api_key, base_url=base_url)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(DB_DIR),
        )
    except Exception as exc:
        print("生成或保存 Chroma 向量库失败。")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    print(f"向量库保存位置：{DB_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
