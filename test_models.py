from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


MODELS_TO_CHECK = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2-thinking",
    "doubao-pro-256k",
    "gpt-4o-mini",
]


def main() -> int:
    """Manually check which chat models are available from the configured endpoint."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("OPENAI_API_KEY is not configured.")
        return 1

    for model_name in MODELS_TO_CHECK:
        print(f"\nTesting model: {model_name}")
        try:
            llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key,
                base_url=base_url or None,
                timeout=60,
                max_retries=0,
            )
            response = llm.invoke("Reply with exactly: OK")
            print(f"[OK] {model_name}")
            print(f"Response: {response.content}")
        except Exception as exc:
            print(f"[FAILED] {model_name}")
            print(f"Error: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
