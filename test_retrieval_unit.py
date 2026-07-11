from __future__ import annotations

import unittest

from langchain_core.documents import Document

import test_retrieval


class TestRetrievalHelpers(unittest.TestCase):
    def test_build_preview_collapses_whitespace_and_limits_length(self) -> None:
        text = "Alpha\n\n   beta\tgamma " + "x" * 600

        preview = test_retrieval.build_preview(text, max_chars=20)

        self.assertEqual(preview, "Alpha beta gamma xxx...")
        self.assertLessEqual(len(preview), 23)

    def test_format_page_uses_unknown_when_missing(self) -> None:
        self.assertEqual(test_retrieval.format_page({"page": 3}), "3")
        self.assertEqual(test_retrieval.format_page({}), "unknown")

    def test_is_placeholder_value_detects_common_placeholders(self) -> None:
        self.assertTrue(
            test_retrieval.is_placeholder_value(
                "your_api_key_here",
                test_retrieval.PLACEHOLDER_API_KEYS,
            )
        )
        self.assertFalse(
            test_retrieval.is_placeholder_value(
                "real-looking-api-key",
                test_retrieval.PLACEHOLDER_API_KEYS,
            )
        )

    def test_retrieve_chunks_prefers_relevance_scores(self) -> None:
        store = FakeVectorStore(relevance_result=[(Document(page_content="hit"), 0.91)])

        results, score_label = test_retrieval.retrieve_chunks(store, "rag", k=5)

        self.assertEqual(score_label, "relevance_score")
        self.assertEqual(results[0][1], 0.91)
        self.assertEqual(store.calls, ["relevance"])

    def test_retrieve_chunks_falls_back_to_raw_score(self) -> None:
        store = FakeVectorStore(
            relevance_error=NotImplementedError("not supported"),
            score_result=[(Document(page_content="hit"), 0.12)],
        )

        results, score_label = test_retrieval.retrieve_chunks(store, "rag", k=5)

        self.assertEqual(score_label, "distance_score")
        self.assertEqual(results[0][1], 0.12)
        self.assertEqual(store.calls, ["relevance", "score"])


class FakeVectorStore:
    def __init__(
        self,
        relevance_result=None,
        score_result=None,
        relevance_error: Exception | None = None,
    ) -> None:
        self.relevance_result = relevance_result or []
        self.score_result = score_result or []
        self.relevance_error = relevance_error
        self.calls: list[str] = []

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        self.calls.append("relevance")
        if self.relevance_error is not None:
            raise self.relevance_error
        return self.relevance_result

    def similarity_search_with_score(self, query: str, k: int):
        self.calls.append("score")
        return self.score_result


if __name__ == "__main__":
    unittest.main()
