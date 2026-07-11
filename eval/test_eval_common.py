from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (
    check_citations,
    extract_display_page,
    normalize_text,
    normalize_rank_config,
    parse_citations,
    score_answer,
    score_group_recall,
    score_must_include,
    score_must_include_any,
    score_must_not_include,
    score_raw_page_recall,
    set_metrics,
)


def test_set_metrics_empty_sets():
    assert set_metrics([], []) == {"exact_match": True, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_set_metrics_partial_match():
    result = set_metrics(["a", "b"], ["b", "c"])
    assert result["exact_match"] is False
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_group_recall_hits_any_page_in_group():
    gold = {"paper.pdf": [{"aspect": "公式", "pages": [4, 5]}]}
    result = score_group_recall(gold, {"paper.pdf": [5]})
    assert result["group_hits_by_paper"]["paper.pdf"][0]["hit"] is True
    assert result["macro_group_recall"] == 1.0


def test_group_recall_multi_paper_macro_average():
    gold = {
        "a.pdf": [{"aspect": "方法", "pages": [1]}],
        "b.pdf": [{"aspect": "实验", "pages": [2]}],
    }
    result = score_group_recall(gold, {"a.pdf": [1], "b.pdf": []})
    assert result["group_recall_by_paper"] == {"a.pdf": 1.0, "b.pdf": 0.0}
    assert result["macro_group_recall"] == 0.5


def test_raw_page_recall():
    result = score_raw_page_recall({"paper.pdf": [1, 2, 3, 4]}, {"paper.pdf": [2, 4, 9]})
    assert result["raw_page_recall_by_paper"]["paper.pdf"] == 0.5


def test_display_page_not_double_incremented():
    assert extract_display_page({"page": 4, "metadata_page": 3}) == 4


def test_metadata_page_fallback_adds_one():
    assert extract_display_page({"metadata_page": 3}) == 4


def test_must_include_all():
    result = score_must_include("LoRA uses low-rank updates and freezes weights.", ["low-rank", "freezes"])
    assert result["passed"] is True


def test_must_include_any_all_policy():
    result = score_must_include_any("模型使用低秩和合并部署。", [["low-rank", "低秩"], ["merge", "合并"]])
    assert result["passed"] is True
    assert result["hit_count"] == 2


def test_must_include_any_at_least_policy():
    result = score_must_include_any(
        "Thought and Action are interleaved.",
        [["Thought", "思考"], ["Action", "动作"], ["Observation", "观察"]],
        {"mode": "at_least", "min_groups": 2},
    )
    assert result["passed"] is True


def test_must_not_include():
    result = score_must_not_include("RAG uses a retriever.", ["RAG 不需要外部检索器", "retriever"])
    assert result["passed"] is False
    assert result["hits"] == ["retriever"]


def test_numeric_relation_does_not_match_citation_number():
    case = {
        "expected_facts": {"attention_heads": 8},
        "fact_match_rules": {"attention_heads": {"mode": "numeric_relation", "aliases": ["h"]}},
    }
    result = score_answer(case, "The model describes h in the architecture [8].")
    assert result["fact_checks"]["attention_heads"]["status"] == "fail"


def test_numeric_relation_matches_heads_8():
    case = {
        "expected_facts": {"attention_heads": 8},
        "fact_match_rules": {"attention_heads": {"mode": "numeric_relation", "aliases": ["h", "heads"]}},
    }
    result = score_answer(case, "The model uses h = 8 attention heads.")
    assert result["fact_checks"]["attention_heads"]["status"] == "pass"


def test_numeric_equivalent_scientific_notation():
    case = {
        "expected_facts": {"lora_learning_rate": "2.00E-04"},
        "fact_match_rules": {"lora_learning_rate": {"mode": "numeric_equivalent"}},
    }
    result = score_answer(case, "The LoRA learning rate is 0.0002.")
    assert result["fact_checks"]["lora_learning_rate"]["status"] == "pass"


def test_numeric_equivalent_multiplication_notation():
    case = {
        "expected_facts": {"lora_learning_rate": "2.00E-04"},
        "fact_match_rules": {"lora_learning_rate": {"mode": "numeric_equivalent"}},
    }
    result = score_answer(case, "The LoRA learning rate is 2 x 10^-4.")
    assert result["fact_checks"]["lora_learning_rate"]["status"] == "pass"


def test_rank_config_latex_normalization():
    assert normalize_rank_config("r_{q} = r_{v} = 8") == normalize_rank_config("rq=rv=8")


def test_rank_subset_min_matches():
    case = {
        "expected_facts": {"typical_rank_configs": ["rq = rv = 1", "rv = 2", "rq = rv = 8"]},
        "fact_match_rules": {
            "typical_rank_configs": {
                "mode": "subset",
                "accepted_values": ["rq = rv = 1", "rv = 2", "rq = rv = 8"],
                "min_matches": 2,
            }
        },
    }
    result = score_answer(case, "The paper compares r_q=r_v=1 and r_{q}=r_{v}=8.")
    assert result["fact_checks"]["typical_rank_configs"]["status"] == "pass"


def test_negative_fact_requires_manual_semantic_review():
    case = {"expected_facts": {"explicit_token_level_nll": False}}
    result = score_answer(case, "The paper does not explicitly say token-level NLL.")
    assert result["fact_checks"]["explicit_token_level_nll"]["status"] == "manual_semantic_required"
    assert result["manual_review_required"] is True


def test_parse_inline_citations():
    parsed = parse_citations("A claim[1][2]\n\n【资料来源】\n[1] a.pdf，第 3 页\n[2] b.pdf，第 4 页")
    assert parsed["inline_reference_ids"] == [1, 2]
    assert parsed["cited_pages_by_paper"] == {"a.pdf": [3], "b.pdf": [4]}


def test_unresolved_citation_detected():
    parsed = parse_citations("A claim[2]\n\n【资料来源】\n[1] a.pdf，第 3 页")
    assert parsed["unresolved_reference_ids"] == [2]


def test_cited_pages_group_recall():
    case = {
        "expected_papers": ["a.pdf"],
        "gold_page_groups": {"a.pdf": [{"aspect": "方法", "pages": [3, 4]}]},
    }
    result = check_citations(case, "A claim[1]\n\n【资料来源】\n[1] a.pdf，第 4 页")
    assert result["citation_group_recall"]["macro_group_recall"] == 1.0


def test_formula_match_accepts_qk_mathsf_t():
    result = score_must_include_any(
        r"attention uses softmax(QK^{\mathsf T} / \sqrt{d_k})",
        [["QK^T", "QK^{\\top}", "QK^\\top", "QKᵀ"]],
    )
    assert result["passed"] is True


def test_formula_match_accepts_multiply_latex_v():
    result = score_must_include_any(r"after softmax, the weights multiply \(V\).", [["与 V 相乘", "乘以 V", "value vector"]])
    assert result["passed"] is True


def test_eval005_answer_would_pass_concept_groups():
    answer = r"""
    Scaled dot-product attention computes softmax(QK^{\mathsf T}/\sqrt{d_k})V.
    Multi-head attention runs several projections in parallel. The output is a weighted sum of values.
    """
    groups = [
        ["scaled dot-product attention", "缩放点积注意力"],
        ["multi-head attention", "多头注意力"],
        ["softmax"],
        ["sqrt(d_k)", "\\sqrt{d_k}", "√d_k", "平方根缩放", "除以 d_k 的平方根"],
        ["QK^T", "QK^{\\top}", "QK^\\top", "QKᵀ", "query-key dot product", "查询和键的点积", "查询与键的点积"],
        ["与 V 相乘", "乘以 V", "value vector", "值向量", "values 的加权和", "值的加权和"],
    ]
    result = score_must_include_any(answer, groups)
    assert result["passed"] is True
