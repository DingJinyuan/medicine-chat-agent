"""IR 检索指标纯函数测试(④ 评估层,离线零依赖)。"""
from evaluation.ir_metrics import (
    recall_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    evaluate_query,
    evaluate_dataset,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5  # 命中 a,共 2 相关
    assert recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0
    assert recall_at_k([], ["a"], k=5) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "a"], ["a"], k=2) == 1.0
    assert hit_at_k(["x", "y"], ["a"], k=2) == 0.0


def test_mrr():
    assert mrr(["a", "b"], ["b"]) == 0.5  # 第一个相关在 rank 2
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["a", "b"], ["c"]) == 0.0


def test_ndcg_at_k():
    # 完美排序:相关文档全在最前
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0
    # 相关文档靠后,NDCG < 1
    assert ndcg_at_k(["c", "b", "a"], ["a", "b"], k=3) < 1.0


def test_evaluate_dataset_average():
    queries = [
        {"retrieved_ids": ["a", "b"], "relevant_ids": ["a"]},
        {"retrieved_ids": ["a", "b"], "relevant_ids": ["b"]},
    ]
    result = evaluate_dataset(queries, k=2)
    assert result["hit@2"] == 1.0
    assert result["mrr"] == 0.75  # (1.0 + 0.5) / 2
