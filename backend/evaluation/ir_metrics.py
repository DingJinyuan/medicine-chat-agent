"""④ IR 检索指标(纯函数,无 LLM judge、零漂移、可复现)。

判断「RRF + 精排到底有没有用」:给定检索返回的 doc_id 列表与标注的正确 doc_id 集合,
计算 recall@k / hit@k / MRR / NDCG@k。二元相关度(命中=1,否则 0)。

输入约定:
- retrieved_ids: 检索系统返回的 doc_id 有序列表(按分数降序)
- relevant_ids: 人工标注的正确 doc_id 集合
"""
import math


def recall_at_k(retrieved_ids: list, relevant_ids, k: int) -> float:
    """前 k 个结果中命中的相关文档数 / 相关文档总数。"""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def hit_at_k(retrieved_ids: list, relevant_ids, k: int) -> float:
    """前 k 个结果中是否有任意一个相关文档(1 或 0)。"""
    return 1.0 if any(d in set(relevant_ids) for d in retrieved_ids[:k]) else 0.0


def mrr(retrieved_ids: list, relevant_ids) -> float:
    """第一个相关文档的倒数排名(Mean Reciprocal Rank 单查询版本)。"""
    relevant = set(relevant_ids)
    for rank, d in enumerate(retrieved_ids, start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list, relevant_ids, k: int) -> float:
    """前 k 个结果的 NDCG(二元相关度,理想排序 = 相关文档全排最前)。"""
    relevant = set(relevant_ids)
    dcg = 0.0
    for i, d in enumerate(retrieved_ids[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(retrieved_ids: list, relevant_ids, k: int = 5) -> dict:
    """单条查询的完整 IR 指标集合。"""
    return {
        f"recall@{k}": round(recall_at_k(retrieved_ids, relevant_ids, k), 4),
        f"hit@{k}": round(hit_at_k(retrieved_ids, relevant_ids, k), 4),
        "mrr": round(mrr(retrieved_ids, relevant_ids), 4),
        f"ndcg@{k}": round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4),
    }


def evaluate_dataset(queries: list[dict], k: int = 5) -> dict:
    """批量评估:queries 每项含 retrieved_ids / relevant_ids。

    返回各指标的平均值。
    """
    agg = {}
    for q in queries:
        m = evaluate_query(q["retrieved_ids"], q["relevant_ids"], k=k)
        for key, val in m.items():
            agg.setdefault(key, []).append(val)
    return {key: round(sum(vals) / len(vals), 4) for key, vals in agg.items()}
