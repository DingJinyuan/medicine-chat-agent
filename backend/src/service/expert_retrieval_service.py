from typing import List, Dict

import structlog

from src.repository.chunk_repo import semantic_search_chunks, bm25_search_chunks

logger = structlog.get_logger(__name__)

RRF_K = 60          # RRF 融合常数
RECALL_K = 20       # 每路粗召回条数
FINAL_TOP_K = 5     # 精排后最终返回条数


def rrf_fusion(result_sets: List[List[Dict]], k: int = RRF_K) -> List[Dict]:
    """RRF 融合多个排序结果集（按 chunk id 去重），返回融合排序后的列表。

    公式：RRF_score(d) = Σ 1/(k + rank_r(d))，k 越大排名差异越平滑。
    """
    fused: Dict[int, Dict] = {}
    for results in result_sets:
        for rank, doc in enumerate(results):
            cid = doc["id"]
            if cid not in fused:
                fused[cid] = dict(doc)
                fused[cid]["score"] = 0.0
            fused[cid]["score"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda x: x["score"], reverse=True)


def hybrid_recall(query_text: str, query_embedding: List[float], recall_k: int = RECALL_K) -> List[Dict]:
    """单路混合粗召回：BM25 + cosine 各自粗召回，RRF 合并。

    BM25 未部署（ParadeDB 缺失）时降级为纯 cosine，保证流水线不中断。
    """
    cosine_hits = semantic_search_chunks(query_embedding, recall_k)
    try:
        bm25_hits = bm25_search_chunks(query_text, recall_k)
    except Exception as e:
        logger.warning("bm25_failed_fallback_cosine", error=str(e))
        return cosine_hits
    return rrf_fusion([bm25_hits, cosine_hits])


def multi_route_retrieve(
    question: str,
    rewritten_query: str,
    keywords: List[str],
    embedder,
    reranker,
    top_k: int = FINAL_TOP_K,
    recall_k: int = RECALL_K,
) -> List[Dict]:
    """多路召回 + 精排。

    流程：
      1. 组装多路 query（改写问题 / 关键词，去重去空）
      2. 每路 hybrid_recall（BM25 + cosine 粗召回 → RRF 合并）
      3. 各路结果 RRF 合并
      4. cross-encoder 用补全后的问题精排 → top_k

    Args:
        embedder: 向量嵌入器（调用方注入）
        reranker: cross-encoder 精排器（调用方注入）
    """
    routes = []
    for q in [rewritten_query or question, " ".join(keywords)]:
        q = (q or "").strip()
        if q and q not in routes:
            routes.append(q)

    result_sets = []
    for route in routes:
        query_embedding = embedder.embed_query(route)
        result_sets.append(hybrid_recall(route, query_embedding, recall_k))

    fused = rrf_fusion(result_sets)
    logger.info("retrieval_fused", route_count=len(routes), candidate_count=len(fused))

    ranked = reranker.rerank(rewritten_query or question, fused, top_k)
    logger.info("retrieval_reranked", result_count=len(ranked))
    return ranked
