from typing import List, Dict

import structlog
from sqlalchemy import text

from src.database.connection import get_db
from src.database.models import Chunk
from src.service.expert_retrieval_service import rrf_fusion, RECALL_K

logger = structlog.get_logger(__name__)

MEDLINEPLUS_PREFIX = "medlineplus_"
FINAL_TOP_K = 5


def _serialize_patient_chunk(chunk: Chunk, score: float) -> Dict:
    """把 Chunk 行序列化成患者版格式：content + topic/url + score（含 id 供 RRF 去重）。"""
    return {
        "id": chunk.id,
        "content": chunk.content,
        "topic": chunk.meta.get("topic", "unknown") if chunk.meta else "unknown",
        "url": chunk.meta.get("url", "") if chunk.meta else "",
        "score": round(score, 4),
    }


def cosine_search_medlineplus(query_embedding: List[float], top_k: int = RECALL_K) -> List[Dict]:
    """pgvector cosine 检索，只查 MedlinePlus 数据（doc_id 前缀过滤）。"""
    with get_db() as db:
        results = db.query(
            Chunk,
            Chunk.embedding.cosine_distance(query_embedding).label("distance")
        ).filter(
            Chunk.doc_id.like(f"{MEDLINEPLUS_PREFIX}%")
        ).order_by("distance").limit(top_k).all()
        return [_serialize_patient_chunk(chunk, 1 - distance) for chunk, distance in results]


def bm25_search_medlineplus(query_text: str, top_k: int = RECALL_K) -> List[Dict]:
    """ParadeDB BM25 检索，只查 MedlinePlus 数据。"""
    sql = text("""
        SELECT c.id, c.content, c.meta, paradedb.score(c.id) AS score
        FROM chunks c
        WHERE c @@@ :query AND c.doc_id LIKE :prefix
        ORDER BY score DESC
        LIMIT :limit
    """)
    with get_db() as db:
        rows = db.execute(sql, {
            "query": query_text,
            "prefix": f"{MEDLINEPLUS_PREFIX}%",
            "limit": top_k,
        }).fetchall()
        return [
            {
                "id": row.id,
                "content": row.content,
                "topic": row.meta.get("topic", "unknown") if row.meta else "unknown",
                "url": row.meta.get("url", "") if row.meta else "",
                "score": round(float(row.score), 4),
            }
            for row in rows
        ]


def retrieve_medlineplus(question: str, embedder, top_k: int = FINAL_TOP_K) -> List[Dict]:
    """患者版检索：BM25 + cosine 混合粗召回 → RRF 融合，不做精排。

    BM25 未部署（ParadeDB 缺失）时降级为纯 cosine。
    """
    query_embedding = embedder.embed_query(question)
    cosine_hits = cosine_search_medlineplus(query_embedding, RECALL_K)
    try:
        bm25_hits = bm25_search_medlineplus(question, RECALL_K)
    except Exception as e:
        logger.warning("bm25_failed_fallback_cosine", error=str(e))
        return cosine_hits[:top_k]
    fused = rrf_fusion([bm25_hits, cosine_hits])
    return fused[:top_k]
