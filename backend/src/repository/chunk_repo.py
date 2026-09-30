from typing import List, Dict
from sqlalchemy import text
from src.database.connection import get_db
from src.database.models import Chunk


def _serialize_chunk(chunk: Chunk, score: float) -> Dict:
    """把 Chunk 行序列化成统一 dict，含 id（供 RRF 去重）"""
    return {
        "id": chunk.id,
        "content": chunk.content,
        "doc_id": chunk.doc_id,
        "score": round(score, 4),
        "source": chunk.meta.get("source", "unknown") if chunk.meta else "unknown",
        "journal": chunk.meta.get("journal", "") if chunk.meta else "",
    }


def semantic_search_chunks(query_embedding: list[float], top_k: int = 20) -> List[Dict]:
    """pgvector 余弦相似度检索（粗召回，多召回一些供后续精排）。

    专家版专用：排除患者版 MedlinePlus 数据（doc_id 前缀 medlineplus_），避免两版数据污染。
    """
    with get_db() as db:
        results = db.query(
            Chunk,
            Chunk.embedding.cosine_distance(query_embedding).label("distance")
        ).filter(
            Chunk.doc_id.notlike("medlineplus_%")
        ).order_by("distance").limit(top_k).all()
        return [_serialize_chunk(chunk, 1 - distance) for chunk, distance in results]


def bm25_search_chunks(query_text: str, top_k: int = 20) -> List[Dict]:
    """ParadeDB pg_search BM25 关键词检索（粗召回）。

    依赖 chunks 表上的 bm25 索引（见 database/connection.init_bm25_index）。
    未部署 ParadeDB 时调用会抛异常，由上层降级为纯向量检索。
    """
    sql = text("""
        SELECT c.id, c.content, c.doc_id, c.meta,
               paradedb.score(c.id) AS score
        FROM chunks c
        WHERE c @@@ :query
          AND c.doc_id NOT LIKE 'medlineplus_%'
        ORDER BY score DESC
        LIMIT :limit
    """)
    with get_db() as db:
        rows = db.execute(sql, {"query": query_text, "limit": top_k}).fetchall()
        return [
            {
                "id": row.id,
                "content": row.content,
                "doc_id": row.doc_id,
                "score": round(float(row.score), 4),
                "source": row.meta.get("source", "unknown") if row.meta else "unknown",
                "journal": row.meta.get("journal", "") if row.meta else "",
            }
            for row in rows
        ]
