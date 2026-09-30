"""患者版知识库摄取：medlineplus_topics.json → 切块 → 嵌入 → pgvector。

把 104 条 MedlinePlus 健康主题（topic/url/summary）切块后用 pubmedbert 嵌入，
存进专家版同款 documents + chunks 表（source='medlineplus'），
与专家版的 pubmed/pdf 数据共用同一套 pgvector 检索。

用法（在 backend/ 下运行）：
    python -m src.ingestion.patient_ingest
"""
import json
from pathlib import Path

import structlog

from src.config import settings
from src.database.connection import get_db, init_db
from src.database.models import Document, Chunk
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import Embedder

logger = structlog.get_logger(__name__)

# 数据路径（相对 backend/）
JSON_PATH = Path("data") / "medlineplus_topics.json"


def load_topics(json_path: Path) -> list[dict]:
    """读 medlineplus_topics.json，返回 [{topic, url, summary}]。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_doc_id(topic: str) -> str:
    """主题名转 doc_id，如 'High Blood Pressure' -> 'medlineplus_high_blood_pressure'。"""
    slug = topic.lower().replace(" ", "_").replace("-", "_").replace("'", "")
    return f"medlineplus_{slug}"


def ingest(json_path: Path = JSON_PATH, batch_size: int = 32) -> int:
    """读 json → 切块 → 嵌入 → 存 documents/chunks 表，返回入库 chunk 数。"""
    topics = load_topics(json_path)
    chunker = TextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    embedder = Embedder()

    # 1. 切块：每条 summary 切成小块（避免嵌入模型 512 token 截断）
    logger.info("patient_ingest_chunking", topic_count=len(topics))
    chunk_entries = []  # (topic_dict, doc_id, TextChunk)
    for topic in topics:
        doc_id = _make_doc_id(topic["topic"])
        for chunk in chunker.chunk_text(topic["summary"], doc_id=doc_id):
            chunk_entries.append((topic, doc_id, chunk))
    logger.info("patient_ingest_chunked", chunk_count=len(chunk_entries))

    # 2. 批量嵌入（pubmedbert 768 维）
    logger.info("patient_ingest_embedding", count=len(chunk_entries))
    text_chunks = [entry[2] for entry in chunk_entries]
    embeddings = embedder.embed_chunks(text_chunks, batch_size=batch_size)
    logger.info("patient_ingest_embedded")

    # 3. 入库：documents + chunks
    with get_db() as db:
        # 预加载已存在的 doc_id -> document.id 映射，避免循环内逐条查库（N+1）
        all_doc_ids = {doc_id for _, doc_id, _ in chunk_entries}
        existing = {
            doc_id: did
            for doc_id, did in db.query(Document.doc_id, Document.id)
            .filter(Document.doc_id.in_(all_doc_ids)).all()
        }
        for (topic, doc_id, chunk), embedding in zip(chunk_entries, embeddings):
            if doc_id in existing:
                # 该文档已摄取过,直接跳过,保证幂等(重复运行不产生重复 chunk)
                continue

            doc = Document(
                source="medlineplus",
                doc_id=doc_id,
                title=topic["topic"],
                meta={"url": topic["url"]},
            )
            db.add(doc)
            db.flush()  # 拿到 doc.id
            document_id = doc.id
            existing[doc_id] = document_id

            db.add(Chunk(
                document_id=document_id,
                doc_id=doc_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embedding,
                token_count=chunk.token_count,
                meta={
                    "topic": topic["topic"],
                    "url": topic["url"],
                    "chunk_index": chunk.chunk_index,
                },
            ))

    logger.info("patient_ingest_done", total_chunks=len(chunk_entries))
    return len(chunk_entries)


if __name__ == "__main__":
    init_db()  # 确保表存在（建 vector 扩展 + 表 + BM25 索引）
    count = ingest()
    print(f"入库完成，共 {count} 个 chunk")
