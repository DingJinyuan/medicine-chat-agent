"""L1 摄取测试:chunker 切块 + patient_ingest 幂等性(重复摄取 bug 回归)。"""
import json

import pytest

from src.ingestion.chunker import TextChunker
from src.ingestion import patient_ingest
from src.database.models import Chunk


# ============ chunker ============
class TestChunker:
    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk_text("Short text.", doc_id="d1")
        assert len(chunks) == 1
        assert chunks[0].doc_id == "d1"
        assert chunks[0].chunk_index == 0

    def test_long_text_continuous_index(self):
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        text = " ".join(f"Sentence number {i} about health." for i in range(40))
        chunks = chunker.chunk_text(text, doc_id="d1")
        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.token_count > 0 for c in chunks)

    def test_overlap_keeps_boundary_context(self):
        # 重叠机制:后一块开头应复用前一块末尾的句子,避免边界信息丢失
        chunker = TextChunker(chunk_size=40, chunk_overlap=10)
        text = " ".join(f"Sentence number {i} about health." for i in range(30))
        chunks = chunker.chunk_text(text, doc_id="d1")
        if len(chunks) > 1:
            last_words_prev = set(chunks[0].content.split())
            first_words_next = set(chunks[1].content.split())
            assert last_words_prev & first_words_next  # 有交集 = 有重叠


# ============ patient_ingest 幂等性 ============
class _FakeQuery:
    def __init__(self, existing):
        self._existing = existing

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._existing


class _FakeSession:
    def __init__(self, existing_doc_ids):
        self._existing = [(doc_id, 1) for doc_id in existing_doc_ids]
        self.added_docs = []
        self.added_chunks = []

    def query(self, *cols):
        return _FakeQuery(self._existing)

    def add(self, obj):
        if isinstance(obj, Chunk):
            self.added_chunks.append(obj)
        else:
            self.added_docs.append(obj)

    def flush(self):
        pass


class _FakeDB:
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, *a):
        pass


class _FakeEmbedder:
    def embed_chunks(self, chunks, batch_size=32):
        return [[0.0] * 768 for _ in chunks]


def _write_topic_json(tmp_path):
    data = [{"topic": "Test Topic", "url": "https://example.com", "summary": "This is a test summary about a medical topic."}]
    p = tmp_path / "topics.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_patient_ingest_idempotent(tmp_path, monkeypatch):
    """第二次摄取同一主题不应重复写 chunk(幂等)。"""
    monkeypatch.setattr(patient_ingest, "Embedder", _FakeEmbedder)
    json_path = _write_topic_json(tmp_path)

    # 第一次:无已存在 doc → 正常写入 Document + Chunk
    session1 = _FakeSession(existing_doc_ids=[])
    monkeypatch.setattr(patient_ingest, "get_db", lambda: _FakeDB(session1))
    count1 = patient_ingest.ingest(json_path=json_path)
    assert count1 >= 1
    assert len(session1.added_chunks) == count1

    # 第二次:doc_id 已存在 → 期望不再新增任何 chunk
    doc_id = patient_ingest._make_doc_id("Test Topic")
    session2 = _FakeSession(existing_doc_ids=[doc_id])
    monkeypatch.setattr(patient_ingest, "get_db", lambda: _FakeDB(session2))
    patient_ingest.ingest(json_path=json_path)
    # 幂等期望:第二次不应新增 chunk(当前 bug 会重复 add)
    assert len(session2.added_chunks) == 0
