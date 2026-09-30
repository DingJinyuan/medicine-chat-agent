"""L1 患者版检索 service 测试(BM25 降级,离线 mock)。"""
from types import SimpleNamespace

from src.service import patient_retrieval_service as svc


def _raise(exc):
    raise exc


def test_retrieve_medlineplus_bm25_fallback(monkeypatch):
    # BM25 未部署 → 降级为纯 cosine[:top_k]
    monkeypatch.setattr(svc, "cosine_search_medlineplus",
                        lambda emb, k: [{"id": 1, "content": "c", "topic": "t", "url": "u", "score": 0.9}])
    monkeypatch.setattr(svc, "bm25_search_medlineplus", lambda q, k: _raise(Exception("paradedb missing")))
    embedder = SimpleNamespace(embed_query=lambda q: [0.0] * 768)
    result = svc.retrieve_medlineplus("q", embedder, top_k=5)
    assert len(result) == 1
    assert result[0]["id"] == 1
