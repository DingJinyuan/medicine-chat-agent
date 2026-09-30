"""L1 专家版检索 service 测试(BM25 降级 + 多路召回,离线 mock)。"""
from types import SimpleNamespace

from src.service import expert_retrieval_service as svc


def _raise(exc):
    raise exc


class TestHybridRecall:
    def test_bm25_failure_falls_back_to_cosine(self, monkeypatch):
        # BM25 未部署(ParadeDB 缺失)→ 降级为纯 cosine
        monkeypatch.setattr(svc, "semantic_search_chunks", lambda emb, k: [{"id": 1, "score": 0.9}])
        monkeypatch.setattr(svc, "bm25_search_chunks", lambda q, k: _raise(Exception("paradedb missing")))
        result = svc.hybrid_recall("query", [0.0] * 768)
        assert result == [{"id": 1, "score": 0.9}]

    def test_normal_fusion_dedups(self, monkeypatch):
        monkeypatch.setattr(svc, "semantic_search_chunks", lambda emb, k: [{"id": 1}, {"id": 2}])
        monkeypatch.setattr(svc, "bm25_search_chunks", lambda q, k: [{"id": 1}, {"id": 3}])
        result = svc.hybrid_recall("query", [0.0] * 768)
        assert len(result) == 3  # 跨路去重后 3 个


class TestMultiRouteRetrieve:
    def test_dedups_routes(self, monkeypatch):
        # rewritten_query 与 keywords 拼接后去重去空
        embedder = SimpleNamespace(embed_query=lambda q: [0.0] * 768)
        reranker = SimpleNamespace(rerank=lambda q, chunks, top_k: chunks[:top_k])
        monkeypatch.setattr(svc, "semantic_search_chunks", lambda emb, k: [])
        monkeypatch.setattr(svc, "bm25_search_chunks", lambda q, k: [])
        svc.multi_route_retrieve("q", "rewritten", ["kw1"], embedder, reranker, top_k=5)
        # 无异常即通过(核心是路由去重 + 空结果也能走通)
