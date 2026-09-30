"""L1 精排器测试(懒加载 + 异常降级 + 排序,离线 mock)。"""
from types import SimpleNamespace

from src.service.expert_rerank import Reranker


def _raise(exc):
    raise exc


class TestReranker:
    def test_empty_chunks(self):
        r = Reranker()
        assert r.rerank("q", [], top_k=5) == []

    def test_fallback_on_predict_error(self, monkeypatch):
        r = Reranker()
        monkeypatch.setattr(r, "_load", lambda: None)
        r.model = SimpleNamespace(predict=lambda pairs: _raise(Exception("predict fail")))
        chunks = [{"id": 1, "content": "a", "score": 0.5}, {"id": 2, "content": "b", "score": 0.4}]
        result = r.rerank("q", chunks, top_k=5)
        assert result == chunks  # 降级为原顺序

    def test_ranks_and_rewrites_score(self, monkeypatch):
        r = Reranker()
        monkeypatch.setattr(r, "_load", lambda: None)
        r.model = SimpleNamespace(predict=lambda pairs: [0.1, 0.9])  # 第二个分数更高
        chunks = [{"id": 1, "content": "a", "score": 0.5}, {"id": 2, "content": "b", "score": 0.4}]
        result = r.rerank("q", chunks, top_k=1)
        assert result[0]["id"] == 2
        assert result[0]["score"] == 0.9
