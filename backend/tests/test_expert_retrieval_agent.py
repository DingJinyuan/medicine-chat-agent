"""L2 专家版检索节点测试(离线 mock 掉 multi_route_retrieve)。"""
from types import SimpleNamespace

from src.agents import expert_retrieval_agent
from src.agents.expert_retrieval_agent import make_retrieval_agent


def test_retrieval_agent_returns_chunks(monkeypatch):
    chunks = [{"id": 1, "content": "c", "score": 0.9}]
    monkeypatch.setattr(expert_retrieval_agent, "multi_route_retrieve",
                        lambda q, rq, kw, emb, rer: chunks)
    node = make_retrieval_agent(SimpleNamespace(), SimpleNamespace())
    result = node({"question": "q", "rewritten_query": "rq", "keywords": ["k"]})
    assert result["retrieved_chunks"] == chunks


def test_retrieval_agent_empty_chunks_no_crash(monkeypatch):
    monkeypatch.setattr(expert_retrieval_agent, "multi_route_retrieve",
                        lambda q, rq, kw, emb, rer: [])
    node = make_retrieval_agent(SimpleNamespace(), SimpleNamespace())
    result = node({"question": "q", "rewritten_query": "rq", "keywords": []})
    assert result["retrieved_chunks"] == []
