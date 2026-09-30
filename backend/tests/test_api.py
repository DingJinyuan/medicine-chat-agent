"""L3 FastAPI 路由测试(monkeypatch lifespan 重依赖 + pipeline,聚焦 API 层)。

覆盖:/health、/api/chat 双版路由、422 边界、429 限流、history、metrics 无鉴权。
pipeline 内部逻辑已由 L2 测试覆盖,这里 mock 掉 invoke 只测 API 层。
"""
import pytest
from fastapi.testclient import TestClient

from fakes import MockClient, MockEmbedder, MockReranker, FakeRiskTool, FakeTriageClassifier
from src.common.security import RateLimiter


class _FakeGraph:
    def __init__(self, result):
        self.result = result

    def invoke(self, *a, **k):
        return self.result


class _FakeDB:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def execute(self, *a, **k):
        return None


@pytest.fixture
def client(monkeypatch):
    import src.main as main_module

    # 0. 关闭鉴权,聚焦路由测试(鉴权逻辑已在 test_security 覆盖)
    monkeypatch.setattr(main_module.settings, "app_api_key", "")

    # 1. monkeypatch lifespan 重依赖,避免加载真实模型 / 连库
    monkeypatch.setattr(main_module, "create_client", lambda settings: MockClient())
    monkeypatch.setattr(main_module, "Embedder", lambda: MockEmbedder())
    monkeypatch.setattr(main_module, "Reranker", lambda: MockReranker())
    monkeypatch.setattr(main_module, "ReadmissionRiskTool", lambda: FakeRiskTool())
    monkeypatch.setattr(main_module, "get_reasoning_model", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "get_triage_classifier", lambda *a, **k: FakeTriageClassifier())

    # 2. monkeypatch DB 相关
    monkeypatch.setattr(main_module, "get_db", lambda: _FakeDB())
    monkeypatch.setattr(main_module, "ensure_session", lambda sid, source: "session-1")
    monkeypatch.setattr(main_module, "add_message", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "get_history", lambda sid: [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ])

    # 3. 限流器给大额度,避免干扰其他测试(429 测试单独覆盖);注意保持单例语义
    _rl = RateLimiter(limit_per_minute=1000)
    monkeypatch.setattr(main_module, "get_rate_limiter", lambda: _rl)

    with TestClient(main_module.app) as c:
        # 4. mock 两版 pipeline 的 invoke,聚焦 API 层
        monkeypatch.setattr(main_module.app.state, "expert_graph", _FakeGraph({
            "answer": "expert answer",
            "citations": ["pubmed_1"],
            "confidence_score": 0.9,
            "is_reliable": True,
            "critique": "ok",
        }))
        monkeypatch.setattr(main_module.app.state, "patient_graph", _FakeGraph({
            "answer": "patient answer",
            "triage_label": "routine",
            "triage_confidence": 0.8,
            "sources": [],
            "guardrail_rewritten": False,
            "injection_flagged": False,
        }))
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_patient_normal(client):
    r = client.post("/api/chat", json={"message": "I have a headache", "mode": "patient"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "patient"
    assert body["answer"] == "patient answer"
    assert body["triage"]["label"] == "routine"


def test_chat_expert_returns_citations(client):
    r = client.post("/api/chat", json={"message": "is metformin safe?", "mode": "expert"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "expert"
    assert body["citations"] == ["pubmed_1"]
    assert body["confidence_score"] == 0.9


def test_chat_rejects_empty_message(client):
    r = client.post("/api/chat", json={"message": "", "mode": "patient"})
    assert r.status_code == 422


def test_chat_rejects_oversized_message(client):
    r = client.post("/api/chat", json={"message": "a" * 5000, "mode": "patient"})
    assert r.status_code == 422


def test_chat_rate_limit_429(client, monkeypatch):
    import src.main as main_module
    _rl = RateLimiter(limit_per_minute=1)
    monkeypatch.setattr(main_module, "get_rate_limiter", lambda: _rl)
    r1 = client.post("/api/chat", json={"message": "hi", "mode": "patient"})
    assert r1.status_code == 200
    r2 = client.post("/api/chat", json={"message": "hi", "mode": "patient"})
    assert r2.status_code == 429


def test_history(client):
    r = client.get("/api/chat/session-1/history")
    assert r.status_code == 200
    body = r.json()
    assert [m["role"] for m in body] == ["user", "assistant"]


def test_history_requires_auth_when_key_set(client, monkeypatch):
    # 配置了 API key 后,/history 也需要鉴权
    import src.main as main_module
    monkeypatch.setattr(main_module.settings, "app_api_key", "correct-key")
    r = client.get("/api/chat/session-1/history")
    assert r.status_code == 401  # 不带 key → 401
    r2 = client.get("/api/chat/session-1/history", headers={"X-API-Key": "correct-key"})
    assert r2.status_code == 200  # 带 key → 200


def test_metrics_no_auth_required(client):
    # 记录当前行为:/metrics 无鉴权(直接可访问)
    r = client.get("/metrics")
    assert r.status_code == 200
