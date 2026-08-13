import pytest
from fastapi.testclient import TestClient

from app import graph as graph_module
from app import main, security


@pytest.fixture
def client(tmp_path, monkeypatch, fake_embeddings, fake_triage_classifier, sample_documents):
    monkeypatch.setattr(main.settings, "db_path", str(tmp_path / "db.sqlite"))
    monkeypatch.setattr(main.settings, "vector_index_path", str(tmp_path / "index"))
    monkeypatch.setattr(security, "_rate_limiter", None)

    # Avoid real network (MedlinePlus fetch, Groq) and real model loads (fastembed, torch) in the default test run.
    monkeypatch.setattr(main, "load_or_fetch_corpus", lambda cache_path: [{"topic": "t", "url": "u", "summary": "s"}])
    monkeypatch.setattr(main, "build_documents", lambda topics, chunk_size, chunk_overlap: sample_documents)
    monkeypatch.setattr(
        main,
        "build_or_load_vector_store",
        lambda index_path, embedding_model, documents=None: __import__("app.vector_store", fromlist=["build_or_load_vector_store"]).build_or_load_vector_store(
            index_path, embedding_model, documents=documents, embeddings=fake_embeddings
        ),
    )
    monkeypatch.setattr(main, "get_triage_classifier", lambda adapter_path, base_model: fake_triage_classifier)
    monkeypatch.setattr(graph_module, "generate_answer", lambda settings, question, context_blocks: "Migraines can cause headaches with light sensitivity.")
    monkeypatch.setitem(main.app.dependency_overrides, security.require_api_key, lambda: None)

    with TestClient(main.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_answer_sources_and_triage(client):
    resp = client.post("/api/chat", json={"message": "What causes migraines with light sensitivity?"})
    assert resp.status_code == 200

    body = resp.json()
    assert "Migraines" in body["answer"]
    assert body["triage"]["label"] == "routine"
    assert body["sources"]
    assert "session_id" in body


def test_chat_emergency_message_short_circuits(client):
    resp = client.post("/api/chat", json={"message": "I have chest pain and can't breathe"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["triage"]["label"] == "emergency"
    assert "emergency" in body["answer"].lower()
    assert body["sources"] == []


def test_chat_persists_and_returns_history(client):
    resp = client.post("/api/chat", json={"message": "What causes migraines?"})
    session_id = resp.json()["session_id"]

    history = client.get(f"/api/chat/{session_id}/history")
    assert history.status_code == 200
    roles = [m["role"] for m in history.json()]
    assert roles == ["user", "assistant"]


def test_chat_rejects_oversized_message(client):
    resp = client.post("/api/chat", json={"message": "x" * 5000})
    assert resp.status_code == 422


def test_chat_rate_limit_returns_429_when_exceeded(client, monkeypatch):
    monkeypatch.setattr(security, "_rate_limiter", security.RateLimiter(limit_per_minute=1))

    first = client.post("/api/chat", json={"message": "one"})
    second = client.post("/api/chat", json={"message": "two"})

    assert first.status_code == 200
    assert second.status_code == 429


def test_chat_rejects_empty_message(client):
    resp = client.post("/api/chat", json={"message": ""})
    assert resp.status_code == 422
