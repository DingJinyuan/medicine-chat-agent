from app import llm_client
from app.config import Settings


def test_build_rag_messages_includes_context_and_question():
    messages = llm_client.build_rag_messages("What is diabetes?", ["Diabetes is high blood sugar."])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == llm_client.SYSTEM_PROMPT
    assert "Diabetes is high blood sugar." in messages[1]["content"]
    assert "What is diabetes?" in messages[1]["content"]


def test_build_rag_messages_handles_empty_context():
    messages = llm_client.build_rag_messages("What is diabetes?", [])
    assert "no relevant reference material" in messages[1]["content"]


class _FakeMessage:
    def __init__(self):
        self.content = "Diabetes info."


class _FakeClient:
    def __init__(self, **kwargs):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def invoke(self, messages, **kwargs):
        return _FakeMessage()


class _FailingClient:
    def __init__(self, **kwargs):
        pass

    def invoke(self, messages, **kwargs):
        raise RuntimeError("all models failed")


def test_generate_answer_returns_content_on_success(monkeypatch):
    monkeypatch.setattr(llm_client, "LLMClient", _FakeClient)
    settings = Settings(llm_model_id="deepseek-chat")
    assert llm_client.generate_answer(settings, "What is diabetes?", ["context"]) == "Diabetes info."


def test_generate_answer_returns_fallback_on_total_failure(monkeypatch):
    monkeypatch.setattr(llm_client, "LLMClient", _FailingClient)
    settings = Settings(llm_model_id="deepseek-chat")
    assert llm_client.generate_answer(settings, "What is diabetes?", ["context"]) == llm_client.FALLBACK_ANSWER
