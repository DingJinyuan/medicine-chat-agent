"""L1 患者版生成 service 测试(消息组装 + 异常降级,离线 mock)。"""
from src.service.patient_llm_service import (
    build_rag_messages,
    generate_answer,
    FALLBACK_ANSWER,
)


class TestBuildRagMessages:
    def test_has_system_and_user(self):
        msgs = build_rag_messages("What is diabetes?", ["ctx1"])
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "What is diabetes?" in msgs[1]["content"]
        assert "ctx1" in msgs[1]["content"]

    def test_empty_context_placeholder(self):
        msgs = build_rag_messages("q", [])
        assert "no relevant reference" in msgs[1]["content"].lower()


class TestGenerateAnswer:
    def test_success_returns_content(self, mock_client):
        mock_client.content = "generated answer"
        assert generate_answer(mock_client, "q", ["ctx"]) == "generated answer"

    def test_failure_returns_fallback(self, mock_client, monkeypatch):
        def boom(*a, **k):
            raise Exception("llm down")

        monkeypatch.setattr(mock_client, "invoke", boom)
        assert generate_answer(mock_client, "q", ["ctx"]) == FALLBACK_ANSWER
