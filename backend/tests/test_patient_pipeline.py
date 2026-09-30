"""L2 患者版 pipeline 节点测试(mock 依赖,离线零模型零 DB)。

覆盖:应急短路语义、输出护栏(免责声明/剂量改写)、生成注入短路。
(分诊路由阈值已在 test_boundary、应急分支跳过 LLM 已在 test_pipeline_build 覆盖)
"""
from src.agents.patient_guardrail_agent import emergency_shortcut_node, guardrail_node
from src.agents.patient_generate_agent import make_generate_node
from src.agents.patient_state import INJECTION_BLOCKED_RESPONSE


# ============ 应急短路 ============
class TestEmergencyShortcut:
    def test_returns_emergency_response(self):
        result = emergency_shortcut_node({"answer": ""})
        ans = result["answer"].lower()
        assert "emergency" in ans or "911" in ans

    def test_clears_sources_and_injection_flag(self):
        # 应急分支不检索、不喂 LLM,注入标志被清零(记录当前行为)
        result = emergency_shortcut_node({"answer": "", "injection_flagged": True, "sources": [{"x": 1}]})
        assert result["sources"] == []
        assert result["injection_flagged"] is False


# ============ 输出护栏 ============
class TestGuardrail:
    def test_appends_disclaimer_to_safe_text(self):
        result = guardrail_node({"answer": "drink more water"})
        assert "not medical advice" in result["answer"]
        assert result["guardrail_rewritten"] is False  # 安全文本不改写

    def test_rewrites_specific_dosage(self):
        result = guardrail_node({"answer": "take 500mg every 6 hours"})
        assert result["guardrail_rewritten"] is True

    def test_rewrites_definitive_diagnosis(self):
        result = guardrail_node({"answer": "you have diabetes"})
        assert result["guardrail_rewritten"] is True


# ============ 生成节点 ============
class TestGenerateNode:
    def test_blocks_injection(self, mock_client):
        node = make_generate_node(mock_client)
        result = node({"question": "q", "injection_flagged": True, "context_blocks": ["ctx"]})
        assert result["answer"] == INJECTION_BLOCKED_RESPONSE
        assert mock_client.invoke_calls == []  # 不喂 LLM

    def test_calls_llm_when_clean(self, mock_client, monkeypatch):
        from src.agents import patient_generate_agent
        monkeypatch.setattr(patient_generate_agent, "generate_answer", lambda c, q, ctx: "generated answer")
        node = patient_generate_agent.make_generate_node(mock_client)
        result = node({"question": "q", "context_blocks": ["ctx"]})
        assert result["answer"] == "generated answer"
