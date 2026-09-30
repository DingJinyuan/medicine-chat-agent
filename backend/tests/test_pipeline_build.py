"""L2 两版 pipeline 构建 + 控制流测试(mock 依赖,离线零模型零 DB)。

- 专家版:build 编译 + 线性节点齐全
- 患者版:build 编译 + 全节点齐全 + 应急分支真实 invoke(短路跳过 LLM)+ 低置信兜底
"""
from src.agents.expert_pipeline import build_pipeline as build_expert
from src.agents.patient_pipeline import build_pipeline as build_patient
from src.agents.patient_guardrail_agent import route_after_triage
from src.config import settings

from fakes import FakeTriageClassifier


def _nodes(graph):
    try:
        return list(graph.get_graph().nodes.keys())
    except Exception:
        return list(graph.nodes.keys())


def test_expert_build_has_linear_nodes(mock_client, mock_embedder, mock_reranker, fake_risk_tool):
    g = build_expert(mock_client, mock_embedder, mock_reranker, fake_risk_tool, mock_client)
    nodes = _nodes(g)
    assert "query_understanding" in nodes
    assert "retrieval" in nodes
    assert "reasoning" in nodes
    assert "critique" in nodes


def test_patient_build_has_all_nodes(mock_client, mock_embedder, fake_triage_classifier):
    g = build_patient(settings, mock_embedder, fake_triage_classifier, mock_client)
    nodes = _nodes(g)
    for n in ["classify_triage", "emergency_shortcut", "retrieve", "generate", "output_guardrail"]:
        assert n in nodes


def test_patient_emergency_branch_skips_llm(mock_client, mock_embedder):
    triage = FakeTriageClassifier(label="emergency", confidence=0.9)
    g = build_patient(settings, mock_embedder, triage, mock_client)
    res = g.invoke({"question": "I have severe chest pain"})
    ans = res["answer"].lower()
    assert "emergency" in ans or "911" in ans
    # 应急分支短路,LLM 完全不被调用
    assert mock_client.invoke_calls == []


def test_patient_low_confidence_falls_back(mock_client, mock_embedder):
    triage = FakeTriageClassifier(label="routine", confidence=0.2)
    g = build_patient(settings, mock_embedder, triage, mock_client)
    res = g.invoke({"question": "feeling unwell"})
    ans = res["answer"].lower()
    assert "emergency" in ans or "911" in ans
    assert mock_client.invoke_calls == []


def test_route_normal_direction():
    # normal 分支(routine@0.9)走 retrieve,需真实 DB,这里只验证路由方向
    assert route_after_triage({"triage_label": "routine", "triage_confidence": 0.9}) == "retrieve"
