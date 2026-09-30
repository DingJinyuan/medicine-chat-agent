"""L2 专家版 pipeline 节点测试(mock 依赖,离线零模型零 DB)。

覆盖:线性节点顺序、query_understanding 结构化提取/降级、reasoning 风险工具调用、
critique 置信度、以及核心遗漏点「数据隔离」(专家版检索排除 medlineplus_)。
"""
import json
from types import SimpleNamespace

from src.agents.expert_pipeline import build_pipeline as build_expert
from src.agents.expert_query_understanding_agent import make_query_understanding_agent
from src.agents.expert_reasoning_agent import make_reasoning_agent
from src.agents.expert_critique_agent import make_critique_agent
from src.repository import chunk_repo


def _tool_call(arguments: dict):
    return SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(arguments)))


# ============ 线性顺序 ============
def test_expert_graph_linear_order(mock_client, mock_embedder, mock_reranker, fake_risk_tool):
    g = build_expert(mock_client, mock_embedder, mock_reranker, fake_risk_tool, mock_client)
    edge_pairs = {(e.source, e.target) for e in g.get_graph().edges}
    for src, dst in [
        ("query_understanding", "retrieval"),
        ("retrieval", "reasoning"),
        ("reasoning", "critique"),
    ]:
        assert (src, dst) in edge_pairs


# ============ query_understanding ============
class TestQueryUnderstanding:
    def test_extracts_structured(self, mock_client):
        mock_client.tool_calls = [_tool_call({
            "rewritten_query": "Is metformin safe in elderly patients with diabetes?",
            "keywords": ["metformin", "diabetes", "elderly"],
        })]
        node = make_query_understanding_agent(mock_client)
        result = node({"question": "is metformin safe for diabetes?"})
        assert result["rewritten_query"] == "Is metformin safe in elderly patients with diabetes?"
        assert result["keywords"] == ["metformin", "diabetes", "elderly"]
        assert result["patient_profile"].get("has_diabetes") == 1  # 画像提取(diabetes 关键词)

    def test_falls_back_on_no_tool_call(self, mock_client):
        mock_client.tool_calls = None  # LLM 没走 function calling
        node = make_query_understanding_agent(mock_client)
        result = node({"question": "is metformin safe?"})
        assert result["rewritten_query"] == "is metformin safe?"  # 降级为原问题
        assert result["keywords"] == []


# ============ reasoning ============
class TestReasoning:
    def test_calls_risk_tool_when_profile_present(self, mock_client, fake_risk_tool):
        mock_client.content = "clinical answer"
        node = make_reasoning_agent(mock_client, fake_risk_tool)
        chunks = [{"doc_id": "pubmed_1", "content": "some content", "score": 0.9}]
        profile = {"age": 72, "has_diabetes": 1}
        result = node({"question": "q", "retrieved_chunks": chunks, "patient_profile": profile})
        assert fake_risk_tool.predict_calls == [profile]  # 画像非空 → 调用风险工具
        assert result["citations"] == ["pubmed_1"]
        assert result["answer"] == "clinical answer"

    def test_skips_risk_tool_when_no_profile(self, mock_client, fake_risk_tool):
        node = make_reasoning_agent(mock_client, fake_risk_tool)
        chunks = [{"doc_id": "pubmed_1", "content": "some content", "score": 0.9}]
        result = node({"question": "q", "retrieved_chunks": chunks, "patient_profile": {}})
        assert fake_risk_tool.predict_calls == []  # 画像空 → 不调用


# ============ critique ============
class TestCritique:
    def test_outputs_confidence_and_reliable(self, mock_client):
        mock_client.tool_calls = [_tool_call({"critique": "well grounded", "confidence": 0.9, "reliable": True})]
        node = make_critique_agent(mock_client)
        chunks = [{"doc_id": "pubmed_1", "content": "content", "score": 0.9}]
        result = node({
            "question": "q", "answer": "answer",
            "retrieved_chunks": chunks, "citations": ["pubmed_1"],
        })
        assert result["confidence_score"] == 0.9
        assert result["is_reliable"] is True
        assert result["critique"] == "well grounded"

    def test_falls_back_on_no_tool_call(self, mock_client):
        mock_client.tool_calls = None
        node = make_critique_agent(mock_client)
        chunks = [{"doc_id": "pubmed_1", "content": "content", "score": 0.9}]
        result = node({
            "question": "q", "answer": "answer",
            "retrieved_chunks": chunks, "citations": ["pubmed_1"],
        })
        assert result["confidence_score"] == 0.5  # 降级默认值
        assert result["is_reliable"] is False

    def test_unreliable_has_prominent_warning(self, mock_client):
        # 增强展示:is_reliable=False 时,核查意见要醒目
        mock_client.tool_calls = [_tool_call({"critique": "claims not grounded", "confidence": 0.3, "reliable": False})]
        node = make_critique_agent(mock_client)
        chunks = [{"doc_id": "pubmed_1", "content": "content", "score": 0.9}]
        result = node({
            "question": "q", "answer": "answer",
            "retrieved_chunks": chunks, "citations": ["pubmed_1"],
        })
        assert result["is_reliable"] is False
        assert result["critique"] == "claims not grounded"


# ============ 数据隔离 ============
def test_semantic_search_excludes_medlineplus(monkeypatch):
    """专家版向量检索必须排除患者版 medlineplus_ 数据。"""
    captured = {}

    class _FakeQuery:
        def filter(self, *conds):
            captured["conds"] = conds
            return self

        def order_by(self, *a):
            return self

        def limit(self, n):
            return self

        def all(self):
            return []

    class _FakeSession:
        def query(self, *cols):
            return _FakeQuery()

    class _FakeDB:
        def __enter__(self):
            return _FakeSession()

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(chunk_repo, "get_db", lambda: _FakeDB())
    chunk_repo.semantic_search_chunks([0.0] * 768, top_k=20)

    cond = captured["conds"][0]
    cond_str = str(cond).lower()
    assert "not like" in cond_str  # 存在 NOT LIKE 过滤
    assert getattr(getattr(cond, "right", None), "value", "") == "medlineplus_%"
