"""L4 端到端真实 query 测试(真实 deepseek LLM + 本地模型 + pgvector)。

默认跳过(@pytest.mark.slow),用 `pytest tests/test_e2e.py --run-slow` 手动回归。
"""
import os

import pytest

from src.config import settings

if settings.hf_home:
    os.environ["HF_HOME"] = settings.hf_home

from src.common.llm_adapter import create_client
from src.ingestion.embedder import Embedder
from src.service.expert_rerank import Reranker
from src.ml.expert_risk_tool import ReadmissionRiskTool
from src.ml.expert_reasoning_model import get_reasoning_model
from src.ml.patient_triage_classifier import get_triage_classifier
from src.agents.expert_pipeline import build_pipeline as build_expert
from src.agents.patient_pipeline import build_pipeline as build_patient


@pytest.fixture(scope="module")
def graphs():
    """构建两版真实 pipeline(慢:加载模型 / 连库)。"""
    client = create_client(settings)
    embedder = Embedder()
    reranker = Reranker()
    risk_tool = ReadmissionRiskTool()
    reasoning_llm = client
    if settings.reasoning_use_local:
        rm = get_reasoning_model(settings.reasoning_model_base_model, settings.reasoning_model_adapter_path)
        reasoning_llm = rm if rm else client
    expert = build_expert(client, embedder, reranker, risk_tool, reasoning_llm)
    triage = get_triage_classifier(settings.triage_adapter_path, settings.triage_base_model)
    patient = build_patient(settings, embedder, triage, client)
    return expert, patient


@pytest.mark.slow
def test_expert_query(graphs):
    expert, _ = graphs
    q = "72 year old male with heart failure and diabetes, emergency admission for 5 days, is metformin still safe?"
    r = expert.invoke({"question": q})
    assert r.get("answer"), "专家版应返回最终回答"
    assert r.get("citations"), "专家版应返回引用"


@pytest.mark.slow
def test_patient_normal(graphs):
    _, patient = graphs
    r = patient.invoke({"question": "I have a mild headache and a slight fever, what should I do?"})
    assert r.get("answer"), "患者版应返回回答"


@pytest.mark.slow
def test_patient_emergency(graphs):
    _, patient = graphs
    r = patient.invoke({"question": "I have severe chest pain, difficulty breathing, and cold sweats"})
    assert r.get("answer"), "紧急分支应返回回答"
    assert r.get("triage_label") in ("emergency", "urgent")  # 紧急/急症都算,不纠结标签
