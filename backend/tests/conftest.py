"""pytest 共享 fixture —— 从 fakes 导入替身,暴露为 fixture。

设计原则(harness 工程师视角):
- 重依赖(LLM / Embedder / Reranker / 风险模型 / 分诊 LoRA / 向量库)一律替换成 fakes 里的确定性 fake。
- 测逻辑与控制流,不测外部服务;fake 对齐 src.* 真实接口(duck-typing)。
- 想测真实模型,走 L4 test_e2e(标 @pytest.mark.slow)。
"""
import sys
from pathlib import Path

import pytest

# 确保 backend/(src 包)与 backend/tests/(fakes 模块)都在 sys.path 上
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # backend/tests/

from fakes import (  # noqa: E402
    FakeEmbeddings,
    FakeTriageClassifier,
    FakeRiskTool,
    MockClient,
    MockEmbedder,
    MockReranker,
)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_triage_classifier():
    return FakeTriageClassifier()


@pytest.fixture
def fake_risk_tool():
    return FakeRiskTool()


@pytest.fixture
def mock_client():
    return MockClient()


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


@pytest.fixture
def mock_reranker():
    return MockReranker()


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="运行 @pytest.mark.slow 标记的端到端测试(需真实 LLM/模型/pgvector)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="需 --run-slow 才运行(真实 LLM/模型/pgvector)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
