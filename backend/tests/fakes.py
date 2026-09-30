"""测试替身(独立模块,conftest 与各 test 文件共用)。

设计原则(harness 工程师视角):
- 所有 fake 对齐 src.* 的真实接口(duck-typing),测逻辑与控制流,不测外部服务。
- 确定性、离线、零网络、零真实模型。
"""
import re
from types import SimpleNamespace

import numpy as np

from src.ml.patient_triage_classifier import TriageResult

DIM = 768  # 与 settings.embedding_dimension 对齐(pubmedbert-base-embeddings)


class FakeEmbeddings:
    """确定性、零网络的嵌入替身(哈希词袋)。"""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[hash(word) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeTriageClassifier:
    """确定性分诊替身:关键词路由,或固定标签,不加载 torch / LoRA。"""

    EMERGENCY_KEYWORDS = (
        "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
        "stroke", "self-harm", "suicide", "ending my life", "slurred speech",
    )

    def __init__(self, label: str = None, confidence: float = None):
        """可选固定标签;不传则按关键词路由。"""
        self._fixed = None if label is None else (label, confidence)

    def classify(self, text: str) -> TriageResult:
        if self._fixed is not None:
            label, conf = self._fixed
            return TriageResult(label=label, confidence=conf)
        t = text.lower()
        if any(k in t for k in self.EMERGENCY_KEYWORDS):
            return TriageResult(label="emergency", confidence=0.95)
        return TriageResult(label="routine", confidence=0.8)


class FakeRiskTool:
    """风险模型替身:不加载 XGBoost,返回固定分层结果。"""

    def __init__(self, available: bool = True, risk_score: float = 0.5):
        self._available = available
        self._risk_score = risk_score
        self.predict_calls: list[dict] = []

    def is_available(self) -> bool:
        return self._available

    def predict(self, patient_profile: dict) -> dict:
        self.predict_calls.append(patient_profile)
        if not self._available:
            return {"error": "not available", "risk_score": None, "risk_level": None}
        level = "HIGH" if self._risk_score >= 0.6 else ("MODERATE" if self._risk_score >= 0.3 else "LOW")
        return {
            "risk_score": round(self._risk_score, 4),
            "risk_level": level,
            "interpretation": "",
            "features_used": patient_profile,
        }


class MockClient:
    """LLM 客户端替身:invoke / invoke_structured 记录调用并返回可配置内容。"""

    def __init__(self, content: str = "mock", tool_calls=None, structured=None):
        self.content = content
        self.tool_calls = tool_calls
        self.structured = structured
        self.invoke_calls: list = []
        self.structured_calls: list = []

    def invoke(self, messages=None, **kw):
        self.invoke_calls.append(kw)
        return SimpleNamespace(content=self.content, tool_calls=self.tool_calls)

    def invoke_structured(self, **kw):
        self.structured_calls.append(kw)
        return self.structured


class MockEmbedder:
    """极简嵌入替身:只提供 embed_query,返回固定向量。"""

    def embed_query(self, q: str) -> list[float]:
        return [0.0] * DIM


class MockReranker:
    """精排替身:保持原顺序截断 top_k。"""

    def rerank(self, q: str, chunks, top_k: int = 5):
        return list(chunks)[:top_k]
