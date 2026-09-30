"""② DeepEval 评估(LLM-as-judge,完整 RAG 指标集)。

患者版(无 ground_truth):Faithfulness + AnswerRelevancy + ContextualRelevancy
专家版(有 ground_truth):Faithfulness + AnswerRelevancy + ContextualPrecision + ContextualRecall

运行:cd backend && python -m evaluation.deepeval_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from src.config import settings
from src.ingestion.embedder import Embedder
from src.ml.patient_triage_classifier import get_triage_classifier
from src.ml.expert_risk_tool import ReadmissionRiskTool
from src.service.expert_rerank import Reranker
from src.common.llm_adapter import create_client
from src.agents.patient_pipeline import build_pipeline as build_patient
from src.agents.expert_pipeline import build_pipeline as build_expert
from src.service.patient_security import DISCLAIMER

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.models import DeepSeekModel  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

PATIENT_CASES = [
    {"question": "What are common symptoms of diabetes?"},
    {"question": "What can trigger a migraine?"},
    {"question": "What are the symptoms of asthma?"},
]


def _strip_disclaimer(text: str) -> str:
    return text.replace(DISCLAIMER, "").strip()


def _load_expert_cases() -> list[dict]:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "eval"
    return json.loads((data_dir / "expert_dataset.json").read_text(encoding="utf-8"))


def _measure(tc, judge, metric_classes) -> dict:
    scores = {}
    for cls in metric_classes:
        m = cls(model=judge, threshold=0.7)
        m.measure(tc)
        scores[cls.__name__] = round(m.score, 2)
    return scores


def main() -> int:
    client = create_client(settings)
    embedder = Embedder()
    triage = get_triage_classifier(settings.triage_adapter_path, settings.triage_base_model)
    patient = build_patient(settings, embedder, triage, client)
    reranker = Reranker()
    risk_tool = ReadmissionRiskTool()
    expert = build_expert(client, embedder, reranker, risk_tool, client)

    judge = DeepSeekModel(model=settings.llm_model_id)
    results = []

    # ===== 患者版 =====
    for case in PATIENT_CASES:
        result = patient.invoke({"question": case["question"]})
        tc = LLMTestCase(
            input=case["question"],
            actual_output=_strip_disclaimer(result["answer"]),
            retrieval_context=[s["text"] for s in result.get("sources", [])],
        )
        s = _measure(tc, judge, [FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric])
        print(f"[patient] {case['question'][:40]!r}: faithfulness={s['FaithfulnessMetric']} "
              f"relevancy={s['AnswerRelevancyMetric']} contextual_relevancy={s['ContextualRelevancyMetric']}")
        results.append({"mode": "patient", "question": case["question"], **s})

    # ===== 专家版 =====
    for case in _load_expert_cases():
        result = expert.invoke({"question": case["question"]})
        tc = LLMTestCase(
            input=case["question"],
            actual_output=result.get("answer", ""),
            retrieval_context=[c.get("content", "") for c in result.get("retrieved_chunks", [])],
            expected_output=case["ground_truth"],
        )
        s = _measure(tc, judge, [
            FaithfulnessMetric, AnswerRelevancyMetric,
            ContextualPrecisionMetric, ContextualRecallMetric,
        ])
        print(f"[expert] {case['question'][:40]!r}: faithfulness={s['FaithfulnessMetric']} "
              f"relevancy={s['AnswerRelevancyMetric']} "
              f"contextual_precision={s['ContextualPrecisionMetric']} contextual_recall={s['ContextualRecallMetric']}")
        results.append({"mode": "expert", "question": case["question"], **s})

    # 写报告(JSON,避免终端编码问题)
    report = Path(__file__).parent / "last_deepeval_report.json"
    report.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
