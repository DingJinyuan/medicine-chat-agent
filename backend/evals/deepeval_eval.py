"""DeepEval-based RAG quality evaluation: faithfulness + answer relevancy.

和 run_evals.py（规则式安全检查）互补——这里用 LLM-as-judge 测量**生成质量**，
回答是否忠实于检索内容、是否切题。需要 pip install deepeval + .env 里的 DEEPSEEK_API_KEY。

运行：python -m evals.deepeval_eval
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.security import DISCLAIMER  # noqa: E402
from app.triage_classifier import get_triage_classifier  # noqa: E402
from app.vector_store import build_or_load_vector_store  # noqa: E402

from deepeval import evaluate  # noqa: E402
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric  # noqa: E402
from deepeval.models import DeepSeekModel  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

# 复用 golden_dataset 里的 groundedness 用例（真实调用 pipeline 拿回答 + 检索上下文）
CASES = [
    {
        "question": "What are common symptoms of diabetes?",
        "corpus": [
            {"topic": "Diabetes Facts", "text": "Diabetes symptoms include increased thirst, frequent urination, unexplained weight loss, and blurry vision. Type 2 diabetes develops when the body cannot use insulin properly."}
        ],
    },
    {
        "question": "What can trigger a migraine?",
        "corpus": [
            {"topic": "Migraine Facts", "text": "Common migraine triggers include stress, certain foods, hormonal changes, and lack of sleep. Migraines often cause throbbing pain and sensitivity to light and sound."}
        ],
    },
    {
        "question": "How can seasonal allergies affect someone with asthma?",
        "corpus": [
            {"topic": "Asthma Facts", "text": "Asthma causes wheezing, shortness of breath, and chest tightness, especially when triggered by allergens."},
            {"topic": "Allergy Facts", "text": "Seasonal allergies are commonly triggered by pollen and can worsen asthma symptoms in people who have both conditions."},
        ],
    },
]


def _strip_disclaimer(text: str) -> str:
    """评估前剥离强制追加的免责声明，让相关性反映真实生成质量而非安全设计"""
    return text.replace(DISCLAIMER, "").strip()


def run_case(settings, triage_classifier, case):
    documents = [Document(page_content=c["text"], metadata={"topic": c["topic"], "url": ""}) for c in case["corpus"]]
    with tempfile.TemporaryDirectory() as tmp:
        vector_store = build_or_load_vector_store(str(Path(tmp) / "index"), settings.embedding_model, documents=documents)
        graph = build_graph(settings, vector_store, triage_classifier)
        result = graph.invoke({"question": case["question"]})

    return LLMTestCase(
        input=case["question"],
        actual_output=_strip_disclaimer(result["answer"]),
        retrieval_context=[s["text"] for s in result.get("sources", [])],
    )


def main() -> int:
    settings = get_settings()
    if not settings.llm_model_id:
        print("LLM_MODEL_ID is not set -- set it in .env and retry.")
        return 1

    triage_classifier = get_triage_classifier(settings.triage_adapter_path, settings.triage_base_model)

    # 用 .env 里的 DeepSeek 模型当 judge（LLM-as-judge）
    judge = DeepSeekModel(model=settings.llm_model_id)

    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7),
        AnswerRelevancyMetric(model=judge, threshold=0.7),
    ]

    test_cases = [run_case(settings, triage_classifier, c) for c in CASES]

    for tc in test_cases:
        faithfulness = FaithfulnessMetric(model=judge, threshold=0.7)
        relevancy = AnswerRelevancyMetric(model=judge, threshold=0.7)
        faithfulness.measure(tc)
        relevancy.measure(tc)
        print(f"{tc.input[:50]!r} -> 忠实度 {faithfulness.score:.2f} / 相关性 {relevancy.score:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
