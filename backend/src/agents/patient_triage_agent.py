"""分诊分类节点：调用 LoRA 分诊分类器，输出标签与置信度。"""
from src import metrics as m
from src.agents.patient_state import ChatState
from src.ml.patient_triage_classifier import TriageClassifier


def make_triage_node(triage_classifier: TriageClassifier):
    """构造【分诊分类】节点闭包，注入外部分类器实例。

    Args:
        triage_classifier: 预初始化的分诊分类器实例

    Returns:
        node: LangGraph 可执行节点函数，输入输出均为 ChatState
    """
    def node(state: ChatState) -> ChatState:
        result = triage_classifier.classify(state["question"])
        # 统计每种分诊标签出现次数，用于监控
        m.TRIAGE_LABELS.labels(label=result.label).inc()
        return {**state, "triage_label": result.label, "triage_confidence": result.confidence}

    return node
