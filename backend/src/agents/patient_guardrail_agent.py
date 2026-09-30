"""应急短路 + 条件路由 + 输出护栏（无依赖纯函数）。"""
from src.agents.patient_state import (
    ChatState,
    EMERGENCY_RESPONSE,
    EMERGENCY_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)
from src.service.patient_security import enforce_medical_guardrails, redact_secrets


def route_after_triage(state: ChatState) -> str:
    """分诊完成后的条件路由，LangGraph conditional_edge 使用。

    分支规则：
        1. 标签=emergency 且置信度>=紧急阈值 → emergency_shortcut
        2. 任意标签，置信度低于低置信阈值 → 保守兜底，走应急短路
        3. 其余情况走正常 RAG 流程 → retrieve
    """
    if state["triage_label"] == "emergency" and state["triage_confidence"] >= EMERGENCY_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"
    if state["triage_confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"   # 任何标签置信度太低，走保守兜底
    return "retrieve"


def emergency_shortcut_node(state: ChatState) -> ChatState:
    """应急短路节点：跳过检索、跳过 LLM 调用，直接写入固定急救提示。

    关键点：LLM 完全不会接触用户输入，避免大模型处理高危医疗输入带来风险。
    """
    return {**state, "answer": EMERGENCY_RESPONSE, "context_blocks": [], "sources": [], "injection_flagged": False}


def guardrail_node(state: ChatState) -> ChatState:
    """输出护栏节点：两条分支都会经过此节点。

    对最终输出执行医疗安全规则校验与改写：
    - 移除给出诊断、给出具体用药剂量的内容
    - 追加医疗免责声明
    """
    # 先脱敏可能泄露的 API 密钥，再执行医疗安全护栏
    redacted = redact_secrets(state["answer"])
    result = enforce_medical_guardrails(redacted)
    return {**state, "answer": result.text, "guardrail_rewritten": result.rewritten}
