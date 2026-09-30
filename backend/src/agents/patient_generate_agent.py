"""生成节点：调用 LLM 生成回答。"""
from src.agents.patient_state import ChatState, INJECTION_BLOCKED_RESPONSE
from src.service.patient_llm_service import generate_answer


def make_generate_node(client):
    """构造【LLM生成回答】节点闭包，注入 LLM 客户端。

    Args:
        client: LLM 客户端（调用方注入）

    Returns:
        node: LangGraph 节点函数
    """
    def node(state: ChatState) -> ChatState:
        # 检测到注入攻击：短路返回固定安全响应，不把用户输入/召回内容喂给 LLM
        if state.get("injection_flagged"):
            return {**state, "answer": INJECTION_BLOCKED_RESPONSE}
        answer = generate_answer(client, state["question"], state["context_blocks"])
        return {**state, "answer": answer}

    return node
