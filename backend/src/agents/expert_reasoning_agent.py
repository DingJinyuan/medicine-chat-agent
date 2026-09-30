import structlog
from src.agents.expert_state import AgentState
from src.service.expert_reasoning_service import generate_clinical_answer

logger = structlog.get_logger(__name__)


def make_reasoning_agent(client, risk_tool):
    """Reasoning Agent 工厂：注入 LLM 客户端与风险工具，返回节点函数。"""
    def node(state: AgentState) -> AgentState:
        question = state["question"]
        chunks = state["retrieved_chunks"]
        patient_profile = state["patient_profile"]

        logger.info("reasoning_start")

        answer, citations = generate_clinical_answer(question, chunks, patient_profile, client, risk_tool)

        return {**state, "answer": answer, "citations": citations}

    return node
