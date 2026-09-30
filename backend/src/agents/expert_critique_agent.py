import structlog
from src.agents.expert_state import AgentState
from src.service.expert_critique_service import run_fact_check

logger = structlog.get_logger(__name__)


def make_critique_agent(client):
    """Critique Agent 工厂：注入 LLM 客户端，返回节点函数。"""
    def node(state: AgentState) -> AgentState:
        question = state["question"]
        answer = state["answer"]
        chunks = state["retrieved_chunks"]

        logger.info("critique_start")

        critique, confidence, reliable = run_fact_check(question, answer, chunks, client)

        return {
            **state,
            "critique": critique,
            "confidence_score": confidence,
            "is_reliable": reliable,
        }

    return node
