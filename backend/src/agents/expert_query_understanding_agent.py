import structlog
from src.agents.expert_state import AgentState
from src.service.expert_query_understanding_service import understand_query
from src.ml.expert_risk_tool import extract_patient_profile_from_query

logger = structlog.get_logger(__name__)


def make_query_understanding_agent(client):
    """Query Understanding Agent 工厂：注入 LLM 客户端，返回节点函数。

    LLM 负责开放式提取（改写问题 / 检索关键词），
    患者画像用关键词匹配（extract_patient_profile_from_query）对照 7 类共病 schema。
    """
    def node(state: AgentState) -> AgentState:
        question = state["question"]
        logger.info("query_understanding_start")

        form = understand_query(question, client)
        patient_profile = extract_patient_profile_from_query(question)

        logger.info("query_understood", rewritten_query=form.rewritten_query[:60])
        logger.info("patient_profile_extracted", profile=patient_profile)

        return {
            **state,
            "rewritten_query": form.rewritten_query,
            "patient_profile": patient_profile,
            "keywords": form.keywords,
        }

    return node
