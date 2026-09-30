import structlog
from src.agents.expert_state import AgentState
from src.service.expert_retrieval_service import multi_route_retrieve

logger = structlog.get_logger(__name__)


def make_retrieval_agent(embedder, reranker):
    """Retrieval Agent 工厂：注入嵌入器与精排器，返回节点函数。

    多路混合召回：BM25 + cosine 各自粗召回 → RRF 合并 → cross-encoder 精排。
    """
    def node(state: AgentState) -> AgentState:
        question = state["question"]
        rewritten_query = state["rewritten_query"]
        keywords = state["keywords"]

        logger.info("retrieval_start")

        retrieved_chunks = multi_route_retrieve(question, rewritten_query, keywords, embedder, reranker)

        logger.info("retrieval_done", chunk_count=len(retrieved_chunks))
        logger.info("retrieval_top_score", score=retrieved_chunks[0]['score'] if retrieved_chunks else 0)

        return {**state, "retrieved_chunks": retrieved_chunks}

    return node
