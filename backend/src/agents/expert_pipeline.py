import os
import structlog
from langgraph.graph import StateGraph, END
from src.config import settings
from src.agents.expert_state import AgentState
from src.agents.expert_query_understanding_agent import make_query_understanding_agent
from src.agents.expert_retrieval_agent import make_retrieval_agent
from src.agents.expert_reasoning_agent import make_reasoning_agent
from src.agents.expert_critique_agent import make_critique_agent

logger = structlog.get_logger(__name__)

# LangSmith追踪环境变量（不能丢）
os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


def build_pipeline(client, embedder, reranker, risk_tool, reasoning_llm):
    """
    构建 LangGraph 多 Agent 临床 RAG 流水线（依赖由调用方注入）。
    执行顺序：query_understanding → retrieval → reasoning → critique → END

    reasoning 节点用本地微调模型（reasoning_llm），其余节点走 API（client）。
    """
    graph = StateGraph(AgentState)

    graph.add_node("query_understanding", make_query_understanding_agent(client))
    graph.add_node("retrieval", make_retrieval_agent(embedder, reranker))
    graph.add_node("reasoning", make_reasoning_agent(reasoning_llm, risk_tool))
    graph.add_node("critique", make_critique_agent(client))

    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "critique")
    graph.add_edge("critique", END)

    return graph.compile()
