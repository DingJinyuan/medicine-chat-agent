"""患者版（MediSense）LangGraph 工作流组装。

    classify_triage --conditional--> emergency_shortcut --> output_guardrail --> END
                     \\-------------> retrieve --> generate --> output_guardrail --> END

应急分支是真正的条件分支：高置信度 "emergency" 分诊预测会跳过检索和生成，
直接返回固定安全响应 —— LLM 完全不接触 emergency 标记的输入。
两条分支都汇聚到 output_guardrail，保证免责声明/剂量/诊断改写规则无条件生效。
"""
import os

from langgraph.graph import END, StateGraph

from src.config import Settings, settings
from src.agents.patient_state import ChatState
from src.agents.patient_triage_agent import make_triage_node
from src.agents.patient_retrieve_agent import make_retrieve_node
from src.agents.patient_generate_agent import make_generate_node
from src.agents.patient_guardrail_agent import (
    route_after_triage,
    emergency_shortcut_node,
    guardrail_node,
)
from src.ml.patient_triage_classifier import TriageClassifier

# LangSmith 追踪环境变量（不能丢）
os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


def build_pipeline(settings: Settings, embedder, triage_classifier: TriageClassifier, client):
    """构建并编译患者版医疗 RAG 状态图。

    Args:
        settings: 患者版全局配置
        embedder: 向量嵌入器实例
        triage_classifier: 分诊分类器实例
        client: LLM 客户端实例

    Returns:
        CompiledGraph: 编译完成的可执行图对象，支持 .invoke / .stream
    """
    graph = StateGraph(ChatState)

    # 注册所有图节点，闭包节点提前注入依赖
    graph.add_node("classify_triage", make_triage_node(triage_classifier))
    graph.add_node("emergency_shortcut", emergency_shortcut_node)
    graph.add_node("retrieve", make_retrieve_node(embedder, settings.retrieval_top_k))
    graph.add_node("generate", make_generate_node(client))
    graph.add_node("output_guardrail", guardrail_node)

    # 设置入口节点：从分诊分类开始执行
    graph.set_entry_point("classify_triage")
    graph.add_conditional_edges(
        "classify_triage",
        route_after_triage,
        {"emergency_shortcut": "emergency_shortcut", "retrieve": "retrieve"},
    )
    # 普通边：固定流转关系
    graph.add_edge("emergency_shortcut", "output_guardrail")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_guardrail")
    graph.add_edge("output_guardrail", END)

    return graph.compile()
