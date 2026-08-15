"""LangGraph state machine for one chat turn.

    classify_triage --conditional--> emergency_shortcut --> output_guardrail --> END
                     \\-------------> retrieve --> generate --> output_guardrail --> END

The emergency branch is a genuine conditional (not just sequential steps):
a high-confidence "emergency" triage prediction skips retrieval and
generation entirely and returns a fixed, safe response -- the LLM never
sees emergency-flagged input. Both branches converge on `output_guardrail`
so the disclaimer/dosage/diagnosis rewrite rules apply unconditionally.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app import metrics as m
from app.config import Settings
from app.llm_client import generate_answer
from app.security import enforce_medical_guardrails, scan_for_injection, wrap_untrusted
from app.triage_classifier import TriageClassifier

# 紧急情况判定置信度阈值：>=该置信度直接触发应急短路分支
EMERGENCY_CONFIDENCE_THRESHOLD = 0.6
# 低置信兜底阈值：任何分类结果置信度低于该值，走保守应急兜底
LOW_CONFIDENCE_THRESHOLD = 0.4

EMERGENCY_RESPONSE = (
    "What you're describing sounds like it could be a medical emergency. "
    "Please call your local emergency number (911 in the US) or go to the "
    "nearest emergency room right now. Do not wait for an online response."
)


class ChatState(TypedDict, total=False):
    """
       对话状态字典，LangGraph 流转的核心上下文
       total=False：字段非必须，节点可以只写入部分key

       Fields:
           question: 用户原始提问
           triage_label: 分诊分类标签，如 emergency / normal / consult
           triage_confidence: 分诊分类器输出置信度 [0~1]
           context_blocks: 检索得到、经过包装的知识库上下文片段列表
           sources: 检索源元数据列表，包含主题、链接、原文、相似度分数，用于溯源展示
           injection_flagged: 检索到的知识库片段是否检测出注入攻击风险
           answer: 大模型生成/应急分支产出的回答文本
           guardrail_rewritten: 输出护栏是否对最终答案做过改写修正
       """
    question: str
    triage_label: str
    triage_confidence: float
    context_blocks: list[str]
    sources: list[dict]
    injection_flagged: bool
    answer: str
    guardrail_rewritten: bool


def _classify_triage_node(triage_classifier: TriageClassifier):
    """
        构造【分诊分类】节点闭包
        注入外部分类器实例，返回真正被LangGraph调用的node函数

        Args:
            triage_classifier: 预初始化的分诊分类器实例

        Returns:
            node: LangGraph可执行节点函数，输入输出均为ChatState
        """
    def node(state: ChatState) -> ChatState:
        result = triage_classifier.classify(state["question"])
        #统计每种分诊标签出现多少次，用于监控：有多少请求走紧急捷径，多少走 RAG 检索链路。
        m.TRIAGE_LABELS.labels(label=result.label).inc()
        return {**state, "triage_label": result.label, "triage_confidence": result.confidence}

    return node


def _route_after_triage(state: ChatState) -> str:
    """
    分诊完成后的条件路由函数，LangGraph conditional_edge 使用
    决定走【应急短路分支】还是【正常RAG检索生成分支】

    分支规则：
        1. 标签=emergency 且置信度>=紧急阈值 → emergency_shortcut
        2. 任意标签，置信度低于低置信阈值 → 保守兜底，走应急短路
        3. 其余情况走正常RAG流程 → retrieve

    Args:
        state: 当前对话状态

    Returns:
        str: 下一跳节点名称，只能是 graph 注册的节点key
    """
    if state["triage_label"] == "emergency" and state["triage_confidence"] >= EMERGENCY_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"
    if state["triage_confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"   # 任何标签置信度太低，走保守兜底
    return "retrieve"


def _emergency_shortcut_node(state: ChatState) -> ChatState:
    """
        应急短路节点：跳过检索、跳过LLM调用
        直接写入固定紧急提示，清空上下文、来源，关闭注入标记
        关键点：LLM 完全不会接触用户输入，避免大模型处理高危医疗输入带来风险

        Args:
            state: 对话状态

        Returns:
            更新后的ChatState，填充answer，清空检索相关字段
        """
    return {**state, "answer": EMERGENCY_RESPONSE, "context_blocks": [], "sources": [], "injection_flagged": False}


def _retrieve_node(retriever, top_k: int):
    """
       构造【知识库检索】节点闭包，注入检索器与召回数量参数

       Args:
           retriever: 向量库检索器实例
           top_k: 召回文档数量，取自配置settings

       Returns:
           node: LangGraph节点函数
       """
    def node(state: ChatState) -> ChatState:
        # 向量库相似度检索，同时返回相似度分数
        docs_with_scores = retriever.similarity_search_with_score(state["question"], k=top_k)

        context_blocks = []
        sources = []
        any_flagged = False  #标记，任意一条召回文档命中注入攻击，则置 True，防御【间接提示注入】。
        for doc, score in docs_with_scores:
            # 对召回回来的知识库文档内容做注入扫描（防间接提示注入）
            scan = scan_for_injection(doc.page_content)  #扫描知识库文档文本，检测是否存在提示注入
            any_flagged = any_flagged or scan.flagged  #只要任意一篇文档标记可疑，整体就标记
            # 包装不可信的文档片段，给LLM的上下文
            context_blocks.append(wrap_untrusted(doc.metadata.get("topic", "unknown"), doc.page_content))
            sources.append(
                {
                    "topic": doc.metadata.get("topic", "unknown"),
                    "url": doc.metadata.get("url", ""),
                    "text": doc.page_content,
                    "score": float(score),
                }  #主题、原文链接、文档文本、相似度分数
            )

        return {**state, "context_blocks": context_blocks, "sources": sources, "injection_flagged": any_flagged}

    return node


def _generate_node(settings: Settings):
    """
        构造【LLM生成回答】节点闭包，注入全局配置

        Args:
            settings: 项目全局配置对象，包含模型参数、温度、最大token等

        Returns:
            node: LangGraph节点函数
        """
    def node(state: ChatState) -> ChatState:
        answer = generate_answer(settings, state["question"], state["context_blocks"])
        return {**state, "answer": answer}

    return node


def _output_guardrail_node(state: ChatState) -> ChatState:
    """
       输出护栏节点：**两条分支都会经过此节点**
       对最终输出执行医疗安全规则校验与改写：
       - 移除给出诊断、给出具体用药剂量的内容
       - 追加医疗免责声明
       - 过滤违规话术

       Args:
           state: 对话状态，内含answer字段

       Returns:
           更新state，覆盖改写后的answer，标记是否执行过改写
       """
    result = enforce_medical_guardrails(state["answer"])
    return {**state, "answer": result.text, "guardrail_rewritten": result.rewritten}


def build_graph(settings: Settings, retriever, triage_classifier: TriageClassifier):
    """
    构建并编译完整的医疗RAG状态图，对外暴露可直接invoke的graph运行时

    Args:
        settings: 全局配置
        retriever: 向量检索器实例
        triage_classifier: 分诊分类器实例

    Returns:
        CompiledGraph: LangGraph编译完成的可执行图对象，支持 .invoke / .stream
    """
    # 初始化状态图，绑定状态定义ChatStat
    graph = StateGraph(ChatState)
    # 注册所有图节点，闭包节点提前注入依赖
    graph.add_node("classify_triage", _classify_triage_node(triage_classifier))
    graph.add_node("emergency_shortcut", _emergency_shortcut_node)
    graph.add_node("retrieve", _retrieve_node(retriever, settings.retrieval_top_k))
    graph.add_node("generate", _generate_node(settings))
    graph.add_node("output_guardrail", _output_guardrail_node)
    # 设置入口节点：从分诊分类开始执行
    graph.set_entry_point("classify_triage")
    graph.add_conditional_edges(
        "classify_triage",
        _route_after_triage,
        {"emergency_shortcut": "emergency_shortcut", "retrieve": "retrieve"},
    )
    # 普通边：固定流转关系
    graph.add_edge("emergency_shortcut", "output_guardrail")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_guardrail")
    graph.add_edge("output_guardrail", END)
    # 编译图，返回可运行实例
    return graph.compile()
