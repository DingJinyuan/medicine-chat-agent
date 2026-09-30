"""检索节点：pgvector 混合检索 + 注入扫描 + 包装不可信片段。"""
from src.agents.patient_state import ChatState
from src.service.patient_security import scan_for_injection, wrap_untrusted
from src.service.patient_retrieval_service import retrieve_medlineplus


def make_retrieve_node(embedder, top_k: int):
    """构造【知识库检索】节点闭包，注入嵌入器与召回数量。

    Args:
        embedder: 向量嵌入器实例（调用方注入）
        top_k: 召回文档数量

    Returns:
        node: LangGraph 节点函数
    """
    def node(state: ChatState) -> ChatState:
        # 扫描用户输入（真正的攻击面），与召回文档一并检测注入
        input_scan = scan_for_injection(state["question"])
        # pgvector 混合检索（BM25 + cosine），返回 {content, topic, url, score}
        chunks = retrieve_medlineplus(state["question"], embedder, top_k)

        context_blocks = []
        sources = []
        any_flagged = input_scan.flagged  # 用户输入或召回文档命中注入即置 True
        for chunk in chunks:
            content = chunk["content"]
            # 对召回回来的知识库文档内容做注入扫描
            scan = scan_for_injection(content)
            any_flagged = any_flagged or scan.flagged
            topic = chunk["topic"]
            url = chunk["url"]
            # 包装不可信的文档片段，告知 LLM 这是数据而非指令
            context_blocks.append(wrap_untrusted(topic, content))
            sources.append(
                {
                    "topic": topic,
                    "url": url,
                    "text": content,
                    "score": chunk["score"],
                }
            )

        return {**state, "context_blocks": context_blocks, "sources": sources, "injection_flagged": any_flagged}

    return node
