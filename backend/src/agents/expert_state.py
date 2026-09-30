# 类型注解：TypedDict(字典类型提示)、列表、可选字段
from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    """
    在流水线所有智能体之间传递的**全局共享状态对象**
    LangGraph DAG工作流的核心数据流容器。
    每一个Agent(检索、NER、推理、校验)读取状态，然后把自己的运行结果写回状态。
    所有节点共用这一份状态，实现数据流转。
    """
    # -------- 用户输入（最开始传入，全程不变） --------
    question: str

    # -------- Query Understanding Agent 节点输出（替换旧 NER） --------
    rewritten_query: str            # LLM 补全后的完整问题，用于检索
    patient_profile: dict           # ML 患者画像，非空字段喂给 risk_tool，空字段走默认值
    keywords: List[str]             # 检索关键词，多路召回用

    # -------- Retrieval Agent 检索节点输出 --------
    retrieved_chunks: List[dict]
    # 从向量库召回回来的知识库切块，每一块包含文本内容、doc_id、页码、向量等溯源信息

    # -------- Reasoning Agent 推理生成节点输出 --------
    answer: str                     # LLM生成的初步回答草稿
    citations: List[str]            # 回答引用的文献溯源标记（PMID、PDF页码等）

    # -------- Critique Agent 批判性校验/幻觉审查节点输出 --------
    critique: str                   # 校验Agent给出的评估意见，指出是否存在幻觉、证据缺口
    confidence_score: float         # 置信度分数 0‑1，衡量回答可靠程度
    is_reliable: bool               # 布尔标记：True=回答证据充足可靠，False=存在幻觉/证据不足
