"""患者版对话状态定义 + 工作流常量。"""
from typing import TypedDict

# 紧急情况判定置信度阈值：>=该置信度直接触发应急短路分支
EMERGENCY_CONFIDENCE_THRESHOLD = 0.6
# 低置信兜底阈值：任何分类结果置信度低于该值，走保守应急兜底
LOW_CONFIDENCE_THRESHOLD = 0.4

EMERGENCY_RESPONSE = (
    "What you're describing sounds like it could be a medical emergency. "
    "Please call your local emergency number (911 in the US) or go to the "
    "nearest emergency room right now. Do not wait for an online response."
)

# 检测到提示注入攻击时的固定安全响应：不把输入喂给 LLM
INJECTION_BLOCKED_RESPONSE = (
    "I couldn't process that request. Please rephrase your question in plain "
    "language describing your symptoms, without any special instructions."
)


class ChatState(TypedDict, total=False):
    """
    对话状态字典，LangGraph 流转的核心上下文
    total=False：字段非必须，节点可以只写入部分 key
    """
    question: str              # 用户原始提问
    triage_label: str          # 分诊分类标签，如 emergency / normal / consult
    triage_confidence: float   # 分诊分类器输出置信度 [0~1]
    context_blocks: list[str]  # 检索得到、经过包装的知识库上下文片段列表
    sources: list[dict]        # 检索源元数据列表（主题、链接、原文、相似度分数）
    injection_flagged: bool    # 检索到的知识库片段是否检测出注入攻击风险
    answer: str                # 大模型生成/应急分支产出的回答文本
    guardrail_rewritten: bool  # 输出护栏是否对最终答案做过改写修正
