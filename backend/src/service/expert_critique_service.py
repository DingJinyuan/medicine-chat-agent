import structlog

from src.common.llm_adapter import invoke_structured

logger = structlog.get_logger(__name__)

# 幻觉校验系统提示词：循证医学事实核查角色、引用纪律、医学越界检查、评分规则
CRITIQUE_PROMPT = """You are a medical fact-checker reviewing an AI-generated clinical answer against its source literature, following evidence-based medicine standards.

Your job:
1. Check every clinical claim is grounded in the provided literature context (citation discipline)
2. Identify any claims not supported by the literature context (potential hallucination)
3. Flag unsafe output: a definitive diagnosis or a specific drug dosage stated without physician judgment
4. Assign a confidence score from 0.0 to 1.0

IMPORTANT: If the answer includes a "Patient Risk Assessment (ML Model)" section, treat that as a SEPARATE, VALID evidence source — it comes from a trained ML classifier, not from the literature. Do NOT flag ML risk scores as unsupported or hallucinated just because they don't appear in the literature context. Only evaluate literature-based claims against the literature context.

Scoring guide:
  0.9 - 1.0 : Answer fully grounded, all literature claims supported by context, proper citations
  0.7 - 0.9 : Mostly grounded, minor gaps
  0.5 - 0.7 : Partially grounded, some unsupported claims
  0.0 - 0.5 : Significant hallucination risk, do not trust
"""

# function calling 的 schema：把事实核查结果定义成 OpenAI function
FACT_CHECK_FUNCTION = {
    "name": "submit_fact_check",
    "description": "提交医学回答的事实核查结果",
    "parameters": {
        "type": "object",
        "properties": {
            "critique": {
                "type": "string",
                "description": "对回答的事实核查评语（指出是否有幻觉、证据缺口）",
            },
            "confidence": {
                "type": "number",
                "description": "置信度分数 0.0 到 1.0",
            },
            "reliable": {
                "type": "boolean",
                "description": "回答是否证据充足、可靠",
            },
        },
        "required": ["critique", "confidence", "reliable"],
    },
}


def build_context_snippet(chunks: list[dict]) -> str:
    """
    从检索切块构建校验用上下文摘要；每段切块截断前300字符，控制prompt长度

    Args:
        chunks: 知识库切块列表

    Returns:
        str: 拼接完成的源上下文字符串
    """
    context = "\n".join([
        f"[Source {i+1}]: {chunk['content'][:300]}"
        for i, chunk in enumerate(chunks)
    ])
    return context


def run_fact_check(
    question: str,
    answer: str,
    chunks: list[dict],
    client,
) -> tuple[str, float, bool]:
    """
    事实核查核心业务函数，Service层纯逻辑，不依赖AgentState
    组装校验Prompt、调用LLM校验接口（function calling 强 schema 约束）、解析返回结果

    Args:
        question: 用户原始问题
        answer: 待核查的AI生成答案
        chunks: 检索返回知识库切块
        client: LLM 客户端（调用方注入）
    """
    context = build_context_snippet(chunks)

    user_prompt = f"""Original Question: {question}

Source Context Used:
{context}

Generated Answer to Review:
{answer}

Please critique this answer and provide your assessment.
"""

    try:
        data = invoke_structured(
            client,
            [
                {"role": "system", "content": CRITIQUE_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            FACT_CHECK_FUNCTION,
            temperature=0.1,
            tag="critique",
        )

        if data:
            critique = data.get("critique", "")
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            reliable = bool(data.get("reliable", False))
        else:
            logger.warning("critique_no_tool_call")
            critique, confidence, reliable = "", 0.5, False

        logger.info("critique_result", confidence=confidence, reliable=reliable)
        return critique, confidence, reliable
    except Exception as e:
        logger.warning("critique_failed", error=str(e))
        return "", 0.5, False
