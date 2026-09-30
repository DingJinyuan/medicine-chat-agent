"""Query Understanding 业务层：把含糊问题填成结构化表单。

用 function calling（强 schema 约束）替代 few-shot JSON：
把 ClinicalQueryForm 定义成 OpenAI function 的 parameters，
让 LLM 强制按 schema 输出，几乎不会出「坏 JSON」。
LLM 客户端由调用方注入（依赖注入），患者画像另走关键词匹配。
"""
from typing import List

import structlog
from pydantic import BaseModel, Field

from src.common.llm_adapter import invoke_structured

logger = structlog.get_logger(__name__)


class ClinicalQueryForm(BaseModel):
    """问题理解表单：LLM 一次调用填写的结构化结果"""
    # ① 补全后的完整问题（检索用）
    rewritten_query: str
    # ② 检索关键词（多路召回用）
    keywords: List[str] = Field(default_factory=list)


# function calling 的 schema：把 ClinicalQueryForm 定义成 OpenAI function
CLINICAL_QUERY_FUNCTION = {
    "name": "extract_clinical_query",
    "description": "从含糊的医学问题中提取补全后的临床问题和检索关键词",
    "parameters": {
        "type": "object",
        "properties": {
            "rewritten_query": {
                "type": "string",
                "description": (
                    "补全后的完整、具体的临床问题。补全隐含的上下文（人群、疾病、用药场景），"
                    "但不要编造未提及的事实。问题已具体则保持接近原文。"
                ),
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "文献检索最有用的关键词（药名、病名、临床概念）",
            },
        },
        "required": ["rewritten_query", "keywords"],
    },
}

SYSTEM_PROMPT = (
    "You are a clinical query understanding assistant. "
    "Expand a vague or incomplete medical question into a complete, "
    "structured form for retrieval."
)


def understand_query(question: str, client) -> ClinicalQueryForm:
    """调用 LLM 把含糊问题填成结构化表单（function calling 强 schema 约束）。

    Args:
        question: 用户原始问题
        client: LLM 客户端（调用方注入）

    失败降级：LLM 没走 function 调用或解析失败时，返回 rewritten_query=原问题，
    保证流水线不因单个环节报错而中断。
    """
    try:
        data = invoke_structured(
            client,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            CLINICAL_QUERY_FUNCTION,
            temperature=0.1,
            tag="query_understanding",
        )

        # 优先用结构化解析结果
        if data:
            form = ClinicalQueryForm(**data)
        else:
            # LLM 没走 function 调用，降级为原问题
            logger.warning("query_understanding_no_tool_call")
            form = ClinicalQueryForm(rewritten_query=question)

        logger.info("query_understood", keywords=form.keywords)
        return form
    except Exception as e:
        logger.warning("query_understanding_failed", error=str(e))
        return ClinicalQueryForm(rewritten_query=question)
