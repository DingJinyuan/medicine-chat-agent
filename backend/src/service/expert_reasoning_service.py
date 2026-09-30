import structlog

from src.skills.loader import load_skill

logger = structlog.get_logger(__name__)

# 系统提示词：循证医学临床推理角色，结构化输出（结论→推理→证据→不确定性）
SYSTEM_PROMPT = """You are a senior evidence-based clinical decision support system assisting a physician with a point-of-care question. Synthesize your answer strictly from the provided research context, following clinical reasoning norms.

## Output structure (follow strictly)
1. **Conclusion** — a concise clinical bottom-line answer.
2. **Clinical Reasoning** — the key evidence points from the context, ordered by clinical relevance.
3. **Evidence & Citations** — cite the source number or document ID for every clinical claim.
4. **Limitations & Uncertainty** — what the context does not cover, and relevant clinical caveats.

## Rules
- Base every statement strictly on the provided context chunks; never fabricate findings, guidelines, dosages, or statistics.
- Cite sources inline (Source N or document ID) whenever you state evidence.
- Use precise clinical terminology (e.g., "reduced hepatic gluconeogenesis" rather than vague phrasing).
- If the context is insufficient to answer, state that explicitly instead of guessing.
- If a Patient Risk Assessment (ML Model) section is provided, integrate it into your recommendations as a separate validated input.
- Frame recommendations as clinical considerations for the treating physician, not as directives to a patient, and never state a definitive diagnosis or specific drug dosage without the treating physician's judgment.
"""


def build_context_and_citations(chunks: list[dict]) -> tuple[str, list[str]]:
    """
    拼接检索得到的知识库上下文文本，并提取文档引用id列表

    Args:
        chunks: 向量检索返回的切块结果列表，每个chunk字典包含content、doc_id、score

    Returns:
        tuple: (拼接完成的上下文字符串, 文档id引用列表)
    """
    context = ""
    citations = []
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i} | {chunk['doc_id']} | score: {chunk['score']}]\n"
        context += chunk["content"] + "\n"
        citations.append(chunk["doc_id"])
    return context, citations


def compute_risk_section(patient_profile: dict | None, risk_tool) -> tuple[str, dict | None]:
    """
    患者再入院风险评估工具调度函数
    画像由 Query Understanding 节点提前填好；画像非空则运行ML风险预测

    Args:
        patient_profile: Query Understanding 提取的患者画像 dict，可空
        risk_tool: 再入院风险工具实例（调用方注入）
    """
    risk_section = ""
    risk_result = None
    # 判断风险工具是否可用
    if risk_tool.is_available():
        if patient_profile:
            logger.info("patient_profile_detected", profile=patient_profile)
            # 调用机器学习模型预测再入院风险
            risk_result = risk_tool.predict(patient_profile)
            risk_section = f"""
Patient Risk Assessment (ML Model):
  Risk Score:  {risk_result['risk_score']}
  Risk Level:  {risk_result['risk_level']}
  Assessment:  {risk_result['interpretation']}
"""
            logger.info("risk_level", level=risk_result['risk_level'])
        else:
            logger.info("risk_skipped_no_profile")
    else:
        logger.info("risk_skipped_no_model")
    return risk_section, risk_result


def generate_clinical_answer(
    question: str,
    chunks: list[dict],
    patient_profile: dict | None,
    client,
    risk_tool,
) -> tuple[str, list[str]]:
    """
    推理模块核心业务函数：组装完整prompt，调用LLM生成临床回答
    Service层纯业务函数，完全不依赖AgentState，可独立单元测试

    Args:
        question: 用户原始问题文本
        chunks: 检索返回知识库切块列表
        patient_profile: Query Understanding 提取的患者画像，可空
        client: LLM 客户端（调用方注入）
        risk_tool: 再入院风险工具实例（调用方注入）
    """
    # 1.拼接检索上下文与文献引用
    context, citations = build_context_and_citations(chunks)
    # 2.执行患者风险评估，拿到风险区块文本
    risk_section, _ = compute_risk_section(patient_profile, risk_tool)

    # 组装发给大模型的用户提示
    user_prompt = f"""Clinical Question: {question}

{risk_section}
Research Context:
{context}

Please provide a comprehensive clinical answer based strictly on the above context. Reference the source numbers when citing evidence. If a patient risk assessment is included above, incorporate it into your recommendations.
"""

    # 调用LLM，temperature=0.1降低创造性，保证临床回答严谨
    # 渐进式加载 skill（运行时按需读 SKILL.md），失败回退到内置 SYSTEM_PROMPT
    try:
        system_prompt = load_skill("clinical-reasoning")
    except FileNotFoundError:
        system_prompt = SYSTEM_PROMPT

    result = client.invoke(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        tag="reasoning",
    )

    answer = result.content
    logger.info("answer_generated")
    return answer, citations
