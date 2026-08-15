from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    对话接口请求体模型

    Fields:
        message: 用户输入问答文本，长度限制 1~4000
        session_id: 会话唯一标识，可为空（匿名会话）
    """
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class SourceChunk(BaseModel):
    """
       RAG 检索溯源片段信息模型

       Fields:
           topic: 医疗主题名称
           url: MedlinePlus 官方溯源链接
           text: 检索召回的原文片段
           score: 向量相似度分数
       """
    topic: str
    url: str
    text: str
    score: float


class TriagePrediction(BaseModel):
    """
    医疗分诊预测结果模型

    Fields:
        label: 分诊标签（emergency / normal 等）
        confidence: 模型预测置信度 [0,1]
    """
    label: str
    confidence: float


class ChatResponse(BaseModel):
    """
        对话接口统一返回体模型

        Fields:
            session_id: 本次会话ID
            answer: 最终经过护栏校验的回答文本
            sources: 检索参考的权威溯源片段列表
            triage: 本次请求分诊模型预测结果
            guardrail_rewritten: 输出护栏是否修改过回答内容
            injection_flagged: 检索内容是否检测到注入风险
        """
    session_id: str
    answer: str
    sources: list[SourceChunk]
    triage: TriagePrediction
    guardrail_rewritten: bool
    injection_flagged: bool


class HealthResponse(BaseModel):
    """
    服务健康检查返回模型

    Fields:
        status: 服务状态标识（ok）
    """
    status: str
