"""两版对话统一 schema：公共基类 + 专家版 / 患者版专属子类。"""
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """对话接口请求体（两版统一）

    Fields:
        message: 用户输入问答文本，长度 1~4000
        session_id: 会话唯一标识，可为空（匿名会话）
        mode: 走哪条工作流，'expert'（专家版）/ 'patient'（患者版）
    """
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    mode: str = "patient"


class BaseChatResponse(BaseModel):
    """两版对话响应的公共字段"""
    session_id: str
    mode: str
    answer: str


class SourceChunk(BaseModel):
    """RAG 检索溯源片段信息（患者版）"""
    topic: str
    url: str
    text: str
    score: float


class TriagePrediction(BaseModel):
    """医疗分诊预测结果模型（患者版）"""
    label: str
    confidence: float


class ExpertChatResponse(BaseChatResponse):
    """专家版响应：多 Agent 临床 RAG + 风险预测结果

    extra="forbid" 保证 Union 响应模型能区分本类与 PatientChatResponse，
    避免患者版字段被专家版的可选默认字段吞掉。
    """
    model_config = ConfigDict(extra="forbid")
    citations: list[str] = []
    confidence_score: float | None = None
    is_reliable: bool | None = None
    critique: str = ""


class PatientChatResponse(BaseChatResponse):
    """患者版响应：分诊安全问答结果"""
    model_config = ConfigDict(extra="forbid")
    sources: list[SourceChunk] = []
    triage: TriagePrediction | None = None
    guardrail_rewritten: bool = False
    injection_flagged: bool = False


class HealthResponse(BaseModel):
    """服务健康检查返回模型

    Fields:
        status: 服务状态标识（ok / degraded）
        database: 数据库是否可连通
    """
    status: str
    database: bool = False
