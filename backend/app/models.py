from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class SourceChunk(BaseModel):
    topic: str
    url: str
    text: str
    score: float


class TriagePrediction(BaseModel):
    label: str
    confidence: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk]
    triage: TriagePrediction
    guardrail_rewritten: bool
    injection_flagged: bool


class HealthResponse(BaseModel):
    status: str
