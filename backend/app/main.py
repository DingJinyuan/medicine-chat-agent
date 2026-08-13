from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import get_settings
from app.errors import AppError, ErrorCode, error_response
from app.logging_config import setup_logging
from app import metrics as m
from app.db import Database
from app.graph import build_graph
from app.ingestion import build_documents, load_or_fetch_corpus
from app.models import ChatRequest, ChatResponse, HealthResponse, SourceChunk, TriagePrediction
from app.security import get_rate_limiter, require_api_key
from app.triage_classifier import get_triage_classifier
from app.vector_store import build_or_load_vector_store

setup_logging()
logger = structlog.get_logger("medisense")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = Database(settings.db_path)

    topics = load_or_fetch_corpus(settings.medlineplus_cache_path)
    documents = build_documents(topics, settings.chunk_size, settings.chunk_overlap)
    vector_store = build_or_load_vector_store(settings.vector_index_path, settings.embedding_model, documents=documents)

    triage_classifier = get_triage_classifier(settings.triage_adapter_path, settings.triage_base_model)

    app.state.vector_store = vector_store
    app.state.triage_classifier = triage_classifier
    app.state.graph = build_graph(settings, vector_store, triage_classifier)
    logger.info("medisense_ready", topics=len(topics), chunks=len(documents))
    yield


app = FastAPI(title="MediSense AI", description="Medical RAG chatbot (portfolio demo, not a clinical tool)", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    structlog.contextvars.clear_contextvars()
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def _app_error_handler(request: Request, exc: AppError):
    return error_response(exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    # 安全：不 str(exc)，避免泄露字段名/内部细节
    return error_response(422, ErrorCode.VALIDATION_ERROR, "Invalid request body or parameters.")


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    # 安全：traceback 进日志，message 保持模糊
    logger.exception("unhandled_error", path=request.url.path)
    return error_response(500, ErrorCode.INTERNAL_ERROR, "An unexpected error occurred.")


def get_db(request: Request) -> Database:
    return request.app.state.db


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.get("/health/ready")
async def readiness(request: Request):
    checks = {
        "triage_classifier": type(request.app.state.triage_classifier).__name__ == "TriageClassifier",
        "vector_store": request.app.state.vector_store is not None,
        "llm_configured": bool(settings.llm_model_id),
    }
    return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request, db: Database = Depends(get_db),
               _: None = Depends(require_api_key)):
    client_id = request.client.host if request.client else "unknown"
    if not get_rate_limiter().check(client_id):
        m.RATE_LIMITED.inc()
        raise AppError(status=429, code=ErrorCode.RATE_LIMITED, message="rate limit exceeded, try again shortly")

    session_id = db.ensure_session(payload.session_id)

    m.REQUEST_COUNT.labels(status="ok").inc()
    with m.REQUEST_LATENCY.labels(path="/api/chat").time():
        result = request.app.state.graph.invoke({"question": payload.message})

    db.add_message(session_id, "user", payload.message)
    db.add_message(session_id, "assistant", result["answer"])

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result.get("sources", [])],
        triage=TriagePrediction(label=result["triage_label"], confidence=result["triage_confidence"]),
        guardrail_rewritten=result.get("guardrail_rewritten", False),
        injection_flagged=result.get("injection_flagged", False),
    )


@app.get("/api/chat/{session_id}/history")
async def chat_history(session_id: str, db: Database = Depends(get_db)):
    return db.get_history(session_id)
