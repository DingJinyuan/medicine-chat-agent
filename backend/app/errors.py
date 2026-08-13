from enum import Enum

from fastapi.responses import JSONResponse


class ErrorCode(str, Enum):
    """机器可读错误码，前端据此分支处理"""
    VALIDATION_ERROR = "validation_error"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"
    LLM_UNAVAILABLE = "llm_unavailable"


class AppError(Exception):
    def __init__(self, status: int = 500, code: ErrorCode = ErrorCode.INTERNAL_ERROR, message: str = "Internal error"):
        self.status = status
        self.code = code.value if isinstance(code, ErrorCode) else code
        self.message = message
        super().__init__(message)


def error_response(status: int, code: ErrorCode | str, message: str) -> JSONResponse:
    code_str = code.value if isinstance(code, ErrorCode) else code
    return JSONResponse(status_code=status, content={"error": {"code": code_str, "message": message}})
