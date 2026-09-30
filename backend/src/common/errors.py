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
    """
    全局自定义业务异常
    统一封装 HTTP 状态码、机器错误码、用户提示信息，用于全局异常捕获返回标准 JSON 错误体
    """
    def __init__(self, status: int = 500, code: ErrorCode = ErrorCode.INTERNAL_ERROR, message: str = "Internal error"):
        """
        Args:
            status: HTTP 响应状态码
            code: 业务枚举错误码
            message: 对外展示的错误描述信息
        """
        self.status = status
        self.code = code.value if isinstance(code, ErrorCode) else code
        self.message = message
        super().__init__(message)


def error_response(status: int, code: ErrorCode | str, message: str) -> JSONResponse:
    """
    统一构造全局标准化错误响应体

    Args:
        status: HTTP 状态码
        code: 业务错误码（支持枚举/字符串）
        message: 错误提示文案

    Returns:
        FastAPI JSONResponse 标准错误返回格式
    """
    code_str = code.value if isinstance(code, ErrorCode) else code
    return JSONResponse(status_code=status, content={"error": {"code": code_str, "message": message}})
