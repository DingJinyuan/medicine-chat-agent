import json

from app.errors import AppError, ErrorCode, error_response


def test_app_error_carries_status_code_and_message():
    r = AppError(status=503, code=ErrorCode.LLM_UNAVAILABLE, message="LLM down")
    assert (r.status, r.code, r.message) == (503, "llm_unavailable", "LLM down")


def test_app_error_defaults():
    r = AppError()
    assert (r.status, r.code, r.message) == (500, "internal_error", "Internal error")


def test_error_response_json():
    resp = error_response(401, ErrorCode.UNAUTHORIZED, "bad key")
    assert resp.status_code == 401
    assert json.loads(resp.body) == {"error": {"code": "unauthorized", "message": "bad key"}}


def test_error_response_accepts_plain_string_code():
    resp = error_response(500, "custom_code", "custom message")
    assert json.loads(resp.body) == {"error": {"code": "custom_code", "message": "custom message"}}
