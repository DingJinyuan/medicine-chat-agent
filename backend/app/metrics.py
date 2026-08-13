"""Prometheus 指标定义。

暴露到 /metrics 端点，供 Prometheus + Grafana 采集画图。
"""
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("medisense_requests_total", "Total chat requests", ["status"])
REQUEST_LATENCY = Histogram("medisense_request_duration_seconds", "Request latency", ["path"])
LLM_ERRORS = Counter("medisense_llm_errors_total", "LLM generation failures")
TRIAGE_LABELS = Counter("medisense_triage_total", "Triage predictions", ["label"])
RATE_LIMITED = Counter("medisense_rate_limited_total", "Rate limit rejections")
TOKEN_USAGE = Counter("medisense_tokens_total", "Token usage", ["kind"])
