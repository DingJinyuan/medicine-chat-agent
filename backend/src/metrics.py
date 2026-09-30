"""Prometheus 指标（两版共用）。

被 patient_graph / patient_llm_service / main 引用，用于监控：
- 分诊标签分布、token 用量、LLM 错误次数、限流、请求计数/耗时。
"""
from prometheus_client import Counter, Histogram

TRIAGE_LABELS = Counter(
    "triage_labels_total", "分诊标签计数", ["label"]
)
TOKEN_USAGE = Counter(
    "token_usage_total", "LLM token 用量", ["kind"]
)
LLM_ERRORS = Counter("llm_errors_total", "LLM 调用错误次数")

RATE_LIMITED = Counter("rate_limited_total", "限流触发次数")
REQUEST_COUNT = Counter("request_count_total", "请求计数", ["status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "请求耗时", ["path"])
