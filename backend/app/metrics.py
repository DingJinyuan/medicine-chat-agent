"""Prometheus 指标定义。

暴露到 /metrics 端点，供 Prometheus + Grafana 采集画图。
"""
from prometheus_client import Counter, Histogram

# 聊天请求总计数器，按status标签区分成功/失败状态
REQUEST_COUNT = Counter("medisense_requests_total", "Total chat requests", ["status"])
# 请求耗时直方图，按path标签区分接口路径
REQUEST_LATENCY = Histogram("medisense_request_duration_seconds", "Request latency", ["path"])
# LLM调用失败总计数
LLM_ERRORS = Counter("medisense_llm_errors_total", "LLM generation failures")
# 分诊预测统计计数器，按label区分分诊类别
TRIAGE_LABELS = Counter("medisense_triage_total", "Triage predictions", ["label"])
# 接口限流拒绝请求计数
RATE_LIMITED = Counter("medisense_rate_limited_total", "Rate limit rejections")
# Token消耗统计，kind区分prompt/completion  给指标打上分类标记，同一个指标，通过不同标签区分不同类型数据
TOKEN_USAGE = Counter("medisense_tokens_total", "Token usage", ["kind"])