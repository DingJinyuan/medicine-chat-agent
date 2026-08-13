# MediSense AI 可靠性 + 可观测 实施计划

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务执行。步骤用 `- [ ]` 勾选跟踪。

**Goal:** 消除 P0 可靠性硬伤（裸 500、单点故障、不可观测），并补上成本可观测、数据漂移、Eval 回归门禁三项已确认的增强。

**Architecture:** 在现有 LangGraph 流水线外层加固，不动 classify_triage → retrieve → generate → guardrail 核心链路。

**Tech Stack:** tenacity（重试）、structlog（结构化日志）、prometheus-client（指标）。

## Global Constraints

- Python ≥3.10（后端 Docker 用 3.12，本地 base-llm 是 3.10）
- 所有新增后端依赖加进 `backend/requirements.txt`，版本号锁定
- 日志统一走 structlog，禁止新增 `print`（CLI 脚本除外）
- **日志禁止记录用户输入原文**（医疗 PHI 红线）：只记 `input_len`、错误类型等元信息
- 错误响应统一走 `app/errors.py` 的 `error_response()`，不再裸抛
- 测试用 pytest，LLM 调用一律 mock，不触发真实网络
- 仓库当前**不是 git 仓库**，执行前先 `git init`（或跳过 commit 步骤）

---

## 任务依赖图

```
阶段一（可并行）：
  1 LLM容错 ─┐
  2 全局异常 ─┼─▶ 相互独立
  3 分类器fallback ─┤
  4 结构化日志 ─┘
        │
        ▼
阶段二：
  5 Metrics 基础 ─┬─▶ 6 成本可观测
                  └─▶ 7 数据漂移
  8 Eval 门禁（独立，新建 CI）
```

---

# 阶段一：可靠性硬伤

## 任务 1：集成多厂商 LLM 适配层（LLMClient），替换 Groq

**Files:**
- Create: `backend/app/llm_adapter.py`（从 sage-research/base/llm_client.py 复制 LLMClient 全文）
- Modify: `backend/app/llm_client.py`
- Modify: `backend/app/config.py`
- Modify: `backend/requirements.txt`
- Modify: `backend/.env.example`
- Modify: `backend/evals/run_evals.py`
- Test: `backend/tests/test_llm_client.py`

**Interfaces:**
- 产出：`generate_answer(settings, question, context_blocks) -> str`
- 引入：`LLMClient`（多厂商适配层：10 厂商前缀匹配 + fallback + 重试 + Claude 协议转换）
- 新增常量：`FALLBACK_ANSWER: str`
- 配置：`llm_model_id` / `llm_fallback_models` / `llm_timeout`（替换 `groq_api_key` / `groq_model`）

> LLMClient 已含：多厂商前缀匹配、主备降级、可重试错误分类、token 统计、Claude 协议转换。本任务只做「集成」，不改 LLMClient 内部逻辑。

- [ ] **Step 1: 复制 LLMClient + 换依赖**

把 `F:\代码学习\llm\sage-research\base\llm_client.py` 全文复制为 `backend/app/llm_adapter.py`（保留 `MODEL_PROFILES`、`LLMClient`、`_is_retryable_error` 等全部内容）。

复制后做两处修改：

（a）**优化重试退避**（参考 `F:\代码学习\llm\self-claudecode\error_recovery.py` 的 `retry_delay`）：把 `invoke` 里的固定线性延时 `delay = 0.5 * (retry + 1)` 换成指数退避 + 抖动：
```python
import random   # 文件顶部补 import
# invoke 内重试处：
base = min(1.0 * (2 ** retry), 32.0)          # 1s → 2s → 4s... 封顶 32s
delay = base + random.uniform(0, base * 0.25) # + 0~25% 抖动
```

（b）**修复 load_dotenv 路径**：把文件顶部的 `load_dotenv()` 改成显式路径，避免从项目根目录启动时找不到 `backend/.env`：
```python
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent / ".env")   # 固定指向 backend/.env
```

`requirements.txt`：
```diff
- langchain-groq==1.1.3
+ openai                    # OpenAI SDK（多厂商 OpenAI 兼容协议）
+ anthropic                 # Claude 原生协议
+ python-dotenv             # LLMClient 用 load_dotenv 读 .env
```
（移除 `tenacity`，重试逻辑 LLMClient 已内置）

- [ ] **Step 2: 改 config.py**

`app/config.py`（删除 `groq_api_key` / `groq_model`）：
```python
llm_model_id: str = ""            # 主模型名，如 deepseek-chat / claude-sonnet-4-6 / gpt-4o
llm_fallback_models: str = ""     # 逗号分隔的备用模型名，如 "gpt-4o,glm-4-flash"
llm_timeout: int = 60             # 单次请求超时秒数
```
厂商 key（`DEEPSEEK_API_KEY`、`ANTHROPIC_API_KEY`、`PROXY_API_KEY` 等）由 LLMClient 内部用 `os.getenv` 读，不进 Settings。

- [ ] **Step 3: 改 llm_client.py**

```python
import logging
from app.llm_adapter import LLMClient

logger = logging.getLogger("medisense")

SYSTEM_PROMPT = """（原文保留不动）"""

FALLBACK_ANSWER = (
    "I'm sorry, I couldn't reach my reference knowledge right now. "
    "Please try again in a moment. If this is urgent, contact a clinician "
    "or call your local emergency number."
)

def build_rag_messages(question, context_blocks):
    """返回 OpenAI 标准消息格式（LLMClient.invoke 的入参）"""
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant reference material found)"
    user_content = f"Reference context:\n{context}\n\nUser question: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def generate_answer(settings, question, context_blocks):
    fallback = [m.strip() for m in settings.llm_fallback_models.split(",") if m.strip()]
    client = LLMClient(
        model=settings.llm_model_id,
        fallback_models=fallback,
        timeout=settings.llm_timeout,
    )
    messages = build_rag_messages(question, context_blocks)
    try:
        return client.invoke(messages).content
    except Exception as exc:
        # 不记录 question 原文（医疗隐私），只记长度
        logger.exception("llm_generation_failed", input_len=len(question))
        return FALLBACK_ANSWER
```

注意：`build_rag_messages` 返回值从 LangChain 消息对象（`SystemMessage`/`HumanMessage`）改成 dict 列表，`generate_answer` 不再依赖 LangChain 的 chat model。

- [ ] **Step 4: 连带改字段引用**

- `evals/run_evals.py`：`if not settings.groq_api_key:` → `if not settings.llm_model_id:`
- `.env.example`：`GROQ_API_KEY`/`GROQ_MODEL` → `LLM_MODEL_ID` + 新增 `LLM_FALLBACK_MODELS`、`LLM_TIMEOUT`，以及厂商 key 示例（`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 等）
- `tests/conftest.py` 及引用 `groq_api_key`/`groq_model` 的地方同步改

- [ ] **Step 5: 重写测试（现有 3 个测试会崩，整体替换）**

现有 `tests/test_llm_client.py` 的 3 个测试依赖 LangChain 消息对象（`.content`）、`get_chat_model`、旧字段 `groq_api_key`/`groq_model`，改完后全部失效。用以下内容**整体替换**整个文件：

```python
from app import llm_client
from app.config import Settings

def test_build_rag_messages_includes_context_and_question():
    messages = llm_client.build_rag_messages("What is diabetes?", ["Diabetes is high blood sugar."])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == llm_client.SYSTEM_PROMPT
    assert "Diabetes is high blood sugar." in messages[1]["content"]
    assert "What is diabetes?" in messages[1]["content"]

def test_build_rag_messages_handles_empty_context():
    messages = llm_client.build_rag_messages("What is diabetes?", [])
    assert "no relevant reference material" in messages[1]["content"]

class _FakeMessage:
    content = "Diabetes info."

class _FakeClient:
    def __init__(self, **kwargs):
        pass
    def invoke(self, messages):
        return _FakeMessage()

class _FailingClient:
    def __init__(self, **kwargs):
        pass
    def invoke(self, messages):
        raise RuntimeError("all models failed")

def test_generate_answer_returns_content_on_success(monkeypatch):
    monkeypatch.setattr(llm_client, "LLMClient", _FakeClient)
    settings = Settings(llm_model_id="deepseek-chat")
    assert llm_client.generate_answer(settings, "What is diabetes?", ["context"]) == "Diabetes info."

def test_generate_answer_returns_fallback_on_total_failure(monkeypatch):
    monkeypatch.setattr(llm_client, "LLMClient", _FailingClient)
    settings = Settings(llm_model_id="deepseek-chat")
    assert llm_client.generate_answer(settings, "What is diabetes?", ["context"]) == llm_client.FALLBACK_ANSWER
```

- [ ] **Step 6: 跑测试确认通过** → `pytest tests/test_llm_client.py -v` PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/llm_adapter.py backend/app/llm_client.py backend/app/config.py backend/requirements.txt backend/.env.example backend/evals/run_evals.py backend/tests/test_llm_client.py
git commit -m "feat(llm): integrate multi-provider LLMClient adapter, replace Groq"
```

---

## 任务 2：全局异常处理器 + 统一错误格式

**Files:**
- Create: `backend/app/errors.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/security.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- 产出：`ErrorCode`（枚举）、`AppError(Exception)`、`error_response(status, code, message) -> JSONResponse`
- 消费：`security.py` 的 `require_api_key` 和限流改抛 `AppError`

> **设计考量**（8 方面浓缩）：
> 1. **错误结构**：统一 `{"error": {"code", "message"}}`，code 机器可读、message 人可读
> 2. **错误分类**：业务错误（AppError）/ 框架错误（RequestValidationError）/ 未预期（Exception 兜底）
> 3. **错误码枚举**：用 `ErrorCode` Enum 集中管理，不散落拼字符串
> 4. **安全**：5xx 绝不泄露 traceback/内部路径；RequestValidationError 用友好 message，不 `str(exc)`
> 5. **日志**：兜底 handler 用 `logger.exception` 记完整 traceback（进日志，不进响应）
> 6. **衔接**：`security.require_api_key` + `main` 限流改抛 `AppError`
> 7. ~~前端对接~~（本轮不做，只改后端）
> 8. **边界**：非法 JSON→422；路由不存在→404（FastAPI 默认）；限流 retry_after 可选

- [ ] **Step 1: 写失败测试**

`tests/test_api.py` 追加：
```python
import json
from app.errors import AppError, ErrorCode, error_response

def test_error_response_shape():
    r = AppError(status=503, code=ErrorCode.LLM_UNAVAILABLE, message="LLM down")
    assert (r.status, r.code, r.message) == (503, "llm_unavailable", "LLM down")

def test_error_response_json():
    resp = error_response(401, ErrorCode.UNAUTHORIZED, "bad key")
    assert resp.status_code == 401
    assert json.loads(resp.body) == {"error": {"code": "unauthorized", "message": "bad key"}}
```

- [ ] **Step 2: 跑测试确认失败** → 预期 FAIL（`app.errors` 不存在）

- [ ] **Step 3: 实现**

`app/errors.py`：
```python
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
```

`app/main.py` 注册三个 handler：
```python
from app.errors import AppError, ErrorCode, error_response
from fastapi.exceptions import RequestValidationError

@app.exception_handler(AppError)
async def _app_error_handler(request, exc: AppError):
    return error_response(exc.status, exc.code, exc.message)

@app.exception_handler(RequestValidationError)
async def _validation_handler(request, exc: RequestValidationError):
    # 安全：不 str(exc)，避免泄露字段名/内部细节
    return error_response(422, ErrorCode.VALIDATION_ERROR, "Invalid request body or parameters.")

@app.exception_handler(Exception)
async def _unhandled_handler(request, exc: Exception):
    # 安全：traceback 进日志，message 保持模糊
    logger.exception("unhandled_error, path=%s", request.url.path)
    return error_response(500, ErrorCode.INTERNAL_ERROR, "An unexpected error occurred.")
```

`app/security.py` 的 `require_api_key`：
```python
from app.errors import AppError, ErrorCode
# 原 raise HTTPException(401, ...) 改为：
raise AppError(status=401, code=ErrorCode.UNAUTHORIZED, message="invalid or missing X-API-Key")
```

`app/main.py` 限流处：
```python
raise AppError(status=429, code=ErrorCode.RATE_LIMITED, message="rate limit exceeded, try again shortly")
```

- [ ] **Step 4: 跑测试确认通过** → `pytest tests/test_api.py -v` PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/errors.py backend/app/main.py backend/app/security.py backend/tests/test_api.py
git commit -m "feat(errors): unified error response + global exception handlers"
```

---

## 任务 3：分类器加载 fallback + 依赖健康检查

**Files:**
- Modify: `backend/app/triage_classifier.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_triage_classifier.py`

**Interfaces:**
- 产出：`ConservativeClassifier`（`classify(text) -> TriageResult(label="emergency", confidence=1.0)`）
- 修改：`get_triage_classifier(adapter_path, base_model)` 失败返回 `ConservativeClassifier()` 而非抛异常

> **补充的 3 个设计点**：
> 1. **lru_cache 不缓存失败结果**：`@lru_cache` 只包「成功加载」，失败返回 ConservativeClassifier 不进缓存，避免 adapter 修好后服务仍卡在保守模式直到重启
> 2. **保守模式语义**：分类器是安全关键件，挂了宁可「全当 emergency 拒答」也不漏放——安全优先 trade-off，会导致所有请求返回急救建议，接受此行为
> 3. **/health/ready 补 LLM 配置检查**：`llm_configured`（`llm_model_id` 是否设置），一眼区分「分类器挂了」vs「LLM 没配」

- [ ] **Step 1: 写失败测试**

`tests/test_triage_classifier.py` 追加：
```python
from app.triage_classifier import ConservativeClassifier, get_triage_classifier

def test_conservative_classifier_always_emergency():
    c = ConservativeClassifier()
    assert c.classify("any text").label == "emergency"
    assert c.classify("any text").confidence == 1.0

def test_get_triage_classifier_falls_back_on_missing_artifact(tmp_path):
    result = get_triage_classifier(str(tmp_path / "nope"), "distilbert-base-uncased")
    assert isinstance(result, ConservativeClassifier)
```

- [ ] **Step 2: 跑测试确认失败** → 预期 FAIL

- [ ] **Step 3: 实现**

`app/triage_classifier.py`（顶部补 `import logging; logger = logging.getLogger("medisense")`）：
```python
class ConservativeClassifier:
    """安全兜底：模型加载失败时，把所有输入判为 emergency。

    刻意的安全优先选择——分类器是安全关键件，挂了宁可全拒（返回急救建议）
    也不漏放真正紧急的情况。代价是后端退化为「只能答急救」，接受此行为。
    """
    def classify(self, text: str) -> TriageResult:
        return TriageResult(label="emergency", confidence=1.0)


@lru_cache
def _load_triage_classifier(adapter_path: str, base_model: str) -> TriageClassifier:
    """只缓存「成功加载」的结果；失败抛异常，不进缓存"""
    return TriageClassifier(adapter_path, base_model)


def get_triage_classifier(adapter_path: str, base_model: str):
    try:
        return _load_triage_classifier(adapter_path, base_model)
    except Exception:
        logger.exception("triage_classifier_load_failed, adapter_path=%s", adapter_path)
        return ConservativeClassifier()   # 失败不缓存，下次请求再试
```

`app/main.py` lifespan 挂 state + 就绪检查（补 llm 检查）：
```python
app.state.triage_classifier = triage_classifier

@app.get("/health/ready")
async def readiness(request: Request):
    checks = {
        "triage_classifier": type(request.app.state.triage_classifier).__name__ == "TriageClassifier",
        "vector_store": request.app.state.vector_store is not None,
        "llm_configured": bool(settings.llm_model_id),   # 新增
    }
    return {"status": "ok" if all(checks.values()) else "degraded", "checks": checks}
```

- [ ] **Step 4: 跑测试确认通过** → `pytest tests/test_triage_classifier.py -v` PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/triage_classifier.py backend/app/main.py backend/tests/test_triage_classifier.py
git commit -m "feat(triage): conservative fallback + readiness endpoint"
```

---

## 任务 4：结构化日志 + 请求 ID（JSON + 文件输出）

**Files:**
- Create: `backend/app/logging_config.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/llm_client.py`
- Modify: `backend/app/triage_classifier.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- 产出：`setup_logging(log_dir="logs")`；中间件注入 `request_id`
- 日志格式：structlog JSON + 文件输出 + 屏蔽第三方（吸收 sage-research/base/logger.py）

> **设计决策**（已定）：
> 1. 格式用 **structlog JSON**（可被日志系统检索，企业级）
> 2. **文件 + 控制台双输出**（吸收 logger.py：文件给本地调试，控制台给 Render 收集）
> 3. **屏蔽第三方冗余**（吸收 logger.py：huggingface/peft/httpx 的 WARNING 以下不刷屏）
> 4. `llm_adapter.py` **保持标准 logging 不改**（复制来的「库代码」，日志照样进文件/控制台）

- [ ] **Step 1: 加依赖**

`requirements.txt` 追加 `structlog`（版本以 pip 解析为准）。

- [ ] **Step 2: 写 logging_config.py**

```python
import logging
import os
import sys

import structlog


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    # 1. 屏蔽第三方库冗余日志（吸收 logger.py）
    for noisy in ["huggingface_hub", "transformers", "peft", "httpx", "urllib3", "datasets", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 2. 标准 logging：文件 + 控制台双输出（吸收 logger.py）
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "medisense.log")

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:   # 防重复初始化
        return

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(file_handler)

    # 3. structlog：JSON 输出到标准 logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

- [ ] **Step 3: 改 main.py**

移除 `logging.basicConfig(...)` 和 `logger = logging.getLogger("medisense")`，改：
```python
import uuid
import structlog
from app.logging_config import setup_logging

setup_logging()
logger = structlog.get_logger("medisense")

@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    structlog.contextvars.clear_contextvars()
    return response
```

- [ ] **Step 4: 改 llm_client.py + triage_classifier.py 的 logger**

`llm_client.py`：
```python
import structlog
logger = structlog.get_logger("medisense")
# logger.exception 改成 structlog 风格（key=value）：
logger.exception("llm_generation_failed", input_len=len(question))
```

`triage_classifier.py`：
```python
import structlog
logger = structlog.get_logger("medisense")
# 同样改成：
logger.exception("triage_classifier_load_failed", adapter_path=adapter_path)
```

（之前 task1/task3 为兼容标准 logging 用了 `%d`/`%s` 格式化，现在换回 structlog 的 key=value）

`llm_adapter.py` **不改**（保持 `logging.getLogger(__name__)` 标准 logging）。

- [ ] **Step 5: 跑测试确认不回归** → `pytest` 全绿

- [ ] **Step 6: Commit**

```bash
git add backend/app/logging_config.py backend/app/main.py backend/app/llm_client.py backend/app/triage_classifier.py backend/requirements.txt
git commit -m "feat(logging): structlog JSON logging + file output + request id"
```

```bash
git add backend/app/logging_config.py backend/app/main.py backend/app/llm_client.py backend/requirements.txt
git commit -m "feat(logging): structlog JSON logging + request id middleware"
```

---

# 阶段二：可观测 + 回归门禁

## 任务 5：Metrics 基础（prometheus 端点 + 核心指标）

**Files:**
- Create: `backend/app/metrics.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/graph.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- 产出：`REQUEST_COUNT`、`REQUEST_LATENCY`、`LLM_ERRORS`、`TRIAGE_LABELS`、`RATE_LIMITED` 五个指标；`/metrics` 端点
- 消费：任务 6、7 复用 `metrics.py`

- [ ] **Step 1: 加依赖**

`requirements.txt` 追加 `prometheus-client==0.21.1`。

- [ ] **Step 2: 实现**

`app/metrics.py`：
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("medisense_requests_total", "Total chat requests", ["status"])
REQUEST_LATENCY = Histogram("medisense_request_duration_seconds", "Request latency", ["path"])
LLM_ERRORS = Counter("medisense_llm_errors_total", "LLM generation failures")
TRIAGE_LABELS = Counter("medisense_triage_total", "Triage predictions", ["label"])
RATE_LIMITED = Counter("medisense_rate_limited_total", "Rate limit rejections")
```

`app/main.py` 暴露端点 + 埋点：
```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from app import metrics as m

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# chat() 内：入口 m.REQUEST_COUNT.labels(status="start").inc()，
# 用 m.REQUEST_LATENCY.labels(path="/api/chat").time() 包裹 graph.invoke，
# 限流触发处 m.RATE_LIMITED.inc()
```

`app/graph.py` 分类节点内 `m.TRIAGE_LABELS.labels(label=result.label).inc()`；`app/llm_client.py` 的 `except` 分支内 `m.LLM_ERRORS.inc()`。

- [ ] **Step 3: 验收（手动）**：`curl /metrics` 能看到 `medisense_*` 指标，发消息后计数增长

- [ ] **Step 4: Commit**

```bash
git add backend/app/metrics.py backend/app/main.py backend/app/graph.py backend/app/llm_client.py backend/requirements.txt
git commit -m "feat(metrics): prometheus metrics for requests, triage, llm errors"
```

---

## 任务 6：成本可观测（token 用量）

**Files:**
- Modify: `backend/app/metrics.py`
- Modify: `backend/app/llm_client.py`

**Interfaces:**
- 产出：`TOKEN_USAGE`（Counter，label `kind` 取值 `prompt`/`completion`）
- 消费：任务 5 的 `metrics.py`

- [ ] **Step 1: 加指标并埋点**

`app/metrics.py` 追加：
```python
TOKEN_USAGE = Counter("medisense_tokens_total", "Token usage", ["kind"])
```

`app/llm_client.py` 在 `generate_answer` 成功后读 usage：
```python
    response = _invoke(model, messages)
    usage = getattr(response, "usage_metadata", None)   # 字段名以实际 LangChain 版本为准
    if usage:
        m.TOKEN_USAGE.labels(kind="prompt").inc(usage.get("input_tokens", 0))
        m.TOKEN_USAGE.labels(kind="completion").inc(usage.get("output_tokens", 0))
    return response.content
```

- [ ] **Step 2: 验收（手动）**：`curl /metrics` 看到 `medisense_tokens_total{kind="prompt"}` 和 `{kind="completion"}`

- [ ] **Step 3: Commit**

```bash
git add backend/app/metrics.py backend/app/llm_client.py
git commit -m "feat(metrics): track LLM token usage for cost observability"
```

---

## 任务 7：数据漂移 + 低置信度处理

**Files:**
- Modify: `backend/app/metrics.py`
- Modify: `backend/app/graph.py`
- Test: `backend/tests/test_graph.py`

**Interfaces:**
- 产出：`TRIAGE_CONFIDENCE`（Histogram）、常量 `LOW_CONFIDENCE_THRESHOLD = 0.4`
- 消费：`graph.py` 路由加低置信度保守分支

- [ ] **Step 1: 加置信度分布指标**

`app/metrics.py` 追加：
```python
TRIAGE_CONFIDENCE = Histogram(
    "medisense_triage_confidence", "Triage confidence", ["label"],
    buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)
```

`app/graph.py` 分类节点内 `m.TRIAGE_CONFIDENCE.labels(label=result.label).observe(result.confidence)`。

- [ ] **Step 2: 低置信度保守路由**

`app/graph.py` 顶部加 `LOW_CONFIDENCE_THRESHOLD = 0.4`，`_route_after_triage` 改为：
```python
def _route_after_triage(state):
    label = state["triage_label"]
    conf = state["triage_confidence"]
    if label == "emergency" and conf >= EMERGENCY_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"
    if conf < LOW_CONFIDENCE_THRESHOLD:   # 分类器"没把握"，走安全兜底
        return "emergency_shortcut"
    return "retrieve"
```

- [ ] **Step 3: 写测试**

`tests/test_graph.py` 追加：构造 `triage_label="routine", triage_confidence=0.3` 的 state，断言 `_route_after_triage` 返回 `"emergency_shortcut"`。

- [ ] **Step 4: 跑测试确认通过** → `pytest tests/test_graph.py -v` PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/metrics.py backend/app/graph.py backend/tests/test_graph.py
git commit -m "feat(triage): confidence drift metric + low-confidence conservative routing"
```

---

## 任务 8：Eval 回归门禁

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `backend/requirements-dev.txt`
- Create: `backend/pyproject.toml`

**Interfaces:**
- 消费：`evals/run_evals.py` 已有退出码（`0 if pass_rate >= 0.8 else 1`）
- 产出：CI 跑 pytest + ruff + evals，改 prompt/检索/分类器后自动回归

- [ ] **Step 1: 建 CI（含 evals job，不含供应链扫描）**

`.github/workflows/ci.yml`：
```yaml
name: CI
on: [push, pull_request]
jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r backend/requirements.txt -r backend/requirements-dev.txt
      - run: ruff check backend/app backend/finetuning backend/evals
      - run: pytest backend
  evals:
    runs-on: ubuntu-latest
    needs: backend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r backend/requirements.txt
      - run: python backend/evals/run_evals.py
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

`backend/requirements-dev.txt`：
```
ruff==0.8.6
pytest-cov==6.0.0
```

`backend/pyproject.toml`：
```toml
[tool.ruff]
line-length = 120
[tool.pytest.ini_options]
addopts = "--cov=app --cov-fail-under=80"
```

- [ ] **Step 2: 仓库配置 secret**

GitHub → Settings → Secrets，新增 `GROQ_API_KEY`（单独建 eval 用 key，别复用生产 key）。

- [ ] **Step 3: 验收（手动）**：本地 `pytest` + `ruff check` 通过；推 PR 后 `backend` 和 `evals` job 都绿

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml backend/requirements-dev.txt backend/pyproject.toml
git commit -m "ci: add pytest + ruff + eval regression gate"
```

---

## 完成标准

- [ ] `pytest` 全绿（含新增用例）
- [ ] Groq 断网/超时时 `/api/chat` 返回友好降级而非 500
- [ ] 删除 triage adapter 后服务仍启动，`/health/ready` 显示 `degraded`
- [ ] 日志为 JSON 行、带 `X-Request-Id`，且**不含用户输入原文**
- [ ] `curl /metrics` 能看到 `medisense_*`（含 `medisense_tokens_total`、`medisense_triage_confidence`）
- [ ] 低置信度输入被路由到保守响应
- [ ] CI 全绿，含 `evals` 回归门禁

> 已按用户要求移除：Prompt 版本管理、供应链扫描（pip-audit/npm audit）。
