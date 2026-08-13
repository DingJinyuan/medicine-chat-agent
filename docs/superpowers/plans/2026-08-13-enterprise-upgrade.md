# MediSense AI 企业级升级计划

**目标：** 把一个「能跑通的作品集 demo」升级为具备生产级可靠性、可观测性、安全性的企业级应用，不改变核心产品形态（RAG + triage 分类器的架构保持不变）。

**架构：** 在现有 LangGraph 流水线外层，补上企业级的横切关注点：容错/降级、可观测性、安全、配置管理、CI/质量。核心链路（classify_triage → retrieve → generate → guardrail）不动，只在外围加固。

**技术栈：** 现有 FastAPI + LangGraph + Groq + FAISS + PEFT，新增 tenacity（重试）、structlog（结构化日志）、prometheus-client（指标）、alembic（迁移）、GitHub Actions（CI）。

---

## 优先级总览

| 优先级 | 维度 | 解决什么 |
|---|---|---|
| **P0** | 可靠性硬伤 | 消除裸 500、单点故障、不可观测的运行态 |
| **P1** | 企业级标配 | 认证审计、配置/密钥管理、CI、数据迁移 |
| **P2** | 增强 | API 版本化、缓存、熔断 |

---

## P0 — 可靠性硬伤（先做）

> 直接对应上一轮指出的四个缺口：LLM 无容错、无全局异常处理、分类器无 fallback、日志只有 `basicConfig`。

### 任务 1：LLM 调用容错（重试 + 超时 + 降级回复）

**现状**：`app/llm_client.py:42-46` 的 `generate_answer` 直接 `model.invoke(messages)`，Groq 超时/限流/抖动时异常冒泡到 `main.py:67`，用户看到裸 500。

**怎么做**：用 `tenacity` 加指数退避重试，外层捕获异常返回友好降级回复。

```python
# app/llm_client.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.exceptions import OutputParserException

FALLBACK_ANSWER = (
    "I'm sorry, I couldn't reach my reference knowledge right now. "
    "Please try again in a moment. If this is urgent, contact a clinician "
    "or call your local emergency number."
)

@retry(
    stop_after_attempt=3,
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=False,
)
def _invoke(model, messages):
    return model.invoke(messages)

def generate_answer(settings, question, context_blocks):
    try:
        model = get_chat_model(settings.groq_api_key, settings.groq_model)
        response = _invoke(model, build_rag_messages(question, context_blocks))
        return response.content
    except Exception as exc:
        logger.error("llm_generation_failed", error=str(exc), question=question[:200])
        return FALLBACK_ANSWER
```

**涉及文件**：`app/llm_client.py`、`requirements.txt`（加 `tenacity`）
**验收**：`test_llm_client.py` 新增用例——mock `model.invoke` 前两次抛异常第三次成功，断言返回正常内容；三次全失败断言返回 `FALLBACK_ANSWER` 且不抛异常。

---

### 任务 2：全局异常处理器 + 统一错误格式

**现状**：`main.py` 没有 `@app.exception_handler`，任何未预期异常 → FastAPI 默认 500（返回 `{"detail": "Internal Server Error"}`，无 traceback、无结构）。

**怎么做**：注册全局异常处理器，统一错误响应体，同时把 traceback 打进日志。

```python
# app/main.py
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.errors import AppError, error_response

@app.exception_handler(AppError)
async def app_error_handler(request, exc: AppError):
    return error_response(status=exc.status, code=exc.code, message=exc.message)

@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    return error_response(status=422, code="validation_error", message=str(exc))

@app.exception_handler(Exception)
async def unhandled_handler(request, exc: Exception):
    logger.exception("unhandled_error", path=request.url.path)
    return error_response(status=500, code="internal_error",
                          message="An unexpected error occurred. Please try again later.")
```

```python
# app/errors.py（新增）
from dataclasses import dataclass
from fastapi.responses import JSONResponse

@dataclass
class AppError(Exception):
    status: int = 500
    code: str = "internal_error"
    message: str = "Internal error"

def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})
```

**涉及文件**：新建 `app/errors.py`、修改 `app/main.py`、`app/security.py`（`require_api_key` 和限流改抛 `AppError` 保持格式一致）
**验收**：任意未捕获异常时，返回 `{"error": {"code": "internal_error", "message": "..."}}`，且日志里有 traceback；`/api/chat` 传非法 JSON 返回 422 结构化错误。

---

### 任务 3：分类器加载 fallback + 依赖健康检查

**现状**：`main.py:32` lifespan 里 `get_triage_classifier()` 抛异常 → 整个应用启动失败，`/health` 都起不来。分类器是安全关键件，却无 fallback。

**怎么做**：分两层——

（a）**启动失败兜底**：分类器加载失败时进入「保守模式」，所有请求直接走 emergency 响应（宁可拒绝，不可漏放）。

```python
# app/triage_classifier.py 新增
class ConservativeClassifier:
    """Fallback when the real model can't load: treat every input as emergency."""
    def classify(self, text: str) -> TriageResult:
        return TriageResult(label="emergency", confidence=1.0)

def get_triage_classifier(adapter_path, base_model):
    try:
        return TriageClassifier(adapter_path, base_model)
    except Exception as exc:
        logger.error("triage_load_failed, using conservative fallback", error=str(exc))
        return ConservativeClassifier()
```

（b）**健康检查分级**：`/health` 返回就绪状态，`/health/ready` 检查所有下游依赖（分类器、向量库、Groq 可达性）。

```python
# app/main.py
@app.get("/health/ready")
async def readiness():
    checks = {
        "triage_classifier": isinstance(request.app.state.triage_classifier, TriageClassifier),
        "vector_store": request.app.state.vector_store is not None,
    }
    ready = all(checks.values())
    return {"status": "ok" if ready else "degraded", "checks": checks}
```

**涉及文件**：`app/triage_classifier.py`、`app/main.py`、`app/graph.py`（注入 classifier 实例而非闭包）
**验收**：删除/改名 adapter 目录后启动，服务仍能起，`/health/ready` 显示 `degraded`，发任意消息返回 emergency 固定回复。

---

### 任务 4：结构化日志 + 请求 ID

**现状**：`main.py:18` 只有 `logging.basicConfig(level=logging.INFO)`，代码里混用 `logger.info` 和 `print`（如 `evals/run_evals.py`）。

**怎么做**：换成 structlog，每条日志带 request_id 和关键字段；中间件给每个请求注入 request_id 并回写响应头。

```python
# app/logging_config.py（新增）
import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )

# app/main.py 中间件
@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response
```

**涉及文件**：新建 `app/logging_config.py`、修改 `app/main.py`、全局把 `print` 换成 `logger`（`evals/run_evals.py` 保留 `print` 因为它也是 CLI）
**验收**：日志输出为 JSON 行，每条含 `request_id`、`level`、`timestamp`；响应头带 `X-Request-Id`。

---

## P1 — 企业级标配

### 任务 5：可观测性（指标 + 追踪）

- **指标**：加 `prometheus-client`，暴露 `/metrics`，采集：请求数、延迟直方图、triage 分类分布、LLM 调用耗时/失败数、速率限制触发数。
- **追踪**：用 request_id 串联一次请求从 `classify_triage` 到 `output_guardrail` 各节点耗时，记进日志（成本最低，暂不引入 OpenTelemetry）。
- **告警**：定义两个黄金指标——`llm_error_rate`（>5% 告警）、`triage_conservative_fallback`（>0 告警）。

### 任务 6：安全强化

- **认证**：现状是单一 `APP_API_KEY` 明文比对。升级为：API key 存 hash（`secrets.compare_digest` 防时序攻击）、支持多 key 轮换。
- **审计**：`app/db.py` 增加 `audit_log` 表，记录每次 triage 决策、guardrail 重写、injection 命中（who/what/when）。
- **限流持久化**：`RateLimiter` 现状是内存字典（多实例失效），升级为可插拔后端，生产用 Redis。

### 任务 7：配置管理

- **环境分离**：`.env` 拆 `dev` / `prod`；`app/config.py` 对生产必填项（`groq_api_key`、`app_api_key`）加非空校验，缺了就启动失败并给出明确报错。
- **密钥**：生产密钥走 Render/Vercel 的 secret 注入，代码里不再有 `"dev-local-key"` 这种默认值落到生产（加 `ENV` 变量区分）。

### 任务 8：CI/CD + 代码质量

- **GitHub Actions**：`.github/workflows/ci.yml` —— push 触发，跑 `pytest` + `ruff` + `mypy` + 前端 `npm run lint`。
- **pre-commit**：`.pre-commit-config.yaml` 挂 ruff、black、mypy。
- **覆盖率门槛**：`pytest --cov=app --cov-fail-under=80`。

### 任务 9：数据库迁移

- **现状**：`app/db.py` 用 `CREATE TABLE IF NOT EXISTS`，改 schema 靠手工。
- **升级**：引入 `alembic`，把 schema 纳入版本管理；任务 6 的 `audit_log` 表就通过第一个 migration 创建。

---

## P2 — 增强

### 任务 10：API 版本化 + OpenAPI 文档
- 路由加 `/api/v1` 前缀，旧路径 301 或保留兼容；补全 OpenAPI 描述和示例。

### 任务 11：缓存层
- LLM 响应缓存（同问题+同检索结果 → 缓存回答，key 用问题 hash）；检索结果缓存。用 Redis 或简单 TTL 内存缓存。

### 任务 12：熔断器
- 对 Groq 调用加熔断器（连续失败 N 次 → 打开熔断，直接返回降级回复，冷却后半开试探）。tenacity 不够时用 `circuitbreaker` 库。

---

## 执行顺序建议

```
P0 任务 1 → 2 → 3 → 4   （一次性消除可靠性硬伤，彼此独立可并行）
        ↓
P1 任务 6 → 7 → 9 → 8 → 5  （安全→配置→迁移→CI→可观测，有部分依赖）
        ↓
P2 按需
```

## 拆分说明

本计划是**路线图**，每项只写到「方案级」。上面 12 个任务中，P0 的 1–4 是彼此独立、可各自展开成完整 bite-sized TDD 计划的最小单元，推荐从这里开始。选定后我会针对该任务写出含具体测试代码、逐条提交的执行计划。
