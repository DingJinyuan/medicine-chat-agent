# MediSense AI 企业级升级（P0 + P1）实现计划

> **For agentic workers:** 使用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务执行。步骤用 `- [ ]` 勾选跟踪。

**Goal:** 给 MediSense AI 补上生产级可靠性（P0）与企业级标配（P1），消除裸 500、单点故障和不可观测的运行态。

**Architecture:** 在现有 LangGraph 流水线外层加固，不改变 classify_triage → retrieve → generate → guardrail 的核心链路。

**Tech Stack:** 新增 tenacity（重试）、structlog（结构化日志）、prometheus-client（指标）、alembic（迁移）、ruff/mypy（质量）、GitHub Actions（CI）。

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
P0-1 LLM容错 ─────────────┐
P0-2 全局异常 ─────────────┤
P0-3 分类器fallback ───────┼──▶ 相互独立，可并行
P0-4 结构化日志 ───────────┘
        │
        ▼
P1-5 可观测性（依赖 P0-4 的 logger）
P1-6 安全强化（审计表依赖 P1-9 的 alembic）
P1-7 配置管理（独立）
P1-8 CI/CD（独立，含供应链扫描）
P1-9 数据库迁移（独立，P1-6 的前置）
P1-10 Eval 回归门禁（依赖 P1-8 的 CI）
P1-11 Prompt 版本管理（独立）
P1-12 成本可观测（依赖 P1-5 的 metrics）
P1-13 数据漂移监控（依赖 P1-5 的 metrics）
```

---

# P0 — 可靠性硬伤

## 任务 1：LLM 调用容错（重试 + 超时 + 降级回复）

**Files:**
- Modify: `backend/app/llm_client.py`
- Modify: `backend/requirements.txt`
- Test: `backend/tests/test_llm_client.py`

**Interfaces:**
- 产出：`generate_answer(settings, question, context_blocks) -> str`，签名不变，失败时返回 `FALLBACK_ANSWER` 常量而非抛异常
- 新增常量：`FALLBACK_ANSWER: str`，供任务 5 的指标和测试引用

- [ ] **Step 1: 加依赖并写失败测试**

`requirements.txt` 追加：
```
tenacity==9.0.0
```

`tests/test_llm_client.py` 追加：
```python
from app import llm_client

def test_generate_answer_retries_then_succeeds(mocker, settings):
    model = mocker.MagicMock()
    model.invoke.side_effect = [RuntimeError("timeout"), RuntimeError("timeout"), mocker.MagicMock(content="ok")]
    mocker.patch.object(llm_client, "get_chat_model", return_value=model)
    assert llm_client.generate_answer(settings, "q", ["ctx"]) == "ok"
    assert model.invoke.call_count == 3

def test_generate_answer_returns_fallback_on_total_failure(mocker, settings):
    model = mocker.MagicMock()
    model.invoke.side_effect = RuntimeError("boom")
    mocker.patch.object(llm_client, "get_chat_model", return_value=model)
    assert llm_client.generate_answer(settings, "q", ["ctx"]) == llm_client.FALLBACK_ANSWER
```

- [ ] **Step 2: 跑测试确认失败**

`pytest tests/test_llm_client.py -k fallback -v` → 预期 FAIL（`generate_answer` 仍会抛异常）

- [ ] **Step 3: 实现**

`app/llm_client.py` 改造 `generate_answer` 并新增常量和重试包装：
```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("medisense")

FALLBACK_ANSWER = (
    "I'm sorry, I couldn't reach my reference knowledge right now. "
    "Please try again in a moment. If this is urgent, contact a clinician "
    "or call your local emergency number."
)

@retry(stop_after_attempt=3, wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _invoke(model, messages):
    return model.invoke(messages)

def generate_answer(settings, question, context_blocks):
    model = get_chat_model(settings.groq_api_key, settings.groq_model)
    messages = build_rag_messages(question, context_blocks)
    try:
        return _invoke(model, messages).content
    except Exception as exc:
        # 不记录 question 原文（医疗隐私），只记长度和错误类型
        logger.exception("llm_generation_failed", input_len=len(question))
        return FALLBACK_ANSWER
```

- [ ] **Step 4: 跑测试确认通过**

`pytest tests/test_llm_client.py -v` → 预期 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/llm_client.py backend/requirements.txt backend/tests/test_llm_client.py
git commit -m "feat(llm): add retry + fallback answer for LLM failures"
```

---

## 任务 2：全局异常处理器 + 统一错误格式

**Files:**
- Create: `backend/app/errors.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/security.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- 产出：`AppError(Exception)`（含 `status/code/message`）、`error_response(status, code, message) -> JSONResponse`
- 消费：`security.py` 的 `require_api_key` 和限流改抛 `AppError`

- [ ] **Step 1: 写失败测试**

`tests/test_api.py` 追加：
```python
from app.errors import AppError

def test_unhandled_error_returns_structured_json(client):
    # 用一个会抛异常的端点模拟（通过 dependency 注入 mock）
    resp = client.post("/api/chat", json={"message": "hi"},
                       headers={"X-API-Key": "dev-local-key"})
    # 未配置真实依赖时走异常路径，断言响应体有 error.code
    assert "error" in resp.json()

def test_error_response_shape():
    r = AppError(status=503, code="llm_unavailable", message="LLM down")
    assert (r.status, r.code, r.message) == (503, "llm_unavailable", "LLM down")
```

- [ ] **Step 2: 跑测试确认失败** → 预期 FAIL（`app.errors` 不存在）

- [ ] **Step 3: 实现**

`app/errors.py`：
```python
from fastapi.responses import JSONResponse

class AppError(Exception):
    def __init__(self, status: int = 500, code: str = "internal_error", message: str = "Internal error"):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)

def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})
```

`app/main.py` 顶部 import 并注册：
```python
from app.errors import AppError, error_response
from fastapi.exceptions import RequestValidationError

@app.exception_handler(AppError)
async def _app_error_handler(request, exc: AppError):
    return error_response(exc.status, exc.code, exc.message)

@app.exception_handler(RequestValidationError)
async def _validation_handler(request, exc):
    return error_response(422, "validation_error", str(exc))

@app.exception_handler(Exception)
async def _unhandled_handler(request, exc: Exception):
    logger.exception("unhandled_error", path=request.url.path)
    return error_response(500, "internal_error", "An unexpected error occurred.")
```

`app/security.py` 两处改抛 AppError：
```python
from app.errors import AppError
# require_api_key 内
raise AppError(status=401, code="unauthorized", message="invalid or missing X-API-Key")
# rate limit 处（main.py）
raise AppError(status=429, code="rate_limited", message="rate limit exceeded, try again shortly")
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
- 修改：`get_triage_classifier(adapter_path, base_model)` 失败时返回 `ConservativeClassifier()` 而非抛异常

- [ ] **Step 1: 写失败测试**

`tests/test_triage_classifier.py` 追加：
```python
from app.triage_classifier import ConservativeClassifier, TriageResult, get_triage_classifier

def test_conservative_classifier_always_emergency():
    c = ConservativeClassifier()
    assert c.classify("any text").label == "emergency"
    assert c.classify("any text").confidence == 1.0

def test_get_triage_classifier_falls_back_on_missing_artifact(tmp_path, monkeypatch):
    # 指向一个不存在的目录 → 返回 ConservativeClassifier
    result = get_triage_classifier(str(tmp_path / "nope"), "distilbert-base-uncased")
    assert isinstance(result, ConservativeClassifier)
```

- [ ] **Step 2: 跑测试确认失败** → 预期 FAIL（`get_triage_classifier` 会抛 `FileNotFoundError`）

- [ ] **Step 3: 实现**

`app/triage_classifier.py`：
```python
class ConservativeClassifier:
    """Fallback: when the real model can't load, treat every input as emergency."""
    def classify(self, text: str) -> TriageResult:
        return TriageResult(label="emergency", confidence=1.0)

@lru_cache
def get_triage_classifier(adapter_path: str, base_model: str):
    try:
        return TriageClassifier(adapter_path, base_model)
    except Exception:
        logger.exception("triage_classifier_load_failed", adapter_path=adapter_path)
        return ConservativeClassifier()
```
（文件顶部补 `import logging; logger = logging.getLogger("medisense")`）

`app/main.py` lifespan 里把 classifier 挂到 state，新增就绪检查：
```python
app.state.triage_classifier = triage_classifier

@app.get("/health/ready")
async def readiness(request: Request):
    checks = {
        "triage_classifier": type(request.app.state.triage_classifier).__name__ == "TriageClassifier",
        "vector_store": request.app.state.vector_store is not None,
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

## 任务 4：结构化日志 + 请求 ID

**Files:**
- Create: `backend/app/logging_config.py`
- Modify: `backend/app/main.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- 产出：`setup_logging()`（在 `main.py` 启动时调用一次）
- 中间件在每条日志和响应头中注入 `request_id`

- [ ] **Step 1: 加依赖**

`requirements.txt` 追加：
```
structlog==25.1.0
```

- [ ] **Step 2: 实现**

`app/logging_config.py`：
```python
import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    )
```

`app/main.py`：移除 `logging.basicConfig(...)`，改：
```python
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

其余 `app/*.py` 里的 `logger = logging.getLogger("medisense")` 逐步替换为 `logger = structlog.get_logger("medisense")`（本任务只改 `main.py` 和 `llm_client.py`，其余在后续任务随手改）。

- [ ] **Step 3: 验收（手动）**

`uvicorn app.main:app --reload`，发一条请求，确认：日志是 JSON 行、含 `request_id`/`level`/`timestamp`，响应头含 `X-Request-Id`。

- [ ] **Step 4: Commit**

```bash
git add backend/app/logging_config.py backend/app/main.py backend/app/llm_client.py backend/requirements.txt
git commit -m "feat(logging): structlog JSON logging + request id middleware"
```

---

# P1 — 企业级标配

## 任务 5：可观测性（指标 + 黄金信号）

**Files:**
- Create: `backend/app/metrics.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/graph.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- 产出：`REQUEST_COUNT`、`REQUEST_LATENCY`、`LLM_ERRORS`、`TRIAGE_LABELS`、`RATE_LIMITED` 五个 prometheus 指标
- 消费：任务 1 的 `generate_answer`、任务 3 的 classifier，在调用点埋点

- [ ] **Step 1: 加依赖**

`requirements.txt` 追加：
```
prometheus-client==0.21.1
```

- [ ] **Step 2: 实现指标模块**

`app/metrics.py`：
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("medisense_requests_total", "Total chat requests", ["status"])
REQUEST_LATENCY = Histogram("medisense_request_duration_seconds", "Request latency", ["path"])
LLM_ERRORS = Counter("medisense_llm_errors_total", "LLM generation failures")
TRIAGE_LABELS = Counter("medisense_triage_total", "Triage predictions", ["label"])
RATE_LIMITED = Counter("medisense_rate_limited_total", "Rate limit rejections")
```

`app/main.py` 暴露端点并在请求前后埋点：
```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app import metrics as m

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# 在 chat() 内：入口 m.REQUEST_COUNT.labels(status="start").inc()，
# 用 m.REQUEST_LATENCY.labels(path="/api/chat").time() 包裹 graph.invoke，
# 限流触发处 m.RATE_LIMITED.inc()
```

`app/graph.py` 分类节点内 `m.TRIAGE_LABELS.labels(label=result.label).inc()`；`app/llm_client.py` 的 `except` 分支内 `m.LLM_ERRORS.inc()`。

- [ ] **Step 3: 验收（手动）**：`curl localhost:8000/metrics` 能看到 `medisense_*` 指标；发几条消息后计数增长。

- [ ] **Step 4: Commit**

```bash
git add backend/app/metrics.py backend/app/main.py backend/app/graph.py backend/app/llm_client.py backend/requirements.txt
git commit -m "feat(metrics): prometheus metrics for requests, triage, llm errors"
```

---

## 任务 6：安全强化（hash 比对 + 多 key + 审计日志）

**Files:**
- Modify: `backend/app/security.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/db.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_security.py`

**Interfaces:**
- 产出：`verify_api_key(key: str) -> bool`（`secrets.compare_digest` 比对 hash）、`record_audit(db, event, detail)`
- 消费：`main.py` 的 `/api/chat` 在 triage 决策和 guardrail 重写后调用 `record_audit`

- [ ] **Step 1: 写失败测试**

`tests/test_security.py` 追加：
```python
from app.security import verify_api_key

def test_verify_api_key_constant_time(monkeypatch, settings):
    monkeypatch.setattr(settings, "app_api_keys", "key1,key2")  # 多 key
    assert verify_api_key("key1") is True
    assert verify_api_key("key2") is True
    assert verify_api_key("wrong") is False
```

- [ ] **Step 2: 实现**

`app/config.py`：新增 `app_api_keys: str = "dev-local-key"`（逗号分隔多 key），保留旧 `app_api_key` 兼容。

`app/security.py`：
```python
import hashlib, secrets as _secrets

def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_api_key(key: str) -> bool:
    settings = get_settings()
    valid = [k.strip() for k in settings.app_api_keys.split(",") if k.strip()]
    return any(_secrets.compare_digest(_hash(key), _hash(v)) for v in valid)
```
`require_api_key` 改用 `verify_api_key(x_api_key)`。

`app/db.py`：新增 `audit_log` 表（见任务 9 迁移，本任务先用 `CREATE TABLE IF NOT EXISTS` 同步建表，任务 9 再纳入 alembic）：
```python
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
def record_audit(self, event: str, detail: str):
    with self._connect() as conn:
        conn.execute("INSERT INTO audit_log (event, detail, created_at) VALUES (?, ?, ?)",
                     (event, detail, _now()))
```

`app/main.py`：triage 结果和 `guardrail_rewritten=True` 时调用 `db.record_audit("triage", ...)` / `db.record_audit("guardrail_rewritten", ...)`。

- [ ] **Step 3: 跑测试确认通过** → `pytest tests/test_security.py -v` PASS

- [ ] **Step 4: Commit**

```bash
git add backend/app/security.py backend/app/config.py backend/app/db.py backend/app/main.py backend/tests/test_security.py
git commit -m "feat(security): multi-key auth with hash compare + audit log"
```

---

## 任务 7：配置管理（环境分离 + 校验）

**Files:**
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- 产出：`Settings.env: str`（`dev`/`prod`），`model_validator` 校验生产必填项

- [ ] **Step 1: 实现**

`app/config.py`：
```python
from pydantic import field_validator, model_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    env: str = "dev"          # 新增
    groq_api_key: str = ""
    app_api_keys: str = "dev-local-key"
    # ...其余不变

    @model_validator(mode="after")
    def _validate_prod(self):
        if self.env == "prod":
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY is required in prod")
            if self.app_api_keys in ("", "dev-local-key"):
                raise ValueError("APP_API_KEYS must not be the dev default in prod")
        return self
```

- [ ] **Step 2: 写测试**

`tests/test_api.py` 追加：
```python
def test_prod_settings_require_secrets(monkeypatch):
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(ValueError):
        Settings()
```

- [ ] **Step 3: 跑测试确认通过** → PASS

- [ ] **Step 4: Commit**

```bash
git add backend/app/config.py backend/tests/test_api.py
git commit -m "feat(config): env separation + prod secret validation"
```

---

## 任务 8：CI/CD + 代码质量

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.pre-commit-config.yaml`
- Create: `backend/pyproject.toml`
- Create: `backend/requirements-dev.txt`

- [ ] **Step 1: 实现**

`backend/pyproject.toml`（ruff + mypy + pytest 配置）：
```toml
[tool.ruff]
line-length = 120
[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
[tool.pytest.ini_options]
addopts = "--cov=app --cov-fail-under=80"
```

`backend/requirements-dev.txt`：
```
ruff==0.8.6
mypy==1.14.1
pytest-cov==6.0.0
```

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
      - run: mypy backend/app
      - run: pytest backend
      - run: pip install pip-audit && pip-audit -r backend/requirements.txt
  frontend:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: frontend } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - run: npm install
      - run: npm run lint
      - run: npm audit --audit-level=high
```

`.pre-commit-config.yaml`：
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks: [{ id: ruff }, { id: ruff-format }]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.1
    hooks: [{ id: mypy }]
```

- [ ] **Step 2: 验收（手动）**：本地 `ruff check backend/app` 跑通；推 GitHub 后 Actions 绿。

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml .pre-commit-config.yaml backend/pyproject.toml backend/requirements-dev.txt
git commit -m "ci: add GitHub Actions, ruff, mypy, coverage gate"
```

---

## 任务 9：数据库迁移（alembic）

**Files:**
- Create: `backend/alembic.ini`
- Create: `backend/migrations/env.py`
- Create: `backend/migrations/versions/0001_initial.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- 产出：`alembic upgrade head` 建出 `sessions`、`messages`、`audit_log` 三表
- 前置：任务 6 已建的 `audit_log` 表结构

- [ ] **Step 1: 加依赖**

`requirements.txt` 追加：
```
alembic==1.14.0
```

- [ ] **Step 2: 实现迁移**

`migrations/versions/0001_initial.py` 的 `upgrade()`：
```python
def upgrade():
    op.create_table("sessions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("created_at", sa.Text(), nullable=False))
    op.create_table("messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False))
    op.create_table("audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False))
```

`migrations/env.py` 用 `app.db.Database` 的 `db_path` 作为连接串（`sqlite:///<path>`）。

- [ ] **Step 3: 验收（手动）**：`alembic upgrade head` 后 `.db` 文件含三张表；`app/db.py` 里的 `SCHEMA` 保留（幂等，不冲突）。

- [ ] **Step 4: Commit**

```bash
git add backend/alembic.ini backend/migrations backend/requirements.txt
git commit -m "feat(db): add alembic migrations"
```

---

## 任务 10：Eval 回归门禁（每次改动自动跑 11 个安全用例）

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `backend/evals/run_evals.py`（确认退出码语义，无需改逻辑）

**Interfaces:**
- 消费：任务 8 的 CI、`run_evals.py` 已有退出码（`0 if pass_rate >= 0.8 else 1`）
- 产出：CI 里 `evals` job，改 prompt/检索/分类器后自动回归

- [ ] **Step 1: CI 加 evals job**

`.github/workflows/ci.yml` 追加：
```yaml
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

`run_evals.py` 已返回 `0 if pass_rate >= PASS_THRESHOLD else 1`，CI 直接用退出码判定，代码不用改。

- [ ] **Step 2: 仓库配置 secret**

GitHub → Settings → Secrets，新增 `GROQ_API_KEY`（单独建一个 eval 用 key，别复用生产 key）。

- [ ] **Step 3: 验收（手动）**

本地 `GROQ_API_KEY=... python -m evals.run_evals`，确认退出码非 0 时 CI 会 fail；推 PR 后 `evals` job 绿。

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add eval regression gate on golden dataset"
```

---

## 任务 11：Prompt 版本管理

**Files:**
- Create: `backend/app/prompts.py`
- Modify: `backend/app/llm_client.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_llm_client.py`

**Interfaces:**
- 产出：`PROMPT_VERSIONS: dict`、`DEFAULT_PROMPT_VERSION = "v1"`、`get_system_prompt(version) -> str`
- 消费：`llm_client.py` 的 `build_rag_messages` 从硬编码 `SYSTEM_PROMPT` 改为按版本读取

- [ ] **Step 1: 抽离 prompt**

`app/prompts.py`：
```python
PROMPT_VERSIONS = {
    "v1": """You are MediSense, a health-information assistant built as a portfolio demo.

Ground rules:
- You are NOT a doctor and this is NOT a real clinical tool...
""",   # 原文从 llm_client.py 原样迁来
}
DEFAULT_PROMPT_VERSION = "v1"

def get_system_prompt(version: str = DEFAULT_PROMPT_VERSION) -> str:
    return PROMPT_VERSIONS[version]
```

`app/config.py` 加：`prompt_version: str = "v1"`

`app/llm_client.py`：删除 `SYSTEM_PROMPT` 常量，`build_rag_messages` 改为：
```python
from app.prompts import get_system_prompt, DEFAULT_PROMPT_VERSION

def build_rag_messages(question, context_blocks, prompt_version=DEFAULT_PROMPT_VERSION):
    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant reference material found)"
    user_content = f"Reference context:\n{context}\n\nUser question: {question}"
    return [SystemMessage(content=get_system_prompt(prompt_version)), HumanMessage(content=user_content)]
```

- [ ] **Step 2: 跑测试确认不回归**

`pytest tests/test_llm_client.py -v` → PASS（prompt 内容未变，只改了来源）

- [ ] **Step 3: Commit**

```bash
git add backend/app/prompts.py backend/app/llm_client.py backend/app/config.py backend/tests/test_llm_client.py
git commit -m "refactor(prompts): extract system prompt to versioned module"
```

> 未来做 A/B：`PROMPT_VERSIONS` 加 `"v2"`，用配置项 `prompt_version` 切换，配合任务 10 的 eval 门禁对比通过率。

---

## 任务 12：成本可观测（token 用量）

**Files:**
- Modify: `backend/app/metrics.py`
- Modify: `backend/app/llm_client.py`

**Interfaces:**
- 产出：`TOKEN_USAGE`（Counter，label `kind` 取值 `prompt`/`completion`）
- 消费：任务 5 的 metrics 模块、任务 1 的 `generate_answer`

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

- [ ] **Step 2: 验收（手动）**

`curl /metrics` 看到 `medisense_tokens_total{kind="prompt"}` 和 `{kind="completion"}`，发消息后计数增长。

- [ ] **Step 3: Commit**

```bash
git add backend/app/metrics.py backend/app/llm_client.py
git commit -m "feat(metrics): track LLM token usage for cost observability"
```

---

## 任务 13：分类器数据漂移 + 低置信度处理

**Files:**
- Modify: `backend/app/metrics.py`
- Modify: `backend/app/graph.py`
- Test: `backend/tests/test_graph.py`

**Interfaces:**
- 产出：`TRIAGE_CONFIDENCE`（Histogram）、常量 `LOW_CONFIDENCE_THRESHOLD = 0.4`
- 消费：`graph.py` 路由，加低置信度保守分支

- [ ] **Step 1: 加置信度分布指标**

`app/metrics.py` 追加：
```python
TRIAGE_CONFIDENCE = Histogram(
    "medisense_triage_confidence", "Triage confidence", ["label"],
    buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)
```

`app/graph.py` 分类节点里 `m.TRIAGE_CONFIDENCE.labels(label=result.label).observe(result.confidence)`。

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

## 执行顺序与完成标准

1. P0 四个任务**可并行**（互不依赖），建议顺序 1→2→3→4
2. P1 中：9（迁移）先于 6（审计表）；5 依赖 4 的 logger；7、11 独立
3. P1 中：5（metrics）先于 12（成本）和 13（漂移）；8（CI）先于 10（eval 门禁）
4. 每个任务完成后跑对应测试；全部完成后 `pytest` 全绿 + `ruff check` 通过 + `python -m evals.run_evals` 仍 11/11（需 GROQ_API_KEY）

## 验证清单（全部完成后）

- [ ] `pytest` 全绿（含新增用例）
- [ ] Groq 断网/超时时 `/api/chat` 返回友好降级而非 500
- [ ] 删除 triage adapter 后服务仍启动，`/health/ready` 显示 `degraded`
- [ ] `curl /metrics` 能看到 `medisense_*` 指标（含 `medisense_tokens_total`、`medisense_triage_confidence`）
- [ ] 日志为 JSON 行，响应头带 `X-Request-Id`；**日志不含用户输入原文**（只有 `input_len`）
- [ ] 低置信度输入被路由到保守响应（不硬走检索）
- [ ] `alembic upgrade head` 成功建表
- [ ] prompt 可通过 `prompt_version` 配置切换
- [ ] CI（GitHub Actions）全绿，含 `evals` 回归门禁和供应链扫描
