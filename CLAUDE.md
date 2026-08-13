# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目是什么

MediSense AI 是一个医疗信息 RAG 问答机器人，定位是**作品集 demo（明确不是临床工具）**。它把 LangGraph 编排的 RAG 流水线（多厂商 LLM + FAISS 检索）和一个 LoRA 微调的分诊分类器组合在一起。知识库是 MedlinePlus 健康主题摘要（美国政府的公共领域内容），通过其 webservices API 抓取。

## 架构

```
Next.js 前端 (Vercel)              FastAPI 后端 (Render, Docker)
  聊天 UI + server route ──X-API-Key──▶  LangGraph 流水线 (app/graph.py)
                                         classify_triage ─▶ emergency_shortcut ─▶ output_guardrail ─▶ END
                                                      \──▶ retrieve ─▶ generate ─▶ output_guardrail ─▶ END
```

请求流是一个真正的 LangGraph 状态机（`app/graph.py`），不是顺序调用的 handler。当分诊分类器以 ≥0.6 的置信度判为 `emergency` 时，检索和生成被**完全跳过**，直接返回固定安全回复——LLM 根本看不到这条输入。

**两个模型，两种职责。** 本地 LoRA 分类器（`app/triage_classifier.py`）是确定性的、快的、不受 prompt 注入影响——它负责把关紧急判断。多厂商 LLM（`app/llm_client.py` 走 `app/llm_adapter.py` 的 LLMClient）只负责生成。要守住这个分离：分类器才是安全关键的那一环。

**三层防线**（`app/security.py`），对**每一条**响应路径都生效：
1. 输入侧——对检索到的 chunk 做 prompt 注入正则扫描
2. 检索侧——`wrap_untrusted()` 把检索文本用「这是数据不是指令」的方式包起来
3. 输出侧——`enforce_medical_guardrails()` 重写确定性诊断和具体剂量的措辞，并强制追加免责声明（连 emergency 短路路径也会跑这一步）

**训练与推理解耦。** `finetuning/` 只产出模型**文件**（权重在 `finetuning/artifacts/triage-lora/`）。运行中的后端只通过 `app/triage_classifier.py` 加载这些文件。重训或改分类器完全不用碰运行时代码——两者只通过 artifact 目录衔接。

**横切关注点**（企业级加固）：
- **统一错误格式**（`app/errors.py`）：所有错误返回 `{"error":{"code","message"}}`，错误码用 `ErrorCode` 枚举
- **可观测性**：structlog JSON 日志（`app/logging_config.py`，含 request_id 中间件）+ Prometheus 指标（`app/metrics.py`，`/metrics` 端点）
- **认证**：`/api/chat` 用 `Depends(require_api_key)` 验证 `X-API-Key`（前后端 key 需一致：后端 `APP_API_KEY` = 前端 `MEDISENSE_API_KEY`）
- **健康检查**：`/health`（存活）+ `/health/ready`（就绪，检查分类器/向量库/LLM 配置）
- **分类器兜底**：加载失败退化为 `ConservativeClassifier`（全当 emergency）；分类置信度 < 0.4 走保守路由

## 常用命令

### 后端（`backend/`）

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # 填入 LLM_MODEL_ID + 厂商 key（如 DEEPSEEK_API_KEY）
uvicorn app.main:app --reload                       # http://localhost:8000
```

注意：`requirements.txt` 把 torch 锁在 CPU wheel（`--extra-index-url .../whl/cpu`）。GPU 训练要用独立环境（比如带 CUDA torch 的 conda 环境）；运行后端本身不需要 CUDA。

### 测试 vs 评估（两码事）

- **`pytest`** —— 53 个离线单元测试，LLM 调用被 mock。检查的是代码路径：guardrail 正则、速率限制、ingestion 的 XML 解析、LangGraph 路由、认证。在 `backend/` 下运行。
- **`python -m evals.run_evals`** —— 端到端评估，调用**真实**多厂商 LLM 和**真实**训练好的分类器，跑 11 个 golden 用例（`evals/golden_dataset.json`）。需要有效的 LLM key（`LLM_MODEL_ID` + 厂商 key）。改了 system prompt、切块、检索或分诊模型后要跑。

```bash
pytest                               # 全部测试
pytest tests/test_security.py        # 单个文件
pytest -k "rate_limit"               # 按关键字
python -m evals.run_evals            # 需要 LLM key
```

### 微调（`backend/`）

```bash
# 微调脚本是 Jupyter notebook（finetuning/ 下三个 .ipynb），
# 在带 CUDA torch 的环境（conda base-llm）里按顺序跑：
#   prepare_dataset.ipynb → train_lora.ipynb → evaluate.ipynb
```

数据集是模板合成的；训练集和测试集用**不相交的句式模板**，所以 held-out 准确率反映的是泛化而非死记硬背。标签是 `emergency` / `urgent` / `routine` / `self_care`（id2label 顺序很重要——它存到 `finetuning/artifacts/triage-lora/label_map.json`，运行时读取）。改分类任务 = 改 `prepare_dataset.py`（LABELS + SYMPTOMS）+ 重训 + 确认 `app/config.py` 的 `triage_adapter_path` / `triage_base_model` 指向新产物。

### 前端（`frontend/`）

```bash
npm install
cp .env.example .env.local    # MEDISENSE_BACKEND_URL + MEDISENSE_API_KEY（须与后端 APP_API_KEY 一致）
npm run dev                   # http://localhost:3000
npm run lint
npm run build
```

## 约定与坑

- **Next.js 16 有 breaking changes。** 前端用的是 Next.js 16 / React 19，和旧版 Next.js 差异较大，可能与训练数据里的认知不符。`frontend/AGENTS.md`（以及 `frontend/CLAUDE.md`）明确要求：写前端代码前先读 `node_modules/next/dist/docs/`。要遵守。
- **API key 绝不进入浏览器。** 前端唯一调用后端的入口是 `src/app/api/chat/route.ts`（server route），它用 `X-API-Key` 代理转发到 FastAPI。key 只存在于服务端环境变量（`MEDISENSE_API_KEY`），不带 `NEXT_PUBLIC_` 前缀。
- **模型烘焙进 Docker 镜像**（`backend/Dockerfile`）：fastembed 的 embedding 模型和 distilbert base 模型在构建期预下载，`HF_HUB_OFFLINE=1` 跳过启动时的 Hub 版本检查。fastembed 的 `cache_dir` 显式放在 `/tmp` 之外，因为 Render 在容器运行时会挂载一个全新的 `/tmp`。改 Dockerfile 或 embedding 加载逻辑时要保留这一点。
- **后端内存吃紧**（分类器要 torch + transformers）。Render 免费层 512MB 在真实负载下可能 OOM；README 里标注了要升到 Starter 层。
- **配置**在 `app/config.py`（pydantic-settings，读 `.env`）。关键项：`llm_model_id` / `llm_fallback_models` / `llm_timeout`、`embedding_model`、`retrieval_top_k`、`triage_adapter_path`、`triage_base_model`、`rate_limit_per_minute`、`cors_origins`。CORS 锁定到部署的前端域名。
- **没有真实患者数据。** 聊天历史只进 SQLite（`app/db.py`），仅为演示连续性。
