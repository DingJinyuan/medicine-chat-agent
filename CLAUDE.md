# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

医学 AI 项目，包含**两套面向不同用户的工作流**，共享同一套技术底座（`common/`）和数据库（pgvector）：

1. **专家版（`expert_*`，给医生）**：多 Agent 临床 RAG（`query_understanding → retrieval → reasoning → critique`）+ 30 天再入院 ML 风险预测。数据源 PubMed 摘要 + 临床指南 PDF。
2. **患者版（`patient_*`，给普通人，MediSense）**：分诊安全问答（`classify_triage → 条件路由 → 应急短路/RAG → 输出护栏`），LoRA 分诊分类器 + 医疗安全护栏。数据源 MedlinePlus 健康主题。

本仓库不是 git 仓库。统一入口是 `backend/src/main.py`（FastAPI，`/api/chat` 按 `mode` 路由到专家版/患者版）。根目录 `main.py` 是 PyCharm 默认模板，与项目无关。

## 架构总览

### 目录结构（`backend/`）

```
backend/
├── src/                    运行时代码
│   ├── agents/             expert_* / patient_* 工作流节点 + build_pipeline
│   ├── service/            expert_* / patient_* 业务逻辑
│   ├── schemas/            chat_schemas.py（两版对话 Pydantic 模型：公共基类 + 专家/患者子类）
│   ├── common/             llm_adapter / logging_config / errors（两版共享）
│   ├── repository/         chunk_repo / chat_repo（数据访问）
│   ├── database/           models / connection（ORM + 引擎）
│   ├── ingestion/          数据摄取（pdf_parser / pubmed_fetcher / chunker / embedder / expert_ingest / patient_ingest）
│   ├── ml/                 推理模型（expert_risk_tool / patient_triage_classifier）
│   └── config.py           统一配置
├── risk_modeling/          专家版训练（expert_risk_model_training.ipynb，自包含 notebook）
├── expert_finetuning/      专家版微调（ai_medical_assistant_fine-tuning.ipynb）
├── patient_finetuning/     患者版微调（train_lora.ipynb 等）
├── tests/                  pytest 测试（15 文件,132 用例,覆盖率 80%）
├── evaluation/             评估脚本（run_evals / deepeval_eval / retrieval_eval / ir_metrics / benchmark）
└── data/                   数据（mimic / medlineplus_topics.json / patient_eval / eval / db）
```

`src` 是 Python 包，所有导入形如 `from src.config import settings`，**任何脚本必须在 `backend/` 目录下运行**。

### 两套工作流

**专家版**（`expert_pipeline.py`，线性）：
```
query_understanding → retrieval → reasoning → critique → END
```

**患者版**（`patient_pipeline.py`，条件分支）：
```
classify_triage ──紧急/低置信──> emergency_shortcut ──> output_guardrail
              └──正常────────> retrieve → generate ─> output_guardrail
```

### 数据摄取 pipeline（离线建库）

- **专家版**：`ingestion/expert_ingest.py` 的 `ExpertIngestionPipeline`（`run_pubmed` + `run_pdf` + `run`），PubMed 摘要 / PDF 指南 → chunker → embedder → pgvector。`run_pubmed` 按 `DEFAULT_QUERIES`（预定义 8 个医学主题）离线预抓 PubMed 摘要入库，**覆盖范围限于这几个主题**。
- **患者版**：`ingestion/patient_ingest.py`（`python -m src.ingestion.patient_ingest`），medlineplus_topics.json → chunker → embedder → pgvector。

### 依赖注入模式（两版统一）

- `build_pipeline(client, embedder, ...)` 接收依赖，节点用 `make_xxx(...)` 工厂函数闭包捕获依赖。
- 重依赖（`LLMClient`、`Embedder`、`Reranker`、`ReadmissionRiskTool`、`TriageClassifier`）由调用方（main）创建后注入，**不写模块级单例**。

### LLM 与外部依赖

- **LLM 统一走 `common/llm_adapter.py` 的 `LLMClient`**，走 `MODEL_PROFILES` 多厂商前缀匹配（deepseek/glm/gemini/gpt/qwen/kimi/minimax/ollama/claude-open/proxy），主备降级 + 重试 + Claude 适配。
- **结构化输出**：`common/llm_adapter.py` 的 `invoke_structured()` 用 function calling 强制 schema 输出；`query_understanding` 和 `critique` 已用它（`ClinicalQueryForm` / `FACT_CHECK_FUNCTION` schema）。
- **日志统一 structlog**（`common/logging_config.py`），按模块名分文件：`logs/expert.log`、`logs/patient.log`、`logs/app.log`（控制台输出全部）。
- **LangSmith 追踪** Agent/LLM 链路：两版 `*_pipeline.py` 顶部都设置 `LANGCHAIN_*` 环境变量（**不能删**）。
- 嵌入模型 `NeuML/pubmedbert-base-embeddings`（768 维，两版共用）；精排模型 `BAAI/bge-reranker-base`（仅专家版）。
- 数据库 PostgreSQL + pgvector + ParadeDB（BM25）。未部署 ParadeDB 时 BM25 降级为纯向量检索。

## 配置

**一个 `config.py`**（两版字段合并），pydantic-settings + `@lru_cache` 的 `get_settings()` + 模块级 `settings` 单例，字段统一小写 snake_case（`env` 大写）。

- 共享：`database_url`（必填）、`llm_model_id`、`llm_fallback_models`、`llm_timeout`、`embedding_model`、`chunk_size`、`langsmith_api_key` 等。
- 专家版专属：`pubmed_*`、`ncbi_api_key`、`rerank_model`、`langchain_*`。
- 患者版专属：`triage_*`、`retrieval_top_k`。API 服务（`cors_origins`/`app_api_key`/`rate_limit_per_minute`）是两版共用入口 `/api/chat` 的配置。

## 数据模型（pgvector，两版共用）

- `documents` / `chunks`：知识库。`documents.source` 区分 `pubmed` / `pdf` / `medlineplus`；患者版 chunk 的 `doc_id` 前缀是 `medlineplus_`。
- `chat_sessions` / `chat_messages`：会话。`chat_sessions.source` 区分 `expert` / `patient`。

## 已知问题 / 注意事项（改代码前先看这里）

1. **`llm_model_id` 默认值**：`config.py` 默认 `gpt-4o-mini`，需在 `.env` 配真实模型 id（走 `MODEL_PROFILES` 对应厂商 key）。
2. **专家版再入院风险模型不可靠**：训练数据是 MIMIC demo 子集（89 有效样本），CV AUC≈0.58 接近随机，需换完整 MIMIC-IV 重训才有效。
3. **本地 Llama 推理仅 GPU**：`reasoning` 节点默认走本地微调 Llama，Windows+CPU 下自动降级为 API client（bitsandbytes 仅 CUDA/Linux）。
4. **数据隔离**：专家版检索（`chunk_repo`）排除 `medlineplus_` 前缀；患者版检索只查 `medlineplus_`。两版数据不混用。
5. **NCBI API key 大小写敏感**：API 请求必须小写（NCBI 页面显示大写）。
6. **英文优先设计**：患者画像提取（`expert_risk_tool.extract_patient_profile_from_query`，英文关键词正则）和患者版分诊分类器（distilbert 英文基座 + 英文微调）只支持英文；中文输入会画像提取为空、分诊置信度低误判。**测试/测评 query 一律用英文。**
7. **专家版数据污染（暂不处理）**：PubMed 摄取未按语言过滤，T2DM 等主题检索到德文文档（DeepEval `contextual_recall=0`）。
8. **无记忆（单轮）**：`chat_repo.get_llm_history` 是死代码，会话历史从不回喂 LLM，两版都是单轮问答。
9. **检索语义理解弱**：只说症状不点主题名（模糊查询）时检索能力弱（MRR 0.37 vs 简单查询 0.90）。

## 测试与评估

### 单元测试（`backend/tests/`，pytest）

15 个测试文件、132 用例、覆盖率 80%，全部离线秒级（mock 掉 LLM/DB/模型）。分层：
- **L1 纯函数**：test_boundary / test_llm_client / test_security / test_ingestion / test_pdf_parser / test_pubmed_fetcher / test_expert_ingest / test_ir_metrics
- **L2 pipeline/节点**：test_pipeline_build / test_expert_pipeline / test_patient_pipeline / test_triage_classifier / test_expert_retrieval_agent / test_patient_retrieve_agent / test_expert_retrieval_service / test_patient_retrieval_service / test_patient_llm_service / test_expert_rerank
- **L3 API**：test_api
- **L4 端到端**：test_e2e（标 `@pytest.mark.slow`，默认跳过）

运行：`cd backend && .venv/Scripts/python -m pytest tests/ -v`（加 `--run-slow` 跑真实端到端）。

### 评估（`backend/evaluation/`，四层）

- ① 规则式 `run_evals.py`：安全护栏/分诊路由/拒答/注入/引用（读 `data/eval/` 的评估集）。
- ② DeepEval `deepeval_eval.py`：LLM-as-judge，忠实度/相关性/检索精准/检索召回（需 `pip install deepeval`）。
- ③ benchmark `benchmark/run_lm_eval.py`：PubMedQA（**已跳过**，`bigbio/pubmed_qa` 数据集已从 HF 下架）。
- ④ IR 指标 `ir_metrics.py` + `retrieval_eval.py`：hit@k / recall@k / MRR / NDCG（真实检索）。

评估集在 `backend/data/eval/`（golden_dataset 患者版 + expert_dataset 专家版）。完整评分见 `TEST_REPORT.md`。

## 数据文件

- `backend/data/mimic/`：MIMIC-IV 四张 CSV，供 `risk_modeling/expert_risk_model_training.ipynb` 训练专家版风险模型。
- `backend/data/medlineplus_topics.json`：患者版知识库（104 条 MedlinePlus 主题），`patient_ingest.py` 的输入。
- `backend/data/patient_eval/`：患者版分诊评估数据（`phase5_eval_context.json`、`base_results.json`、`finetuned_v2_results.json`）。
- `backend/data/db/mlflow.db`：MLflow 追踪的 SQLite，非运行必需。
- `data/guidelines/`（PDF 指南）和 `models/`（模型输出）当前不存在，运行时自动创建或需先放文件。
