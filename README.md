# Medicine AI Pro

> 📖 **文档导航**：[功能说明](FEATURES.md) · [面试 QA](INTERVIEW_QA.md) · [手工测试](MANUAL_TEST.md)

医学 AI 项目，包含**两套面向不同用户的工作流**，共享同一套技术底座和数据库（PostgreSQL + pgvector）。

| | 专家版（医生） | 患者版（普通人） |
|---|---|---|
| 定位 | 临床决策支持 | 健康信息助手（MediSense） |
| 工作流 | 多 Agent RAG + ML 风险预测 | 分诊 → 安全问答 |
| 数据源 | PubMed 摘要 + 临床指南 PDF | MedlinePlus 健康主题 |
| 特色 | 文献引用、事实核查、再入院风险 | 紧急分诊短路、医疗安全护栏 |

## 架构

**专家版**（`expert_pipeline.py`，线性）：
```
query_understanding → retrieval → reasoning → critique → END
```
- 检索：多路召回（改写问题/关键词）→ BM25(ParadeDB) + cosine(pgvector) → RRF → cross-encoder 精排
- reasoning：检索上下文 + ML 风险 → LLM 生成带引用回答

**患者版**（`patient_pipeline.py`，条件分支）：
```
classify_triage ──紧急/低置信──> emergency_shortcut ──> output_guardrail
              └──正常────────> retrieve → generate ─> output_guardrail
```
- 分诊：LoRA 微调分类器判断 emergency/routine
- 检索：BM25 + cosine → RRF（无精排）
- 护栏：禁止诊断/剂量 + 强制免责声明 + 注入扫描

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活（Windows）
.venv\Scripts\activate
# 或 Git Bash
source .venv/Scripts/activate

# 3. 安装依赖（首次下载 torch + 模型，较慢）
pip install -r requirements.txt

# 4. 配置环境变量：backend/.env
#    必填 database_url；llm_model_id 走 MODEL_PROFILES 多厂商（配对应厂商的 API key）

# 5. 初始化数据库（建表 + pgvector 扩展 + BM25 索引）
cd backend
python -c "from src.database.connection import init_db; init_db()"

# 6. 导入患者版知识库（MedlinePlus → chunks 表，source='medlineplus'）
python -m src.ingestion.patient_ingest

# 7. 跑专家版流水线（当前无 CLI，直接调函数）
python -c "from src.agents.expert_pipeline import run_pipeline; print(run_pipeline('你的临床问题'))"
```

> 说明：代码包根是 `backend/`，所有脚本需在 `backend/` 下运行（`from src.xxx` 才能解析）。患者版 `build_pipeline` 的 main 入口尚未写，需自行创建依赖（`LLMClient`、`Embedder`、`TriageClassifier`）后调用。

## 前端

前端是 Next.js 16（App Router）+ TypeScript + Tailwind + shadcn/ui。入口是首页选择界面 `/`，点选进入专家版 `/expert` 或患者版 `/patient`。

```bash
cd frontend
npm install   # 下载前端依赖（node_modules）
npm run dev   # 启动开发服务器 http://localhost:3000
```

前端 `src/app/api/chat/route.ts` 作为代理转发到后端 `/api/chat`（server-only 环境变量，API key 不暴露给浏览器）。在 `frontend/.env.local` 配置：

```bash
MEDISENSE_BACKEND_URL=http://localhost:8000   # 后端地址
MEDISENSE_API_KEY=medisense-secret-2026       # 必须 = 后端 .env 的 APP_API_KEY
```

> 后端 `.env` 的 `APP_API_KEY` 必须与前端 `MEDISENSE_API_KEY` 一致，否则 `/api/chat` 返回 401。

## 测试

```bash
cd backend
../.venv/Scripts/python -m pytest tests/ -v                       # 全量 132 用例（mock LLM/DB/模型，离线秒级）
../.venv/Scripts/python -m pytest tests/ -m slow --run-slow -v   # 端到端（真实 LLM + 模型 + 数据库）
```

完整手工测试步骤见 [MANUAL_TEST.md](MANUAL_TEST.md)。

## 怎么提问最符合

> ⚠️ **提问请用英文**：患者画像提取和患者版分诊分类器只支持英文，中文会导致画像提取为空、分诊误判。

### 专家版（医生）

系统会自动补全含糊问题，但**信息越完整结果越准**：

- **说具体药名/病名**：`metformin` 比 `降糖药` 准，`heart failure` 比 `心脏病` 准。
- **风险预测要提供患者画像**（缺失用默认值兜底，但默认值 = 猜测）：

| 信息 | 示例 |
|---|---|
| 年龄 | `72岁` |
| 性别 | `男` / `女` |
| 是否急诊 | `急诊入院` |
| 住院天数 | `住院 5 天` |
| 共病史（7 类） | 糖尿病 / 心衰 / 高血压 / 肾病 / 肺炎 / 败血症 / 慢阻肺 |

示例：`72 岁男性，有心衰和糖尿病，急诊入院 5 天，metformin 还安全吗`

### 患者版（普通人）

分诊分类器会先判断紧不紧急，**描述症状要具体**：

- 说清**症状和部位**：`胸口闷痛、呼吸急促` 比 `不舒服` 更容易被正确分诊
- 紧急症状（胸痛、呼吸困难、中风迹象、大出血）会被自动识别并引导就医

## 数据准备

- **专家版知识库**：临床指南 PDF 放 `data/guidelines/`，PubMed 摘要用 `ingestion/expert_ingest.py`（`ExpertIngestionPipeline.run()`）抓取并入库——`DEFAULT_QUERIES` 预定义 8 个主题，覆盖范围限于这 8 个。
- **患者版知识库**：`data/medlineplus_topics.json`（104 条），跑 `python -m src.ingestion.patient_ingest` 导入。
- **MIMIC 数据**：`risk_modeling/mimic/*.csv`（demo 子集），供 `risk_modeling/expert_risk_model_training.ipynb` 训练再入院风险模型。
- **BM25 检索**：需 ParadeDB `pg_search`（Docker 镜像 `paradedb/paradedb`）；未部署时自动降级为纯向量检索。

## 需要下载的模型与依赖

### HuggingFace 模型（运行时自动下载，需联网）

| 模型 | 用途 | 体积参考 |
|---|---|---|
| `NeuML/pubmedbert-base-embeddings` | 嵌入（两版共用，768 维） | ~450MB |
| `BAAI/bge-reranker-base` | 精排（仅专家版） | ~1.1GB |
| `distilbert-base-uncased` | 分诊基座（患者版） | ~260MB |

首次运行会从 HuggingFace 拉取，可用 `HF_HOME` 指定缓存目录（默认 `~/.cache/huggingface`）。

### 本地模型文件（训练产物，非下载）

需先训练生成（或在仓库中已存在），否则对应节点会降级/报错：

| 路径 | 用途 |
|---|---|
| `backend/patient_finetuning/artifacts/triage-lora/` | 患者版分诊 LoRA + `label_map.json` |
| `backend/expert_finetuning/medical_lora_adapter` | 专家版 reasoning LoRA（可选，仅 GPU） |
| `backend/risk_modeling/readmission_model.json` | 再入院风险模型（XGBoost） |

### 数据库

PostgreSQL + pgvector + ParadeDB（BM25），Docker 镜像 `paradedb/paradedb`；未部署 ParadeDB 时 BM25 自动降级为纯向量检索。
