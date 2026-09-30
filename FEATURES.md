# 医学 AI 功能说明（面试用）

> 📖 **文档导航**：[README](README.md)（项目概览）· [功能说明](FEATURES.md)（本文）· [面试 QA](INTERVIEW_QA.md)

本文每个模块都是「**介绍（干什么、为什么）→ 代码（怎么实现）→ 设计要点（关键决策）**」三层，面试时既能讲清思路，也能落到代码。

---

## 一、项目介绍

**是什么**：一个医学 AI 项目，包含**两套面向不同用户的工作流**，共享同一套技术底座（`common/` + pgvector）：

1. **专家版**（给医生，`mode=expert`）：多 Agent 临床 RAG——`query_understanding → retrieval → reasoning → critique` 四节点，加一个 30 天再入院 ML 风险预测。数据源 PubMed 摘要 + 临床指南 PDF。
2. **患者版**（给普通人，MediSense，`mode=patient`）：分诊安全问答——先用 LoRA 分诊分类器判断紧急程度，紧急的走固定急救回复（大模型看不到输入），非紧急的走 RAG 检索 MedlinePlus 再生成，最后过输出安全护栏。

**为什么做**：医疗是 LLM 应用里「最不能出错」的领域。这个项目探索两条路线——**面向医生**：如何用多 Agent 协作 + 可追溯引用，把检索增强做到临床可用；**面向患者**：如何用工程手段把大模型的不可控性收敛到安全范围，确定性分类器把关安全关键决策、多层降级保证「宁可拒答、不可误答」。

**技术栈**：FastAPI + LangGraph（编排）、PostgreSQL + pgvector + ParadeDB BM25（检索）、XGBoost + distilbert/LoRA（ML 模型）、自研 LLMClient（10 厂商）、structlog + Prometheus + LangSmith（可观测）、Next.js 16（前端）。

**核心设计**：**双模型架构**贯穿两版——分类器管「紧急判断」（确定、快、防注入），LLM 管「生成」（有据可依）；专家版额外用「多 Agent 拆解 + 事实核查」把临床回答的可靠性从「一次生成」提升到「生成后再质疑」。

---

## 二、整体架构

```
浏览器 → Next.js server route（代理，key 留服务端）
  → FastAPI 后端统一入口 /api/chat（认证 → 限流 → 按 mode 路由）
  ├─ mode=expert  → expert_pipeline：query_understanding → retrieval → reasoning → critique
  └─ mode=patient → patient_pipeline：classify_triage → 条件路由 → 应急短路 / RAG → output_guardrail
  → 返回 answer +（专家版 citations/confidence/critique | 患者版 sources/triage）
```

一次请求对应一条 pipeline。两版共用：`LLMClient`、`Embedder`、pgvector 数据库、安全护栏、可观测性；靠 `doc_id` 前缀隔离数据（专家版检索排除 `medlineplus_`，患者版只查 `medlineplus_`）。

**依赖注入模式**：`build_pipeline(client, embedder, ...)` 接收依赖，节点用 `make_xxx(...)` 工厂函数闭包捕获依赖，重依赖由 `main.py` 创建后注入，**不写模块级单例**。

---

## 三、专家版 Pipeline：多 Agent 临床 RAG（`expert_pipeline.py`）

**介绍**：给医生的回答，不是「检索 + 一次生成」就完事，而是四个 Agent 串成一条线：先理解临床问题 → 多路检索证据 → 基于证据生成 → 再让另一个 Agent 质疑自己。为什么要拆：临床问答对「有据可依」要求极高，把「理解、检索、生成、质疑」拆成独立节点，每个节点职责单一、可独立测试、出问题可定位到具体节点。

```
拓扑（线性 DAG）：
    query_understanding → retrieval → reasoning → critique → END
```

```python
def build_pipeline(client, embedder, reranker, risk_tool, reasoning_llm):
    graph = StateGraph(AgentState)
    graph.add_node("query_understanding", make_query_understanding_agent(client))
    graph.add_node("retrieval", make_retrieval_agent(embedder, reranker))
    graph.add_node("reasoning", make_reasoning_agent(reasoning_llm, risk_tool))  # 注入 reasoning_llm
    graph.add_node("critique", make_critique_agent(client))
    graph.add_edge("query_understanding", "retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "critique")
    graph.add_edge("critique", END)
    return graph
```

**全局状态 `AgentState`**（TypedDict，字段）：
- 输入：`question`
- query_understanding 输出：`rewritten_query`、`patient_profile`、`keywords`
- retrieval 输出：`retrieved_chunks`
- reasoning 输出：`answer`、`citations`
- critique 输出：`critique`、`confidence_score`、`is_reliable`

### 节点 1 — query_understanding（临床问题理解）

**介绍**：用户问「72 岁心衰合并糖尿病，急诊住院 5 天，二甲双胍还安全吗」——这句话里有临床画像（年龄/共病/住院），也有检索意图。这个节点用 function calling 把「原始问题」重写成「更适合检索的 query」+ 提取关键词，同时用关键词正则提取患者画像（给下游风险模型用）。

```python
class ClinicalQueryForm(BaseModel):
    rewritten_query: str
    keywords: List[str] = Field(default_factory=list)

# function calling schema：强制 LLM 返回结构化字段
CLINICAL_QUERY_FUNCTION = {
    "name": "extract_clinical_query",
    "parameters": {..., "required": ["rewritten_query", "keywords"]},
}

# invoke_structured 用 tool_choice 锁定函数名强制 schema 输出
form = invoke_structured(client, messages, CLINICAL_QUERY_FUNCTION, temperature=0.1)
# 降级：无 tool_call 或异常 → 用原问题兜底
```

**设计要点**：
- **结构化输出走 function calling**：不是让 LLM 返回一段「看起来像 JSON」的文本再解析，而是 `tool_choice` 锁定函数名强制走 schema，返回就是合法的 `ClinicalQueryForm`
- **降级用原问题**：重写失败不影响主流程，检索照常进行

### 节点 2 — retrieval（多路混合检索 + 精排）

**介绍**：单一向量检索在医学场景不够——同义词、专有名词、罕见缩写会漏。这里用「多路召回 + RRF 融合 + 交叉编码器精排」三级检索，把召回率和精度都拉起来。

```python
RRF_K = 60          # RRF 融合常数
RECALL_K = 20       # 每路粗召回数量
FINAL_TOP_K = 5     # 精排后返回数量

def rrf_fusion(result_sets, k=RRF_K):
    # Σ 1/(k + rank + 1)，对多路结果按排名打分融合
    ...

def hybrid_recall(query):
    semantic = semantic_search_chunks(query)   # pgvector cosine
    bm25 = bm25_search_chunks(query)           # ParadeDB BM25（失败降级纯 cosine）
    return semantic, bm25

def multi_route_retrieve(question, rewritten_query, keywords, embedder, reranker):
    queries = list(dict.fromkeys([rewritten_query or question, " ".join(keywords)]))
    fused = rrf_fusion([hybrid_recall(q) for q in queries])
    return reranker.rerank(rewritten_query or question, fused, top_k=FINAL_TOP_K)
```

**设计要点**：
- **多路召回**：语义（向量）+ 关键词（BM25）互补，RRF 融合不依赖分数绝对大小、只比排名，融合稳健
- **BM25 降级**：未部署 ParadeDB 时自动降级纯向量，服务不因依赖缺失而挂
- **交叉编码器精排**：`BAAI/bge-reranker-base`（CrossEncoder）对融合结果重排，精排分数回写 chunk；reranker 加载失败降级 `chunks[:top_k]`

### 节点 3 — reasoning（基于证据生成 + 风险预测）

**介绍**：把检索到的证据拼成带 `[Source N]` 标记的上下文，再让 LLM 生成四段式临床回答；同时若患者画像非空且风险模型可用，插入一段 ML 风险预测。生成用「渐进式 skill 加载」——从 `src/skills/clinical-reasoning/SKILL.md` 读临床推理系统提示词，文件不存在才回退内置 prompt。

```python
def build_context_and_citations(chunks):
    # 每块格式 [Source {i} | {doc_id} | score: {score}]，citations 收集 doc_id
    ...

def compute_risk_section(patient_profile, risk_tool):
    if risk_tool.is_available() and patient_profile:
        return risk_tool.predict(patient_profile)  # 生成 ML 风险区块
    return None

result = client.invoke(messages, temperature=0.1, tag="reasoning")
```

**设计要点**：
- **reasoning 节点注入 `reasoning_llm`**：默认走 API client；`REASONING_USE_LOCAL=true` 时才切本地微调 Llama（`unsloth/llama-3.2-3b` + LoRA），无 GPU 自动降级 API
- **引用可追溯**：每个 claim 带 `[Source N]`，citations 收集 PubMed/PDF 的 `doc_id` 返回前端

### 节点 4 — critique（事实核查）

**介绍**：生成完不是直接返回，而是让**另一个 Agent 当裁判**——把「问题 + 回答 + 证据」喂回去，用 function calling 强制它给出「批评 + 置信度 + 是否可靠」，作为最终回答的可靠性标签。

```python
FACT_CHECK_FUNCTION = {
    "name": "submit_fact_check",
    "parameters": {
        "critique": {"type": "string"},        # required
        "confidence": {"type": "number"},       # 0.0~1.0, required
        "reliable": {"type": "boolean"},        # required
    },
}

critique, confidence, reliable = run_fact_check(question, answer, chunks, client)
# 置信度钳制 max(0.0, min(1.0, ...))；降级值 ("", 0.5, False)
```

**设计要点**：
- **自质疑机制**：用「LLM 质疑 LLM」降低幻觉——即使 reasoning 生成了不可靠内容，critique 也会给出低置信度 + 不可靠标签，前端据此降级展示
- **ML 风险区块被标记为独立证据源**：critique 的 prompt 明确「风险预测来自独立 ML 模型，不得判为幻觉」

---

## 四、患者版 Pipeline：分诊安全问答（`patient_pipeline.py`）

**介绍**：给普通人的健康问答，安全优先。先分诊——紧急高置信度直接短路（LLM 看不到输入），低置信度保守兜底，只有「比较确定是普通症状」才走 RAG。两条路径最后都过输出护栏。

```
classify_triage ──紧急(≥0.6)/低置信度(<0.4)──> emergency_shortcut ──> output_guardrail → END
              └──正常──────────────────────> retrieve → generate ──> output_guardrail → END
```

```python
EMERGENCY_CONFIDENCE_THRESHOLD = 0.6   # 高置信度才敢短路
LOW_CONFIDENCE_THRESHOLD = 0.4         # 低置信度不敢硬走

def route_after_triage(state):
    if state["triage_label"] == "emergency" and state["triage_confidence"] >= EMERGENCY_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"
    if state["triage_confidence"] < LOW_CONFIDENCE_THRESHOLD:
        return "emergency_shortcut"      # 任何标签低置信度 → 保守兜底
    return "retrieve"

def emergency_shortcut_node(state):
    # 写死 EMERGENCY_RESPONSE（指向 911/急诊室），清空 sources，LLM 完全不接触输入
    return {**state, "answer": EMERGENCY_RESPONSE, "context_blocks": [], "sources": [], "injection_flagged": False}
```

**设计要点**：
- **两个阈值**：`0.6` 是「高置信度才敢短路」，`0.4` 是「低置信度不敢硬走」——覆盖安全决策的两个方向
- **两条分支汇聚到 output_guardrail**：紧急短路和正常路径都过输出护栏，保证安全审查对每条路径生效
- **分诊节点埋指标**：`m.TRIAGE_LABELS.labels(label=...).inc()` 统计分诊分布

---

## 五、LoRA 分诊分类器（`ml/patient_triage_classifier.py`）

**介绍**：用 LoRA 把 distilbert（67M 参数）微调成 4 分类（emergency/urgent/routine/self_care）的症状紧急程度分类器。**为什么不用 LLM 判断紧急**：延迟（本地毫秒级 vs 网络）、可靠性（确定性 vs 可被注入带偏）、安全（独立小模型是最后防线）。

```python
class TriageClassifier:
    def __init__(self, adapter_path, base_model):
        label_map = json.loads(Path(adapter_path, "label_map.json").read_text())
        self.id2label = {int(k): v for k, v in label_map["id2label"].items()}
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=4)
        self.model = PeftModel.from_pretrained(base, adapter_path)   # 挂 LoRA 适配器
        self.model.eval()

    @torch.no_grad()
    def classify(self, text):
        inputs = self.tokenizer(text, max_length=64, return_tensors="pt")
        probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        idx = torch.argmax(probs).item()
        return TriageResult(label=self.id2label[idx], confidence=float(probs[idx]))
```

**设计要点**：
- **id2label 从文件读**：训练时存了 `label_map.json`，推理端读同一个文件，保证标签顺序一致
- **兜底**：加载失败退化为 `ConservativeClassifier`（全判 emergency，置信度 1.0），且 **lru_cache 只缓存成功结果**（失败抛异常不进缓存，避免 adapter 修好后服务仍卡在保守模式）

---

## 六、再入院风险模型（`ml/expert_risk_tool.py`）

**介绍**：专家版的一个差异化能力——从临床问句里提取患者画像，喂给 XGBoost 模型预测 **30 天再入院概率**，作为回答里的 `Patient Risk Assessment` 区块。这是「LLM 之外的确定性信号」，与生成内容互补。

```python
# 15 个特征（顺序固定）
FEATURE_COLS = ["age", "gender_m", "admission_type_emergency", "los_days",
                "diagnosis_count", "has_diabetes", "has_heart_failure",
                "has_hypertension", "has_renal_disease", "has_pneumonia",
                "has_sepsis", "has_copd", "num_icu_stays", "total_icu_los", "max_icu_los"]

def predict(self, profile):
    score = self.model.predict_proba(features)[0][1]   # 30 天再入院概率
    level = "HIGH" if score >= 0.6 else ("MODERATE" if score >= 0.3 else "LOW")
    return {"risk_score": round(score, 4), "risk_level": level, "interpretation": ...}
```

**画像提取 `extract_patient_profile_from_query`（英文关键词/正则）**：
- 年龄 `(\d+)[- ]?year[s]?[- ]?old`；性别 `male`/`female`（用词边界避免 female 误匹配 male）
- 急诊 `"emergency"`/`"urgent"`；住院天数正则（要求住院语境，兼容三种写法）
- 7 类共病映射：diabetes / heart failure / hypertension / renal disease / pneumonia / sepsis / copd

**设计要点**：
- **缺失用保守默认值**（age=65、los_days=5 等），画像提取不到也能出结果
- **诚实标注**：训练数据是 MIMIC demo 子集（89 有效样本），CV AUC≈0.58 接近随机——**风险分数当前只作演示，换完整 MIMIC-IV 重训才有效**（这是面试里主动暴露的不足，见第十六节）

---

## 七、多厂商 LLM 适配层（`common/llm_adapter.py`）

**介绍**：自研的多厂商适配层，10 个厂商前缀路由 + 主备降级 + 重试 + 原生 Claude 协议转换。**为什么不用 LangChain 的封装**：需要「多厂商 + 自定义 fallback + 协议转换 + function calling 强 schema」的控制力。

```python
MODEL_PROFILES = {
    "deepseek": {"api_key": "DEEPSEEK_API_KEY"},       # extra_body 关 thinking
    "glm":      {"api_key": "GLM_API_KEY"},
    "gemini":   {"api_key": "GOOGLE_API_KEY"},
    "gpt":      {"api_key": "OPENAI_API_KEY"},          # 无 base_url
    "qwen":     {"api_key": "QWEN_API_KEY"},            # enable_thinking=False
    "kimi":     {"api_key": "MOONSHOT_API_KEY"},
    "minimax":  {"api_key": "MINIMAX_API_KEY"},
    "ollama":   {"api_key": "OLLAMA_API_KEY"},
    "claude-open": {"api_key": "CLAUDE_OPEN_API_KEY"},  # OpenAI 兼容中转
    "proxy":    {"api_key": "PROXY_API_KEY"},
}

def _resolve_model_env(model):
    for prefix, profile in MODEL_PROFILES.items():
        if model.lower().startswith(prefix):
            return os.getenv(profile["api_key"]), ...
    raise ValueError(f"未知模型 '{model}'")
```

**主备降级链 + 重试 + 指数退避**：

```python
def invoke(self, messages, ...):
    for model in [self.model] + self.fallback_models:      # 主模型 → 备用模型列表
        self._init_client(model)
        for retry in range(self.max_retries):              # max_retries=2
            try:
                return self._call(messages, ...)
            except Exception as e:
                if not _is_retryable_error(e): break       # 不可重试，切备用
                base = min(1.0 * (2 ** retry), 32.0)       # 指数退避封顶 32s
                time.sleep(base + random.uniform(0, base * 0.25))  # + 抖动
    raise RuntimeError("所有模型调用失败")
```

**结构化输出 `invoke_structured`**：把 schema 包成 `tools=[{"type":"function","function":schema}]`，`tool_choice` 锁定函数名**强制**该函数，返回 `tool_calls[0].function.arguments` 的 JSON；无 tool_call 返回 None 由调用方降级。

**设计要点**：
- **可重试错误分类**：`json`/`401`/`400`/`402`/`model not found` → 不重试（重试没用）；超时/5xx/429 → 重试
- **指数退避 + 抖动**：抖动防止大量请求同时重试打挂服务（雪崩）
- **Claude 协议转换**：`is_claude`（且非 `claude-open`）走原生 anthropic SDK，`_translate_messages` 做消息互转（system 抽顶层、tool 结果包成 user 的 tool_result），上层无感知底层厂商

---

## 八、RAG 链路：三数据源 + 多路混合检索

**介绍**：整条链路分两半——**离线**（建库）把三源知识备成 pgvector 向量 + BM25 索引，**在线**（每次查询）走「多路召回 → RRF 融合 → 精排 → 生成」。知识源按用户分：
- 专家版：PubMed 摘要（8 个预定义主题）+ 临床指南 PDF
- 患者版：MedlinePlus 健康主题（104 条）

### 离线：数据摄取（`ingestion/`）

**① PubMed 抓取**（`pubmed_fetcher.py` + `expert_ingest.py`）：`DEFAULT_QUERIES` 8 个主题（heart failure / type 2 diabetes / sepsis / community acquired pneumonia / hypertension / chronic kidney disease / acute myocardial infarction / stroke），esearch 拿 PMID → efetch 拿 XML → 解析出 `PubMedArticle`（pmid/title/abstract/authors/doi），按 PMID 去重 + `_exists(doc_id=f"pubmed_{pmid}")` 幂等跳过。限流：有 NCBI key 10 req/s，无 key 3 req/s。

**② PDF 解析**（`pdf_parser.py`）：PyMuPDF（fitz）`page.get_text("text")`，按页打 `[Page N]` 标记，空文本判为扫描版返回 None，`source="pdf"`、`doc_id=f"pdf_{stem}"`。

**③ MedlinePlus**（`patient_ingest.py`）：读 `data/medlineplus_topics.json`（104 条），`_make_doc_id` 生成 `medlineplus_{slug}`，chunk meta 带 `{topic,url,chunk_index}`，按 doc_id 幂等跳过。

**④ 切块 + 向量化**（`chunker.py` + `embedder.py`）：

```python
chunker = TextChunker(chunk_size=512, chunk_overlap=50)      # tiktoken cl100k_base
embedder = SentenceTransformer("NeuML/pubmedbert-base-embeddings")  # 768 维，L2 归一化
```

**设计要点**：
- **三源 doc_id 前缀隔离**：`pubmed_` / `pdf_` / `medlineplus_`，专家版检索 `notlike 'medlineplus_%'`、患者版 `like 'medlineplus_%'`，两版数据不混用
- **幂等摄取**：按 `doc_id` 查存在跳过，重复跑不产生重复数据
- **切块用 tiktoken 按 token 数**：比按字符数更接近模型实际切分，`_split_sentences` 用正则 `(?<=[.!?])\s+(?=[A-Z])` 避免 `Dr.`/`mg.` 误切

---

## 九、数据库设计（`database/models.py`，pgvector）

**介绍**：业务数据（聊天历史）和知识库（向量）统一放 PostgreSQL + pgvector，不再分「SQLite + FAISS 文件」两套。**四个实体（表）分两组**：知识侧 `documents` / `chunks`（管「证据从哪来」），对话侧 `chat_sessions` / `chat_messages`（管「谁在问、说了啥」）。

### 实体关系（ER）

```
知识侧      documents 1 ────── N chunks          （一个文档切多个 chunk）
对话侧      chat_sessions 1 ─── N chat_messages   （一个会话多条消息）

关联方式：
· chunks.document_id      →  documents.id         逻辑外键（未声明 DB 级 FK，靠应用层维护）
· chunks.doc_id           →  documents.doc_id     冗余存 doc_id，检索命中后直接拿引用、免 join
· chat_messages.session_id → chat_sessions.id      真实 FK（ondelete=CASCADE，删会话级联删消息）
```

### 四个实体

| 实体 | 职责 | 关键字段 | 区分维度 |
|---|---|---|---|
| `documents` | 知识文档（一篇 PubMed 摘要 / PDF 指南 / MedlinePlus 主题） | `source`、`doc_id`(unique)、`title`、`authors`、`abstract`、`doi` | `source`：pubmed / pdf / medlineplus |
| `chunks` | 文档切块 + 向量（documents 的 1:N 子表） | `embedding`(Vector(768))、`document_id`、`doc_id`、`chunk_index` | `doc_id` 前缀：pubmed_ / pdf_ / medlineplus_ |
| `chat_sessions` | 一次对话会话 | `id`(PK, 无自增)、`created_at` | `source`：expert / patient |
| `chat_messages` | 会话里的一条消息（chat_sessions 的 1:N 子表） | `session_id`(FK)、`role`、`content` | `role`：user / assistant |

**两版数据隔离不靠分表，靠 `doc_id` 前缀**：专家版检索 `doc_id NOT LIKE 'medlineplus_%'`（只命中 pubmed/pdf），患者版 `doc_id LIKE 'medlineplus_%'`（只命中健康科普）。注意 `documents.source` 和 `chat_sessions.source` 是两个**不同维度**——前者区分「知识来源」，后者区分「对话版本」，别混。

```python
class Document(Base):       # documents：知识文档（实体，1 端）
    source = Column(String(50))          # 'pubmed' / 'pdf' / 'medlineplus'
    doc_id = Column(String(100), unique=True)   # pubmed_xxx / pdf_xxx / medlineplus_xxx
    title, authors(JSON), abstract, publication_date, journal, doi, full_text, meta(JSON)

class Chunk(Base):          # chunks：切块 + 向量（documents 的 N 端）
    document_id = Column(Integer)        # 逻辑外键 → documents.id（未声明 FK 约束）
    doc_id = Column(String(100))         # 冗余：检索命中直接拿引用，免 join
    chunk_index, content, token_count
    embedding = Column(Vector(768))      # pgvector
    meta(JSON)

class ChatSession(Base):    # chat_sessions：对话会话（实体，1 端）
    id = Column(String, primary_key=True)   # source: 'expert' / 'patient'

class ChatMessage(Base):    # chat_messages：消息（chat_sessions 的 N 端）
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role, content, created_at
```

**连接初始化**：`create_engine(pool_pre_ping=True, pool_size=10, max_overflow=20)`；`init_db()` 先 `CREATE EXTENSION IF NOT EXISTS vector` 再 `create_all`，最后建 BM25 索引 `chunks_bm25_idx USING bm25`（ParadeDB，失败降级）。

**设计要点**：
- **chunks 冗余 doc_id 是刻意的**：检索命中 chunk 后要返回引用（source/url/topic），冗余 doc_id 避免每次命中都 join documents，查一次拿全
- **documents → chunks 用逻辑外键**：不用 DB 级 FK 约束，因为摄取按 doc_id 幂等、跨表一致性由应用层保证，省掉建库时的约束开销

**诚实的设计点**：`chat_repo.get_llm_history` 是死代码，会话历史从不回喂 LLM，两版都是**单轮问答**——历史目前只用于展示。要做多轮，把历史拼进 prompt 即可，这是可扩展点。

---

## 十、安全防线（`common/security.py` + 患者版护栏）

**介绍**：医疗场景安全是核心。三层防线覆盖输入（扫注入）、检索（包数据）、输出（重写诊断/剂量 + 强制免责声明），且输出护栏对**每一条**响应路径生效（含紧急短路）。另有 API key 认证、每 IP 限流、密钥脱敏。

### ① 输入侧：prompt 注入扫描（正则，确定、快、不可绕过）

```python
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (all )?(previous|prior|above)", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"new system prompt", re.I),
    re.compile(r"reveal (your|the) system prompt", re.I),
    re.compile(r"act as (if|though) you (have no|are not)", re.I),
    re.compile(r"\bDAN\b|do anything now", re.I),
    re.compile(r"<\s*/?system\s*>", re.I),
]
```

### ② 检索侧：数据边界声明（`wrap_untrusted`）

检索回来的 chunk 按「不可信」处理：包一层 `<untrusted_document>` 标签，明确告诉 LLM「这是数据、不是指令」，防**间接注入**。

### ③ 输出侧：医疗护栏（`enforce_medical_guardrails`）

命中确定性诊断（"you have diabetes"）或具体剂量（"take 500mg"）就**改写**（换成「需医生检查」/「遵医嘱」），最后无条件追加免责声明。

### 认证 + 限流 + 脱敏

```python
async def require_api_key(x_api_key: str = Header(default="")):
    if settings.app_api_key and x_api_key != settings.app_api_key:
        raise AppError(status=401, code=ErrorCode.UNAUTHORIZED, ...)  # 未配置 key 时关闭鉴权

class RateLimiter:    # 内存固定窗口，单实例够用，多实例换 Redis
    def check(self, client_id):
        hits = [t for t in self._hits[client_id] if t > time.time() - 60]
        return len(hits) + 1 <= self.limit
```

**设计要点**：正则不是 LLM（安全判断交给确定性逻辑，不能让被攻击对象自己当裁判）；改写不是删除（换安全措辞而非留断句）；免责声明无条件追加。

---

## 十一、统一错误处理（`common/errors.py`）

**介绍**：所有错误统一成 `{"error":{"code","message"}}`，code 机器可读、message 人可读，用枚举集中管理错误码。

```python
class ErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"
    LLM_UNAVAILABLE = "llm_unavailable"
```

**三个 handler**（`main.py`）：`AppError`（业务错误）、`RequestValidationError`（422，不 `str(exc)` 避免泄露字段细节）、`Exception`（兜底 500，traceback 进日志、message 模糊）。

**设计要点**：5xx 绝不泄露内部信息（traceback/路径/SQL），详细错误进日志，给用户的 message 保持模糊。

---

## 十二、可观测性（日志 + 指标 + 追踪）

**介绍**：三层——structlog JSON 日志（按模块分文件）+ Prometheus 指标（埋点 + `/metrics`）+ LangSmith 追踪 Agent/LLM 链路。日志不记录用户输入原文（医疗 PHI 红线）。

- **日志**（`common/logging_config.py`）：`logs/expert.log`、`logs/patient.log`、`logs/app.log`，全 JSON；`X-Request-Id` 中间件用 `structlog.contextvars.bind_contextvars` 绑定，一个请求从头到尾所有日志带同一 id。
- **指标**（`src/metrics.py`）：`REQUEST_COUNT`、`REQUEST_LATENCY`、`LLM_ERRORS`、`TRIAGE_LABELS`、`RATE_LIMITED`、`TOKEN_USAGE`；`/metrics` 端点暴露。
- **追踪**：两版 pipeline 顶部设置 `LANGCHAIN_TRACING_V2` / `LANGCHAIN_PROJECT=clinicalagent`，LangSmith 追踪 Agent 节点调用链路。

**要点**：不记用户原文（LLM 失败只记 `input_len`）；指标覆盖成本（token 用量）与安全（分诊分布、限流、LLM 错误）。

---

## 十三、测试与评估

**介绍**：132 个单元测试（mock LLM/DB/模型，离线秒级）+ 5 个 slow 端到端，覆盖率 80%；评估分四层——规则式（安全/路由确定性）、DeepEval（LLM-as-judge 生成质量）、benchmark（底座模型知识）、IR 指标（检索质量）。

### 单元测试（`tests/`，20 个文件）

分层：
- **L1 纯函数**：`test_boundary`（44 项：护栏正则/限流/RRF/风险分层/画像提取）、`test_llm_client`、`test_security`、`test_ingestion`、`test_pdf_parser`、`test_pubmed_fetcher`、`test_ir_metrics`
- **L2 pipeline/节点**：`test_pipeline_build`、`test_expert_pipeline`、`test_patient_pipeline`、`test_triage_classifier`、`test_expert_retrieval_agent/service`、`test_patient_retrieve_agent` 等
- **L3 API**：`test_api`（health/chat/history/限流/metrics）
- **L4 端到端**：`test_e2e`（标 `@pytest.mark.slow`，真实 LLM + pgvector + 模型，`--run-slow` 手动跑）

### 四层评估（`evaluation/`）

| 层 | 工具 | 测什么 |
|---|---|---|
| ① 规则式 | `run_evals.py` | 安全护栏/分诊路由/拒答/注入/引用（golden 12 + expert 5 case，PASS 阈值 0.8） |
| ② LLM-judge | `deepeval_eval.py` | DeepEval 忠实度/相关性/检索精准/召回（judge=DeepSeek，阈值 0.7） |
| ③ benchmark | `benchmark/run_lm_eval.py` | PubMedQA 底座模型医学知识（**已跳过**，数据集下架） |
| ④ IR 指标 | `ir_metrics.py` + `retrieval_eval.py` | hit@k / recall@k / MRR / NDCG（真实检索） |

---

## 十四、降级链（完整）

**介绍**：从外到内多层降级，原则「宁可降级到安全，也不降级到错误」。

```
LLM 层      重试(max_retries=2,指数退避+抖动) → 切备用模型 → FALLBACK_ANSWER 降级文案
分类器层    加载失败 → ConservativeClassifier（全判 emergency）
路由层      置信度 < 0.4 → 保守兜底（不给瞎猜的标签进生成）
检索层      BM25 失败 → 降级纯向量；reranker 失败 → 降级 chunks[:top_k]
风险模型层  is_available() False → 跳过 ML 风险区块
异常层      未预期异常 → 全局异常处理器 → 统一 500 格式
```

---

## 十五、设计决策速查

| 决策 | 选择 | 理由 |
|---|---|---|
| 编排 | LangGraph | 专家版多节点线性、患者版安全分支，用图显式表达 |
| 紧急判断 | LoRA 小分类器 | 快、确定、防注入，安全关键件独立 |
| 专家版生成 | 多 Agent（生成 + 质疑） | 用「LLM 质疑 LLM」降低幻觉，给出置信度标签 |
| 检索 | pgvector + ParadeDB BM25 + RRF + 精排 | 多路互补 + 交叉编码器提精度 |
| 向量库 | pgvector（非 FAISS） | 与业务库同库、支持 BM25 混合检索 |
| 知识源 | PubMed + PDF + MedlinePlus | 专家版用临床证据，患者版用公共领域健康科普 |
| 结构化输出 | function calling 强 schema | `tool_choice` 锁定函数，返回合法 schema |
| LLM 接入 | 自研 10 厂商 LLMClient | 多厂商 + 自定义 fallback + Claude 协议转换 |
| 风险预测 | XGBoost（MIMIC-IV 训练） | 轻量、可解释、独立于 LLM |
| 日志 | structlog JSON + request_id | 生产可观测、可检索 |
| 追踪 | LangSmith | 追踪多 Agent 节点调用链路 |
| 质量评估 | DeepEval + IR 指标 | LLM-judge 测语义，规则/指标测确定项 |

---

## 十六、诚实的不足

1. **专家版风险模型不可靠**：MIMIC demo 子集训练，CV AUC≈0.58 接近随机，需换完整 MIMIC-IV 重训
2. **中文失效**：患者画像提取和分诊分类器只支持英文，中文输入会误判/画像为空
3. **无记忆（单轮）**：`get_llm_history` 是死代码，会话历史从不回喂 LLM
4. **检索语义理解弱**：只描述症状不点主题名（模糊查询）时 MRR 0.37（vs 简单查询 0.90）
5. **专家版数据污染**：PubMed 摄取未按语言过滤，部分主题混入非英文文档
6. **eval 没进 CI**：改 prompt 有安全退化风险
7. **限流内存版**：多实例部署要换 Redis
8. **本地 Llama 仅 GPU**：Windows+CPU 下 reasoning 自动降级 API（bitsandbytes 仅 CUDA/Linux）
