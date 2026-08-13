# MediSense AI 面试材料

> 📖 **文档导航**：[README](README.md)（项目概览）· [功能说明](FEATURES.md)（架构细节）· [面试 QA](INTERVIEW_QA.md)（本文）

## A. 简历项目条目

**医疗信息问答机器人（全栈）** — 设计并实现端到端 RAG 问答系统，包含 LangGraph 编排、LoRA 微调分诊分类器、FAISS 向量检索、多厂商 LLM 适配和医疗安全三层护栏。

**双模型安全架构** — 用 LoRA 微调 distilbert（67M 参数，held-out 97.4%）做紧急分诊，确定性小模型把关安全关键决策，多厂商 LLM 只负责生成；紧急短路让危险输入到不了生成步骤。

**多厂商 LLM 适配层** — 自研前缀匹配注册表 + 主备降级链 + 可重试错误分类 + 指数退避，统一 DeepSeek/OpenAI/Claude 等 10+ 厂商（含 Claude 协议转换）。

**医疗安全三层防线** — 8 种注入模式扫描 + 检索数据边界包裹 + 输出诊断/剂量重写 + 无条件免责声明；53 项单元测试 + 11 项端到端评估 + DeepEval 生成质量评估。

**全栈可观测工程** — structlog JSON 日志 + request_id 串联、Prometheus 6 项指标（含 token 成本）、统一错误格式、分级健康检查，前后端分离部署（Render + Vercel）。

## B. 1 分钟项目介绍

"我做了个医疗信息问答机器人，全栈项目。核心思路是**双模型设计**——用一个 LoRA 微调的小分类器做紧急分诊，而不是让大模型判断。

用户输入症状，分类器先判紧急程度：紧急的直接返回急救建议，**大模型根本看不到这条输入**；非紧急的走 RAG 检索 MedlinePlus 医学知识库，再让大模型生成回答，最后过输出安全审查。

我特别设计了**多层降级**——大模型挂了重试切备用模型、分类器坏了退化成保守模式、没把握就走安全兜底。整个系统有 structlog 结构化日志、Prometheus 指标、统一错误处理和健康检查。

前端 React 能跑完整 Demo 流程，配置 DeepSeek Key 就能切真实 AI。"

## C. 面试高频问答（44 问）

### 一、项目介绍

**Q1：介绍一下这个项目？**
医疗信息问答机器人，覆盖完整大模型应用技术栈。流程：用户症状 → LoRA 分类器判紧急程度（本地毫秒级）→ 紧急走固定急救回复（LLM 不看输入）→ 非紧急走 RAG 检索 → LLM 生成 → 输出安全审查。技术栈：FastAPI + LangGraph、FAISS + fastembed、自研 LLMClient 多厂商、distilbert + LoRA、structlog + Prometheus、Next.js 16。亮点是双模型设计。

**Q2：最有挑战的是什么？**
微调数据质量的坑：合成数据里 "headache + light sensitivity" 只在 emergency 标签出现，模型学到「这词组合 = emergency」的假关联，把典型偏头痛误判紧急。这让我意识到数据质量比模型调参更重要。

**Q3：这个项目你一个人做完的？分工？**
全栈独立完成：后端（FastAPI + LangGraph + RAG + 分类器 + 安全）、前端（Next.js 16）、微调（LoRA）、测试评估、部署配置。

### 二、架构设计

**Q4：为什么用 LangGraph 而不是手写 if？**
项目里有安全关键的条件分支——紧急短路。分类器 ≥0.6 置信度判 emergency 时，完全跳过检索和生成，LLM 不看输入。这种分支用 if 写会埋在业务代码深处，用 LangGraph 表达成图里的节点和条件边，逻辑显式、可审查。

**Q5：双模型设计怎么分工？为什么？**
分类器管「紧急判断」（确定性、快、防注入），LLM 管「生成」。紧急判断不交给 LLM 的三个原因：延迟（毫秒级 vs 网络）、可靠性（确定性 vs 可被注入带偏）、安全（独立小模型是最后防线）。

**Q6：为什么紧急要短路，而不是让 LLM 生成「去看医生」？**
因为「让 LLM 看到危险输入」本身就是风险——可能被带偏、被注入绕过、生成不当内容。短路本质是用确定性逻辑替代 LLM 判断，把危险输入挡在 LLM 之外。

**Q7：为什么选 FAISS + fastembed？**
规模匹配。518 个 chunk、单用户 demo，FAISS 成熟快、CPU 友好，fastembed 轻量。没必要上 Milvus/Weaviate 分布式向量库。

**Q8：为什么知识库用 MedlinePlus？**
美国政府作品、公共领域，没有版权问题。很多医疗 QA 数据集会因版权删掉非政府来源的答案正文。

**Q9：为什么自己写 LLM 适配层，不用 LangChain 的封装？**
因为需要「多厂商 + 自定义 fallback + 协议转换」的控制力。LangChain 的统一抽象反而碍事——它已经有自己的 fallback 机制，跟我的 fallback_models 重叠。而且 Claude 的协议转换我需要自己掌控。

### 三、代码实现

**Q10：LangGraph 状态怎么定义？**
`TypedDict`（total=False），字段有 question、triage_label、triage_confidence、context_blocks、sources、injection_flagged、answer、guardrail_rewritten。节点函数接收 state 返回部分更新的 dict，LangGraph 自动合并。

**Q11：两个阈值（0.6 和 0.4）分别干嘛的？**
`emergency ≥ 0.6`：紧急高置信度才短路。`confidence < 0.4`：任何标签低置信度都走保守兜底。两个阈值覆盖「高置信度敢短路」和「低置信度不敢硬走」两个安全方向。

**Q12：分类器的 id2label 为什么重要？**
训练时 `label2id = {emergency:0, urgent:1, routine:2, self_care:3}` 存到 `label_map.json`，推理时靠它把 argmax 索引映射回标签。训练和推理顺序不一致，分类结果全错。

**Q13：分类器加载失败怎么办？**
退化为 `ConservativeClassifier`（全判 emergency、置信度 1.0），宁可拒答不漏放。关键细节：lru_cache 只缓存成功结果——拆成 `_load_triage_classifier`（lru_cache，失败抛异常不进缓存）+ `get_triage_classifier`（捕获异常返回兜底）。

**Q14：lru_cache 失败缓存的坑具体是什么？**
早期实现 `get_triage_classifier` 直接加 `@lru_cache` 且函数内捕获异常返回 ConservativeClassifier——导致失败结果也被缓存：adapter 文件修好后，服务仍返回缓存的保守分类器，直到重启。修复是拆两层，lru_cache 只包成功路径。

**Q15：统一错误格式怎么设计？**
`ErrorCode` 枚举（validation_error / unauthorized / rate_limited / internal_error / llm_unavailable）+ `AppError`（status/code/message）+ `error_response` 输出 `{"error":{"code","message"}}`。三个 handler：AppError、RequestValidationError（422，不 `str(exc)` 避免泄露字段细节）、Exception（兜底 500，traceback 进日志、message 模糊）。

**Q16：request_id 中间件怎么实现？**
`@app.middleware("http")` 里从请求头取 `X-Request-Id`，没有就生成 uuid，`structlog.contextvars.bind_contextvars` 绑定到上下文，响应头回写。该请求所有日志自动带 request_id，能串联一个请求从头到尾的日志。

**Q17：token 用量怎么统计？**
LLMClient 内部 `_track_usage` 累加 total 统计，`generate_answer` 每次 new 一个 LLMClient，调用后读累计值就是单次用量，埋到 Prometheus `TOKEN_USAGE`（label 区分 prompt/completion）。成本可观测。

**Q18：指数退避重试怎么实现？**
`min(1.0 * 2**retry, 32.0) + random.uniform(0, base*0.25)`——指数递增封顶 32 秒 + 0~25% 随机抖动。抖动防止大量请求同时重试打挂服务（雪崩）。

**Q19：可重试错误怎么分类？**
`_is_retryable_error`：401/400/参数错误/JSON 解析失败 → 不可重试（重试没用，切备用）；429/529/超时/连接 → 可重试（服务方临时问题）。关键是「重试会不会变好」。

**Q20：多厂商适配层的前缀匹配怎么实现？**
`MODEL_PROFILES` 里模型名前缀 → 厂商 key/base_url 环境变量。`_resolve_model_env(model)` 遍历前缀，`model.lower().startswith(prefix)` 匹配，读对应环境变量。比如 "deepseek-chat" 匹配 "deepseek"，读 `DEEPSEEK_API_KEY` + `DEEPSEEK_BASE_URL`。

**Q21：Claude 协议转换怎么做的？**
Claude 是唯一不兼容 OpenAI 协议的。`_translate_messages()` 做消息格式互转：system 字段独立抽出、tool call 格式转换（OpenAI function ↔ Claude tool_use/tool_result）。用 anthropic SDK 调用，返回结果再转回 OpenAI 的 ChatCompletionMessage。

**Q22：数据摄入怎么做的？**
从 MedlinePlus webservices API 抓 104 个主题，XML 解析出 title + summary，`RecursiveCharacterTextSplitter` 切块（800 字符 / 120 重叠），缓存到本地 JSON。单个主题抓取失败（网络/解析错误）跳过继续。

### 四、安全

**Q23：医疗安全三层防线？**
输入侧（8 种注入正则）+ 检索侧（`wrap_untrusted` 声明数据边界）+ 输出侧（重写诊断/剂量 + 无条件免责声明）。输出护栏对每一条路径生效，免责声明不赌模型表现。

**Q24：prompt 注入怎么防？**
RAG 特有风险是检索文档里藏注入指令。防御是「扫描 + 包裹」：先过注入正则，再用 `wrap_untrusted` 显式声明数据边界。核心是用格式告诉模型哪些是数据、哪些是指令，不依赖模型自己判断。

**Q25：API key 怎么保证安全？**
key 不进浏览器。前端唯一调后端入口是 Next.js server route，key 只在服务端环境变量（无 `NEXT_PUBLIC_` 前缀），浏览器只跟 server route 通信。后端 `Depends(require_api_key)` 验证 + 每 IP 限流 + CORS 白名单。

**Q26：为什么日志不记用户原文？**
医疗 PHI（受保护健康信息）红线。日志会散落到各处（错误日志、日志系统、监控平台），管控难度远大于数据库。排查故障只需要知道「有个请求出错了、输入多长」，不需要知道具体内容。所以只记 `input_len`。

### 五、测试与评估

**Q27：测试分几层？**
三层：单元测试（pytest 53 个，离线 mock LLM，测代码路径）、端到端评估（run_evals 11 用例，真实 LLM，测安全行为）、生成质量（DeepEval，LLM-as-judge，测忠实度/相关性）。安全用规则保证确定性，质量用 LLM-judge 衡量语义。

**Q28：单元测试和端到端评估的区别？**
单元测试离线、mock LLM、测「代码对不对」（正则、限流、解析、路由、认证）；端到端评估真实调用、测「系统行为对不对」（是否用检索事实、是否泄露剂量/诊断、是否路由急救、注入是否拦截）。改 prompt 后要跑 eval，因为单测 mock 了 LLM 测不出 prompt 变化的影响。

**Q29：DeepEval 是什么？为什么用它？**
confident-ai 的开源 LLM 评估框架，用 LLM 当裁判打分。测生成质量：忠实度（回答是否忠实于检索内容）、回答相关性（是否切题）。安全关键点用规则（确定性），生成质量用 LLM-judge（语义），分层清晰。

**Q30：写测试抓到什么真实 bug？**
两个护栏正则 bug：一个字符类没包含数字漏掉 "type 2 diabetes"，一个要求 "have" 和病名之间有填充词放过 "you have diabetes for sure"。正则这种「看起来简单」的逻辑，边界最容易漏。

### 六、降级容错

**Q31：降级链是怎么设计的？**
从外到内五层：LLM 层（重试→切备用→降级文案）、分类器层（加载失败→保守模式）、路由层（置信度<0.4→保守兜底）、检索层（miss→明说不知道）、异常层（全局异常处理统一 500）。原则：宁可降级到安全，也不降级到错误。

**Q32：LLM 挂了怎么兜底？**
`LLMClient.invoke` 内部：每个模型重试 3 次（指数退避），失败切 fallback 列表下一个，全挂抛异常。上层 `generate_answer` catch 后返回 `FALLBACK_ANSWER` 降级文案，埋 `LLM_ERRORS` 指标 + 记日志。

**Q33：容错机制有哪些？**
缓存优先（语料/索引持久化）、离线启动（HF_HUB_OFFLINE=1 模型烘焙）、局部失败跳过（ingestion 单主题失败 continue）、幂等写入（INSERT OR IGNORE）、前端降级（后端不可达返回 502）、防雪崩抖动。

**Q34：健康检查怎么做？**
`/health`（存活）+ `/health/ready`（就绪，检查分类器是不是真 TriageClassifier、向量库有没有、LLM 配置有没有）。降级时 `/health/ready` 返回 degraded，一眼看出是哪个依赖挂了。

### 七、生产实践

**Q35：生产部署架构？**
后端 Render（Docker）：构建时烘焙 embedding + distilbert 模型，HF_HUB_OFFLINE=1 免冷启动下载。前端 Vercel：server route 代理。环境变量在各自面板配。

**Q36：生产怎么监控？**
Prometheus `/metrics` 暴露请求数、延迟、分类分布、LLM 错误、限流、token 用量，接 Grafana。JSON 日志进 ELK/Loki，request_id 串联请求。

**Q37：性能瓶颈在哪？怎么优化？**
唯一网络瓶颈是 LLM 调用（超时 60s + 重试兜底）。分类器 67M 参数 CPU 毫秒级、检索 518 向量微秒级都不是瓶颈。内存是隐形成本（torch + transformers），Render 免费 512MB 可能 OOM。

**Q38：限流怎么实现？**
内存版固定窗口限流（每 IP 每分钟 N 次），单实例够用。多实例部署要换 Redis 共享存储。

**Q39：日志为什么用 JSON？**
JSON 可被日志系统按字段检索、按 request_id 拉出一个请求的所有日志；文本只能 grep。代价是学 structlog 写法，但生产可观测性值得。

**Q40：分类器在 CPU 还是 GPU 跑？为什么？**
后端推理用 CPU。因为 distilbert 67M 参数、64 token 输入，CPU 毫秒级，GPU 收益可忽略（20ms vs 5ms 用户无感），还多占显存。训练才用 GPU（base-llm 环境的 RTX 3060），因为训练计算量是推理的几千倍。训练和推理分开：CUDA torch vs CPU torch。

### 八、反思

**Q41：项目有什么不足？怎么改进？**
数据漂移监控没做、eval 没进 CI、限流内存版、知识库覆盖有限、微调数据是合成的。优先级：eval 进 CI → 数据漂移监控 → 限流换 Redis。

**Q42：如果重做会怎么改？**
先把 eval 接进 CI（自动化安全回归），再补数据漂移监控（积累真实流量后），限流上 Redis。微调数据会想办法找真实标注数据，而不是纯模板合成。

**Q43：这个项目跟生产级医疗应用差距在哪？**
数据（合成 vs 真实临床标注）、合规（无 HIPAA/GDPR 认证）、规模（单实例 vs 分布式）、评估（手动 vs CI 自动化）。核心是「安全关键路径的确定性」已经做到了，差的是「规模化和合规化」。

**Q44：为什么选 DeepSeek 作为 LLM？**
性价比高、OpenAI 兼容协议（适配层直接接）、中文和英文都支持。适配层本来就是多厂商的，选 DeepSeek 只是因为当前用它跑 Demo，换成 OpenAI/Claude 只要改 .env 的模型名和 key。

### Q45：数据库是怎么设计的？

SQLite + 标准库 sqlite3（两张表不值得引 ORM）。两张表：`sessions`（会话）+ `messages`（消息，外键关联会话）。用途是存聊天历史，前端展示会话连续性。`ensure_session` 用 `INSERT OR IGNORE` 幂等。

一个诚实的设计点：`get_llm_history` 虽写了，但当前 `generate_answer` 是**无状态**的（只传当前问题 + 检索上下文，不传历史），所以历史目前只用于展示，没真正用于多轮对话。要做多轮，把历史拼进 `build_rag_messages` 即可。

### Q46：RAG 链路是怎样的？

标准三段式：**检索 → 增强 → 生成**。

离线阶段（启动时）：MedlinePlus 104 主题 → 切块 518 chunk → fastembed 向量化 → FAISS 索引持久化。

在线阶段（一次查询）：
1. **检索**：问题向量化 → FAISS 相似度 → top-4 chunk（同时做注入扫描 + 数据包裹）
2. **增强**：top-4 chunk + 问题拼进 prompt 的 Reference context
3. **生成**：LLM 基于增强 prompt 生成回答

关键设计：紧急短路在 RAG 前（分类器判 emergency 就跳过整个 RAG）；检索 miss 时 context 空、LLM 明说不知道；chunk 带 topic+url 作为 sources 返回、引用可追溯。
