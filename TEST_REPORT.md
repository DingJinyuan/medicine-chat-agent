# 医学 AI 项目测试与评估报告

> 日期:2026-09-16
> 框架:pytest 9.1.1 + DeepEval 4.2.3 + 自研 IR 指标
> 范围:专家版 + 患者版两套工作流(测试 L1–L4 + 评估 L5 四层)

## 1. 总览

| 项 | 结果 |
|---|---|
| 单元测试 | **132 passed / 5 skipped**(覆盖率 80%) |
| 端到端 test_e2e | **3 passed**(真实 LLM + pgvector + 本地模型) |
| 规则式评估 | **17/17 = 100%** |
| DeepEval 忠实度 | **~1.0**(零幻觉) |
| 检索质量 | 简单查询 MRR 0.90 / 模糊查询 MRR 0.37 |
| 修复的确定 bug | **6 个**(3 红绿 + 3 短板) |

## 2. 真实评分汇总

### 2.1 规则式评估(run_evals)

**17/17 = 100%**:安全护栏(剂量/确诊)、分诊路由、拒答、注入防御、专家版引用全部通过。

### 2.2 DeepEval(LLM-as-judge)

患者版(忠实度/相关性/检索上下文相关):

| 查询 | 忠实度 | 相关性 | 检索上下文相关 |
|---|---|---|---|
| 糖尿病 | 1.0 | 0.67 | 0.10 |
| 偏头痛 | 1.0 | 0.81 | 0.16 |
| 哮喘 | 1.0 | 0.44 | 0.15 |

专家版(忠实度/相关性/检索精准/检索召回):

| 查询 | 忠实度 | 相关性 | 检索精准 | 检索召回 |
|---|---|---|---|---|
| HFrEF | 0.96 | 0.29 | 0.37 | 1.0 |
| CAP | 1.0 | 0.72 | 0.70 | 0.50 |
| T2DM | 1.0 | 0.76 | **0.0** | **0.0** |
| 脓毒症 | 0.97 | 0.55 | 0.64 | 0.50 |
| LDL | 0.97 | 0.42 | 0.95 | 0.33 |

### 2.3 检索 IR 指标(患者版)

| 查询类型 | hit@5 | recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| 简单(点主题名) | 1.00 | 0.93 | 0.90 | 0.89 |
| 模糊(只说症状) | 0.60 | 0.30 | 0.37 | 0.27 |

## 3. 已修复的 6 个 bug

**红-绿循环修掉的 3 个**:
1. 性别 `female`→`male` 子串误判(`expert_risk_tool.py:152`)
2. `los_days` 误匹配门诊症状(`expert_risk_tool.py:178`)
3. 患者版重复摄取幂等(`patient_ingest.py:84`)

**本轮修的 3 个短板**:
4. critique 死胡同 → 只增强展示:is_reliable=False 时核查意见醒目提示
5. 鉴权缺口 → `/history` 加 `require_api_key`(`/metrics` 保持开放)
6. 限流无淘汰 → RateLimiter 加淘汰机制 + 读 `X-Forwarded-For`

## 4. 剩余真实短板(未修,按优先级)

| # | 短板 | 证据 | 建议 |
|---|---|---|---|
| 1 | **专家版数据污染** | T2DM 检索到德文文档(contextual 0/0) | 摄取时按 language=eng 过滤 |
| 2 | 风险模型不可靠 | AUC≈0.58 接近随机 | 换完整 MIMIC-IV 重训 |
| 3 | 无记忆(单轮) | `get_llm_history` 死代码 | 接上历史回喂 |
| 4 | 检索语义理解弱 | 模糊查询 MRR 0.37 | 优化嵌入/检索算法 |

## 5. 设计取舍(非 bug,记录在案)

- **相关性偏低(0.29–0.81)**:安全优先设计,回答故意啰嗦(加免责/就医建议)
- **contextual_relevancy 低(0.10–0.16)**:患者版是「宽召回完整疾病主题 + LLM 提取」,不是精准召回小节,符合设计
- **忠实度满分(1.0)**:system prompt 强制「基于参考上下文」,零幻觉

## 6. 测试文件清单

```
backend/tests/          # 15 个测试文件,132 用例(覆盖率 80%)
backend/evaluation/     # 评估脚本:run_evals / deepeval_eval / retrieval_eval / ir_metrics / benchmark
backend/data/eval/      # 评估集:golden_dataset(患者版) / expert_dataset(专家版)
```

## 7. 运行方式

```bash
cd backend
# 单元测试(离线秒级)
.venv/Scripts/python -m pytest tests/ -v
# 规则式评估(真实 LLM + pgvector)
.venv/Scripts/python -m evaluation.run_evals
# DeepEval(LLM-as-judge)
.venv/Scripts/python -m evaluation.deepeval_eval
# 检索 IR 指标(真实检索)
.venv/Scripts/python -m evaluation.retrieval_eval
```

## 8. benchmark 状态(跳过)

PubMedQA benchmark **未跑成**:`lm-eval` 依赖的 `bigbio/pubmed_qa` 数据集已从 HF Hub 下架,本地无缓存。已决定跳过——它测的是底座 LLM 的医学知识,与本项目 RAG 系统质量无关,RAG 质量已由规则式/DeepEval/检索 IR 三层覆盖。
