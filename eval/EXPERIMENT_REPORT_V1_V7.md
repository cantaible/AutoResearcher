# RAG 月报实验报告（v1-v7）

> 版本编号说明：仓库中没有显式的 `v1-v7` 标签。本报告按现有日志与评测文件的实验演进重新编号，覆盖从 RAG 子图探索、检索评测，到端到端报告与结构化引用评测的主要结果。

## 结论

1. **RAG 检索层已证明可行**：在 20 个模型发布事件家族的 Ground Truth 上，最佳离线检索实验达到 **75.0% Event Recall**。
2. **端到端报告仍是短板**：最终报告的 Finding Recall 最高只有 **20.0%**，说明“检索命中”没有稳定转化为“报告写出”。
3. **Evidence Pool 解决的是证据账本问题**：它把 Evidence Support Rate 从不可用/0% 提升到 **100%**，但不能直接提高召回。
4. **结构化引用有价值但需控噪**：v7 报告引用命中 **14/20 事件（Reference Recall 70.0%）**，但 Article Precision 只有 **26.5%**，说明引用覆盖广但噪声多。
5. **下一步重点不是继续加 memory**：优先做 chunk 检索接入、RAG 压缩到报告的事实保真、引用白名单裁剪。

## 版本结果总览

| 版本 | 日期 / Run | 主要变化 | 关键结果 | 主要问题 | 详情 |
|---|---|---|---|---|---|
| v1 | 2026-04-09 RAG 子图宽召回 | 按“模型类型 × 周时间窗”拆成 21 个子查询，覆盖基础模型、图像、视频、Agent、代码、LMArena | 21 子查询，47 次工具调用，耗时 537.3s，压缩摘要 5848 字符 | 查询过宽，部分非榜单 query 被 LMArena 补搜污染；成本偏高 | [RAG_SUBGRAPH_ANALYSIS](../logs/ragtest搜索本地新闻数据库查找2026年3月1-20260409-194019/RAG_SUBGRAPH_ANALYSIS.md) |
| v2 | 2026-04-10 基础模型聚焦 | 缩小到基础大语言模型，按“新闻 / 产品发布 / 榜单变化 × 周时间窗”拆 15 个子查询 | 15 子查询，29 搜索轮次，摘要 3943 字符 | query 重复度高，多轮 retry 后仍偏泛，OpenSearch 噪声明显 | [RAG_REPORT](../logs/ragtest搜索本地新闻数据库查找2026年3月1-20260410-114207/RAG_REPORT.md) |
| v3 | 2026-04-10 轻量日期窗 | 只保留 5 个日期窗口，统一 query 为“新发布 + 头部厂商” | 5 子查询，7 搜索轮次，耗时 67.4s，摘要 2890 字符 | 速度快但召回维度不足，Top 命中混入产品发布和产业新闻 | [RAG_REPORT](../logs/ragtest搜索本地新闻数据库查找2026年3月1-20260410-120136/RAG_REPORT.md) |
| v4 | 2026-04-10 意图/厂商规划 | 改成 10 个正交子查询：总体、OpenAI、Google、Anthropic、Meta、xAI、国内厂商、开源、多模态、推理/Agent | 10 子查询，13 搜索调用，耗时 289.5s，摘要 4300 字符 | 子查询更合理，但 evaluator 对“不相关结果”判断偏宽，仍会接受错位命中 | [RAG_REPORT](../logs/ragtest搜索本地新闻数据库查找2026年3月1-20260410-124455/RAG_REPORT.md) |
| v5 | 2026-04-10 事件级检索评测 | 建立 RAG 检索评测，使用旧口径 45 个发布事件 | Event Recall **40.0%**（18/45），Article Precision **41.38%**，Article Recall **27.27%**，NDCG@20 **0.4531** | GT 口径过细，子型号/小众事件过多，难以反映月报口径 | [EVAL_REPORT](../logs/eval搜索本地新闻数据库查找2026年3月1-20260410-185342/EVAL_REPORT.md), [EXPERIMENTS](EXPERIMENTS.md) |
| v6 | 2026-04-11 事件家族评测 | 将 Ground Truth 合并为 20 个模型家族，更贴近月报口径 | Event Recall **75.0%**（15/20），Article Precision **27.08%**，Article Recall **39.0%**，NDCG@20 **0.2750** | 召回提升但纯度下降；仍漏 ASMR、MAI-Image-2、Midjourney V8、Seedance 2.0、Vidu Q3 | [EVAL_REPORT](../logs/eval搜索本地新闻数据库查找2026年3月1-20260411-114812/EVAL_REPORT.md), [PHASE1](PHASE1_SUMMARY.md), [PHASE2](PHASE2_SUMMARY.md) |
| v7 | 2026-05-16 端到端 + Evidence Pool + article 引用 | 接入 `evidence_pool.json`、`[article:ID]` 引用、Finding 评测和 Reference 评测 | Finding Recall **20.0%**（4/20），Finding Precision **57.1%**，Evidence Support **100%**；Reference Recall **70.0%**，Article Precision **26.5%** | 报告召回仍低；引用覆盖比 finding 更高，说明抽取/报告结构与 GT 对齐仍有损耗；完整 evidence_pool 不适合进上下文 | [FINDING_REPORT](../logs/搜索本地新闻数据库查找2026年3月1日至3月31日期间发布-20260516-111017/FINDING_REPORT.md), [REFERENCE_EVAL](../logs/搜索本地新闻数据库查找2026年3月1日至3月31日期间发布-20260516-111017/REFERENCE_EVAL.json), [EVAL_REPORT](../logs/搜索本地新闻数据库查找2026年3月1日至3月31日期间发布-20260516-111017/EVAL_REPORT.md) |

## 指标解读

- **检索指标看 v6**：v6 是当前最能代表 RAG 检索能力的离线评测，因为 GT 已合并为月报口径的 20 个事件家族。
- **报告指标看 v7**：v7 是当前最能代表端到端月报效果的实验，因为它包含最终报告、Finding 评测、Evidence Pool 和结构化引用评测。
- **v6 与 v7 不可直接比较**：v6 评的是 RAG 子图 raw retrieval；v7 评的是完整报告链路，中间经过 Supervisor、压缩、报告写作和引用约束。

## 关键发现

### 1. Query 规划比“搜索次数”更重要

v1 查询覆盖很宽、调用很多，但噪声高；v3 很快但维度不足；v4 的意图/厂商拆分更像可持续方向。后续应保留 v4 的正交拆分思路，同时加入模型类型覆盖和缺口补搜。

### 2. 检索与写作之间存在信息损耗

v6 的 RAG Event Recall 达到 75%，但 v7 最终 Finding Recall 只有 20%。这说明瓶颈已经不只在检索，还在：

- RAG compress 是否保留所有关键事件；
- Supervisor 是否把 RAG 输出继续传递给 final report；
- final report prompt 是否鼓励完整列举，而不是选择性总结；
- Finding extractor 是否能抽出报告里的全部事实。

### 3. Evidence Pool 应保留为外部账本

真实样本中 `evidence_pool.json` 约 340-393 条证据，完整加载约 **13.7 万到 16.4 万 tokens**；即使压成 `[article:ID] title` 也约 **1.7 万到 1.9 万 tokens**。因此它适合落盘做审计/评测，不适合作为完整上下文输入。

### 4. 引用评测暴露了“覆盖广但不够准”

v7 的 Reference Recall 是 70%，高于 Finding Recall 20%，说明报告引用确实覆盖了更多 GT 事件。但 Article Precision 只有 26.5%，说明引用里有大量 GT 外文章或当前 GT 未覆盖文章，需要做引用白名单、事件归属校验或引用后处理。

## 下一步

1. **接入 `news_chunks` 检索**：当前构建了 chunk 索引，但运行时主要查文章级标题/摘要；长文事实应接入 chunk recall。
2. **保留 v4 规划方式，增加缺口补搜**：按厂商/模型类型拆分后，对 GT 常漏项方向（视频、图像、代码模型）做专门查询。
3. **RAG compress 改成结构化 findings**：先抽结构化事件，再写报告，减少从检索到最终报告的信息损耗。
4. **引用白名单裁剪**：final report 只允许引用压缩摘要中实际出现、或 rerank top evidence 中的 article ID。
5. **继续使用 Evidence Pool，但不进完整上下文**：只在评测、引用校验、debug 中读取完整文件。

## 关联文档

- [RAG 检索评估实验记录](EXPERIMENTS.md)
- [Ground Truth 构建总结](PHASE1_SUMMARY.md)
- [Retrieval Details 评测总结](PHASE2_SUMMARY.md)
- [Finding 评测总结](PHASE3_SUMMARY.md)
- [评测使用指南](EVALUATION_GUIDE.md)
- [Evidence Pool 设计 TODO](../docs/EVIDENCE_POOL_TODO.md)
