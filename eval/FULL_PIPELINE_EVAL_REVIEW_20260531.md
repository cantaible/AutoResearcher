# RAG 全链路评测改造复盘（2026-05-31）

## 结论

本次改造完成了 RAG 链路从 `Plan 子查询 → Dense/Chroma + Sparse/OpenSearch → Merge → Rerank → RAG Compress → Supervisor → Final Report` 的分阶段评测闭环。

主要结果：

1. 已先将原有未提交改动提交并推送到 GitHub：`5be6929 docs: summarize rag experiments v1-v7`。
2. 完整主图现在会保存分阶段评测所需中间产物：`retrieval_details.json`、`rag_outputs.json`、`rag_sub_queries.json`、`supervisor_tool_calls.json`、`final_report_notes.json`。
3. 新增统一评测入口：`eval/eval_full_pipeline.py`。
4. 无人工 Plan 标准答案时，支持用 OpenAI 兼容接口合成 `synthetic_plan_rubric.json`。
5. 实测发现当前最大损耗不在 Dense/Sparse 召回，而在 `Rerank` 和 `RAG Compress`：
   - 完整 smoke run 中 Merge Event Recall 为 **90.0%**，Rerank 后降到 **80.0%**。
   - RAG Compress 事件保留率只有 **62.5%**。
   - Supervisor 交接和引用白名单表现正常，均为 **100.0%**。

## 本次改动

### 1. 先提交当前工作区

开始开发前，工作区只有两处已有改动：

- `eval/EXPERIMENT_REPORT_V1_V7.md`
- `docs/PROGRESS.md`

`.env` 被 `.gitignore` 忽略，没有进入 Git。

已执行提交并推送：

```bash
git add docs/PROGRESS.md eval/EXPERIMENT_REPORT_V1_V7.md
git commit -m "docs: summarize rag experiments v1-v7"
git push origin main
```

提交：

```text
5be6929 docs: summarize rag experiments v1-v7
```

### 2. 补齐完整主图中间产物

修改文件：

- `src/state.py`
- `src/graph.py`
- `src/runner.py`

新增或透传的数据：

| 文件 | 内容 | 用途 |
|---|---|---|
| `retrieval_details.json` | dense/sparse/merged/reranked 的 article ids | 检索、合并、重排评测 |
| `rag_outputs.json` | RAG compressed、raw_results、sub_queries 等 | 压缩层评测 |
| `rag_sub_queries.json` / `sub_queries.json` | Plan 产出的子查询 | Plan 评测 |
| `supervisor_tool_calls.json` | Supervisor 调用工具的结构化记录 | 调度层评测 |
| `final_report_notes.json` | final writer 接收到的 notes 快照 | Supervisor 到最终报告的信息保留评测 |

关键点：

- `ResearcherOutputState` 增加 `retrieval_details`、`sub_queries`、`raw_results`，让 RAG 子图结果能向上暴露。
- `SupervisorState` 和 `AgentState` 增加 RAG/Supervisor 评测字段。
- `supervisor_tools()` 汇总 RAG 结果时保存检索详情、子查询、压缩结果、证据数和工具调用状态。
- `final_report_generation()` 在清空 `notes` 前保存 `final_report_notes` 快照。
- `runner.py` 在最终 run 目录落盘上述 JSON。

### 3. 新增分阶段评测脚本

新增文件：

- `eval/eval_full_pipeline.py`

支持的评测层：

| 阶段 | 指标 |
|---|---|
| Plan | 子查询数、日期正确率、分类正确率、重复率、rubric 覆盖率、query 命中事件 |
| Dense/Chroma | Event Recall、Article Precision、Article Recall、NDCG@K |
| Sparse/OpenSearch | Event Recall、Article Precision、Article Recall、NDCG@K |
| Merge | 相对 Dense/Sparse 的召回增益、dense-only/sparse-only/both 事件 |
| Rerank | Event Recall@K、Article Precision@K、NDCG@K、从 merged 保留的事件比例、被误杀事件 |
| RAG Compress | raw 命中事件、compressed 引用命中事件、事件保留率、引用保留率、压缩比、丢失事件 |
| Supervisor | RAG 调用数、Web 调用数、evidence_pool 数量、handoff retention、引用白名单率 |

使用方式：

```bash
python eval/eval_full_pipeline.py --run-dir logs/<run_dir>
```

使用大模型合成 Plan rubric：

```bash
python eval/eval_full_pipeline.py \
  --run-dir logs/<run_dir> \
  --synthesize-plan-rubric
```

输出：

- `STAGE_EVAL_REPORT.md`
- `stage_eval_metrics.json`
- `synthetic_plan_rubric.json`（可选）

### 4. 补 Finding 评测兼容

修改文件：

- `eval/finding_extractor.py`
- `eval/eval_findings.py`

改动：

- `finding_extractor.py` 现在读取 `OPENAI_BASE_URL`，支持当前 OpenAI 兼容代理。
- 支持通过 `FINDING_EXTRACTOR_MODEL` 覆盖抽取模型。
- `eval_findings.py` 新增 `--target`：
  - `--target auto`
  - `--target compressed`
  - `--target report`

这样可以显式区分“评 RAG 压缩摘要”还是“评最终报告”。

### 5. 修复 reranker 离线加载问题

修改文件：

- `rag/reranker.py`

问题：

虽然原逻辑在 `get_reranker()` 里设置了 `HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE`，但 `CrossEncoder` 初始化过程中 tokenizer 路径仍尝试访问 Hugging Face，触发：

```text
SSLError: HTTPSConnectionPool(host='huggingface.co' ...)
```

处理：

- 在导入 `sentence_transformers.CrossEncoder` 之前设置离线环境变量。
- 验证 `get_reranker()` 可以本地初始化。

验证命令：

```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('rag').resolve()))
from reranker import get_reranker
model = get_reranker()
print(type(model).__name__)
PY
```

输出：

```text
CrossEncoder
```

### 6. 更新评测指南

修改文件：

- `eval/EVALUATION_GUIDE.md`

新增内容：

- 全链路分阶段评测入口。
- 完整主图需要保存的中间产物。
- 大模型合成 Plan rubric 的命令。
- Finding 评测 `--target` 用法。

## 验证过程

### 1. 语法检查

执行：

```bash
python -m py_compile \
  eval/eval_full_pipeline.py \
  eval/eval_findings.py \
  eval/finding_extractor.py \
  rag/reranker.py \
  src/state.py \
  src/graph.py \
  src/runner.py
```

结果：通过。

### 2. 独立 RAG 子图评测

运行命令：

```bash
python eval/eval_rag.py --topic "搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻，看看有哪些新的大模型发布了，尤其要关注头部厂商。"
```

Run 目录：

```text
logs/eval搜索本地新闻数据库查找2026年3月1-20260531-110457
```

基础 RAG 检索结果：

| 指标 | 结果 |
|---|---:|
| Event Recall | 70.00% (14/20) |
| Article Precision | 28.33% (34/120) |
| Article Recall | 34.00% (34/100) |
| F1 | 30.91% |
| NDCG@10 | 0.2817 |
| NDCG@20 | 0.2883 |
| NDCG@All | 0.3267 |
| 耗时 | 368.6s |

未命中事件：

- ASMR
- Gemini 3.1 Flash-Lite
- MAI-Image-2
- Midjourney V8
- Seedance 2.0
- Vidu Q3

分阶段评测：

```bash
python eval/eval_full_pipeline.py \
  --run-dir "logs/eval搜索本地新闻数据库查找2026年3月1-20260531-110457"
```

结果：

| 阶段 | Event Recall | Article Precision | NDCG@20 |
|---|---:|---:|---:|
| Dense/Chroma | 85.0% | 20.9% | 0.415 |
| Sparse/OpenSearch | 85.0% | 17.9% | 0.509 |
| Merge | 85.0% | 15.9% | 0.415 |
| Rerank | 70.0% | 28.3% | 0.318 |

额外发现：

- Merge 并没有相对 Dense/Sparse 增加事件召回，两路都达到 85%。
- Rerank 把候选池事件召回从 85% 降到 70%，误杀事件包括：
  - Gemini 3.1 Flash-Lite
  - MAI-Image-2
  - Seedance 2.0
- RAG Compress 事件保留率为 71.4%，从 raw 命中的 14 个事件压缩后只保留 10 个。
- Compress 丢失事件：
  - Composer
  - GLM-5-Turbo
  - GPT-5.3 Instant
  - SkyReels V4

### 3. 完整主图 smoke run

为了验证 Supervisor 产物落盘，跑了一个轻量端到端 smoke。

运行命令：

```bash
MAX_RAG_RETRIES=0 MAX_RESEARCHER_ITERATIONS=2 python - <<'PY'
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
from runner import run_research

async def on_event(evt):
    if evt.get('type') in {'report', 'clarify', 'error'}:
        print(evt.get('type'), str(evt.get('content') or evt.get('question') or evt.get('message'))[:200])

async def main():
    topic = '只使用本地新闻数据库RAG，不要使用网络搜索。查找2026年3月1日至3月31日期间发布的大模型相关新闻，列出新发布的大模型，并保留[article:ID]引用。'
    run_dir = await run_research(topic, on_event=on_event)
    print(run_dir)

asyncio.run(main())
PY
```

Run 目录：

```text
logs/只使用本地新闻数据库RAG不要使用网络搜索查找2026年3月-20260531-111213
```

该 run 成功生成了完整主图新增产物：

- `retrieval_details.json`
- `rag_outputs.json`
- `rag_sub_queries.json`
- `sub_queries.json`
- `supervisor_tool_calls.json`
- `final_report_notes.json`
- `evidence_pool.json`
- `report.md`

分阶段评测命令：

```bash
python eval/eval_full_pipeline.py \
  --run-dir "logs/只使用本地新闻数据库RAG不要使用网络搜索查找2026年3月-20260531-111213" \
  --synthesize-plan-rubric
```

分阶段结果：

| 阶段 | 核心指标 | 结果 |
|---|---:|---:|
| Plan | 覆盖率 / 重复率 | 100.0% / 5.3% |
| Dense/Chroma | Event Recall | 90.0% |
| Sparse/OpenSearch | Event Recall | 90.0% |
| Merge | Event Recall | 90.0% |
| Rerank | Event Recall | 80.0% |
| Compress | 事件保留率 | 62.5% |
| Supervisor | RAG 到 notes 保留率 | 100.0% |
| Final Report | 引用白名单率 | 100.0% |

细节：

| 阶段 | 候选数 | Event Recall | Article Precision | NDCG@20 |
|---|---:|---:|---:|---:|
| Dense/Chroma | 327 | 90.0% | 19.3% | 0.381 |
| Sparse/OpenSearch | 468 | 90.0% | 13.2% | 0.544 |
| Merge | 612 | 90.0% | 12.6% | 0.381 |
| Rerank | 103 | 80.0% | 28.2% | 0.404 |

Rerank 丢失事件：

- Midjourney V8
- Mistral Small 4

Compress 丢失事件：

- GLM-5-Turbo
- Gemini 3.1 Flash-Lite
- MAI-Image-2
- MiMo
- Seedance 2.0
- SkyReels V4

Supervisor 指标：

| 指标 | 结果 |
|---|---:|
| Supervisor 工具调用数 | 4 |
| ConductRAGResearch 调用数 | 2 |
| ConductResearch 调用数 | 0 |
| Evidence Pool 条数 | 103 |
| RAG compressed 到 final notes 的事件保留率 | 100.0% |
| final notes 到 report 的事件保留率 | 100.0% |
| 报告引用白名单率 | 100.0% |

说明：这是轻量 smoke，不是生产配置 benchmark。它使用了 `MAX_RAG_RETRIES=0`，并要求只用本地 RAG，不使用网络搜索。

### 4. 引用评测

命令：

```bash
python eval/eval_references.py \
  --run-dir "logs/只使用本地新闻数据库RAG不要使用网络搜索查找2026年3月-20260531-111213"
```

结果：

| 指标 | 结果 |
|---|---:|
| Reference Event Recall | 50.0% (10/20) |
| Reference Article Precision | 50.0% (20/40) |

命中事件：

- Composer
- DeepSeek V4
- GPT-5.3 Instant
- GPT-5.4
- Gemini Embedding 2
- Grok 4.20
- LongCat-Flash-Prover
- MiniMax M2.7
- Nemotron 3
- Qwen3.5

未命中事件：

- ASMR
- GLM-5-Turbo
- Gemini 3.1 Flash-Lite
- MAI-Image-2
- MiMo
- Midjourney V8
- Mistral Small 4
- Seedance 2.0
- SkyReels V4
- Vidu Q3

### 5. Finding 评测

命令：

```bash
python eval/eval_findings.py \
  --run-dir "logs/只使用本地新闻数据库RAG不要使用网络搜索查找2026年3月-20260531-111213" \
  --target report
```

结果：

| 指标 | 结果 |
|---|---:|
| Finding Recall | 35.0% (7/20) |
| Finding Precision | 53.8% (7/13) |
| Finding F1 | 42.4% |
| Evidence Support Rate | 100.0% |

正确识别事件：

- GPT-5.4
- Gemini Embedding 2
- Nemotron 3
- Grok 4.20
- MiniMax M2.7
- LongCat-Flash-Prover
- Qwen3.5

漏报事件：

- ASMR
- Composer
- DeepSeek V4
- GLM-5-Turbo
- GPT-5.3 Instant
- Gemini 3.1 Flash-Lite
- MAI-Image-2
- MiMo
- Midjourney V8
- Mistral Small 4
- Seedance 2.0
- SkyReels V4
- Vidu Q3

## 踩坑记录

### 1. `.env` 不能提交

用户消息里包含有效 API key，但 `.env` 当前被 `.gitignore` 忽略，没有进入提交。

处理原则：

- 不在报告里复述密钥。
- 不把 `.env` 加入 Git。
- 代码只读取环境变量。

### 2. 旧日志缺少分阶段数据

旧 run 只有 `raw_results.json`、`report.md`、`evidence_pool.json` 等文件，缺少：

- `retrieval_details.json`
- `sub_queries.json`
- `supervisor_tool_calls.json`
- `final_report_notes.json`

因此旧 run 无法完整评估 Dense/Sparse/Merge/Rerank/Supervisor，只能做引用、Finding 或部分 compression 评测。

处理：

- 新增主图状态透传和 runner 落盘。
- `eval_full_pipeline.py` 遇到缺失 Supervisor 产物时显示 `N/A`，不再误报 0。

### 3. reranker 离线模式设置位置不够早

问题：

`CrossEncoder` 初始化时仍尝试访问 Hugging Face，导致 SSL 错误。

原因：

环境变量在函数内部设置，晚于部分 transformers/sentence_transformers 初始化路径。

处理：

- 将 `HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE` 提前到 `rag/reranker.py` 顶部，在导入 CrossEncoder 之前设置。

### 4. Mac `find` 不支持 GNU `-printf`

验证日志文件时使用了：

```bash
find ... -printf
```

macOS 自带 `find` 不支持该参数。

处理：

- 后续改用 `ls -1` 或普通 `find` 输出。

### 5. 大模型合成 rubric 容易给抽象关键词

第一次合成的 Plan rubric 中，“头部厂商”维度只给了“top vendor”“big tech”等抽象词，没有 OpenAI/Google/Anthropic 这类具体可匹配关键词，导致覆盖率被低估。

处理：

- 调整 `eval_full_pipeline.py` 的合成 prompt，明确要求每个维度同时包含抽象词和具体例子。

### 6. 完整主图 smoke 输出延迟

端到端 smoke 一度看起来卡住，原因是 CPU rerank 和 final writer 阶段 stdout 较少；最后实际完成。

处理：

- 没有杀进程，等待最终输出。
- 复盘中明确该 smoke 只验证产物链路，不作为生产质量 benchmark。

### 7. Pydantic serializer warning

运行中多次出现：

```text
PydanticSerializationUnexpectedValue(Expected `none` ...)
```

这类 warning 来自 LangGraph/LangChain 事件序列化结构化输出，不影响本次 run 完成。

后续可单独处理，但不是本次评测链路的阻塞项。

## 最终判断

当前链路的瓶颈更明确了：

1. **Plan 基本可用**  
   子查询能覆盖主要维度，日期和分类也正确。但完整主图中出现了 20 个子查询，说明 Supervisor 两次调用 RAG 后存在一定重复，需要控制调用次数和 query 去重。

2. **Dense 和 Sparse 召回都不差**  
   完整 smoke 中 Dense/Sparse/Merge 都达到 90% Event Recall，说明底层候选池能覆盖大部分 GT。

3. **Merge 没带来额外事件召回**  
   Dense 和 Sparse 单路都 90%，Merge 仍是 90%。双路提升主要可能体现在文章覆盖，不体现在事件覆盖。

4. **Rerank 有明显误杀**  
   Merge 90% → Rerank 80%。Rerank 提升了 Article Precision，但牺牲了事件召回。

5. **Compress 是最大损耗点**  
   RAG raw 命中 16 个事件，compressed 引用只覆盖 10 个，事件保留率 62.5%。后续应优先改结构化压缩，而不是继续扩大召回。

6. **Supervisor 交接不是当前主要瓶颈**  
   在 smoke run 中，RAG compressed 到 final notes、final notes 到 report 都是 100%。当前问题主要发生在 Rerank 和 Compress。

7. **最终报告仍漏报较多**  
   Finding Recall 35%，Reference Event Recall 50%。即使检索层能到 80-90%，最终“写出来并可被结构化抽取”的事件仍明显不足。

## 后续建议

优先级从高到低：

1. **RAG Compress 改为结构化事件列表**
   - 先输出 `[{event_name, vendor, model_name, date, article_ids, evidence}]`
   - 再交给 final writer 写报告。
   - 目标：Compression Retention Rate 从 62.5% 提升到 85%+。

2. **Rerank 增加 recall guardrail**
   - 如果某个事件只在 merged 里出现 1-2 篇，不应被 rerank 全部踢出 Top K。
   - 可做 event-aware 或 diversity-aware rerank。

3. **Supervisor 控制重复 RAG 调用**
   - 当前 smoke 中 ConductRAGResearch 调了 2 次，合计 20 个子查询。
   - 建议限制同一月报任务只调用 1 次 RAG，或对子查询做跨调用去重。

4. **引用评测升级为 finding-level citation**
   - 现在只能看报告引用了哪些文章。
   - 下一步应评每个 finding 是否至少有 1 个 GT evidence article 支撑。

5. **把 `eval_full_pipeline.py` 接入批量实验**
   - 每次改检索、rerank、compress 后自动生成 `STAGE_EVAL_REPORT.md`。
   - 用同一套 `stage_eval_metrics.json` 做横向对比。

## 相关产物

- 分阶段评测脚本：`eval/eval_full_pipeline.py`
- 评测指南：`eval/EVALUATION_GUIDE.md`
- 独立 RAG run：`logs/eval搜索本地新闻数据库查找2026年3月1-20260531-110457`
- 完整主图 smoke run：`logs/只使用本地新闻数据库RAG不要使用网络搜索查找2026年3月-20260531-111213`
- 完整主图分阶段报告：`logs/只使用本地新闻数据库RAG不要使用网络搜索查找2026年3月-20260531-111213/STAGE_EVAL_REPORT.md`
- 完整主图 Finding 报告：`logs/只使用本地新闻数据库RAG不要使用网络搜索查找2026年3月-20260531-111213/FINDING_REPORT.md`
- 完整主图引用评测：`logs/只使用本地新闻数据库RAG不要使用网络搜索查找2026年3月-20260531-111213/REFERENCE_EVAL.json`
