# RAG 检索评估实验记录

## 1. 实验设计

**目标**：量化评估 RAG 子图对本地新闻数据库的检索能力——给定一个研究主题，RAG 能找到多少相关的模型发布事件。

**测试主题**：`搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻`

**流程**：运行 RAG 子图（Plan → Execute → Compress） → 提取检索结果中的 ArticleID → 与 Ground Truth 对比计算指标。

## 2. 数据处理

### 2.1 数据源

MariaDB `news_reader.news_article` 表，筛选条件：`category = 'AI'`，`created_at` 在 2026年3月，共 **2262 篇**。

### 2.2 自动打标

脚本 `eval/label_articles.py` 调用 GPT-5.4，为每篇文章标注：
- **event_type**：`model_release` / `product_launch` / `funding_business` / `industry_news` / `other`（5 类正交标签）
- **entities**：开放式实体列表（模型名、公司名等）

### 2.3 人工审阅与家族合并

1. **审阅 model_release**：通过交互式 HTML 工具（`eval/event_review.html`）人工审查 160 篇 model_release 文章，移除 28 篇误标的。
2. **事件归并**：调用 LLM 为每篇 model_release 文章标注标准化的 `release_event` 名称，将重复报道聚合为独立事件。
3. **家族合并**：通过交互式工具（`eval/family_merge.html`）将同一模型家族的子型号合并（如 MiMo / MiMo-V2 / MiMo-V2-Pro → "MiMo"），并删除过于小众的事件。

最终 Ground Truth：**20 个模型家族**，覆盖 **100 篇**文章。

### 2.4 关键文件

| 文件 | 说明 |
|------|------|
| `eval/article_labels.json` | 全量标签（2262 篇） |
| `eval/event_families.json` | 家族定义（20 个家族及其包含的子事件和文章） |
| `eval/eval_rag.py` | 评估脚本（运行 RAG + 计算指标） |
| `eval/label_articles.py` | 自动打标脚本 |

## 3. 指标设计

### 3.1 核心指标：Event Recall

```
Event Recall = 命中家族数 / 全部家族数
```

**命中判定**：检索结果中包含该家族的**任意一篇**文章即算命中。同一模型的多篇重复报道不重复计数。

### 3.2 辅助指标

| 指标 | 公式 | 说明 |
|------|------|------|
| Article Precision | 检索到的 model_release 文章数 / 检索总数 | 返回结果的纯度 |
| Article Recall | 检索到的 model_release 文章数 / 全库 model_release 总数 | 文章级召回 |
| F1 | 2 × P × R / (P + R) | 调和平均 |
| NDCG@K | 归一化折损累积增益（二值相关度） | 排序质量 |

### 3.3 对齐机制

RAG 搜索结果输出中强制包含 `ArticleID` 字段（在 `rag/rag_search.py` 中注入），评估脚本通过正则提取该字段与 `article_labels.json` 的 `article_id` 进行匹配。

## 4. 实验结果

### Exp #1 — Baseline（2026-04-10）

**配置**：Plan 模型 gpt-5.4 | Execute 模型 gpt-4.1-mini | top_k=10 | max_retries=2 | 11 个子查询

| 指标 | 值 |
|:-----|:---:|
| 🎯 **Event Recall** | **55.00%** (11/20) |
| Article Precision | 34.48% |
| Article Recall | 30.00% |
| F1 | 32.09% |
| NDCG@10 | 0.3700 |
| NDCG@20 | 0.4135 |

**命中 11 个**：DeepSeek V4, GPT-5.3 Instant, GPT-5.4, Gemini Embedding 2, Grok 4.20, LongCat-Flash-Prover, MiMo, MiniMax M2.7, Mistral Small 4, Nemotron 3, Qwen3.5

**未命中 9 个**：ASMR, Composer, Gemini 3.1 Flash-Lite, GLM-5-Turbo, MAI-Image-2, Midjourney V8, Seedance 2.0, SkyReels V4, Vidu Q3

**Run 目录**：`logs/eval搜索本地新闻数据库查找2026年3月1-20260410-185342/`
