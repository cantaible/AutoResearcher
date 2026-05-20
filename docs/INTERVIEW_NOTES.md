# 面试回答模板

## Evidence Support Rate 如何计算

**定义**：衡量 findings 是否有高质量证据支撑。

**公式**：
```
Evidence Support Rate = 在 gold evidence 中的文章数 / 总引用文章数
```

**计算步骤**：
1. 为每个 finding 找到引用的文章（基于事件匹配）
2. 检查这些文章是否在 gold evidence 中
3. 汇总：sum(在gold中) / sum(总引用)

**vs Faithfulness**：
- Evidence Support Rate：检查是否引用权威文章（轻量）
- Faithfulness：检查内容是否忠实于文档（需要 LLM）

---

## AutoResearcher 评测体系

**两层评测**：

**Layer 1: RAG 检索**
- 对象：Dense/Sparse/Merged/Reranked 四阶段
- 指标：Event Recall, Article Recall
- 输出：RETRIEVAL_REPORT.md

**Layer 2: Finding 评测**
- 对象：最终报告
- 指标：Finding Recall/Precision/F1, Evidence Support Rate
- 方法：LLM 抽取 + 模糊匹配
- 输出：FINDING_REPORT.md

**优化方向**：
1. 提高检索召回
2. 完善 Ground Truth
3. 优化 Finding 抽取策略

---

## Ground Truth 构建

**数据**：
- 2262 篇标注文章
- 20 个事件家族

**增强**：
- 生成 canonical_name 和 aliases（规则生成，零成本）
- 构建反向索引（article_id → event_name）

**示例**：
```json
{
  "canonical_name": "GPT-5.4",
  "aliases": ["GPT 5.4", "gpt-5.4", "OpenAI GPT-5.4"]
}
```
