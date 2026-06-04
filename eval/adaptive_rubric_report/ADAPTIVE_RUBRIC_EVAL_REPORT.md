# Adaptive Rubric 评测报告

## 结论
- 可迭代/可学习 rubric 在主指标上优于固定 rubric。
- 最优方法 `adaptive_forest` 的 Kendall Tau 为 0.1735，固定 rubric 为 0.0341，提升 +0.1393。
- 当前证明的是“从候选特征中学习组合方式”优于固定权重，不是完整证明 LLM 自动生成文字 rubric 更好。

## 实验定义

- 数据：80 个 execute，来自 5 个历史 run；每个 execute 是一次 `query -> 候选文章 -> rerank TopK`。
- 输入 X：不查看真值即可计算的 rubric/统计特征，例如 impact、事件多样性、来源多样性、rerank 分数和噪音率。
- 主目标 y：只有一个，即 `composite_quality`。其他 y 只用于诊断提升来自哪个方面，不参与主结论。
- 验证：按 run 做 leave-one-run-out；每轮用 4 个 run 训练，用剩余 1 个 run 测试，避免同一 run 泄漏。

`composite_quality` 的选择原因：研究型 RAG 不能只追求召回率，还要兼顾文章准确性、排序和 rerank 不误删事件。

```text
composite_quality = 0.45*event_recall + 0.20*article_precision
                  + 0.15*article_recall + 0.15*ndcg10
                  + 0.05*event_retention
```

## 主结果

| 方法 | Kendall Tau | Spearman | Pairwise Acc | MAE |
|---|---:|---:|---:|---:|
| adaptive_forest | 0.1735 | 0.2505 | 0.5868 | 0.0919 |
| semantic_rerank_only | 0.1515 | 0.2152 | 0.5756 | 0.3100 |
| adaptive_ridge | 0.0459 | 0.0839 | 0.5230 | 0.1262 |
| adaptive_pls | 0.0421 | 0.0610 | 0.5211 | 0.1518 |
| adaptive_two_stage_pls_top5 | 0.0384 | 0.0467 | 0.5160 | 0.1082 |
| fixed_rubric | 0.0341 | 0.0515 | 0.5169 | 0.2131 |
| adaptive_top5_ridge | -0.0448 | -0.0609 | 0.4745 | 0.1197 |
| adaptive_elasticnet | -0.1556 | -0.2009 | 0.3749 | 0.1089 |

指标含义：

- `Kendall Tau`：预测排序与真值排序的一致性，范围 -1 到 1，越高越好；本报告的主判定指标。
- `Spearman`：另一种排序相关性指标，越高越好。
- `Pairwise Acc`：任取两个 execute，判断谁更好时的准确率，越高越好。
- `MAE`：预测质量分与真值分的平均绝对误差，越低越好。

方法含义与选择原因：

- `fixed_rubric`：当前项目的手写公式，作为基线。
- `semantic_rerank_only`：只用 bge reranker 平均语义分，检验业务 rubric 是否必要。
- `adaptive_pls`：所有候选特征经标准化后用 PLS 学习线性组合，最接近 AutoMetrics。
- `adaptive_ridge`：用 Ridge 学习稳定的线性权重，作为共线特征下的线性备选。
- `adaptive_elasticnet`：用 ElasticNet 自动压缩无用特征，检验稀疏线性组合。
- `adaptive_forest`：用随机森林学习非线性关系，检验固定线性公式是否过于简单。
- `adaptive_top5_ridge`：先按相关性选 Top5 特征，再用 Ridge 拟合。
- `adaptive_two_stage_pls_top5`：先用 PLS 选 Top5 特征，再重新拟合 PLS，对应论文的两阶段思路。

## 诊断目标

以下目标不是多个主真值，而是用于解释主结果的分项真值。

| 目标 | 最优方法 | 固定 Rubric Tau | 最优 Tau | 提升 |
|---|---|---:|---:|---:|
| composite_quality | adaptive_forest | 0.0341 | 0.1735 | +0.1393 |
| event_recall | semantic_rerank_only | 0.0577 | 0.1581 | +0.1004 |
| article_precision | adaptive_forest | 0.0739 | 0.2192 | +0.1453 |
| ndcg10 | adaptive_forest | 0.0067 | 0.3723 | +0.3656 |
| event_retention | adaptive_pls | 0.0943 | 0.3380 | +0.2437 |

- `composite_quality`：唯一主目标 y；综合衡量事件覆盖、文章准确性、排序质量和 rerank 保留能力。
- `event_recall`：应找到的相关事件中，rerank TopK 实际找到了多少。
- `article_precision`：rerank TopK 中，人工标注为相关的文章占比。
- `ndcg10`：前 10 名中相关文章是否排得更靠前。
- `event_retention`：merged 阶段已经召回的相关事件，有多少没有被 rerank 丢掉。

## 权重解释

以下权重来自全量数据上拟合的 Ridge 模型，仅用于解释候选特征与主目标的关系，不用于主结果，也不能直接上线。绝对值越大表示关联越强；正负号可能受特征共线性影响。

| 特征 | 权重 | 含义与选择原因 |
|---|---:|---|
| event_diversity | -0.0211 | Top10 中不同已识别事件数 / 文章数；用于描述事件覆盖的分散程度。 |
| penalty_score | 0.0182 | 同一来源占比过高时的惩罚分；用于描述来源刷榜风险。 |
| max_source_share | 0.0100 | Top10 中占比最高的单一来源比例；用于描述来源集中度。 |
| source_diversity | -0.0091 | Top10 中不同来源数 / 文章数；用于描述来源多样性。 |
| known_event_rate | -0.0083 | Top10 中能映射到已知事件的文章比例；用于描述事件可识别性。 |
| tier2_vendor_rate | -0.0075 | 涉及二级厂商的文章比例；用于描述厂商构成。 |
| duplicate_event_rate | 0.0064 | Top10 中重复事件文章的比例；用于描述同一事件挤占结果的问题。 |
| hard_noise_rate | 0.0061 | 包含教程、指南、下载榜等强噪音词的文章比例。 |

## 限制

- 80 个 execute 只覆盖 32 个唯一 query，且来自少量历史 run，仍需更多独立主题验证。
- 当前真值来自文章与事件标注，适合验证检索质量；若要验证最终研究体验，还应增加人工 1-5 分整体质量标签。