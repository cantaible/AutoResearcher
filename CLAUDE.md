# Claude 工作指南

## 沟通风格

- 简明扼要，结构清晰，避免冗长
- 先说结论，再说细节
- 用数据和代码说话，少用描述性文字
- 简单问题 30 秒内回答，复杂问题不超过 2 分钟

## 代码风格

- 突出核心逻辑，省略样板代码
- 中文注释（注释说明 why，不说明 what）
- 单次改动控制在 20 行以内（注释不计）

## 项目运行

```bash
# TUI 交互模式
python src/tui.py

# CLI 模式
python -m src.main --topic "你的研究主题"

# RAG 评测
python eval/eval_findings.py --run-dir logs/<run_dir>

# 检索评测
python eval/eval_rag.py
```

## 目录结构

- `src/` — 核心代码（graph, state, runner, rag_subgraph）
- `rag/` — RAG 检索模块（OpenSearch 混合检索 + Rerank）
- `eval/` — 评测框架（finding 抽取、匹配、指标计算）
- `logs/` — 运行日志和输出（report.md, evidence_pool.json）
- `docs/` — 项目文档和设计记录
- `scripts/` — 工具脚本

## 关键约定

- 触发 RAG 检索：查询中需包含"本地新闻数据库"或"本地数据库"
- evidence_pool 仅在 ConductRAGResearch 路径下生成
- State 使用 `Annotated[list, operator.add]` 做跨节点累加
