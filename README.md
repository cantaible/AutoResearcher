# My Deep Research

> 完全复现 [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) 的学习项目。

## 项目目标

以学习为目的，逐步复现 open_deep_research —— 一个基于 LangGraph 的三层嵌套自动化深度研究 Agent：

```
用户输入 → 意图澄清 → 研究简报 → Supervisor(管理) → Researcher×N(并行搜索) → 压缩 → 最终报告
```

## 当前进度

查看 [docs/PROGRESS.md](docs/PROGRESS.md) 了解当前阶段和待办项。

## 复现方案

分 4 个阶段渐进式构建，每阶段独立可测试：

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 基础设施（state/config/prompts/utils） | 已完成 |
| Phase 2 | Researcher 子图（ReAct 循环 + 压缩） | 已完成 |
| Phase 3 | Supervisor 子图 + 报告生成 | 已完成 |
| Phase 4 | 完整主图 + 端到端验证 | 未开始 |

详见 [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)。

## 项目结构

```
├── src/                  # 实现代码
│   ├── state.py          # 状态定义
│   ├── configuration.py  # 配置体系
│   ├── prompts.py        # Prompt 模板
│   ├── utils.py          # 工具函数
│   └── graph.py          # LangGraph 图定义
├── reference/            # 原版源码参考（用于 diff 对比）
├── tests/                # 每阶段的测试用例
└── docs/                 # 进度追踪和复现指南
```

## 快速开始

```bash
conda activate base
pip install -e ".[dev]"
pytest tests/
```

## AI 协作入口

- **Codex**: 自动读取 `AGENTS.md`
- **Antigravity**: 使用 `/resume` 恢复工作
- **其他 AI**: 请先阅读 `docs/PROGRESS.md` 和 `docs/REPRODUCTION_GUIDE.md`
