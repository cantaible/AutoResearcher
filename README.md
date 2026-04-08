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
| Phase 4 | 完整主图 + TUI | 已完成 |

详见 [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md)。

## 使用方式

### Terminal UI（推荐）

```bash
conda activate base
python src/tui.py "给我调研一下2026年3月1日到3月31日都有哪些大模型发布了，调研的对象包括基础大语言模型，图像生成模型，视频生成模型，和Agent&coding模型，重点关注头部厂商，和Lmarena leaderboard榜单当时上榜的模型"
```

- 实时显示研究过程（节点状态、LLM 流式输出、工具调用）
- 支持交互式澄清（AI 追问时可在终端输入回答）
- 所有中间状态自动保存到 `logs/{topic}-{timestamp}/` 目录

### 纯文本脚本（适合 AI agent）

```bash
python tests/test_phase4.py
```

### 运行产出

每次运行自动创建独立目录：

```
logs/{topic-slug}-{timestamp}/
├── checkpoints.db    # LangGraph checkpoint（可回放每一步）
├── events.jsonl      # 所有事件流（实时写入）
├── run_meta.json     # 运行元数据（耗时、token 用量等）
└── report.md         # 最终研究报告
```

## 项目结构

```
├── src/                  # 实现代码
│   ├── state.py          # 状态定义
│   ├── configuration.py  # 配置体系
│   ├── prompts.py        # Prompt 模板
│   ├── utils.py          # 工具函数
│   ├── graph.py          # LangGraph 图定义
│   ├── runner.py         # 核心运行器（事件采集 + 持久化）
│   └── tui.py            # Terminal UI（Rich）
├── reference/            # 原版源码参考（用于 diff 对比）
├── tests/                # 测试用例
└── docs/                 # 进度追踪和复现指南
```

## 环境配置

```bash
conda activate base
pip install -e ".[dev]"
```

需要在 `.env` 文件中配置：
```
OPENAI_API_KEY=sk-xxx
TAVILY_API_KEY=tvly-xxx
```

## AI 协作入口

- **Codex**: 自动读取 `AGENTS.md`
- **Antigravity**: 使用 `/resume` 恢复工作
- **其他 AI**: 请先阅读 `docs/PROGRESS.md` 和 `docs/REPRODUCTION_GUIDE.md`
