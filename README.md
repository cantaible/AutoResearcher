# agent_hello_world

一个最小可运行的中文 LangGraph Agent 基础项目。

## 这个项目现在做什么

第一版只做一件事：

- 接收用户输入
- 进入唯一的 `chat` 节点
- 拼接中文系统提示词
- 调用模型
- 返回中文回答

也就是最简单的单节点流程：

`用户输入 -> chat 节点 -> 输出结果`

## 为什么先做成单节点

- 最容易跑通
- 最适合拿来继续改造成别的 Agent
- 不会一开始就把结构做重
- 后续加工具、加路由、加多 Agent 都很自然

## 项目结构

```text
agent_hello_world/
├── .env.example
├── AGENTS.md
├── README.md
├── environment.yml
├── langgraph.json
├── pyproject.toml
├── examples/
│   └── run_local.py
├── tests/
│   └── test_smoke.py
└── src/
    ├── cli.py
    ├── configuration.py
    ├── debug_trace.py
    ├── graph.py
    ├── model_factory.py
    ├── prompts.py
    └── state.py
```

## 环境使用

按你的偏好，默认推荐直接使用 conda 的 `base` 环境：

```bash
conda activate base
cd /path/to/agent_hello_world
pip install -r requirements-dev.txt
```

如果以后你想切独立环境，也可以：

```bash
conda env create -f environment.yml
conda activate agent_hello_world
```

如果你只想安装运行依赖，不装测试和 lint 工具：

```bash
pip install -r requirements.txt
```

## 密钥配置

先复制：

```bash
cp .env.example .env
```

然后在 `.env` 中填写：

- `OPENAI_API_KEY`

如果你使用代理或兼容网关，也可以填写：

- `OPENAI_BASE_URL`

## 运行方式

本地单次运行：

```bash
python examples/run_local.py --message "请用中文介绍一下你自己"
```

如果你想直接走模块入口，也可以：

```bash
PYTHONPATH=src python -m cli --message "请用中文介绍一下你自己"
```

运行测试：

```bash
pytest -q
```

启动 LangGraph 本地开发服务：

```bash
langgraph dev --allow-blocking
```

## 规则文件

项目根目录下的 `AGENTS.md` 记录了开发规则，比如：

- 优先使用 conda 的 `base` 环境
- 所有用户可见文案默认使用中文
- 第一版优先保持单节点简单结构
- 密钥统一写在 `.env`

后续继续改这个项目时，建议先读这个文件。
