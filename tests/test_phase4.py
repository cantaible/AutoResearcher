"""Phase 4 验证：完整主图端到端测试（纯文本版，适合 AI agent 运行）。

与 tui.py 共用 runner.py 核心，产出相同的 logs 目录结构。
需要环境变量：OPENAI_API_KEY (或 ANTHROPIC_API_KEY), TAVILY_API_KEY
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


async def on_event(evt: dict):
    """纯文本事件渲染，AI agent 友好。"""
    t = evt["type"]
    node = evt.get("node", "")

    if t == "node_start":
        print(f"\n▶ [{node}]")
    elif t == "llm_start":
        print(f"  🤖 模型: {evt.get('model', '')}")
    elif t == "llm_end":
        usage = evt.get("usage", {})
        total = usage.get("total_tokens", 0)
        print(f"  💰 {total} tokens | 内容: {evt.get('content', '')[:100]}")
    elif t == "tool_start":
        print(f"  🔧 {evt.get('tool', '')} {evt.get('args', '')[:80]}")
    elif t == "tool_end":
        print(f"     └─ {evt.get('result', '')[:100]}")
    elif t == "clarify":
        print(f"  🤔 AI 追问: {evt['question']}")
    elif t == "report":
        print(f"\n📋 最终报告（前500字）:\n{evt['content'][:500]}")


async def on_clarify(question: str) -> str:
    """从终端获取用户输入。"""
    return input("你的回答: ")


async def main():
    from runner import run_research

    topic = "2025年诺贝尔物理学奖的研究内容是什么，有什么实际应用前景"
    print(f"📋 研究主题: {topic}")
    print("⏳ 启动完整主图...\n")

    run_dir = await run_research(
        topic, on_event=on_event, on_clarify=on_clarify
    )
    print(f"\n📁 结果已保存: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
