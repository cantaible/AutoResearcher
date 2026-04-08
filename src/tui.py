"""Deep Research Terminal UI.

用法：python src/tui.py "研究主题"
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent))

console = Console()
_streaming = False  # 是否正在流式输出 LLM token


async def on_event(event: dict):
    """将标准化事件渲染到终端。每种事件类型对应不同的显示样式。"""
    global _streaming
    t = event["type"]

    if t == "node_start":
        # 新节点开始 → 打印节点名（如果之前在流式输出先换行）
        if _streaming:
            print()
            _streaming = False
        console.print(f"\n▶ [bold cyan]{event.get('node', '')}[/]")

    elif t == "llm_start":
        console.print(f"  🤖 [dim]{event.get('model', '')}[/]")

    elif t == "llm_stream":
        # 逐 token 打字效果：直接写到 stdout，不换行
        sys.stdout.write(event["token"])
        sys.stdout.flush()
        _streaming = True

    elif t == "llm_end":
        if _streaming:
            print()  # 流式结束，换行
            _streaming = False
        usage = event.get("usage", {})
        total = usage.get("total_tokens", 0)
        if total:
            console.print(f"  [dim]💰 {total} tokens[/]")

    elif t == "tool_start":
        console.print(f"  🔧 [yellow]{event.get('tool', '')}[/] [dim]{event.get('args', '')[:80]}[/]")

    elif t == "tool_end":
        console.print(f"  [dim]   └─ {event.get('result', '')[:100]}[/]")

    elif t == "clarify":
        console.print(Panel(event["question"], title="🤔 需要澄清", border_style="yellow"))

    elif t == "report":
        console.print(Panel(Markdown(event["content"]),
                            title="📋 最终报告", border_style="green"))


async def on_clarify(question: str) -> str:
    """暂停渲染，等待用户在终端输入回答。"""
    # asyncio 中不能直接用 input()（会阻塞事件循环），用 run_in_executor 包装
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(
        None, lambda: console.input("[bold yellow]你的回答：[/] ")
    )
    return answer


async def main(topic: str):
    """启动完整研究流水线。"""
    from runner import run_research

    console.print(Panel(
        f"[bold]{topic}[/]",
        title="🔬 Deep Research Agent",
        border_style="blue",
    ))

    run_dir = await run_research(
        topic, on_event=on_event, on_clarify=on_clarify
    )
    console.print(f"\n📁 [dim]结果已保存: {run_dir}[/]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = console.input("[bold]请输入研究主题：[/] ")
    asyncio.run(main(topic))
