"""命令行入口。"""

from __future__ import annotations

import argparse
import asyncio

from graph import deep_researcher as agent_graph


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 AutoResearcher 单节点 Agent。")
    parser.add_argument(
        "--message",
        default="",
        help="要发送给 Agent 的用户消息。若不传，会进入交互输入。",
    )
    return parser.parse_args()


async def _run_once(message: str) -> str:
    """执行一次图调用并返回最后一条消息内容。"""
    result = await agent_graph.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
    )
    return str(result["messages"][-1].content)


def main() -> None:
    """CLI 主入口。"""
    args = parse_args()
    message = args.message.strip() or input("[用户] 请输入内容：").strip()
    if not message:
        raise SystemExit("输入不能为空。")

    output = asyncio.run(_run_once(message))
    print(f"\n[AutoResearcher]\n{output}\n")


if __name__ == "__main__":
    main()
