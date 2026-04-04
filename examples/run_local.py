"""本地执行一次图调用。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="本地运行 agent_hello_world 图。")
    parser.add_argument("--message", required=True, help="要发送给 Agent 的用户消息。")
    return parser.parse_args()


async def main() -> None:
    """调用图并打印最后一条消息。"""
    from graph import agent_graph

    args = parse_args()
    result = await agent_graph.ainvoke(
        {"messages": [{"role": "user", "content": args.message}]},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
