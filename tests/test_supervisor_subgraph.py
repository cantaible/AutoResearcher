"""Phase 3 最小验证：Supervisor 子图的退出行为。"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END

sys.path.insert(0, "src")


def _save_phase3_run(payload: dict) -> Path:
    """保存一次 Phase 3 调试运行结果，便于后续复盘。"""
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    run_id = f"phase3-{datetime.now():%Y%m%d-%H%M%S}"
    log_path = logs_dir / f"{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return log_path


def check_supervisor_tools_exits_and_collects_notes() -> None:
    """当没有新的工具调用时，Supervisor 应结束并汇总已有笔记。"""
    from graph import supervisor_tools

    state = {
        "supervisor_messages": [
            ToolMessage(
                content="研究笔记 A",
                name="ConductResearch",
                tool_call_id="tool-1",
            ),
            AIMessage(content="当前信息已足够"),
        ],
        "research_brief": "调研某主题并输出结论",
        "research_iterations": 1,
    }

    command = asyncio.run(supervisor_tools(state, config={}))

    assert command.goto == END
    assert command.update["research_brief"] == "调研某主题并输出结论"
    assert command.update["notes"] == ["研究笔记 A"]


def check_supervisor_tools_runs_conduct_research() -> None:
    """当收到 ConductResearch 调用时，应运行子图并回到 supervisor。"""
    import graph

    class DummySubgraph:
        async def ainvoke(self, state, config):
            assert state["research_topic"] == "研究子任务"
            return {
                "compressed_research": "子研究摘要",
                "raw_notes": ["原始笔记 A"],
            }

    original_subgraph = graph.researcher_subgraph
    graph.researcher_subgraph = DummySubgraph()

    state = {
        "supervisor_messages": [
            AIMessage(
                content="开始委派研究",
                tool_calls=[{
                    "name": "ConductResearch",
                    "args": {"research_topic": "研究子任务"},
                    "id": "call-1",
                }],
            ),
        ],
        "research_brief": "总体研究简报",
        "research_iterations": 1,
    }

    try:
        command = asyncio.run(graph.supervisor_tools(state, config={}))
    finally:
        graph.researcher_subgraph = original_subgraph

    assert command.goto == "supervisor"
    assert command.update["raw_notes"] == ["原始笔记 A"]
    assert command.update["supervisor_messages"][0].content == "子研究摘要"


def check_final_report_generation_success() -> None:
    """报告生成成功时，应返回 final_report 并清空 notes。"""
    import graph

    class DummyModel:
        def with_config(self, config):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="最终报告内容")

    original_model = graph.configurable_model
    graph.configurable_model = DummyModel()

    state = {
        "messages": [],
        "notes": ["研究发现 A", "研究发现 B"],
        "research_brief": "输出最终总结",
    }

    try:
        result = asyncio.run(graph.final_report_generation(state, config={}))
    finally:
        graph.configurable_model = original_model

    assert result["final_report"] == "最终报告内容"
    assert result["messages"][0].content == "最终报告内容"
    assert result["notes"] == {"type": "override", "value": []}


async def _run_phase3_demo() -> None:
    """运行一次带状态历史的最小 Phase 3 场景，并写入 logs/。"""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import graph

    class DummyModel:
        def __init__(self):
            self.calls = 0
        def bind_tools(self, tools): return self
        def with_retry(self, **kwargs): return self
        def with_config(self, config): return self
        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="开始委派研究", tool_calls=[{
                    "name": "ConductResearch",
                    "args": {"research_topic": "研究子任务"},
                    "id": "call-1",
                }])
            return AIMessage(content="当前信息已足够")

    class DummySubgraph:
        async def ainvoke(self, state, config):
            return {"compressed_research": "子研究摘要", "raw_notes": ["原始笔记 A"]}

    graph.configurable_model = DummyModel()
    graph.researcher_subgraph = DummySubgraph()
    thread_id = f"phase3-{datetime.now():%Y%m%d-%H%M%S}"
    db_path = str(Path(__file__).resolve().parent.parent / "logs" / "checkpoints.db")
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        subgraph = graph.supervisor_builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = await subgraph.ainvoke({
            "supervisor_messages": [HumanMessage(content="总体研究任务")],
            "research_brief": "调研某主题并输出结论",
        }, config=config)
        history = []
        async for snapshot in subgraph.aget_state_history(config):
            history.append({
                "step": snapshot.metadata.get("step", -1),
                "node": snapshot.metadata.get("source", "unknown"),
                "writes_keys": list((snapshot.metadata.get("writes") or {}).keys()),
                "msg_count": len(snapshot.values.get("supervisor_messages", [])),
                "iterations": snapshot.values.get("research_iterations", 0),
            })

    log_path = _save_phase3_run({
        "case": "supervisor_subgraph_demo",
        "thread_id": thread_id,
        "notes": result.get("notes", []),
        "research_brief": result.get("research_brief", ""),
        "raw_notes": result.get("raw_notes", []),
        "steps": history,
    })
    print(f"📁 Phase 3 运行记录已保存: {log_path}")


if __name__ == "__main__":
    check_supervisor_tools_exits_and_collects_notes()
    check_supervisor_tools_runs_conduct_research()
    check_final_report_generation_success()
    print("✅ Phase 3 基础检查通过")
    asyncio.run(_run_phase3_demo())
