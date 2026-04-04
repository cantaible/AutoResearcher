"""最小可运行的单节点 LangGraph 图。"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from configuration import Configuration
from debug_trace import log_trace
from model_factory import build_chat_model
from prompts import DEFAULT_SYSTEM_PROMPT
from state import AgentInputState, AgentState


async def chat_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list]:
    """执行唯一的对话节点。"""
    settings = Configuration.from_runnable_config(config)
    model = build_chat_model(
        settings.agent_model,
        settings.agent_model_max_tokens,
        config,
    )
    messages = [SystemMessage(content=_build_system_prompt(settings))] + state["messages"]
    response = await model.ainvoke(messages)

    log_trace(
        "chat_response",
        {
            "message_count": len(messages),
            "response": response,
        },
        config,
    )
    return {"messages": [response]}


def _build_system_prompt(settings: Configuration) -> str:
    """拼接系统提示词。"""
    if settings.extra_system_prompt:
        return f"{DEFAULT_SYSTEM_PROMPT}\n\n{settings.extra_system_prompt}"
    return DEFAULT_SYSTEM_PROMPT


builder = StateGraph(AgentState, input_schema=AgentInputState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
agent_graph = builder.compile()
