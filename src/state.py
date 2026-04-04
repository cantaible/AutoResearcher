"""图状态定义。"""

from langgraph.graph import MessagesState


class AgentInputState(MessagesState):
    """图输入状态。"""


class AgentState(MessagesState):
    """图运行状态。"""
