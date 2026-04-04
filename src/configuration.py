"""运行时配置。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

for env_path in (PROJECT_ROOT / ".env", WORKSPACE_ROOT / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)


class Configuration(BaseModel):
    """从环境变量和 LangGraph 运行配置中读取设置。"""

    agent_model: str = "openai:gpt-4o-mini"
    agent_model_max_tokens: int = 4096
    debug_trace: bool = False
    extra_system_prompt: str = ""

    @classmethod
    def from_runnable_config(
        cls,
        config: RunnableConfig | None = None,
    ) -> Configuration:
        """优先从运行配置读取，其次读取环境变量。"""
        configurable = config.get("configurable", {}) if config else {}

        values: dict[str, Any] = {
            "agent_model": configurable.get("agent_model") or os.getenv("AGENT_MODEL"),
            "agent_model_max_tokens": configurable.get("agent_model_max_tokens")
            or os.getenv("AGENT_MODEL_MAX_TOKENS"),
            "debug_trace": configurable.get("debug_trace") or os.getenv("DEBUG_TRACE"),
            "extra_system_prompt": configurable.get("extra_system_prompt")
            or os.getenv("EXTRA_SYSTEM_PROMPT"),
        }

        cleaned = {key: value for key, value in values.items() if value not in (None, "")}
        return cls(**cleaned)
