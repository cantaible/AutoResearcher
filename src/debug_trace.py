"""本地调试 trace 日志。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig


def trace_enabled(config: RunnableConfig | None) -> bool:
    """判断是否启用本地 trace。"""
    if config:
        configurable = config.get("configurable", {})
        if "debug_trace" in configurable and configurable["debug_trace"] not in (None, ""):
            return _parse_bool(configurable["debug_trace"])
    return _parse_bool(os.getenv("DEBUG_TRACE"))


def log_trace(event: str, payload: Any, config: RunnableConfig | None) -> None:
    """把一条事件写入本地日志。"""
    if not trace_enabled(config):
        return

    trace_dir = Path.cwd() / ".debug_runs"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / "events.jsonl"
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _parse_bool(value: Any) -> bool:
    """解析常见布尔写法。"""
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
