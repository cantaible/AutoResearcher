"""模型工厂，统一处理 OpenAI 兼容接口。"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig


def build_chat_model(
    model_name: str,
    max_tokens: int | None,
    config: RunnableConfig | None,
) -> BaseChatModel:
    """创建聊天模型实例。"""
    provider, bare_model_name = split_model_identifier(model_name)
    init_kwargs: dict[str, Any] = {"model": bare_model_name}

    if provider:
        init_kwargs["model_provider"] = provider
    if max_tokens is not None:
        init_kwargs["max_tokens"] = max_tokens

    api_key = get_api_key_for_model(model_name, config)
    if api_key:
        init_kwargs["api_key"] = api_key

    if provider == "openai":
        base_url = get_runtime_option(config, "openai_base_url", "OPENAI_BASE_URL")
        if base_url:
            init_kwargs["base_url"] = base_url

        if parse_bool(
            get_runtime_option(
                config,
                "openai_use_responses_api",
                "OPENAI_USE_RESPONSES_API",
            )
        ):
            init_kwargs["use_responses_api"] = True

        streaming_value = get_runtime_option(config, "openai_streaming", "OPENAI_STREAMING")
        if streaming_value is not None:
            init_kwargs["streaming"] = parse_bool(streaming_value)

        reasoning_effort = get_runtime_option(
            config,
            "openai_reasoning_effort",
            "OPENAI_REASONING_EFFORT",
        )
        if reasoning_effort:
            if init_kwargs.get("use_responses_api"):
                init_kwargs["reasoning"] = {"effort": reasoning_effort}
            else:
                init_kwargs["reasoning_effort"] = reasoning_effort

        extra_body = parse_json_object(
            get_runtime_option(config, "openai_extra_body_json", "OPENAI_EXTRA_BODY_JSON")
        )
        if extra_body:
            init_kwargs["extra_body"] = extra_body

        default_headers = parse_json_object(
            get_runtime_option(config, "openai_default_headers_json", "OPENAI_DEFAULT_HEADERS_JSON")
        )
        if default_headers:
            init_kwargs["default_headers"] = default_headers

    return init_chat_model(**init_kwargs)


def split_model_identifier(model_name: str) -> tuple[str | None, str]:
    """拆分类似 openai:gpt-4o-mini 的模型标识。"""
    if ":" not in model_name:
        return None, model_name
    provider, bare_model_name = model_name.split(":", 1)
    return provider, bare_model_name


def get_runtime_option(config: RunnableConfig | None, key: str, env_var: str) -> Any:
    """优先从运行配置读取，再回退到环境变量。"""
    if config:
        configurable = config.get("configurable", {})
        if key in configurable and configurable[key] not in (None, ""):
            return configurable[key]
    return os.getenv(env_var)


def parse_bool(value: Any) -> bool:
    """解析布尔值。"""
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_json_object(value: Any) -> dict[str, Any] | None:
    """把字符串解析成 JSON 对象。"""
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def get_api_key_for_model(model_name: str, config: RunnableConfig | None) -> str | None:
    """从运行配置或环境变量中解析 API Key。"""
    configurable = config.get("configurable", {}) if config else {}
    api_keys = configurable.get("apiKeys", {})
    model_name = model_name.lower()

    if model_name.startswith("openai:"):
        return api_keys.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if model_name.startswith("anthropic:"):
        return api_keys.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if model_name.startswith("google:"):
        return api_keys.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return None
