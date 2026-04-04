from __future__ import annotations

from configuration import Configuration
from graph import agent_graph


def test_graph_compiles() -> None:
    assert agent_graph is not None


def test_configuration_defaults() -> None:
    config = Configuration()
    assert config.agent_model
    assert config.agent_model_max_tokens > 0
