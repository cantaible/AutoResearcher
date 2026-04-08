from __future__ import annotations

from configuration import Configuration
from graph import deep_researcher as agent_graph

def test_graph_compiles() -> None:
    assert agent_graph is not None

def test_configuration_defaults() -> None:
    config = Configuration()
    assert config.simple_model
    assert config.simple_model_max_tokens > 0

