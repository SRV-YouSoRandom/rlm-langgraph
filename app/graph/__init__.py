"""
LangGraph graph definition for the RLM agent.
Replaces the AgentExecutor loop from the original rlm-agent.
"""
from graph.builder import build_graph

__all__ = ["build_graph"]