"""
Graph edges — the routing logic between nodes.

This is what replaces the implicit "keep looping until done"
behaviour of AgentExecutor. You now explicitly control:
  - When to continue calling tools
  - When to stop and return the final answer
  - When to stop due to hitting the max iteration limit
"""
import logging
from langchain_core.messages import AIMessage
from graph.state import AgentState
from core.config import get_settings

logger = logging.getLogger("rlm_agent")


def should_continue(state: AgentState) -> str:
    """
    Conditional edge — called after every agent_node execution.

    Returns:
        "tools"     → LLM wants to call a tool, route to tools_node
        "end"       → LLM gave a final answer (no tool calls), stop
        "end"       → Hit max iterations safety limit, force stop

    This is the core of the graph's control flow.
    In rlm-agent this was hidden inside AgentExecutor.
    Now you own it completely.
    """
    settings = get_settings()
    last_message = state["messages"][-1]
    depth = state.get("recursion_depth", 0)

    # Safety: enforce max iterations
    if depth >= settings.rlm_agent_max_iterations:
        logger.warning(f"[edges] Max iterations ({settings.rlm_agent_max_iterations}) reached, forcing end")
        return "end"

    # If the last message has tool_calls → route to tools node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(f"[edges] Routing to tools (depth={depth})")
        return "tools"

    # Otherwise the LLM gave a final answer → stop
    logger.info(f"[edges] Final answer reached (depth={depth})")
    return "end"