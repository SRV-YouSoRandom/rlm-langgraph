"""
Graph builder — assembles the StateGraph.

This is where all the pieces come together:
  - Nodes (agent_node, tools_node)
  - Edges (should_continue routing)
  - Checkpointer (memory/persistence)

The compiled graph is a drop-in replacement for AgentExecutor.
Call graph.invoke() or graph.stream() exactly like you called
agent_executor.invoke() / agent_executor.stream().
"""
import logging
from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes import agent_node, tools_node_with_logging
from graph.edges import should_continue
from services.memory import get_checkpointer

logger = logging.getLogger("rlm_agent")

# Module-level compiled graph cache
_graph = None


def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Graph structure:
        START → agent_node
        agent_node → [should_continue] → tools_node  (if tool call)
        agent_node → [should_continue] → END          (if final answer)
        tools_node → agent_node                        (always loop back)

    The checkpointer makes this stateful:
        Each invoke/stream call with the same thread_id
        resumes from the last saved checkpoint.
    """
    global _graph
    if _graph is not None:
        return _graph

    logger.info("Building LangGraph StateGraph...")

    workflow = StateGraph(AgentState)

    # --- Add nodes ---
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node_with_logging)

    # --- Entry point ---
    workflow.set_entry_point("agent")

    # --- Conditional edge: after agent, decide what to do ---
    workflow.add_conditional_edges(
        "agent",          # from this node
        should_continue,  # call this function to decide
        {
            "tools": "tools",  # if returns "tools" → go to tools node
            "end": END,        # if returns "end"   → stop
        }
    )

    # --- Unconditional edge: after tools, always go back to agent ---
    workflow.add_edge("tools", "agent")

    # --- Compile with checkpointer for persistence ---
    checkpointer = get_checkpointer()
    _graph = workflow.compile(checkpointer=checkpointer)

    logger.info("LangGraph StateGraph compiled successfully")
    return _graph