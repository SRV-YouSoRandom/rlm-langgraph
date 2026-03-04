"""
Graph state definition.
This is the shared 'memory' object that flows between all nodes.
Every node reads from it and writes back to it.

Think of it as a replacement for the AgentExecutor's internal state,
but now fully visible and controllable by you.
"""
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The state object passed between nodes in the graph.

    Fields:
        messages: Full conversation history including tool calls and results.
                  add_messages is a reducer — it appends new messages rather
                  than replacing the whole list. This is how LangGraph
                  accumulates the agent's reasoning across loop iterations.

        session_id: The session this run belongs to (= LangGraph thread_id).

        collection_name: Which Qdrant collection to search in.

        agent_steps: Human-readable log of tool calls made this run,
                     for the API response (matches your existing AgentStep schema).

        recursion_depth: How deep the current reasoning loop has gone.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    collection_name: str
    agent_steps: list[dict]
    recursion_depth: int