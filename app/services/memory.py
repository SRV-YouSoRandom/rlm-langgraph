"""
Session memory using LangGraph's checkpointer system.
Replaces ConversationBufferWindowMemory from the old rlm-agent.

The checkpointer saves the entire graph state after every node execution.
Thread ID = session_id, so each user session has isolated, persistent memory.

LangGraph v1.x checkpointer packages:
  - MemorySaver        → from langgraph.checkpoint.memory (in langgraph-checkpoint)
  - SqliteSaver        → from langgraph.checkpoint.sqlite (in langgraph-checkpoint-sqlite)
  - AsyncSqliteSaver   → from langgraph.checkpoint.sqlite.aio
  - PostgresSaver      → from langgraph.checkpoint.postgres (in langgraph-checkpoint-postgres)

NOTE: Since v0.2, SQLite and Postgres checkpointers are SEPARATE packages.
      They are NOT bundled inside langgraph itself anymore.
      We install them via langgraph-checkpoint-sqlite / langgraph-checkpoint-postgres.
"""
import uuid
import logging
from typing import Optional
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("rlm_agent")

# Global checkpointer instance — shared across all requests
_checkpointer: Optional[MemorySaver] = None


def get_checkpointer() -> MemorySaver:
    """
    Return the global checkpointer instance.
    Default: MemorySaver (in-memory, lost on restart).
    For persistence: swap to SqliteSaver (see upgrade path at bottom of guide).
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
        logger.info("Initialized LangGraph MemorySaver checkpointer")
    return _checkpointer


def create_session_id() -> str:
    """Generate a new session ID (= LangGraph thread_id)."""
    return str(uuid.uuid4())


def get_thread_config(session_id: str) -> dict:
    """
    Return the LangGraph config dict for a given session.
    This is passed to graph.invoke() / graph.stream() as the 'config' argument.
    LangGraph uses thread_id to isolate and persist state per session.
    """
    return {"configurable": {"thread_id": session_id}}


def get_session_history(session_id: str) -> list[dict]:
    """
    Retrieve conversation history from checkpointer for a session.
    Returns list of {role, content} dicts for API compatibility.
    """
    checkpointer = get_checkpointer()
    config = get_thread_config(session_id)
    
    try:
        checkpoint = checkpointer.get(config)
        if not checkpoint:
            return []
        
        messages = checkpoint.get("channel_values", {}).get("messages", [])
        history = []
        for msg in messages:
            if hasattr(msg, "type"):
                role = "user" if msg.type == "human" else "assistant"
                history.append({"role": role, "content": msg.content})
        return history
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        return []


def list_active_sessions() -> list[str]:
    """
    List all thread IDs that have checkpoints saved.
    MemorySaver stores these in-memory; SqliteSaver would persist across restarts.
    """
    checkpointer = get_checkpointer()
    try:
        # MemorySaver exposes internal storage
        if hasattr(checkpointer, "storage"):
            return list(checkpointer.storage.keys())
        return []
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return []


def clear_session(session_id: str):
    """Clear all checkpoints for a session thread."""
    checkpointer = get_checkpointer()
    try:
        if hasattr(checkpointer, "storage") and session_id in checkpointer.storage:
            del checkpointer.storage[session_id]
            logger.info(f"Cleared session: {session_id}")
    except Exception as e:
        logger.error(f"Failed to clear session {session_id}: {e}")