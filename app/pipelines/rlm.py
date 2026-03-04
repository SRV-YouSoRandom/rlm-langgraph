"""
RLM pipeline — now a thin wrapper over the LangGraph graph.

The heavy lifting (agent loop, tool execution, memory) is all
handled by the graph. This file just:
  1. Prepares the initial state
  2. Calls graph.invoke() or graph.stream()
  3. Formats the response to match existing API schemas

Your FastAPI routes (chat.py, stream.py) call this exactly
as they called the old rlm.py — no route changes needed.
"""
import logging
import time
import uuid
from langchain_core.messages import HumanMessage
from graph.builder import build_graph
from services.memory import get_thread_config, create_session_id
from core.config import get_settings
from core.metrics import get_metrics_collector, QueryMetrics

logger = logging.getLogger("rlm_agent")


def run_rlm(
    question: str,
    collection_name: str = None,
    session_id: str = None,
    max_depth: int = None,
) -> dict:
    """
    Run the RLM agent graph for a single question.
    Returns a dict matching the ChatResponse schema.
    """
    settings = get_settings()
    metrics_collector = get_metrics_collector()
    col = collection_name or settings.qdrant_collection

    if not session_id:
        session_id = create_session_id()

    metrics = QueryMetrics(
        session_id=session_id,
        question=question,
        collection_name=col,
    )
    start_time = time.time()

    try:
        graph = build_graph()
        thread_config = get_thread_config(session_id)

        # Initial state for this run
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "session_id": session_id,
            "collection_name": col,
            "agent_steps": [],
            "recursion_depth": 0,
        }

        # Invoke the graph — checkpointer handles memory automatically
        result = graph.invoke(initial_state, config=thread_config)

        # Extract final answer from last AI message
        answer = ""
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                answer = msg.content
                break
            # Also handle AIMessage with content but no tool_calls
            if hasattr(msg, "tool_calls") and not msg.tool_calls and msg.content:
                answer = msg.content
                break

        agent_steps = result.get("agent_steps", [])

        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.num_docs_retrieved = len(agent_steps)
        metrics.success = True
        metrics_collector.record_query(metrics)

        return {
            "answer": answer or "No answer generated.",
            "sources": [],
            "collection_name": col,
            "session_id": session_id,
            "agent_steps": agent_steps,
            "recursion_depth_reached": result.get("recursion_depth", 0),
            "pipeline_used": "langgraph",
        }

    except Exception as e:
        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.success = False
        metrics.error = str(e)
        metrics_collector.record_query(metrics)
        logger.error(f"LangGraph RLM pipeline failed: {e}", exc_info=True)
        raise


def run_rlm_streaming(
    question: str,
    collection_name: str = None,
    session_id: str = None,
):
    """
    Streaming version — yields agent step events then final answer.
    Used by the /chat/stream endpoint.
    LangGraph's .stream() gives us much richer streaming than before.
    """
    settings = get_settings()
    col = collection_name or settings.qdrant_collection

    if not session_id:
        session_id = create_session_id()

    graph = build_graph()
    thread_config = get_thread_config(session_id)

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "session_id": session_id,
        "collection_name": col,
        "agent_steps": [],
        "recursion_depth": 0,
    }

    # stream_mode="updates" yields the state delta after each node
    for event in graph.stream(initial_state, config=thread_config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if node_name == "tools":
                # Tool was executed — show what ran
                messages = node_output.get("messages", [])
                for msg in messages:
                    yield f"[TOOL RESULT]: {str(msg.content)[:200]}\n"

            elif node_name == "agent":
                # Agent step — show tool calls or final answer
                messages = node_output.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield f"[TOOL: {tc['name']}] {str(tc['args'])[:100]}\n"
                    elif msg.content:
                        yield f"\n[ANSWER]: {msg.content}"