"""
Graph nodes — the actual work happens here.

Two nodes:
  1. agent_node  — calls the LLM, which decides whether to call a tool or finish
  2. tools_node  — executes whichever tool the LLM chose

This is equivalent to the AgentExecutor loop in rlm-agent,
but now each step is an explicit, inspectable node.
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from services.tools import get_all_tools
from services.repl import REPLEnvironment
from services.document_store import get_all_documents_text
from services.tools import set_repl
from graph.state import AgentState
from core.config import get_settings

logger = logging.getLogger("rlm_agent")

ROOT_SYSTEM_PROMPT = """You are an intelligent document analysis assistant. Documents have been ingested and are available for you to query.

IMPORTANT RULES:
1. ALWAYS use vector_search first — it is the fastest and most reliable tool
2. After vector_search, use grep_context to find specific terms if needed
3. Use divide_and_analyze("your question|||FULL_CONTEXT") to scan the whole document
4. ALWAYS provide a detailed answer based on what the tools return — never say documents aren't available
5. Cite page numbers or sections when the text includes them

TOOL USAGE:
- vector_search("your query") → best first step for any question
- grep_context("keyword") → find specific terms quickly
- divide_and_analyze("question|||FULL_CONTEXT") → broad analysis of whole document
- sub_llm_analyze("question|||some text you found") → deep analysis of a specific passage
- repl_execute("print(...)") → custom Python exploration of document text

You have access to the full document text. Always answer the question using the document content."""


def _get_llm_with_tools():
    """Build the LLM instance bound to all tools."""
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=settings.rlm_max_tokens_per_call,
    )
    tools = get_all_tools()
    return llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """
    The agent node — calls the LLM with the current message history.

    The LLM sees all previous messages (including tool results) and decides:
      - Call another tool → returns an AIMessage with tool_calls
      - Give a final answer → returns an AIMessage with plain content

    This replaces the inner reasoning step of AgentExecutor.
    """
    logger.info(f"[agent_node] messages in state: {len(state['messages'])}")

    # Initialize REPL with document context on first call
    # (subsequent calls reuse the same state, REPL is already set)
    if state["recursion_depth"] == 0:
        doc_text = get_all_documents_text(state.get("collection_name"))
        repl = REPLEnvironment(document_text=doc_text)
        repl.set_variable("collection_name", state.get("collection_name", ""))
        set_repl(repl)
        logger.info(f"[agent_node] REPL initialized with {len(doc_text)} chars")

    llm_with_tools = _get_llm_with_tools()

    # Prepend system message to give the LLM its instructions
    messages_with_system = [SystemMessage(content=ROOT_SYSTEM_PROMPT)] + state["messages"]

    response = llm_with_tools.invoke(messages_with_system)
    logger.info(f"[agent_node] LLM responded, tool_calls: {len(response.tool_calls) if hasattr(response, 'tool_calls') else 0}")

    return {
        "messages": [response],
        "recursion_depth": state["recursion_depth"] + 1,
    }


def tools_node_with_logging(state: AgentState) -> dict:
    """
    The tools node — executes the tool(s) the LLM chose.

    Wraps LangGraph's built-in ToolNode to also log agent_steps
    so the API response includes visible tool call history
    (matching your existing AgentStep schema).
    """
    tools = get_all_tools()
    tool_node = ToolNode(tools)

    # Get the last AI message (which contains tool_calls)
    last_message = state["messages"][-1]
    current_steps = state.get("agent_steps", [])

    # Log each tool call for the response
    new_steps = list(current_steps)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tc in last_message.tool_calls:
            new_steps.append({
                "step_number": len(new_steps) + 1,
                "tool_used": tc["name"],
                "input_summary": str(tc["args"])[:150],
                "output_summary": "pending...",
                "recursion_depth": state["recursion_depth"],
            })

    # Execute the tools via LangGraph's ToolNode
    result = tool_node.invoke(state)

    # Update output_summary now that we have results
    tool_messages = result.get("messages", [])
    for i, tm in enumerate(tool_messages):
        idx = len(current_steps) + i
        if idx < len(new_steps):
            new_steps[idx]["output_summary"] = str(tm.content)[:150]

    logger.info(f"[tools_node] executed {len(tool_messages)} tool(s)")

    return {
        "messages": result.get("messages", []),
        "agent_steps": new_steps,
    }