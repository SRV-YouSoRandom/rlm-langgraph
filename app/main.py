from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.logging import setup_logging
from core.tracing import setup_langsmith_tracing
from api.routes import ingest, chat, stream, collections, sessions, metrics
from services.vector_store import ensure_collection
from graph.builder import build_graph
import logging

setup_logging()
setup_langsmith_tracing()
logger = logging.getLogger("rlm_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RLM-LangGraph Agent v2.0.0...")
    ensure_collection()
    # Pre-build and cache the graph at startup
    build_graph()
    logger.info("LangGraph StateGraph ready.")
    yield
    logger.info("Shutting down RLM-LangGraph Agent.")


app = FastAPI(
    title="RLM-LangGraph Agent",
    description="RLM Agent powered by LangGraph: Stateful graph | Checkpointing | REPL | Sub-LLM | Vector Search",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(stream.router, prefix="/api/v1", tags=["Streaming"])
app.include_router(collections.router, prefix="/api/v1", tags=["Collections"])
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "mode": "langgraph"}


@app.get("/graph/structure")
def graph_structure():
    """Returns the graph node/edge structure for debugging."""
    graph = build_graph()
    return {
        "nodes": list(graph.nodes.keys()) if hasattr(graph, "nodes") else [],
        "description": "agent → [should_continue] → tools → agent (loop) | END"
    }