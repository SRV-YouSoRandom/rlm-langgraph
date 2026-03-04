# RLM-LangGraph Agent

A production-grade **Recursive Language Model (RLM)** system rebuilt on **LangGraph v1.x** — evolving the original [rlm-agent](https://github.com/SRV-YouSoRandom/rlm-agent) from a black-box `AgentExecutor` loop into a fully explicit, stateful, resumable graph.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.7-blueviolet)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)

> **Evolved from [rlm-agent](https://github.com/SRV-YouSoRandom/rlm-agent)** — the RLM core is unchanged. The orchestration layer is completely replaced: `AgentExecutor` → `StateGraph`. All tools, services, and infrastructure carry forward.

---

## What's New vs rlm-agent

| Feature          | rlm-agent                        | rlm-langgraph                                |
| ---------------- | -------------------------------- | -------------------------------------------- |
| Orchestration    | `AgentExecutor` (black box)      | `StateGraph` (explicit nodes + edges)        |
| Memory           | `ConversationBufferWindowMemory` | LangGraph `MemorySaver` checkpointer         |
| State visibility | Hidden inside executor           | Full `AgentState` TypedDict, inspectable     |
| Loop control     | `max_iterations` config          | `should_continue` edge you own               |
| Crash recovery   | Work is lost                     | Resumes from last checkpoint                 |
| Streaming        | Custom event parsing             | Native `graph.stream()` with per-node deltas |
| Debug endpoint   | None                             | `GET /graph/structure`                       |

---

## 🚀 Features

### Core RLM Pipeline (Unchanged)

- **Sandboxed REPL** — LLM writes Python to explore document text stored as a variable
- **Recursive Sub-LLM Calls** — Root LLM delegates focused analysis to sub-LLMs on text snippets
- **Divide & Conquer** — Splits large documents into segments and queries each in parallel
- **Keyword Grep** — Fast line-level keyword search across all ingested documents
- **Vector Search Tool** — Semantic retrieval as one of many tools the agent can choose

### LangGraph Additions

- **Explicit Graph** — Agent loop is now a visible `StateGraph` you control completely
- **Checkpointed State** — Every node execution is saved; crash mid-run and resume exactly where you left off
- **Stateful Sessions** — `thread_id = session_id` — LangGraph manages conversation history natively
- **Per-Node Streaming** — Stream tool calls, tool results, and the final answer node-by-node
- **Human-in-the-Loop Ready** — Add `interrupt_before` on any node for approval workflows

---

## Architecture

```
User Request (FastAPI)
        │
        ▼
pipelines/rlm.py  (thin wrapper)
        │
        ▼
graph/builder.py  ──── assembles ────►  StateGraph
                                              │
                              ┌───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │       START        │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │    agent node      │  ← LLM decides: call tool or finish
                    │  (graph/nodes.py)  │    same LLM + prompt + tools as before
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  should_continue   │  ← conditional edge you own
                    │  (graph/edges.py)  │
                    └──────┬─────┬───────┘
                     tools?│     │done?
                           │     │
               ┌───────────▼┐    ▼
               │ tools node │   END
               │(nodes.py)  │
               │            │  executes the chosen tool from
               │vector_search│  services/tools.py (zero changes)
               │grep_context │
               │repl_execute │
               │sub_llm      │
               │divide_analyze│
               └──────┬──────┘
                      │
                      └──────────► (loops back to agent node)

Checkpointing (NEW):
  Every node execution → saved to MemorySaver (upgradeable to SQLite/Postgres)
  thread_id = session_id → state is isolated and persistent per user session
  Crash mid-run → graph resumes from the last completed node automatically
```

---

## Project Structure

```
rlm-langgraph/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ingest.py            # ← COPIED from rlm-agent
│   │   │   ├── chat.py              # ← MODIFIED (uses new memory)
│   │   │   ├── stream.py            # ← MODIFIED (uses graph.stream)
│   │   │   ├── collections.py       # ← COPIED from rlm-agent
│   │   │   ├── sessions.py          # ← MODIFIED (uses checkpointer)
│   │   │   └── metrics.py           # ← COPIED from rlm-agent
│   │   ├── __init__.py
│   │   └── dependencies.py          # ← COPIED from rlm-agent
│   ├── core/
│   │   ├── config.py                # ← MODIFIED (+ checkpointer path)
│   │   ├── logging.py               # ← COPIED from rlm-agent
│   │   ├── metrics.py               # ← COPIED from rlm-agent
│   │   └── tracing.py               # ← COPIED from rlm-agent
│   ├── graph/                       # ← NEW — replaces AgentExecutor
│   │   ├── __init__.py
│   │   ├── state.py                 # AgentState TypedDict
│   │   ├── nodes.py                 # agent_node + tools_node
│   │   ├── edges.py                 # should_continue routing logic
│   │   └── builder.py               # StateGraph assembly + compile
│   ├── pipelines/
│   │   ├── ingest.py                # ← COPIED from rlm-agent
│   │   └── rlm.py                   # ← NEW (thin wrapper over graph)
│   ├── services/
│   │   ├── memory.py                # ← NEW (LangGraph checkpointer)
│   │   ├── tools.py                 # ← COPIED from rlm-agent (zero changes)
│   │   ├── repl.py                  # ← COPIED from rlm-agent
│   │   ├── sub_llm.py               # ← COPIED from rlm-agent
│   │   ├── vector_store.py          # ← COPIED from rlm-agent
│   │   ├── document_store.py        # ← COPIED from rlm-agent
│   │   ├── embedder.py              # ← COPIED from rlm-agent
│   │   ├── parser.py                # ← COPIED from rlm-agent
│   │   ├── chunker.py               # ← COPIED from rlm-agent
│   │   └── hasher.py                # ← COPIED from rlm-agent
│   ├── schemas/                     # ← COPIED (chat.py minor update)
│   ├── main.py                      # ← MODIFIED (pre-builds graph at startup)
│   ├── Dockerfile
│   └── requirements.txt
├── document_storage/                # Raw full-text store (mounted volume)
├── qdrant_storage/                  # Persisted vector data (mounted volume)
├── checkpoints/                     # ← NEW — LangGraph checkpoint storage
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Tech Stack

| Component               | Technology                               | Purpose                                 |
| ----------------------- | ---------------------------------------- | --------------------------------------- |
| **Framework**           | FastAPI (Python 3.11)                    | High-performance async API              |
| **Graph Orchestration** | LangGraph 1.0.7                          | Stateful agent graph with checkpointing |
| **LLM Tooling**         | LangChain 1.2.0                          | Tool definitions, prompts, LLM bindings |
| **Vector Database**     | Qdrant                                   | Persistent vector storage               |
| **LLM Provider**        | OpenRouter                               | Access to hosted LLMs                   |
| **Embeddings**          | sentence-transformers (all-MiniLM-L6-v2) | Local embedding generation              |
| **REPL Sandbox**        | RestrictedPython                         | Safe code execution environment         |
| **Checkpointer**        | LangGraph MemorySaver → SqliteSaver      | Graph state persistence                 |
| **Containerization**    | Docker & Docker Compose                  | Reproducible deployment                 |
| **Observability**       | Custom Metrics + LangSmith (optional)    | Performance monitoring                  |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenRouter API Key ([get one free](https://openrouter.ai))

> ⚠️ **Model Note:** The agent requires a model that supports tool/function calling. Not all OpenRouter models do — check before configuring `LLM_MODEL`.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SRV-YouSoRandom/rlm-langgraph.git
cd rlm-langgraph

# 2. Configure environment
cp .env.example .env
nano .env   # Add your OPENROUTER_API_KEY

# 3. Add your user to the docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# 4. Create required directories
mkdir -p document_storage qdrant_storage checkpoints

# 5. Build and run
docker compose up --build -d

# 6. Verify
curl http://localhost:8000/health
# {"status":"ok","version":"2.0.0","mode":"langgraph"}

# 7. Inspect graph structure
curl http://localhost:8000/graph/structure
```

The API is available at `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

---

## API Usage

### 1. Ingest Documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@document.pdf" \
  -F "collection_name=my_docs"
```

**Response:**

```json
{
  "filename": "document.pdf",
  "doc_hash": "a3f5c2...",
  "total_chunks": 42,
  "new_chunks_indexed": 42,
  "collection_name": "my_docs",
  "raw_text_stored": true,
  "raw_text_length": 84321,
  "message": "Document ingested successfully."
}
```

### 2. Chat — LangGraph RLM Mode

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarise all financial figures mentioned in the document",
    "collection_name": "my_docs",
    "use_rlm": true
  }'
```

**Response:**

```json
{
  "answer": "Based on my analysis of the document...",
  "sources": [],
  "collection_name": "my_docs",
  "session_id": "a1b2c3d4-...",
  "agent_steps": [
    {
      "step_number": 1,
      "tool_used": "grep_context",
      "input_summary": "revenue",
      "output_summary": "Found 6 matches: Q1 revenue $2.1M...",
      "recursion_depth": 0
    },
    {
      "step_number": 2,
      "tool_used": "sub_llm_analyze",
      "input_summary": "List all financial figures|||Q1 revenue...",
      "output_summary": "Q1: $2.1M, Q2: $3.4M, Q3: $4.2M...",
      "recursion_depth": 1
    }
  ],
  "recursion_depth_reached": 2,
  "pipeline_used": "langgraph"
}
```

### 3. Chat — Vector Fallback Mode

For simple questions where speed matters more than deep reasoning.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the document about?", "use_rlm": false}'
```

### 4. Streaming Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key risks?", "session_id": "a1b2c3d4-..."}' \
  --no-buffer
```

Streams per-node output:

```
[TOOL: grep_context] risk
[TOOL RESULT]: Found 4 matches: "key risks include..."
[TOOL: sub_llm_analyze] List all risks|||key risks include...
[TOOL RESULT]: 1. Market risk  2. Operational risk...
[ANSWER]: The document identifies three key risks...
```

### 5. Multi-Turn Conversations

LangGraph manages conversation history automatically via `thread_id`. Just pass back the `session_id`:

```bash
# First turn
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Q3 revenue?"}'

# Follow-up — graph resumes the same thread
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does that compare to Q2?", "session_id": "<returned_session_id>"}'
```

### 6. Collection Management

```bash
curl -X POST http://localhost:8000/api/v1/collections -H "Content-Type: application/json" -d '{"name": "project_alpha"}'
curl http://localhost:8000/api/v1/collections
curl http://localhost:8000/api/v1/collections/project_alpha
curl -X DELETE http://localhost:8000/api/v1/collections/project_alpha
```

### 7. Session Management

```bash
curl http://localhost:8000/api/v1/sessions
curl http://localhost:8000/api/v1/sessions/{session_id}/history
curl -X DELETE http://localhost:8000/api/v1/sessions/{session_id}
```

### 8. Metrics

```bash
curl http://localhost:8000/api/v1/metrics/summary
curl http://localhost:8000/api/v1/metrics/recent?limit=20
```

### 9. Graph Debug (New)

```bash
curl http://localhost:8000/graph/structure
```

---

## Configuration

```env
# LLM
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=qwen/qwen3-vl-30b-a3b-thinking

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=rlm_docs

# App
APP_ENV=production
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5

# RLM Settings
RLM_MAX_RECURSION_DEPTH=5
RLM_REPL_TIMEOUT_SECONDS=10
RLM_MAX_TOKENS_PER_CALL=1024
RLM_SNIPPET_SIZE=2000
RLM_SANDBOX_MODE=true
RLM_AGENT_MAX_ITERATIONS=10
RLM_FALLBACK_TO_VECTOR=true
DOCUMENT_STORAGE_PATH=/app/document_storage

# LangGraph Checkpointer
CHECKPOINTER_DB_PATH=/app/checkpoints/graph.db

# Observability (Optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=rlm-langgraph
```

---

## RLM Agent Tools

The root LLM chooses from five tools at each reasoning step — identical to rlm-agent:

```
vector_search       → Fast semantic chunk retrieval from Qdrant
grep_context        → Line-level keyword search across all document text
repl_execute        → Execute Python against full document text in sandbox
sub_llm_analyze     → Focused sub-LLM call on a specific snippet
divide_and_analyze  → Split document into N segments, query each in parallel
```

---

## RLM vs Vector Fallback

| Scenario              | Recommended      | Reason                             |
| --------------------- | ---------------- | ---------------------------------- |
| Simple factual Q&A    | `use_rlm: false` | Faster, single LLM call (~1–2s)    |
| Multi-step reasoning  | `use_rlm: true`  | Agent breaks down the problem      |
| Very long documents   | `use_rlm: true`  | Divide-and-conquer scales better   |
| Keyword/number lookup | `use_rlm: true`  | Grep + sub-LLM more precise        |
| Low latency required  | `use_rlm: false` | ~1–2s vs 5–30s                     |
| Exploratory analysis  | `use_rlm: true`  | REPL allows open-ended exploration |

---

## Performance Benchmarks

Tested on modest hardware (4 vCPU, 8GB RAM):

| Operation                    | Latency | Notes                             |
| ---------------------------- | ------- | --------------------------------- |
| PDF Ingestion (10 pages)     | ~2–3s   | Chunks + embeds + stores raw text |
| Vector Fallback Query        | ~1–2s   | Single LLM call                   |
| RLM Simple Query (2–3 steps) | ~5–10s  | grep + sub-LLM                    |
| RLM Complex Query (5+ steps) | ~15–30s | divide-and-analyze + recursion    |
| Max depth query (depth 5)    | ~45–60s | Full recursive chain              |

---

## Docker Commands

```bash
# View logs
docker compose logs -f app

# Rebuild after code changes
docker compose up --build -d

# Stop services
docker compose down

# Shell into container
docker exec -it rlm-langgraph-app bash

# View Qdrant dashboard
# Open http://localhost:6333/dashboard

# Nuclear reset (deletes all data including checkpoints)
docker compose down -v && rm -rf qdrant_storage/ document_storage/ checkpoints/
```

---

## Production Upgrade Path

### Persistent Checkpoints — SqliteSaver

Swap `MemorySaver` for `SqliteSaver` in `services/memory.py` to persist graph state across container restarts:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

def get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        conn = sqlite3.connect(settings.checkpointer_db_path, check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    return _checkpointer
```

### Multi-Worker — PostgresSaver

Add `langgraph-checkpoint-postgres` to `requirements.txt` and update `services/memory.py`:

```python
from langgraph.checkpoint.postgres import PostgresSaver

def get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = PostgresSaver.from_conn_string(
            "postgresql://user:pass@localhost/rlm_checkpoints"
        )
    return _checkpointer
```

Then update `docker-compose.yml` to add a `postgres` service and change `--workers 1` to `--workers 4` in the Dockerfile.

### Human-in-the-Loop

Add `interrupt_before` in `graph/builder.py` to pause before any tool execution for human review:

```python
_graph = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"]   # pause before every tool call
)
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

query_counter = Counter('rlm_queries_total', 'Total queries')
query_latency = Histogram('rlm_query_latency_seconds', 'Query latency')
```

---

## Key Design Decisions

| Decision                          | Reasoning                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------ |
| **StateGraph over AgentExecutor** | Explicit, inspectable, resumable — you own the loop                            |
| **MemorySaver default**           | Zero-dependency start; swap to SQLite/Postgres when ready                      |
| **thread_id = session_id**        | LangGraph handles conversation memory natively per thread                      |
| **graph/ folder**                 | Clean separation — graph wiring never touches business logic                   |
| **tools.py unchanged**            | `@tool` decorated functions are framework-agnostic                             |
| **workers = 1**                   | MemorySaver is not multi-process safe; upgrade to SqliteSaver for multi-worker |
| **Graph built at startup**        | Compiled and cached once at `lifespan`, not per request                        |

---

## Future Enhancements

- [ ] **SqliteSaver** — Swap MemorySaver for persistence across restarts
- [ ] **Human-in-the-Loop** — Pause graph at tool node for approval
- [ ] **Async Sub-LLM Calls** — Parallelize `divide_and_analyze` with `asyncio.gather`
- [ ] **Source Attribution** — Extract document/page citations from agent steps
- [ ] **Cost Tracking** — Token counting per query across all recursive LLM calls
- [ ] **Prometheus Metrics** — Industry-standard observability with Grafana dashboards
- [ ] **Rate Limiting** — API throttling with SlowAPI
- [ ] **Authentication** — JWT-based user authentication
- [ ] **LangGraph Studio** — Visual graph debugging and prototyping

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **MIT CSAIL** — Original RLM research (Alex Zhang & Omar Khattab)
- **LangChain / LangGraph** — Graph orchestration framework
- **Qdrant** — High-performance vector database
- **sentence-transformers** — State-of-the-art embeddings
- **FastAPI** — Modern Python web framework
- **OpenRouter** — Unified LLM API access
- **[rlm-agent](https://github.com/SRV-YouSoRandom/rlm-agent)** — The foundation this project evolves from

---

**Built with ❤️ for the AI/ML community**
