"""
server.py — FastAPI backend for the memory dashboard.

Reads/writes the existing DuckDB knowledge base via the memory package.
Serves the Next.js static export from /static when available.
"""
from __future__ import annotations

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Ensure memory package is importable ────────────────────────────────
MEMORY_PKG = Path.home() / ".claude" / "memory"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if MEMORY_PKG.exists():
    sys.path.insert(0, str(MEMORY_PKG.parent))
# Also support running from the repo directly
sys.path.insert(0, str(REPO_ROOT))

from memory import db as memdb  # noqa: E402

# ── Write lock (DuckDB is single-writer) ──────────────────────────────
write_lock = threading.Lock()


def get_read_conn():
    return memdb.get_connection(read_only=True)


def get_write_conn():
    return memdb.get_connection(read_only=False)


# ── App ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify DB is accessible
    try:
        conn = get_read_conn()
        conn.close()
    except Exception as e:
        print(f"[dashboard] Warning: DB not accessible: {e}", file=sys.stderr)
    yield


app = FastAPI(
    title="Memory Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:9222", "http://127.0.0.1:9222"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ───────────────────────────────────────────────────
from .routes import stats, facts, decisions, entities, relationships  # noqa: E402
from .routes import guardrails, procedures, error_solutions  # noqa: E402
from .routes import observations, sessions, scopes, search  # noqa: E402
from .routes import ideas, questions  # noqa: E402
from .routes import code_graph  # noqa: E402
from .routes import knowledge_graph, chat  # noqa: E402

app.include_router(stats.router, prefix="/api/v1")
app.include_router(facts.router, prefix="/api/v1")
app.include_router(decisions.router, prefix="/api/v1")
app.include_router(entities.router, prefix="/api/v1")
app.include_router(relationships.router, prefix="/api/v1")
app.include_router(guardrails.router, prefix="/api/v1")
app.include_router(procedures.router, prefix="/api/v1")
app.include_router(error_solutions.router, prefix="/api/v1")
app.include_router(observations.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(scopes.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(ideas.router, prefix="/api/v1")
app.include_router(questions.router, prefix="/api/v1")
app.include_router(code_graph.router, prefix="/api/v1")
app.include_router(knowledge_graph.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")

# ── Serve static frontend (if built) ─────────────────────────────────
static_dir = Path(__file__).parent.parent / "frontend" / "out"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


@app.get("/api/health")
def health():
    return {"status": "ok"}
