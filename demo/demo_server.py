#!/usr/bin/env python3
"""
demo_server.py — Read-only demo dashboard server.

Serves the dashboard with a pre-built demo database. Write operations are disabled.
Run: python demo/demo_server.py
"""
import os
import sys
from pathlib import Path

# Setup paths
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Point memory config at the demo DB
DEMO_DB = DEMO_DIR / "knowledge.duckdb"
if not DEMO_DB.exists():
    print("Demo database not found. Run: python demo/seed_demo_db.py")
    sys.exit(1)

import memory.config as _cfg
_cfg.DB_PATH = DEMO_DB

# Import and configure the FastAPI app
from dashboard.backend.server import app
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Allow all origins for the demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if built
static_dir = PROJECT_ROOT / "dashboard" / "frontend" / "out"
if static_dir.exists():
    # Mount at root — must be last
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="demo-static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting demo server on port {port}")
    print(f"Database: {DEMO_DB}")
    print(f"Frontend: {static_dir if static_dir.exists() else 'not built'}")
    uvicorn.run(app, host="0.0.0.0", port=port)
