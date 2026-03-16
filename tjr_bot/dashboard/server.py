"""
server.py — TJR Bot Dashboard FastAPI + WebSocket Server
Serves the frontend, polls MT5 every 500ms, and broadcasts via WebSocket.
Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
 or: python server.py
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Set

import aiofiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from mt5_bridge import MT5Bridge

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
bridge = MT5Bridge()
connected_clients: Set[WebSocket] = set()
POLL_INTERVAL = 0.5  # seconds — broadcast every 500 ms

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background polling task on startup."""
    logger.info("TJR Dashboard starting...")
    bridge.connect()
    task = asyncio.create_task(broadcast_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    bridge.disconnect()
    logger.info("TJR Dashboard stopped.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TJR Bot Dashboard",
    description="Real-time trading dashboard for the TJR strategy EA",
    version="3.0.0",
    lifespan=lifespan,
)

# Static file serving for frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ---------------------------------------------------------------------------
# Background broadcast loop
# ---------------------------------------------------------------------------

async def broadcast_loop():
    """Poll MT5 every 500ms and broadcast to all connected WebSocket clients."""
    while True:
        try:
            await asyncio.sleep(POLL_INTERVAL)
            if not connected_clients:
                continue

            data = await asyncio.get_event_loop().run_in_executor(
                None, bridge.get_live_data
            )
            payload = json.dumps(data, default=str)

            dead: Set[WebSocket] = set()
            for ws in connected_clients.copy():
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            connected_clients -= dead

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Broadcast loop error: %s", exc)
            await asyncio.sleep(1)

# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard index.html."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        async with aiofiles.open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=await f.read())
    return HTMLResponse(
        content="<h1>TJR Dashboard</h1><p>Frontend not found. "
                "Ensure dashboard/frontend/index.html exists.</p>",
        status_code=404,
    )


@app.get("/api/live")
async def api_live():
    """Return the current live data snapshot."""
    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, bridge.get_live_data
        )
        return JSONResponse(content=data)
    except Exception as exc:
        logger.exception("api_live error: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/history")
async def api_history():
    """Return full trade history with statistics."""
    try:
        trades = await asyncio.get_event_loop().run_in_executor(
            None, bridge.get_trade_history
        )
        stats = await asyncio.get_event_loop().run_in_executor(
            None, bridge.get_statistics
        )
        return JSONResponse(content={"trades": trades, "statistics": stats})
    except Exception as exc:
        logger.exception("api_history error: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/account")
async def api_account():
    """Return account balance, equity, and margin info."""
    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, bridge.get_account_info
        )
        return JSONResponse(content=data)
    except Exception as exc:
        logger.exception("api_account error: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/positions")
async def api_positions():
    """Return currently open MT5 positions."""
    try:
        positions = await asyncio.get_event_loop().run_in_executor(
            None, bridge.get_positions
        )
        return JSONResponse(content={"positions": positions})
    except Exception as exc:
        logger.exception("api_positions error: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/statistics")
async def api_statistics():
    """Return aggregated strategy statistics."""
    try:
        stats = await asyncio.get_event_loop().run_in_executor(
            None, bridge.get_statistics
        )
        return JSONResponse(content=stats)
    except Exception as exc:
        logger.exception("api_statistics error: %s", exc)
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket feed — pushes live data every 500ms."""
    await websocket.accept()
    connected_clients.add(websocket)
    client = websocket.client
    logger.info("WebSocket client connected: %s", client)

    try:
        # Send an immediate snapshot on connect
        data = await asyncio.get_event_loop().run_in_executor(
            None, bridge.get_live_data
        )
        await websocket.send_text(json.dumps(data, default=str))

        # Keep alive — read any client messages (ping/pong or config)
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", client)
    except Exception as exc:
        logger.debug("WebSocket error: %s", exc)
    finally:
        connected_clients.discard(websocket)


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
