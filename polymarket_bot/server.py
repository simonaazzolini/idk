"""
Standalone dashboard server + bot process manager.

Run this instead of main.py:

    cd /path/to/idk
    python polymarket_bot/server.py

Dashboard: http://localhost:8765/dashboard

Use the Start / Stop / Pause / Resume buttons on the dashboard to
control the bot.  The bot is launched as a subprocess; this server
keeps running even when the bot is stopped, so the dashboard is
always reachable.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent          # repo root  (idk/)
BOT_DIR  = Path(__file__).parent                 # polymarket_bot/
sys.path.insert(0, str(BOT_DIR))

from api.server import APIServer, _scalar, _execute
from config.settings import settings as _settings
from utils.logger import setup_logging

setup_logging(_settings.log_level or "INFO")
logger = logging.getLogger(__name__)

DB_PATH  = str(BOT_DIR / "polymarket_bot.db")
MAIN_PY  = BOT_DIR / "main.py"


# ── BotProcessManager ─────────────────────────────────────────────────────────

class BotProcessManager:
    """Manages the bot as an asyncio subprocess."""

    def __init__(self):
        self._proc: asyncio.subprocess.Process | None = None
        self._start_ts: float | None = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self, mode: str = "paper", budget: float | None = None) -> dict:
        if self.is_running:
            return {"success": False, "message": f"Bot is already running (PID {self._proc.pid})"}

        b = budget or _settings.budget
        cmd = [sys.executable, str(MAIN_PY), "--mode", mode, "--budget", str(b)]
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(BOT_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._start_ts = time.time()
            asyncio.create_task(self._pipe_output(), name="bot_stdout")
            logger.info("Bot started  PID=%s  mode=%s  budget=%.2f", self._proc.pid, mode, b)
            return {"success": True, "message": f"Bot started (PID {self._proc.pid})"}
        except Exception as exc:
            logger.error("Failed to start bot: %s", exc)
            return {"success": False, "message": str(exc)}

    async def stop(self) -> dict:
        if not self.is_running:
            return {"success": False, "message": "Bot is not running"}
        try:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=12.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
            logger.info("Bot stopped")
            return {"success": True, "message": "Bot stopped"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}
        finally:
            self._proc = None
            self._start_ts = None

    async def _pipe_output(self) -> None:
        """Forward bot stdout/stderr to the server's logger."""
        if not self._proc or not self._proc.stdout:
            return
        try:
            async for raw in self._proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line:
                    logger.info("[bot] %s", line)
        except Exception:
            pass
        rc = self._proc.returncode if self._proc else None
        logger.info("Bot process exited (rc=%s)", rc)
        self._proc = None
        self._start_ts = None


# ── Background tasks ──────────────────────────────────────────────────────────

async def _poll_paper_eval(server: APIServer) -> None:
    """
    Watch bot_state['paper_eval_go'].  When the bot sets it to '1' after
    passing GO criteria, broadcast the paper_evaluation WS event so the
    dashboard auto-opens the live-trading confirmation modal.
    """
    while True:
        await asyncio.sleep(5)
        try:
            val = await _scalar(DB_PATH, "SELECT value FROM bot_state WHERE key='paper_eval_go'")
            if val == "1":
                await server.mgr.broadcast("paper_evaluation", {"go": True, "stats": {}})
                await _execute(DB_PATH,
                               "INSERT OR REPLACE INTO bot_state(key,value) VALUES('paper_eval_go','0')")
                logger.info("Broadcasted paper_evaluation GO event to dashboard")
        except Exception as exc:
            logger.debug("_poll_paper_eval: %s", exc)


async def _poll_new_events(server: APIServer) -> None:
    """
    Detect new rows in trades/paper_trades and whale_trades since last
    broadcast, and push them over WebSocket so the dashboard event feed
    updates in real time without waiting for the 15 s polling interval.
    """
    last_trade_id:  int = 0
    last_whale_id:  int = 0

    # Seed with current max IDs so we don't re-broadcast old rows on startup
    try:
        last_trade_id = int(await _scalar(
            DB_PATH, "SELECT COALESCE(MAX(id),0) FROM ("
                     "SELECT id FROM trades UNION ALL SELECT id FROM paper_trades)") or 0)
        last_whale_id = int(await _scalar(
            DB_PATH, "SELECT COALESCE(MAX(id),0) FROM whale_trades") or 0)
    except Exception:
        pass

    while True:
        await asyncio.sleep(8)
        try:
            # New trades
            rows = await _scalar(
                DB_PATH,
                "SELECT COUNT(*) FROM ("
                "  SELECT id FROM trades      WHERE id > ? "
                "  UNION ALL "
                "  SELECT id FROM paper_trades WHERE id > ?)",
                (last_trade_id, last_trade_id))
            if rows:
                new_max = int(await _scalar(
                    DB_PATH,
                    "SELECT MAX(id) FROM ("
                    "  SELECT id FROM trades UNION ALL SELECT id FROM paper_trades)"
                ) or last_trade_id)
                if new_max > last_trade_id:
                    await server.mgr.broadcast("cycle", {"new_trades": int(rows)})
                    last_trade_id = new_max

            # New whale trades
            new_whale_id = int(await _scalar(
                DB_PATH,
                "SELECT COALESCE(MAX(id),0) FROM whale_trades WHERE id > ?",
                (last_whale_id,)) or last_whale_id)
            if new_whale_id > last_whale_id:
                wt = await _scalar(
                    DB_PATH,
                    "SELECT wallet || '|' || market_slug || '|' || outcome || '|' || size_usdc"
                    " FROM whale_trades WHERE id=?",
                    (new_whale_id,))
                if wt:
                    parts = str(wt).split("|")
                    await server.mgr.broadcast("whale_alert", {
                        "wallet":      parts[0] if len(parts) > 0 else "",
                        "short_address": (parts[0][:6] + "…" + parts[0][-4:]) if len(parts[0]) > 10 else parts[0],
                        "market_slug": parts[1] if len(parts) > 1 else "",
                        "outcome":     parts[2] if len(parts) > 2 else "",
                        "size_usdc":   float(parts[3]) if len(parts) > 3 else 0,
                        "alert_level": "HIGH",
                    })
                last_whale_id = new_whale_id

        except Exception as exc:
            logger.debug("_poll_new_events: %s", exc)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    pm = BotProcessManager()

    server = APIServer(
        settings=_settings,
        db_path=DB_PATH,
        bot_ref=None,
        process_manager=pm,
    )

    await server.start()
    logger.info("Dashboard server running at http://localhost:8765/dashboard")
    logger.info("Open the dashboard and press ▶ Start to launch the bot.")

    asyncio.create_task(_poll_paper_eval(server),   name="poll_paper_eval")
    asyncio.create_task(_poll_new_events(server),   name="poll_new_events")

    # Keep the event loop alive
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Server shutting down…")
        if pm.is_running:
            await pm.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
