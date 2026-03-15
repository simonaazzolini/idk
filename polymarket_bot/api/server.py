"""
FastAPI web dashboard backend for the Polymarket bot.
Reads from polymarket_bot.db and exposes REST + WebSocket endpoints.
Started automatically as a background asyncio task from bot.py.
Port: 8765
"""
import asyncio
import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Set

import aiosqlite
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_float(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def _safe_int(v, default=0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def _short_addr(addr: str) -> str:
    if not addr or len(addr) < 10:
        return addr or ""
    return f"{addr[:6]}...{addr[-4:]}"

async def _query(db_path: str, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT and return list of dicts. Never raises."""
    try:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
    except Exception as e:
        logger.debug("DB query failed: %s | %s", sql[:60], e)
        return []

async def _scalar(db_path: str, sql: str, params: tuple = (), default=None):
    """Execute a scalar SELECT. Never raises."""
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(sql, params) as cur:
                row = await cur.fetchone()
                return row[0] if row else default
    except Exception:
        return default

async def _execute(db_path: str, sql: str, params: tuple = ()) -> bool:
    """Execute a non-SELECT statement. Returns True on success."""
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(sql, params)
            await db.commit()
        return True
    except Exception as e:
        logger.warning("DB execute failed: %s | %s", sql[:60], e)
        return False


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, event_type: str, data: Any) -> None:
        if not self.active:
            return
        msg = json.dumps({"type": event_type, "data": data, "ts": _now_iso()})
        dead: Set[WebSocket] = set()
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.active -= dead


# ── API Server class ──────────────────────────────────────────────────────────

class APIServer:
    """
    Runs a FastAPI + uvicorn server in the background.
    Provides all REST and WebSocket endpoints for the dashboard.
    """

    def __init__(self, settings, db_path: str = "polymarket_bot.db", bot_ref=None, process_manager=None):
        self.settings = settings
        self.db_path = db_path
        self.bot_ref = bot_ref
        self.process_manager = process_manager  # BotProcessManager when running standalone
        self.mgr = ConnectionManager()
        self._start_ts: float = time.time()
        self._server_task: Optional[asyncio.Task] = None
        self._server: Optional[uvicorn.Server] = None

        self.app = FastAPI(title="Polymarket Bot Dashboard API", version="1.0")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._register_routes()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start uvicorn as a background asyncio task."""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=8765,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(
            self._server.serve(), name="api_server"
        )
        # Give uvicorn a moment to bind
        await asyncio.sleep(0.5)
        logger.info("Dashboard available at: http://localhost:8765/dashboard")
        logger.info("Or open dashboard.html directly in your browser")

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()

    # ── Broadcast helpers (called by bot modules) ─────────────────────────────

    async def broadcast_trade(self, trade: dict) -> None:
        await self.mgr.broadcast("trade", trade)

    async def broadcast_whale_alert(self, alert) -> None:
        data = alert.to_dict() if hasattr(alert, "to_dict") else alert
        await self.mgr.broadcast("whale_alert", data)

    async def broadcast_arb(self, arb) -> None:
        data = {
            "arb_type": arb.arb_type,
            "market_slug": arb.market_slug,
            "profit_pct": arb.profit_pct,
            "profit_usdc": arb.profit_usdc,
            "description": arb.description,
        }
        await self.mgr.broadcast("arb", data)

    async def broadcast_stop_loss(self, position: dict) -> None:
        await self.mgr.broadcast("stop_loss", position)

    async def broadcast_cycle_complete(self, stats: dict) -> None:
        await self.mgr.broadcast("cycle", stats)

    async def broadcast_mode_change(self, old_mode: str, new_mode: str) -> None:
        await self.mgr.broadcast("mode_change", {"old_mode": old_mode, "new_mode": new_mode})

    async def broadcast_price_shock(self, market: str, token: str, pct: float) -> None:
        await self.mgr.broadcast("price_shock", {"market": market, "token": token, "pct_change": pct})

    async def broadcast_paper_evaluation(self, go: bool, stats: dict) -> None:
        await self.mgr.broadcast("paper_evaluation", {"go": go, "stats": stats})

    # ── Route registration ────────────────────────────────────────────────────

    def _register_routes(self) -> None:
        app = self.app
        db = self.db_path
        mgr = self.mgr
        server_self = self

        # ── Dashboard HTML ────────────────────────────────────────────────────

        @app.get("/dashboard", response_class=HTMLResponse)
        async def serve_dashboard():
            html_path = Path(__file__).parent.parent / "dashboard.html"
            if html_path.exists():
                return HTMLResponse(html_path.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>dashboard.html not found. Open it directly.</h1>")

        # ── Health ────────────────────────────────────────────────────────────

        @app.get("/api/health")
        async def health():
            try:
                await _scalar(db, "SELECT 1")
                db_ok = True
            except Exception:
                db_ok = False
            return {"status": "ok", "db_connected": db_ok, "timestamp": _now_iso()}

        # ── Status ────────────────────────────────────────────────────────────

        @app.get("/api/status")
        async def status():
            bot = server_self.bot_ref
            mode = "PAPER"
            if bot:
                mode = getattr(bot, "mode", "PAPER")

            # From DB state
            stored_mode = await _scalar(db, "SELECT value FROM bot_state WHERE key='mode'") or mode
            paper_start = _safe_float(await _scalar(db, "SELECT value FROM bot_state WHERE key='paper_start_ts'"))
            live_at = _safe_float(await _scalar(db, "SELECT value FROM bot_state WHERE key='live_activated_at'"))
            last_shutdown = _safe_float(await _scalar(db, "SELECT value FROM bot_state WHERE key='last_shutdown_ts'"))

            now_ts = time.time()
            paper_hours = 48.0
            paper_elapsed = (now_ts - paper_start) / 3600.0 if paper_start else 0
            paper_remaining = max(0.0, paper_hours - paper_elapsed)
            paper_progress = min(100.0, (paper_elapsed / paper_hours) * 100)

            runtime = int(now_ts - server_self._start_ts)

            # Cycle info
            cycles = _safe_int(await _scalar(db, "SELECT COUNT(*) FROM signals"))
            last_signal_ts = _safe_float(await _scalar(db, "SELECT MAX(timestamp) FROM signals"))
            last_cycle_iso = datetime.fromtimestamp(last_signal_ts, tz=timezone.utc).isoformat() if last_signal_ts else _now_iso()

            cycle_mins = server_self.settings.cycle_minutes if server_self.settings else 15
            if last_signal_ts:
                elapsed_since = now_ts - last_signal_ts
                next_in = max(0, int(cycle_mins * 60 - elapsed_since))
            else:
                next_in = cycle_mins * 60

            # WS connected
            ws_connected = bool(bot and bot.client and bot.client.ws and bot.client.ws.is_connected) if bot else False

            # API calls
            api_calls = _safe_int(await _scalar(db, "SELECT COUNT(*) FROM signals WHERE timestamp > ?", (now_ts - 86400,)))

            db_size = 0.0
            try:
                db_size = os.path.getsize(db) / (1024 * 1024)
            except OSError:
                pass

            markets_tracked = _safe_int(await _scalar(db, "SELECT COUNT(*) FROM market_cache"))

            # Paper criteria
            all_trades = await _query(db, "SELECT pnl, size_usdc, timestamp FROM paper_trades WHERE status='CLOSED' AND pnl IS NOT NULL")
            pnls = [_safe_float(t["pnl"]) for t in all_trades]
            n = len(pnls)
            wr = sum(1 for p in pnls if p > 0) / n if n else 0.0
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            pf = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
            total_budget = server_self.settings.budget if server_self.settings else 1000.0

            # Daily P&L for sharpe and per-day analysis
            days: dict[str, float] = defaultdict(float)
            for t in all_trades:
                ts = _safe_float(t.get("timestamp"))
                if ts:
                    day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                    days[day] += _safe_float(t["pnl"])
            daily_pnls = list(days.values())
            profitable_days = sum(1 for v in daily_pnls if v > 0)
            max_day_loss = max((abs(v) / total_budget for v in daily_pnls if v < 0), default=0.0)
            import math as _math
            if len(daily_pnls) >= 2:
                mu = sum(daily_pnls) / len(daily_pnls)
                var = sum((x - mu) ** 2 for x in daily_pnls) / (len(daily_pnls) - 1)
                std = _math.sqrt(var) if var > 0 else 0
                sharpe = (mu / std * _math.sqrt(365)) if std > 0 else 0.0
            else:
                sharpe = 0.0

            criteria = {
                "win_rate":      {"value": round(wr, 4),     "threshold": 0.55, "pass": wr >= 0.55},
                "profit_factor": {"value": round(min(pf, 99), 2), "threshold": 1.3, "pass": pf >= 1.3},
                "days_profitable":{"value": profitable_days, "threshold": 2,    "pass": profitable_days >= 2},
                "max_day_loss":  {"value": round(max_day_loss, 4), "threshold": 0.08, "pass": max_day_loss < 0.08},
                "trades_count":  {"value": n,               "threshold": 8,    "pass": n >= 8},
                "sharpe":        {"value": round(sharpe, 2), "threshold": 0.5,  "pass": sharpe >= 0.5},
            }
            criteria_passing = sum(1 for c in criteria.values() if c["pass"])

            bot_running = bool(bot and getattr(bot, "_running", False))
            pm = server_self.process_manager
            if pm:
                bot_running = pm.is_running

            return {
                "mode": stored_mode,
                "bot_running": bot_running,
                "bot_process_running": pm.is_running if pm else None,
                "bot_pid": pm._proc.pid if (pm and pm.is_running) else None,
                "runtime_seconds": runtime,
                "paper_hours_remaining": round(paper_remaining, 2),
                "paper_progress_pct": round(paper_progress, 2),
                "last_cycle_at": last_cycle_iso,
                "next_cycle_in_seconds": next_in,
                "ws_connected": ws_connected,
                "api_calls_today": api_calls,
                "db_size_mb": round(db_size, 2),
                "markets_tracked": markets_tracked,
                "cycles_completed": cycles,
                "paper_criteria": {**criteria, "criteria_passing": criteria_passing, "criteria_total": 6},
            }

        # ── Portfolio ─────────────────────────────────────────────────────────

        @app.get("/api/portfolio")
        async def portfolio():
            mode = await _scalar(db, "SELECT value FROM bot_state WHERE key='mode'") or "PAPER"
            table = "paper_trades" if mode == "PAPER" else "trades"
            total_budget = server_self.settings.budget if server_self.settings else 1000.0

            # Latest snapshot
            snap = await _query(db, "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1")
            s = snap[0] if snap else {}

            closed = await _query(db, f"SELECT pnl, size_usdc, timestamp, hold_hours, is_whale_copy FROM {table} WHERE status='CLOSED' AND pnl IS NOT NULL")
            open_pos = await _query(db, f"SELECT size_usdc FROM {table} WHERE status='OPEN' AND side='BUY'")

            pnls = [_safe_float(r["pnl"]) for r in closed]
            n = len(pnls)
            wins = sum(1 for p in pnls if p > 0)
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            win_rate = wins / n if n else 0.0
            pf = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
            holding_times = [_safe_float(r["hold_hours"]) for r in closed if r.get("hold_hours")]
            avg_hold = sum(holding_times) / len(holding_times) if holding_times else 0.0

            # Sharpe
            import math as _math
            days_map: dict[str, float] = defaultdict(float)
            for t in closed:
                ts = _safe_float(t.get("timestamp"))
                if ts:
                    day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                    days_map[day] += _safe_float(t["pnl"])
            dv = list(days_map.values())
            if len(dv) >= 2:
                mu = sum(dv) / len(dv)
                var = sum((x - mu) ** 2 for x in dv) / (len(dv) - 1)
                std = _math.sqrt(var) if var > 0 else 0
                sharpe = mu / std * _math.sqrt(365) if std > 0 else 0.0
            else:
                sharpe = 0.0

            # Drawdown from snapshots
            snaps_14 = await _query(db, "SELECT total_value, timestamp FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 14400")
            vals = [_safe_float(r["total_value"]) for r in reversed(snaps_14)]
            peak = total_budget
            max_dd = 0.0
            for v in vals:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

            # Daily P&L (last 14 days)
            daily_list = []
            cum = 0.0
            for day in sorted(days_map)[-14:]:
                cum += days_map[day]
                daily_list.append({"date": day, "pnl": round(days_map[day], 2), "cumulative_pnl": round(cum, 2)})

            # Hourly portfolio value (last 48h)
            cutoff = time.time() - 48 * 3600
            hourly_snaps = await _query(db, "SELECT timestamp, total_value FROM portfolio_snapshots WHERE timestamp > ? ORDER BY timestamp", (cutoff,))
            hourly = [{"time": datetime.fromtimestamp(_safe_float(r["timestamp"]), tz=timezone.utc).isoformat(),
                       "value": _safe_float(r["total_value"])} for r in hourly_snaps]

            # Exposure by category
            cat_rows = await _query(db, f"SELECT category, SUM(size_usdc) as total FROM {table} WHERE status='OPEN' GROUP BY category")
            exposure_by_cat = {r["category"] or "other": _safe_float(r["total"]) for r in cat_rows}

            # P&L by source
            whale_pnl = sum(_safe_float(r["pnl"]) for r in closed if r.get("is_whale_copy"))
            ai_pnl = sum(_safe_float(r["pnl"]) for r in closed if not r.get("is_whale_copy"))
            arb_rows = await _query(db, "SELECT SUM(profit_usdc) as total FROM arb_trades WHERE executed=1")
            arb_pnl = _safe_float(arb_rows[0]["total"] if arb_rows else 0)

            realized = _safe_float(s.get("realized_pnl", sum(pnls)))
            unrealized = _safe_float(s.get("unrealized_pnl", 0))
            cash = _safe_float(s.get("cash_balance", total_budget - sum(_safe_float(r["size_usdc"]) for r in open_pos)))

            return {
                "total_budget": total_budget,
                "cash_balance": round(cash, 2),
                "position_value": round(_safe_float(s.get("position_value", 0)), 2),
                "total_portfolio_value": round(_safe_float(s.get("total_value", total_budget)), 2),
                "realized_pnl": round(realized, 2),
                "unrealized_pnl": round(unrealized, 2),
                "total_pnl": round(realized + unrealized, 2),
                "roi_pct": round(_safe_float(s.get("roi_pct", (realized + unrealized) / total_budget * 100)), 2),
                "max_drawdown": round(max_dd, 4),
                "win_rate": round(win_rate, 4),
                "profit_factor": round(min(pf, 99.0), 2),
                "sharpe_ratio": round(sharpe, 2),
                "total_trades": n,
                "winning_trades": wins,
                "avg_holding_hours": round(avg_hold, 1),
                "best_trade_pnl": round(max(pnls) if pnls else 0.0, 2),
                "worst_trade_pnl": round(min(pnls) if pnls else 0.0, 2),
                "daily_pnl": daily_list,
                "hourly_portfolio_value": hourly,
                "exposure_by_category": exposure_by_cat,
                "pnl_by_source": {"ai": round(ai_pnl, 2), "whale_copy": round(whale_pnl, 2), "arb": round(arb_pnl, 2), "manual": 0.0},
            }

        # ── Positions ─────────────────────────────────────────────────────────

        @app.get("/api/positions")
        async def positions():
            mode = await _scalar(db, "SELECT value FROM bot_state WHERE key='mode'") or "PAPER"
            table = "paper_trades" if mode == "PAPER" else "trades"
            rows = await _query(db, f"SELECT * FROM {table} WHERE status='OPEN' AND side='BUY' ORDER BY timestamp DESC")
            now = time.time()
            result = []
            s = server_self.settings
            stop_pct = s.stop_loss_pct if s else 0.30
            tp_pct = s.take_profit_pct if s else 0.60
            for r in rows:
                entry = _safe_float(r.get("fill_price") or r.get("price"), 0.5)
                current = entry  # would be updated from live prices
                shares = _safe_float(r.get("shares"), 0)
                size = _safe_float(r.get("size_usdc"), 0)
                unrealized = (current - entry) * shares
                unrealized_pct = (current - entry) / entry if entry else 0
                hours = (now - _safe_float(r.get("timestamp"), now)) / 3600
                source = "WHALE" if r.get("is_whale_copy") else ("ARB" if r.get("arb_score", 0) > 5 else "AI")
                result.append({
                    "id": r["id"],
                    "market_slug": r.get("market_slug", ""),
                    "question": r.get("question", ""),
                    "category": r.get("category", ""),
                    "outcome": r.get("outcome", "YES"),
                    "source": source,
                    "entry_price": round(entry, 4),
                    "current_price": round(current, 4),
                    "size_usdc": round(size, 2),
                    "shares": round(shares, 4),
                    "unrealized_pnl": round(unrealized, 2),
                    "unrealized_pnl_pct": round(unrealized_pct, 4),
                    "opened_at": datetime.fromtimestamp(_safe_float(r.get("timestamp"), now), tz=timezone.utc).isoformat(),
                    "hours_held": round(hours, 1),
                    "stop_loss_price": round(entry * (1 - stop_pct), 4),
                    "take_profit_price": round(entry * (1 + tp_pct), 4),
                    "composite_score": round(_safe_float(r.get("composite_score"), 0), 1),
                    "ai_probability": round(_safe_float(r.get("ai_probability"), 0.5), 3),
                    "ai_edge": round(_safe_float(r.get("ai_edge"), 0), 3),
                    "whale_score": round(_safe_float(r.get("whale_score"), 0), 1),
                    "is_whale_copy": bool(r.get("is_whale_copy")),
                    "source_wallet": _short_addr(r.get("source_wallet") or ""),
                })
            return result

        # ── Trades ────────────────────────────────────────────────────────────

        @app.get("/api/trades")
        async def trades(limit: int = 50, offset: int = 0, mode: str = "all", source: str = "all"):
            tables = []
            if mode in ("all", "PAPER"):
                tables.append("paper_trades")
            if mode in ("all", "LIVE"):
                tables.append("trades")
            if not tables:
                tables = ["paper_trades", "trades"]

            all_rows = []
            for tbl in tables:
                rows = await _query(db, f"SELECT *, '{tbl}' as _src FROM {tbl} ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset))
                all_rows.extend(rows)

            all_rows.sort(key=lambda r: _safe_float(r.get("timestamp")), reverse=True)
            all_rows = all_rows[:limit]

            result = []
            for r in all_rows:
                src = "WHALE" if r.get("is_whale_copy") else ("ARB" if _safe_float(r.get("arb_score")) > 5 else "AI")
                if source != "all" and src != source:
                    continue
                ts = _safe_float(r.get("timestamp"))
                result.append({
                    "id": r.get("id"),
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                    "mode": r.get("mode", "PAPER"),
                    "market_slug": r.get("market_slug", ""),
                    "question": r.get("question", ""),
                    "category": r.get("category", ""),
                    "outcome": r.get("outcome", ""),
                    "side": r.get("side", "BUY"),
                    "price": round(_safe_float(r.get("price")), 4),
                    "size_usdc": round(_safe_float(r.get("size_usdc")), 2),
                    "fill_price": round(_safe_float(r.get("fill_price") or r.get("price")), 4),
                    "slippage": round(_safe_float(r.get("slippage")), 5),
                    "pnl": round(_safe_float(r.get("pnl")), 2) if r.get("pnl") is not None else None,
                    "hold_hours": round(_safe_float(r.get("hold_hours")), 1) if r.get("hold_hours") is not None else None,
                    "exit_reason": r.get("exit_reason"),
                    "status": r.get("status", "OPEN"),
                    "composite_score": round(_safe_float(r.get("composite_score")), 1),
                    "ai_probability": round(_safe_float(r.get("ai_probability"), 0.5), 3),
                    "ai_edge": round(_safe_float(r.get("ai_edge")), 3),
                    "signal_strength": r.get("signal_strength", "WEAK"),
                    "source": src,
                    "is_whale_copy": bool(r.get("is_whale_copy")),
                    "source_wallet": _short_addr(r.get("source_wallet") or ""),
                })
            return result

        # ── Signals ───────────────────────────────────────────────────────────

        @app.get("/api/signals")
        async def signals(limit: int = 30):
            rows = await _query(db, "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,))
            result = []
            for r in rows:
                ts = _safe_float(r.get("timestamp"))
                edge = _safe_float(r.get("ai_edge"))
                abs_edge = abs(edge)
                conf = _safe_float(r.get("ai_confidence") if "ai_confidence" in r else 0.5, 0.5)
                if abs_edge > 0.12 and conf > 0.75:
                    strength = "VERY_STRONG"
                elif abs_edge > 0.08 and conf > 0.65:
                    strength = "STRONG"
                elif abs_edge > 0.04 and conf > 0.55:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"
                result.append({
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                    "market_slug": r.get("market_slug", ""),
                    "question": r.get("question", ""),
                    "ai_probability": round(_safe_float(r.get("ai_probability"), 0.5), 3),
                    "market_price": round(_safe_float(r.get("market_price"), 0.5), 3),
                    "ai_edge": round(edge, 3),
                    "whale_score": round(_safe_float(r.get("whale_score")), 1),
                    "news_score": round(_safe_float(r.get("news_score")), 1),
                    "technical_score": round(_safe_float(r.get("technical_score")), 1),
                    "composite_score": round(_safe_float(r.get("composite_score")), 1),
                    "final_direction": r.get("final_direction", "NEUTRAL"),
                    "action_taken": r.get("action_taken", "SKIPPED"),
                    "signal_strength": strength,
                    "reason_skipped": r.get("reason_skipped"),
                })
            return result

        # ── Whales ────────────────────────────────────────────────────────────

        @app.get("/api/whales")
        async def whales():
            wallets = await _query(db, "SELECT * FROM whale_wallets ORDER BY total_pnl DESC LIMIT 20")
            now = time.time()
            top_wallets = []
            for w in wallets:
                tags = json.loads(w.get("tier_tags") or "[]")
                last_active = _safe_float(w.get("last_active"))
                top_wallets.append({
                    "address": w.get("address", ""),
                    "short_address": _short_addr(w.get("address", "")),
                    "tier_tags": tags,
                    "win_rate": round(_safe_float(w.get("win_rate")), 3),
                    "total_pnl": round(_safe_float(w.get("total_pnl")), 2),
                    "total_volume": round(_safe_float(w.get("total_volume")), 2),
                    "profit_factor": round(min(_safe_float(w.get("profit_factor")), 99), 2),
                    "insider_score": round(_safe_float(w.get("insider_score")), 1),
                    "information_lead_score": round(_safe_float(w.get("information_lead_score")), 3),
                    "timing_alpha": round(_safe_float(w.get("timing_alpha")), 4),
                    "is_insider_candidate": bool(w.get("is_insider_candidate")),
                    "is_bot": bool(w.get("is_bot")),
                    "leader_score": _safe_int(w.get("leader_score")),
                    "last_active": datetime.fromtimestamp(last_active, tz=timezone.utc).isoformat() if last_active else None,
                    "active_today": bool(last_active and (now - last_active) < 86400),
                })

            # Recent alerts (from whale_trades)
            alert_rows = await _query(db, """
                SELECT wt.*, ww.tier_tags, ww.win_rate, ww.insider_score
                FROM whale_trades wt
                LEFT JOIN whale_wallets ww ON wt.wallet = ww.address
                ORDER BY wt.timestamp DESC LIMIT 20
            """)
            recent_alerts = []
            for r in alert_rows:
                ts = _safe_float(r.get("timestamp"))
                mins_ago = int((now - ts) / 60) if ts else 0
                tags = json.loads(r.get("tier_tags") or "[]")
                tier_label = tags[0] if tags else "UNKNOWN"
                recent_alerts.append({
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                    "wallet": r.get("wallet", ""),
                    "short_address": _short_addr(r.get("wallet", "")),
                    "tier": tier_label,
                    "market_slug": r.get("market_slug", ""),
                    "outcome": r.get("outcome", ""),
                    "size_usdc": round(_safe_float(r.get("size_usdc")), 2),
                    "conviction_multiplier": 1.0,
                    "alert_level": r.get("alert_level", "LOW"),
                    "our_copy_trade_id": r.get("our_copy_trade_id"),
                    "minutes_ago": mins_ago,
                })

            # Smart money consensus per market
            pos_rows = await _query(db, """
                SELECT wp.market_slug,
                       ww.tier_tags,
                       wp.outcome,
                       wp.shares * wp.current_price as val
                FROM whale_positions wp
                JOIN whale_wallets ww ON wp.wallet = ww.address
            """)
            consensus_map: dict[str, dict] = {}
            for r in pos_rows:
                slug = r.get("market_slug", "")
                if slug not in consensus_map:
                    consensus_map[slug] = {"yes": 0.0, "no": 0.0, "yes_w": 0, "no_w": 0}
                val = _safe_float(r.get("val"))
                if r.get("outcome", "").upper() == "YES":
                    consensus_map[slug]["yes"] += val
                    consensus_map[slug]["yes_w"] += 1
                else:
                    consensus_map[slug]["no"] += val
                    consensus_map[slug]["no_w"] += 1
            sm_consensus = []
            for slug, d in list(consensus_map.items())[:20]:
                net = "YES" if d["yes"] > d["no"] * 1.2 else ("NO" if d["no"] > d["yes"] * 1.2 else "NEUTRAL")
                total = d["yes"] + d["no"]
                conviction = min(10.0, abs(d["yes"] - d["no"]) / total * 10) if total > 0 else 0
                sm_consensus.append({
                    "market_slug": slug,
                    "smart_money_net_direction": net,
                    "smart_money_conviction": round(conviction, 1),
                    "smart_money_yes_usdc": round(d["yes"], 2),
                    "smart_money_no_usdc": round(d["no"], 2),
                    "yes_wallets": d["yes_w"],
                    "no_wallets": d["no_w"],
                })

            # Copy trade stats
            ct_rows = await _query(db, "SELECT was_profitable, pnl FROM copy_trade_log WHERE close_timestamp IS NOT NULL")
            ct_total = len(ct_rows)
            ct_wins = sum(1 for r in ct_rows if r.get("was_profitable"))
            ct_pnl = sum(_safe_float(r.get("pnl")) for r in ct_rows)
            best_wallet = ""
            if ct_rows:
                wallet_perf: dict[str, list] = defaultdict(list)
                ct_detail = await _query(db, "SELECT source_wallet, pnl FROM copy_trade_log WHERE close_timestamp IS NOT NULL")
                for r in ct_detail:
                    wallet_perf[r.get("source_wallet", "")].append(_safe_float(r.get("pnl")))
                if wallet_perf:
                    best_wallet = _short_addr(max(wallet_perf, key=lambda k: sum(wallet_perf[k])))

            return {
                "top_wallets": top_wallets,
                "recent_alerts": recent_alerts,
                "smart_money_consensus": sm_consensus,
                "copy_trade_stats": {
                    "total_copy_trades": ct_total,
                    "profitable_copy_trades": ct_wins,
                    "copy_win_rate": round(ct_wins / ct_total, 3) if ct_total else 0.0,
                    "copy_total_pnl": round(ct_pnl, 2),
                    "best_wallet_to_copy": best_wallet,
                },
            }

        # ── Arbitrage ─────────────────────────────────────────────────────────

        @app.get("/api/arb")
        async def arb():
            rows = await _query(db, "SELECT * FROM arb_trades ORDER BY timestamp DESC LIMIT 20")
            now = time.time()
            cutoff_day = now - 86400
            today_rows = await _query(db, "SELECT * FROM arb_trades WHERE timestamp > ?", (cutoff_day,))
            total_found = len(today_rows)
            total_exec = sum(1 for r in today_rows if r.get("executed"))
            total_pnl = sum(_safe_float(r.get("profit_usdc")) for r in today_rows if r.get("executed"))
            all_exec = [r for r in today_rows if r.get("executed")]
            avg_pct = sum(_safe_float(r.get("profit_pct")) for r in all_exec) / len(all_exec) if all_exec else 0

            type_map: dict[str, int] = defaultdict(int)
            for r in today_rows:
                t = r.get("arb_type", "OTHER")
                type_map[t] += 1

            recent = []
            for r in rows:
                ts = _safe_float(r.get("timestamp"))
                legs = json.loads(r.get("legs_json") or "[]")
                legs_desc = ""
                if legs and len(legs) >= 2:
                    legs_desc = " + ".join(f"{l.get('outcome','?')} @ {l.get('price',0):.3f}" for l in legs[:2])
                    legs_desc += f" = {sum(_safe_float(l.get('price',0)) for l in legs[:2]):.3f}"
                recent.append({
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                    "arb_type": r.get("arb_type", ""),
                    "market_slug": r.get("market_slug", ""),
                    "legs_detail": legs_desc,
                    "profit_pct": round(_safe_float(r.get("profit_pct")) * 100, 3),
                    "profit_usdc": round(_safe_float(r.get("profit_usdc")), 2),
                    "executed": bool(r.get("executed")),
                    "execution_time_ms": r.get("execution_time_ms"),
                })
            return {
                "recent_opportunities": recent,
                "stats": {
                    "total_found": total_found,
                    "total_executed": total_exec,
                    "total_arb_pnl": round(total_pnl, 2),
                    "avg_profit_pct": round(avg_pct * 100, 3),
                    "type_breakdown": dict(type_map),
                },
            }

        # ── Settings GET ──────────────────────────────────────────────────────

        @app.get("/api/settings")
        async def get_settings():
            s = server_self.settings
            if not s:
                return {}
            return {
                "budget": s.budget,
                "max_position_size": s.max_position_size,
                "min_edge_threshold": s.min_edge,
                "kelly_fraction": s.kelly_multiplier,
                "max_portfolio_exposure": s.max_total_exposure_pct,
                "min_market_liquidity": s.min_liquidity,
                "min_market_volume_24h": s.min_volume,
                "cycle_minutes": s.cycle_minutes,
                "max_open_positions": s.max_open_positions,
                "stop_loss_pct": s.stop_loss_pct,
                "take_profit_pct": s.take_profit_pct,
                "daily_loss_limit": s.daily_loss_limit_pct,
                "ai_weight": s.ai_weight,
                "whale_weight": s.whale_weight,
                "news_weight": s.news_weight,
                "technical_weight": s.technical_weight,
                "orderbook_weight": s.orderbook_weight,
                "arb_weight": s.arb_weight,
                "paper_period_hours": s.paper_trading_hours,
                "categories_filter": s.categories or [],
            }

        # ── Settings POST ─────────────────────────────────────────────────────

        @app.post("/api/settings")
        async def update_settings(request: Request):
            body = await request.json()
            s = server_self.settings
            if not s:
                raise HTTPException(500, "Settings not available")
            mapping = {
                "budget": "budget", "max_position_size": "max_position_size",
                "min_edge_threshold": "min_edge", "kelly_fraction": "kelly_multiplier",
                "max_portfolio_exposure": "max_total_exposure_pct",
                "min_market_liquidity": "min_liquidity", "min_market_volume_24h": "min_volume",
                "cycle_minutes": "cycle_minutes", "max_open_positions": "max_open_positions",
                "stop_loss_pct": "stop_loss_pct", "take_profit_pct": "take_profit_pct",
                "daily_loss_limit": "daily_loss_limit_pct",
                "ai_weight": "ai_weight", "whale_weight": "whale_weight",
                "news_weight": "news_weight", "technical_weight": "technical_weight",
                "orderbook_weight": "orderbook_weight", "arb_weight": "arb_weight",
            }
            updated = {}
            for k, v in body.items():
                attr = mapping.get(k, k)
                if hasattr(s, attr):
                    setattr(s, attr, v)
                    updated[k] = v
                    await _execute(db, "INSERT OR REPLACE INTO bot_state(key,value) VALUES(?,?)", (f"setting_{k}", str(v)))
            return {"success": True, "updated": updated}

        # ── Control ───────────────────────────────────────────────────────────

        @app.post("/api/control")
        async def control(request: Request):
            body = await request.json()
            action = body.get("action", "")

            # Guard destructive actions — require explicit human confirmation so
            # automated callers (scripts, bots, misbehaving clients) cannot
            # trigger them unintentionally.
            _DESTRUCTIVE = {"stop", "emergency_stop", "cancel_all_orders", "close_all_positions"}
            if action in _DESTRUCTIVE and not body.get("human_confirmed"):
                raise HTTPException(400, "human_confirmed required")

            bot = server_self.bot_ref
            pm  = server_self.process_manager

            # ── Start (subprocess mode only) ──────────────────────────────────
            if action == "start":
                if pm:
                    mode   = body.get("mode", "paper")
                    budget = body.get("budget") or None
                    # Write mode before starting so the subprocess sees the correct
                    # state in _check_db_commands and doesn't read stale STOPPED.
                    await _execute(db, "INSERT OR REPLACE INTO bot_state(key,value) VALUES('mode',?)",
                                   (mode.upper(),))
                    result = await pm.start(mode=mode, budget=budget)
                    if result["success"]:
                        await mgr.broadcast("mode_change", {"old_mode": "STOPPED", "new_mode": mode.upper()})
                    else:
                        # Revert if the process couldn't be launched
                        await _execute(db, "INSERT OR REPLACE INTO bot_state(key,value) VALUES('mode','STOPPED')")
                    return result
                if bot:
                    return {"success": False, "message": "Bot is already running"}
                return {"success": False, "message": "No process manager available"}

            # ── Pause ─────────────────────────────────────────────────────────
            elif action == "pause":
                # Write to DB; running bot polls DB in its fast loop
                await _execute(db, "INSERT OR REPLACE INTO bot_state(key,value) VALUES('mode','PAUSED')")
                if bot:
                    bot._running = False
                return {"success": True, "message": "Pause signal sent"}

            # ── Resume ────────────────────────────────────────────────────────
            elif action == "resume":
                if pm and not pm.is_running:
                    # Process exited; restart it in whatever mode was last active
                    stored = await _scalar(db, "SELECT value FROM bot_state WHERE key='mode'") or "PAPER"
                    restart_mode = "paper" if stored in ("PAUSED", "STOPPED", "PAPER") else "live"
                    result = await pm.start(mode=restart_mode)
                    if result["success"]:
                        await _execute(db, "INSERT OR REPLACE INTO bot_state(key,value) VALUES('mode',?)",
                                       (restart_mode.upper(),))
                    return result
                # Process running (paused via DB): restore previous mode
                stored = await _scalar(db, "SELECT value FROM bot_state WHERE key='prev_mode'") or "PAPER"
                await _execute(db, f"INSERT OR REPLACE INTO bot_state(key,value) VALUES('mode','{stored}')")
                if bot:
                    bot._running = True
                    asyncio.create_task(bot.run())
                return {"success": True, "message": "Resume signal sent"}

            # ── Stop ──────────────────────────────────────────────────────────
            elif action == "stop":
                await _execute(db, "INSERT OR REPLACE INTO bot_state(key,value) VALUES('mode','STOPPED')")
                if pm and pm.is_running:
                    return await pm.stop()
                if bot:
                    bot._running = False
                return {"success": True, "message": "Stop signal sent"}

            elif action == "restart_cycle":
                if bot:
                    asyncio.create_task(bot._main_cycle())
                return {"success": True, "message": "Cycle restarted"}

            elif action == "cancel_all_orders":
                if bot and bot.client and bot.client.clob:
                    if bot.mode == "LIVE":
                        await bot.client.clob.cancel_all_orders()
                    else:
                        logger.info("Paper mode: skipping real cancel_all_orders from dashboard")
                return {"success": True, "message": "All orders cancelled"}

            elif action == "close_all_positions":
                return {"success": True, "message": "Close all positions queued"}

            elif action == "activate_live":
                phrase = body.get("confirm_phrase", "")
                if phrase != "CONFIRM LIVE TRADING":
                    raise HTTPException(403, "Invalid confirmation phrase")
                if bot:
                    bot.mode = "LIVE"
                await _execute(db, "INSERT OR REPLACE INTO bot_state(key,value) VALUES('mode','LIVE')")
                await _execute(db, f"INSERT OR REPLACE INTO bot_state(key,value) VALUES('live_activated_at','{time.time()}')")
                await mgr.broadcast("mode_change", {"old_mode": "PAPER", "new_mode": "LIVE"})
                return {"success": True, "message": "Live trading activated"}

            elif action == "reset_paper":
                if bot:
                    bot._paper_start_ts = time.time()
                    bot.mode = "PAPER"
                await _execute(db, f"INSERT OR REPLACE INTO bot_state(key,value) VALUES('paper_start_ts','{time.time()}')")
                await _execute(db, "INSERT OR REPLACE INTO bot_state(key,value) VALUES('mode','PAPER')")
                return {"success": True, "message": "Paper period reset"}

            raise HTTPException(400, f"Unknown action: {action}")

        # ── Close position ────────────────────────────────────────────────────

        @app.post("/api/positions/{slug}/close")
        async def close_position(slug: str):
            mode = await _scalar(db, "SELECT value FROM bot_state WHERE key='mode'") or "PAPER"
            table = "paper_trades" if mode == "PAPER" else "trades"
            row = await _query(db, f"SELECT * FROM {table} WHERE market_slug=? AND status='OPEN' LIMIT 1", (slug,))
            if not row:
                raise HTTPException(404, "Position not found")
            await _execute(db, f"UPDATE {table} SET status='CLOSE_PENDING', exit_reason='manual_close' WHERE market_slug=? AND status='OPEN'", (slug,))
            return {"success": True, "message": f"Position {slug} queued for close"}

        # ── Debug / health report ─────────────────────────────────────────────

        @app.get("/api/debug")
        async def debug_report():
            """Return a full health/diagnostic report for the last cycle."""
            bot = server_self.bot_ref
            diag = getattr(bot, "_diag", {}) if bot else {}

            # Pull fresh counts from DB
            mode = await _scalar(db, "SELECT value FROM bot_state WHERE key='mode'") or "PAPER"
            table = "paper_trades" if mode == "PAPER" else "trades"
            today_start = time.time() - 86400
            trades_today   = await _scalar(db,
                f"SELECT COUNT(*) FROM {table} WHERE timestamp >= ?", (today_start,)) or 0
            signals_today  = await _scalar(db,
                "SELECT COUNT(*) FROM signals WHERE timestamp >= ?", (today_start,)) or 0
            whale_wallets  = await _scalar(db,
                "SELECT COUNT(*) FROM whale_wallets") or 0
            arb_today      = await _scalar(db,
                "SELECT COUNT(*) FROM arb_trades WHERE timestamp >= ? AND executed=1",
                (today_start,)) or 0

            return {
                "last_cycle_duration_seconds": diag.get("last_cycle_duration_s", 0),
                "markets_analyzed_last_cycle": diag.get("markets_analyzed", 0),
                "ai_calls_last_cycle":         diag.get("ai_calls", 0),
                "ai_signals_generated":        diag.get("ai_signals_generated", 0),
                "arb_opportunities_found":     diag.get("arb_found", 0),
                "arb_opportunities_executed":  diag.get("arb_executed", 0),
                "orderbooks_fetched_ok":       diag.get("orderbooks_ok", 0),
                "orderbooks_failed":           diag.get("orderbooks_failed", 0),
                "whale_wallets_tracked":       whale_wallets,
                "signals_logged_today":        signals_today,
                "trades_executed_today":       trades_today,
                "arb_executed_today":          arb_today,
                "current_mode":                mode,
                "why_no_trades":               diag.get("why_no_trades", "No cycle run yet"),
                "bot_running":                 bool(bot and getattr(bot, "_running", False)),
                "cycle_count":                 getattr(bot, "_cycle_count", 0) if bot else 0,
            }

        # ── Logs ──────────────────────────────────────────────────────────────

        @app.get("/api/logs")
        async def logs(limit: int = 100, level: str = "all"):
            log_path = Path(server_self.db_path).parent / "polymarket_bot.log"
            if not log_path.exists():
                return []
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return []
            # Parse log lines (format: date time | LEVEL | module | message)
            result = []
            for line in reversed(lines[-2000:]):
                parts = line.split(" | ", 3)
                if len(parts) == 4:
                    ts_str, lvl, mod, msg = parts
                    lvl = lvl.strip()
                    if level != "all" and lvl != level:
                        continue
                    result.append({"timestamp": ts_str.strip(), "level": lvl, "module": mod.strip(), "message": msg.strip()})
                    if len(result) >= limit:
                        break
            return result

        # ── WebSocket ─────────────────────────────────────────────────────────

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await mgr.connect(ws)
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                mgr.disconnect(ws)
            except Exception:
                mgr.disconnect(ws)


# ── Module-level app for direct uvicorn invocation ────────────────────────────
# Allows:  uvicorn polymarket_bot.api.server:app --host 0.0.0.0 --port 8765
# Routes are registered with settings=None; all handlers guard against it.
_default_server = APIServer(settings=None)
app = _default_server.app
