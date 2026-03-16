"""
mt5_bridge.py — MetaTrader 5 Python API Bridge
Connects to MT5, reads live data, and returns a unified data object.
"""

import os
import json
import csv
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

logger = logging.getLogger(__name__)


class MT5Bridge:
    """Bridges the MT5 terminal and the dashboard backend."""

    def __init__(self):
        self.connected = False
        self.terminal_data_path: Optional[str] = None
        self.last_live_data: dict = {}
        self.trade_history: list = []
        self._equity_history: list = []  # (timestamp, equity) pairs
        self._connect_attempt_time: float = 0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize MT5 connection. Returns True on success."""
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 package not installed — running in demo mode")
            self.connected = False
            return False

        if self.connected:
            return True

        now = time.time()
        # Rate-limit reconnect attempts to once per 5 seconds
        if now - self._connect_attempt_time < 5:
            return False
        self._connect_attempt_time = now

        try:
            if not mt5.initialize():
                logger.error("MT5 initialize() failed: %s", mt5.last_error())
                self.connected = False
                return False

            info = mt5.terminal_info()
            if info is None:
                logger.error("Could not retrieve MT5 terminal info")
                self.connected = False
                return False

            self.terminal_data_path = info.data_path
            self.connected = True
            logger.info("MT5 connected — terminal: %s", self.terminal_data_path)
            return True

        except Exception as exc:
            logger.exception("Exception connecting to MT5: %s", exc)
            self.connected = False
            return False

    def disconnect(self):
        if MT5_AVAILABLE and self.connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
        self.connected = False

    def ensure_connected(self) -> bool:
        if not self.connected:
            return self.connect()
        # Ping with a lightweight call
        try:
            if MT5_AVAILABLE and mt5.terminal_info() is None:
                self.connected = False
                return self.connect()
        except Exception:
            self.connected = False
            return self.connect()
        return True

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    def _json_path(self) -> Optional[str]:
        if self.terminal_data_path:
            return os.path.join(
                self.terminal_data_path, "MQL5", "Files", "tjr_live_data.json"
            )
        # Fallback — search common MT5 data locations
        candidates = [
            os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\Common\Files\tjr_live_data.json"),
            "tjr_live_data.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _csv_path(self) -> Optional[str]:
        if self.terminal_data_path:
            return os.path.join(
                self.terminal_data_path, "MQL5", "Files", "tjr_trade_history.csv"
            )
        candidates = [
            os.path.expandvars(r"%APPDATA%\MetaQuotes\Terminal\Common\Files\tjr_trade_history.csv"),
            "tjr_trade_history.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    # ------------------------------------------------------------------
    # Live data
    # ------------------------------------------------------------------

    def get_live_data(self) -> dict:
        """Return the full live data dict, merging EA JSON + MT5 API data."""
        if not self.ensure_connected():
            return self._demo_data()

        ea_data = self._read_ea_json()
        api_data = self._read_mt5_api()
        merged = {**ea_data, **api_data}
        self.last_live_data = merged

        # Append equity history point every ~60 s
        now_ts = datetime.utcnow().isoformat()
        equity = merged.get("account_equity", 0)
        if equity and (
            not self._equity_history
            or (datetime.utcnow() - datetime.fromisoformat(
                    self._equity_history[-1]["t"])).seconds >= 60
        ):
            self._equity_history.append({"t": now_ts, "e": equity})
            # Keep last 480 points (8 hours at 1 min each)
            if len(self._equity_history) > 480:
                self._equity_history = self._equity_history[-480:]

        merged["equity_history"] = self._equity_history
        return merged

    def _read_ea_json(self) -> dict:
        """Read tjr_live_data.json written by the EA."""
        path = self._json_path()
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.debug("Could not read EA JSON: %s", exc)
            return {}

    def _read_mt5_api(self) -> dict:
        """Read live data directly from MT5 Python API."""
        if not MT5_AVAILABLE or not self.connected:
            return {}

        try:
            account = mt5.account_info()
            if account is None:
                return {}

            positions = mt5.positions_get()
            pos_list = []
            if positions:
                for p in positions:
                    pos_list.append({
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "type": "BUY" if p.type == 0 else "SELL",
                        "volume": p.volume,
                        "price_open": p.price_open,
                        "sl": p.sl,
                        "tp": p.tp,
                        "price_current": p.price_current,
                        "profit": p.profit,
                        "comment": p.comment,
                    })

            return {
                "account_balance":    account.balance,
                "account_equity":     account.equity,
                "account_margin":     account.margin,
                "account_free_margin": account.margin_free,
                "account_profit":     account.profit,
                "account_currency":   account.currency,
                "open_positions":     pos_list,
                "mt5_connected":      True,
            }
        except Exception as exc:
            logger.debug("MT5 API read error: %s", exc)
            return {"mt5_connected": False}

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    def get_trade_history(self) -> list:
        """Read CSV trade history and return as list of dicts with stats."""
        path = self._csv_path()
        trades = []
        if path and os.path.exists(path):
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trades.append(row)
            except Exception as exc:
                logger.debug("Could not read trade CSV: %s", exc)

        self.trade_history = trades
        return trades

    def get_statistics(self) -> dict:
        """Calculate statistics from trade history."""
        trades = self.get_trade_history()
        closed = [t for t in trades if t.get("exit", "0") not in ("0", "")]

        if not closed:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_r": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_r": 0.0,
                "max_consec_losses": 0,
                "r_distribution": {},
                "session_stats": {},
            }

        wins   = [t for t in closed if float(t.get("result_r", 0)) > 0]
        losses = [t for t in closed if float(t.get("result_r", 0)) <= 0]
        gross_profit = sum(float(t.get("result_usd", 0)) for t in wins)
        gross_loss   = abs(sum(float(t.get("result_usd", 0)) for t in losses))

        r_values = [float(t.get("result_r", 0)) for t in closed]
        avg_r = sum(r_values) / len(r_values) if r_values else 0

        # Max drawdown in R
        running_r = 0
        peak_r = 0
        max_dd_r = 0
        for r in r_values:
            running_r += r
            if running_r > peak_r:
                peak_r = running_r
            dd = peak_r - running_r
            if dd > max_dd_r:
                max_dd_r = dd

        # Max consecutive losses
        max_consec = 0
        cur_consec = 0
        for t in closed:
            if float(t.get("result_r", 0)) <= 0:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0

        # R distribution histogram
        r_dist: dict[str, int] = {}
        for r in r_values:
            bucket = str(round(r))
            r_dist[bucket] = r_dist.get(bucket, 0) + 1

        # Session performance
        session_stats: dict[str, dict] = {}
        for t in closed:
            sess = t.get("session", "UNKNOWN")
            if sess not in session_stats:
                session_stats[sess] = {"trades": 0, "wins": 0, "total_r": 0.0}
            session_stats[sess]["trades"] += 1
            r = float(t.get("result_r", 0))
            if r > 0:
                session_stats[sess]["wins"] += 1
            session_stats[sess]["total_r"] += r

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) * 100 if closed else 0,
            "avg_r": avg_r,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
            "max_drawdown_r": max_dd_r,
            "max_consec_losses": max_consec,
            "r_distribution": r_dist,
            "session_stats": session_stats,
        }

    # ------------------------------------------------------------------
    # Account / positions (direct API)
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict:
        if not self.ensure_connected() or not MT5_AVAILABLE:
            return {"mt5_connected": False}
        try:
            a = mt5.account_info()
            if a is None:
                return {"mt5_connected": False}
            return {
                "mt5_connected": True,
                "login":         a.login,
                "server":        a.server,
                "currency":      a.currency,
                "balance":       a.balance,
                "equity":        a.equity,
                "margin":        a.margin,
                "free_margin":   a.margin_free,
                "profit":        a.profit,
                "leverage":      a.leverage,
            }
        except Exception as exc:
            logger.debug("get_account_info error: %s", exc)
            return {"mt5_connected": False}

    def get_positions(self) -> list:
        if not self.ensure_connected() or not MT5_AVAILABLE:
            return []
        try:
            positions = mt5.positions_get()
            if not positions:
                return []
            result = []
            for p in positions:
                result.append({
                    "ticket":       p.ticket,
                    "symbol":       p.symbol,
                    "type":         "BUY" if p.type == 0 else "SELL",
                    "volume":       p.volume,
                    "price_open":   p.price_open,
                    "sl":           p.sl,
                    "tp":           p.tp,
                    "price_current": p.price_current,
                    "profit":       p.profit,
                    "swap":         p.swap,
                    "comment":      p.comment,
                    "time":         str(datetime.fromtimestamp(p.time)),
                })
            return result
        except Exception as exc:
            logger.debug("get_positions error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Demo / fallback data
    # ------------------------------------------------------------------

    def _demo_data(self) -> dict:
        """Return realistic demo data when MT5 is not connected."""
        return {
            "timestamp":        datetime.utcnow().isoformat(),
            "symbol":           "XAUUSD",
            "phase":            "AWAIT_SWEEP",
            "htf_bias":         "BULLISH",
            "session":          "NY_KILLZONE",
            "sweep_detected":   False,
            "sweep_direction":  "BULLISH",
            "sweep_level":      0.0,
            "m5_bos_confirmed": False,
            "entry_zone_active": False,
            "checklist_score":  1,
            "checklist":        [True, True, False, False, False, False, False, False],
            "trade_active":     False,
            "trade_direction":  "NONE",
            "entry_price":      0.0,
            "stop_loss":        0.0,
            "tp1":              0.0,
            "tp2":              0.0,
            "open_pnl_r":       0.0,
            "open_pnl_usd":     0.0,
            "trades_today":     0,
            "wins_today":       0,
            "losses_today":     0,
            "daily_pnl_pct":    0.0,
            "daily_pnl_r":      0.0,
            "consec_losses":    0,
            "account_balance":  10000.0,
            "account_equity":   10000.0,
            "account_margin":   0.0,
            "account_free_margin": 10000.0,
            "account_profit":   0.0,
            "account_currency": "USD",
            "asia_high":        0.0,
            "asia_low":         0.0,
            "london_high":      0.0,
            "london_low":       0.0,
            "prev_day_high":    0.0,
            "prev_day_low":     0.0,
            "h4_swing_high":    0.0,
            "h4_swing_low":     0.0,
            "fvg_active":       False,
            "fvg_high":         0.0,
            "fvg_low":          0.0,
            "ob_active":        False,
            "ob_high":          0.0,
            "ob_low":           0.0,
            "eq_level":         0.0,
            "open_positions":   [],
            "mt5_connected":    False,
            "equity_history":   self._equity_history,
            "_demo_mode":       True,
        }
