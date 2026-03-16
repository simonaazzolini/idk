"""
performance_tracker.py — Tracks all trade performance metrics.

Logs detailed trade data and computes aggregate statistics including
win rate, Sharpe ratio, and P&L curves.
"""
import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PAPER_TRADES_PATH = Path(__file__).parent / "paper_trades.json"
LIVE_TRADES_PATH = Path(__file__).parent / "live_trades.json"


@dataclass
class TradeRecord:
    """Immutable record of a completed trade."""
    trade_id: str
    mode: str
    asset: str
    outcome: str                # YES or NO
    posterior_at_entry: float
    market_price_at_entry: float
    divergence: float
    evidence_breakdown: dict
    prior_used: float
    regime_at_entry: str
    session_at_entry: str
    resolution_outcome: Optional[bool]
    pnl_usd: float
    pnl_pct: float
    hold_time_minutes: float
    fill_price: float
    slippage: float
    exit_reason: str
    question: str
    entry_time: str
    exit_time: str


@dataclass
class PerformanceStats:
    """Aggregate performance statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_divergence_wins: float = 0.0
    avg_divergence_losses: float = 0.0
    avg_hold_minutes: float = 0.0
    pnl_by_regime: Dict[str, float] = field(default_factory=dict)
    pnl_by_session: Dict[str, float] = field(default_factory=dict)
    calibration_error: float = 0.0  # avg |posterior - win_rate_at_posterior|

    @property
    def is_ready_for_live(self) -> bool:
        """Check if paper trading metrics meet go-live criteria."""
        return (
            self.total_trades >= 40
            and self.win_rate >= 0.55
            and self.sharpe_ratio >= 1.0
        )

    @property
    def readiness_score(self) -> dict:
        return {
            "trades_done": self.total_trades,
            "trades_needed": 40,
            "trades_pct": min(100, int(self.total_trades / 40 * 100)),
            "win_rate": self.win_rate,
            "win_rate_target": 0.55,
            "win_rate_ok": self.win_rate >= 0.55,
            "sharpe": self.sharpe_ratio,
            "sharpe_target": 1.0,
            "sharpe_ok": self.sharpe_ratio >= 1.0,
            "ready": self.is_ready_for_live,
        }


class PerformanceTracker:
    """Tracks and computes performance metrics for all trades."""

    def __init__(self, mode: str = "paper"):
        self.mode = mode
        self._trades_cache: Optional[List[dict]] = None

    def _get_path(self) -> Path:
        return PAPER_TRADES_PATH if self.mode == "paper" else LIVE_TRADES_PATH

    def load_trades(self, status: Optional[str] = "closed") -> List[dict]:
        """Load trades from disk, optionally filtered by status."""
        path = self._get_path()
        if not path.exists():
            return []
        try:
            with open(path) as f:
                all_trades = json.load(f)
            if status:
                return [t for t in all_trades if t.get("status") == status]
            return all_trades
        except Exception as e:
            logger.warning("Failed to load trades: %s", e)
            return []

    def record_trade(self, position) -> None:
        """Record a completed trade (called from TradeExecutor on close)."""
        logger.info(
            "Trade recorded: %s | P&L=$%.2f (%.1f%%) | outcome=%s",
            position.trade_id,
            position.pnl_usd,
            position.pnl_pct * 100,
            "WIN" if position.pnl_usd > 0 else "LOSS"
        )
        self._trades_cache = None  # Invalidate cache

    def compute_stats(self, n_recent: Optional[int] = None) -> PerformanceStats:
        """Compute aggregate performance statistics from closed trades."""
        trades = self.load_trades(status="closed")

        if n_recent:
            trades = trades[-n_recent:]

        stats = PerformanceStats()

        if not trades:
            return stats

        pnls = [t.get("pnl_usd", 0.0) for t in trades]
        stats.total_trades = len(trades)
        stats.total_pnl = sum(pnls)
        stats.winning_trades = sum(1 for p in pnls if p > 0)
        stats.losing_trades = sum(1 for p in pnls if p <= 0)
        stats.win_rate = stats.winning_trades / stats.total_trades if stats.total_trades > 0 else 0.0
        stats.avg_pnl = statistics.mean(pnls) if pnls else 0.0

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        stats.avg_win = statistics.mean(wins) if wins else 0.0
        stats.avg_loss = statistics.mean(losses) if losses else 0.0

        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        stats.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Sharpe ratio (daily, assuming trades complete within minutes)
        if len(pnls) >= 2:
            pnl_std = statistics.stdev(pnls)
            stats.sharpe_ratio = (stats.avg_pnl / pnl_std) * math.sqrt(252) if pnl_std > 0 else 0.0
            stats.sharpe_ratio = max(-10.0, min(10.0, stats.sharpe_ratio))

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        stats.max_drawdown = max_dd

        # Avg divergence on wins vs losses
        win_divs = [t.get("divergence_at_entry", 0) for t, p in zip(trades, pnls) if p > 0]
        loss_divs = [t.get("divergence_at_entry", 0) for t, p in zip(trades, pnls) if p <= 0]
        stats.avg_divergence_wins = statistics.mean(win_divs) if win_divs else 0.0
        stats.avg_divergence_losses = statistics.mean(loss_divs) if loss_divs else 0.0

        # Avg hold time
        hold_times = []
        for t in trades:
            et = t.get("entry_time")
            xt = t.get("exit_time")
            if et and xt:
                try:
                    entry = datetime.fromisoformat(et)
                    exit_ = datetime.fromisoformat(xt)
                    mins = (exit_ - entry).total_seconds() / 60.0
                    hold_times.append(mins)
                except Exception:
                    pass
        stats.avg_hold_minutes = statistics.mean(hold_times) if hold_times else 0.0

        # P&L by regime
        regimes = {}
        for t, p in zip(trades, pnls):
            regime = t.get("regime_at_entry", "unknown")
            regimes[regime] = regimes.get(regime, 0.0) + p
        stats.pnl_by_regime = regimes

        # P&L by session
        sessions = {}
        for t, p in zip(trades, pnls):
            session = t.get("session_at_entry", "unknown")
            sessions[session] = sessions.get(session, 0.0) + p
        stats.pnl_by_session = sessions

        # Calibration error: bucket posteriors by 10% bands, compare to actual win rate
        stats.calibration_error = self._compute_calibration_error(trades, pnls)

        return stats

    def _compute_calibration_error(self, trades: list, pnls: list) -> float:
        """
        Compute Brier-like calibration error.
        Groups trades by predicted probability bucket, measures how well
        predicted probability matches actual win rate.
        """
        buckets: Dict[float, List[float]] = {}
        for t, p in zip(trades, pnls):
            posterior = t.get("posterior_at_entry", 0.5)
            outcome = t.get("outcome", "YES")
            # For YES trades, win = resolution YES
            # For NO trades, win = resolution NO
            resolved = t.get("resolution_outcome")
            if resolved is None:
                continue
            if outcome == "YES":
                won = resolved
            else:
                won = not resolved

            bucket = round(posterior * 10) / 10  # 0.1 buckets
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(1.0 if won else 0.0)

        if not buckets:
            return 0.0

        errors = []
        for prob, outcomes in buckets.items():
            actual_rate = sum(outcomes) / len(outcomes)
            errors.append(abs(prob - actual_rate))

        return statistics.mean(errors) if errors else 0.0

    def get_calibration_data(self) -> List[dict]:
        """
        Return calibration data for reliability diagram.
        Each bucket: predicted_prob, actual_win_rate, count
        """
        trades = self.load_trades(status="closed")
        buckets: Dict[float, List] = {}

        for t in trades:
            posterior = t.get("posterior_at_entry", 0.5)
            resolved = t.get("resolution_outcome")
            outcome = t.get("outcome", "YES")
            if resolved is None:
                continue

            won = resolved if outcome == "YES" else not resolved
            bucket = round(posterior * 10) / 10
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(1.0 if won else 0.0)

        result = []
        for prob in sorted(buckets.keys()):
            outcomes = buckets[prob]
            result.append({
                "predicted": prob,
                "actual": sum(outcomes) / len(outcomes),
                "count": len(outcomes)
            })
        return result

    def get_pnl_curve(self) -> List[dict]:
        """Return cumulative P&L curve data."""
        trades = self.load_trades(status="closed")
        trades.sort(key=lambda t: t.get("exit_time", ""))

        curve = []
        cumulative = 0.0
        for t in trades:
            cumulative += t.get("pnl_usd", 0.0)
            curve.append({
                "time": t.get("exit_time", ""),
                "pnl": round(cumulative, 2),
                "trade_id": t.get("trade_id", ""),
            })
        return curve

    def get_recent_trades(self, n: int = 20) -> List[dict]:
        """Return the N most recent closed trades."""
        trades = self.load_trades(status=None)
        trades.sort(key=lambda t: t.get("exit_time") or t.get("entry_time", ""), reverse=True)
        return trades[:n]

    def get_stats_summary(self) -> dict:
        """Return a summary dict for the dashboard."""
        stats = self.compute_stats()
        stats_10 = self.compute_stats(n_recent=10)
        stats_50 = self.compute_stats(n_recent=50)

        return {
            "all_time": {
                "trades": stats.total_trades,
                "win_rate": round(stats.win_rate, 4),
                "total_pnl": round(stats.total_pnl, 2),
                "profit_factor": round(stats.profit_factor, 2),
                "sharpe": round(stats.sharpe_ratio, 2),
                "max_drawdown": round(stats.max_drawdown, 4),
                "avg_hold_min": round(stats.avg_hold_minutes, 1),
                "avg_div_wins": round(stats.avg_divergence_wins, 4),
                "avg_div_losses": round(stats.avg_divergence_losses, 4),
                "calibration_error": round(stats.calibration_error, 4),
            },
            "last_10": {
                "trades": stats_10.total_trades,
                "win_rate": round(stats_10.win_rate, 4),
                "total_pnl": round(stats_10.total_pnl, 2),
            },
            "last_50": {
                "trades": stats_50.total_trades,
                "win_rate": round(stats_50.win_rate, 4),
                "total_pnl": round(stats_50.total_pnl, 2),
            },
            "readiness": stats.readiness_score,
            "pnl_by_regime": stats.pnl_by_regime,
            "pnl_by_session": stats.pnl_by_session,
        }
