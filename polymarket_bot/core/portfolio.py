"""
Portfolio & Risk Manager — Module 11.
Real-time portfolio state, risk enforcement, position management.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings
from data.database import Database
from execution.safety import EmergencyStop
from utils.helpers import max_drawdown, now_ts, pct_change, safe_div, sharpe_ratio

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    total_budget: float = 1000.0
    cash_balance: float = 1000.0
    total_position_value: float = 0.0
    total_portfolio_value: float = 1000.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    current_exposure: float = 0.0
    peak_value: float = 1000.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    daily_start_value: float = 1000.0
    positions_by_category: dict = field(default_factory=dict)
    open_positions: list = field(default_factory=list)
    open_positions_count: int = 0
    daily_loss_limit_breached: bool = False
    max_drawdown_breached: bool = False
    win_rate: float = 0.0
    mode: str = "PAPER"

    def to_dict(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "cash_balance": self.cash_balance,
            "total_position_value": self.total_position_value,
            "total_portfolio_value": self.total_portfolio_value,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_pnl": self.total_pnl,
            "roi_pct": self.roi_pct,
            "win_rate": self.win_rate,
            "current_exposure": self.current_exposure,
            "peak_value": self.peak_value,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "open_positions_count": self.open_positions_count,
            "daily_loss_limit_breached": self.daily_loss_limit_breached,
            "max_drawdown_breached": self.max_drawdown_breached,
            "mode": self.mode,
            "positions_by_category": self.positions_by_category,
        }


class PortfolioManager:
    """
    Manages portfolio state, risk limits, and position lifecycle.
    """

    def __init__(self, db: Database, settings: Settings, mode: str, budget: float):
        self.db = db
        self.settings = settings
        self.state = PortfolioState(
            total_budget=budget,
            cash_balance=budget,
            total_portfolio_value=budget,
            daily_start_value=budget,
            peak_value=budget,
            mode=mode,
        )
        self.emergency_stop = EmergencyStop(settings)
        self._portfolio_value_history: list[float] = [budget]
        self._daily_pnl_series: list[float] = []
        # Set to an APIServer instance to enable live WebSocket pushes after
        # each trade.  Left as None when running without the API server.
        self.api_server = None

    async def initialize(self) -> None:
        """Load state from database on startup."""
        open_trades = await self.db.get_open_trades(self.state.mode)
        self.state.open_positions = open_trades
        self.state.open_positions_count = len(open_trades)

        # Compute current exposure
        deployed = sum(float(t.get("size_usdc", 0)) for t in open_trades)
        self.state.current_exposure = deployed
        self.state.cash_balance = max(0.0, self.state.total_budget - deployed)

        # Restore realized P&L from ALL closed trades so it survives restarts.
        # Without this, total_realized_pnl resets to 0 every time the bot starts.
        closed = await self.db.get_all_closed_trades(self.state.mode)
        self.state.total_realized_pnl = sum(
            float(t.get("pnl") or 0) for t in closed if t.get("pnl") is not None
        )
        self.state.total_pnl = self.state.total_realized_pnl  # unrealized = 0 at init
        self.state.roi_pct = (
            safe_div(self.state.total_pnl, self.state.total_budget) * 100
        )

        logger.info(
            "Portfolio initialized: budget=$%.2f, exposure=$%.2f, positions=%d, realized_pnl=$%.2f",
            self.state.total_budget, deployed, len(open_trades), self.state.total_realized_pnl,
        )

    async def on_trade_opened(self, trade_id: int, size_usdc: float, fill_price: float) -> None:
        """Update portfolio state after a new trade is opened."""
        self.state.cash_balance = max(0.0, self.state.cash_balance - size_usdc)
        self.state.current_exposure += size_usdc
        # Full recalculation from DB so every derived metric is consistent.
        await self._recalculate_after_trade()
        logger.info(
            "Trade opened: $%.2f | cash=$%.2f | exposure=$%.2f | roi=%.2f%% | win_rate=%.1f%%",
            size_usdc, self.state.cash_balance, self.state.current_exposure,
            self.state.roi_pct, self.state.win_rate * 100,
        )

    async def on_trade_closed(
        self,
        trade_id: int,
        pnl: float,
        size_usdc: float,
    ) -> None:
        """Update portfolio state after a trade is closed."""
        self.state.total_realized_pnl += pnl
        self.state.total_pnl = self.state.total_realized_pnl + self.state.total_unrealized_pnl
        self.state.cash_balance += size_usdc + pnl
        self.state.current_exposure = max(0.0, self.state.current_exposure - size_usdc)
        self.state.total_portfolio_value = self.state.cash_balance + self.state.total_position_value
        self.state.roi_pct = safe_div(self.state.total_pnl, self.state.total_budget) * 100

        # Update daily PnL
        self.state.daily_pnl += pnl

        # Peak tracking for drawdown
        if self.state.total_portfolio_value > self.state.peak_value:
            self.state.peak_value = self.state.total_portfolio_value
        if self.state.peak_value > 0:
            self.state.current_drawdown = (
                (self.state.peak_value - self.state.total_portfolio_value) / self.state.peak_value
            )
            if self.state.current_drawdown > self.state.max_drawdown:
                self.state.max_drawdown = self.state.current_drawdown

        # Full recalculation reloads open positions, updates win_rate, saves
        # the snapshot, and pushes the state to the dashboard.
        await self._recalculate_after_trade()

    async def update_mark_to_market(
        self,
        market_prices: dict[str, float],
    ) -> None:
        """
        Update unrealized PnL for all open positions using current prices.
        market_prices: {token_id or market_slug → current_price}
        """
        total_position_value = 0.0
        total_unrealized = 0.0
        positions_to_exit: list[dict] = []

        open_trades = await self.db.get_open_trades(self.state.mode)

        for trade in open_trades:
            slug = trade.get("market_slug", "")
            outcome = str(trade.get("outcome", "YES")).upper()
            entry_price = float(trade.get("fill_price") or trade.get("price") or 0.5)
            shares = float(trade.get("shares") or 0)
            size_usdc = float(trade.get("size_usdc") or 0)

            # Try to get current price
            current_price = (
                market_prices.get(slug) or
                market_prices.get(f"{slug}_{outcome}") or
                entry_price
            )

            current_value = shares * current_price
            unrealized = current_value - size_usdc
            total_position_value += current_value
            total_unrealized += unrealized

            # Check stop-loss and take-profit
            pnl_pct = safe_div(unrealized, size_usdc)
            should_exit = False
            exit_reason = ""

            if pnl_pct <= -self.settings.stop_loss_pct:
                should_exit = True
                exit_reason = f"STOP_LOSS ({pnl_pct:.1%})"
            elif pnl_pct >= self.settings.take_profit_pct:
                should_exit = True
                exit_reason = f"TAKE_PROFIT ({pnl_pct:.1%})"

            if should_exit:
                positions_to_exit.append({
                    "trade_id": trade["id"],
                    "token_id": trade.get("token_id", ""),
                    "shares": shares,
                    "current_price": current_price,
                    "exit_reason": exit_reason,
                    "entry_price": entry_price,
                    "size_usdc": size_usdc,
                })

        self.state.total_position_value = total_position_value
        self.state.total_unrealized_pnl = total_unrealized
        self.state.total_pnl = self.state.total_realized_pnl + total_unrealized
        self.state.total_portfolio_value = self.state.cash_balance + total_position_value
        self.state.roi_pct = safe_div(self.state.total_pnl, self.state.total_budget) * 100

        # Check emergency conditions
        halted, halt_reason = self.emergency_stop.check_emergency(
            self.state.to_dict(), self.state.mode
        )
        if halted:
            self.state.daily_loss_limit_breached = True
            self.state.max_drawdown_breached = True
            logger.critical("TRADING HALTED: %s", halt_reason)

        # Update peak and drawdown
        if self.state.total_portfolio_value > self.state.peak_value:
            self.state.peak_value = self.state.total_portfolio_value
        if self.state.peak_value > 0:
            self.state.current_drawdown = max(
                0.0,
                (self.state.peak_value - self.state.total_portfolio_value) / self.state.peak_value
            )
            self.state.max_drawdown = max(self.state.max_drawdown, self.state.current_drawdown)

        # Check daily loss limit
        daily_loss_threshold = -self.state.total_budget * self.settings.daily_loss_limit_pct
        if self.state.daily_pnl < daily_loss_threshold:
            self.state.daily_loss_limit_breached = True
            logger.warning(
                "Daily loss limit breached: $%.2f < $%.2f",
                self.state.daily_pnl, daily_loss_threshold
            )

        self._portfolio_value_history.append(self.state.total_portfolio_value)

        return positions_to_exit

    async def on_new_day(self) -> None:
        """Reset daily counters at UTC midnight."""
        self._daily_pnl_series.append(self.state.daily_pnl)
        self.state.daily_pnl = 0.0
        self.state.daily_start_value = self.state.total_portfolio_value
        self.state.daily_loss_limit_breached = False

        if not self.emergency_stop.is_halted:
            self.emergency_stop.reset()

        # Save daily performance
        await self._save_daily_performance()

        logger.info("New trading day started. Portfolio value: $%.2f", self.state.total_portfolio_value)

    async def _recalculate_after_trade(
        self, market_prices: Optional[dict] = None
    ) -> None:
        """
        Recompute all portfolio metrics from the database after a trade is
        inserted, then persist a snapshot and push the updated state to the
        dashboard over WebSocket.

        Steps:
            1. Recalculate total_position_value from all open positions.
            2. Update unrealized_pnl for each position using current_price
               (falls back to entry price when no live quote is available).
            3. Update roi_pct = total_pnl / total_budget * 100.
            4. Update win_rate from all closed trades in the trades table.
            5. Save snapshot + push to dashboard.

        Args:
            market_prices: optional {market_slug → current_price} mapping.
                           Pass bot.portfolio_manager's latest price map for
                           accurate mark-to-market; omit to use entry prices.
        """
        mode = self.state.mode
        prices = market_prices or {}

        # ── 1 & 2. Position value + unrealized PnL ───────────────────────────
        open_trades = await self.db.get_open_trades(mode)
        total_position_value = 0.0
        total_unrealized = 0.0

        for t in open_trades:
            slug = t.get("market_slug", "")
            outcome = str(t.get("outcome", "YES")).upper()
            entry_price = float(t.get("fill_price") or t.get("price") or 0.5)
            shares = float(t.get("shares") or 0)
            size_usdc = float(t.get("size_usdc") or 0)

            # Live price if available; at entry the value equals cost (unrealized=0)
            current_price = (
                prices.get(slug)
                or prices.get(f"{slug}_{outcome}")
                or entry_price
            )
            current_value = shares * current_price
            total_position_value += current_value
            total_unrealized += current_value - size_usdc

        self.state.total_position_value = total_position_value
        self.state.open_positions = open_trades
        self.state.open_positions_count = len(open_trades)
        self.state.total_unrealized_pnl = total_unrealized

        # ── 3. ROI ────────────────────────────────────────────────────────────
        self.state.total_pnl = self.state.total_realized_pnl + total_unrealized
        self.state.total_portfolio_value = (
            self.state.cash_balance + total_position_value
        )
        self.state.roi_pct = (
            safe_div(self.state.total_pnl, self.state.total_budget) * 100
        )

        # ── 4. Win rate from all closed trades ────────────────────────────────
        closed = await self.db.get_all_closed_trades(mode)
        pnls = [float(t.get("pnl") or 0) for t in closed if t.get("pnl") is not None]
        self.state.win_rate = (
            safe_div(sum(1 for p in pnls if p > 0), len(pnls)) if pnls else 0.0
        )

        # ── 5. Persist + push ─────────────────────────────────────────────────
        await self._save_snapshot()
        await self._push_to_dashboard()
        logger.debug(
            "Portfolio recalculated | pos_value=$%.2f unrealized=$%.2f "
            "roi=%.2f%% win_rate=%.1f%% positions=%d",
            total_position_value, total_unrealized,
            self.state.roi_pct, self.state.win_rate * 100,
            len(open_trades),
        )

    async def _push_to_dashboard(self) -> None:
        """
        Broadcast the current portfolio state over WebSocket via the API server.
        No-ops silently when api_server has not been wired in.
        """
        if self.api_server is None:
            return
        try:
            await self.api_server.broadcast_portfolio_update(self.state.to_dict())
        except Exception as exc:
            logger.debug("Portfolio dashboard push failed: %s", exc)

    async def _save_snapshot(self) -> None:
        """Save current portfolio state to the database."""
        await self.db.insert_portfolio_snapshot({
            "timestamp": now_ts(),
            "mode": self.state.mode,
            "cash_balance": self.state.cash_balance,
            "position_value": self.state.total_position_value,
            "total_value": self.state.total_portfolio_value,
            "realized_pnl": self.state.total_realized_pnl,
            "unrealized_pnl": self.state.total_unrealized_pnl,
            "roi_pct": self.state.roi_pct,
            "drawdown": self.state.current_drawdown,
            "open_positions_count": self.state.open_positions_count,
        })

    async def _save_daily_performance(self) -> None:
        """Save daily performance summary."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # day_start_ts is the Unix timestamp of UTC midnight for today.
        today_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_ts = today_dt.timestamp()
        closed_today = await self.db.get_trades_in_range(
            mode=self.state.mode,
            start_ts=day_start_ts,
            end_ts=now_ts(),
        )
        pnls = [float(t.get("pnl") or 0) for t in closed_today if t.get("pnl") is not None]

        from utils.helpers import profit_factor as pf_fn, win_rate as wr_fn
        wr = wr_fn(pnls)
        pf = pf_fn(pnls)
        best_trade = max(pnls) if pnls else 0.0
        worst_trade = min(pnls) if pnls else 0.0

        await self.db.upsert_performance_daily({
            "date": today,
            "mode": self.state.mode,
            "starting_value": self.state.daily_start_value,
            "ending_value": self.state.total_portfolio_value,
            "pnl": self.state.daily_pnl,
            "win_rate": wr,
            "trades_count": len(closed_today),
            "avg_edge": 0.0,
            "sharpe_daily": 0.0,
            "max_drawdown": self.state.max_drawdown,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "whale_copy_pnl": sum(
                float(t.get("pnl") or 0) for t in closed_today
                if t.get("is_whale_copy")
            ),
            "ai_only_pnl": sum(
                float(t.get("pnl") or 0) for t in closed_today
                if not t.get("is_whale_copy")
            ),
            "arb_pnl": 0.0,
        })

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is currently allowed."""
        if self.state.daily_loss_limit_breached:
            return False, "Daily loss limit breached"
        if self.state.max_drawdown_breached:
            return False, "Max drawdown breached"
        if self.emergency_stop.is_halted:
            return False, self.emergency_stop.halt_reason
        if self.state.open_positions_count >= self.settings.max_open_positions:
            return False, f"Max positions ({self.settings.max_open_positions}) reached"
        return True, "OK"

    def get_category_exposure(self, category: str) -> float:
        """Get current exposure in a specific category."""
        return self.state.positions_by_category.get(category, 0.0)

    def sharpe_ratio(self) -> float:
        """Compute session-level Sharpe ratio from value history."""
        if len(self._portfolio_value_history) < 2:
            return 0.0
        returns = [
            pct_change(self._portfolio_value_history[i], self._portfolio_value_history[i+1])
            for i in range(len(self._portfolio_value_history) - 1)
        ]
        return sharpe_ratio(returns, annualize_factor=365.0)
