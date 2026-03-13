"""
Pre-trade safety checks and rate limiting — Module 10.
All checks that must pass before any order is placed.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import Settings
from utils.helpers import now_ts

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    passed: bool
    reason: str
    checks_run: list[str]
    checks_failed: list[str]


class SafetyChecker:
    """
    Enforces all pre-trade safety checks.
    All checks must pass for any order to be placed in live mode.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._order_timestamps: list[float] = []

    async def check_all(
        self,
        mode: str,
        market: dict,
        size_usdc: float,
        current_price: float,
        signal_price: float,
        portfolio_state: dict,
        orderbook: Optional[dict] = None,
        usdc_balance: float = 0.0,
    ) -> SafetyCheckResult:
        """
        Run all safety checks. In PAPER mode most checks still run
        (to simulate real behavior) but balance checks use simulated balance.
        """
        checks_run: list[str] = []
        checks_failed: list[str] = []

        # ── Check 1: USDC balance ─────────────────────────────────────────────
        checks_run.append("usdc_balance")
        if mode == "LIVE":
            buffer = size_usdc * 0.05  # 5% buffer for fees
            if usdc_balance < size_usdc + buffer:
                checks_failed.append("usdc_balance")
                return SafetyCheckResult(
                    False,
                    f"Insufficient USDC: have ${usdc_balance:.2f}, need ${size_usdc + buffer:.2f}",
                    checks_run, checks_failed
                )

        # ── Check 2: Market still active ─────────────────────────────────────
        checks_run.append("market_active")
        if not market.get("active", True) or market.get("closed", False):
            checks_failed.append("market_active")
            return SafetyCheckResult(
                False, "Market is not active",
                checks_run, checks_failed
            )

        # ── Check 3: Price hasn't moved > 3% since signal ──────────────────
        checks_run.append("price_movement")
        if signal_price > 0:
            price_drift = abs(current_price - signal_price) / max(signal_price, 0.001)
            if price_drift > 0.03:
                checks_failed.append("price_movement")
                return SafetyCheckResult(
                    False,
                    f"Price moved {price_drift:.1%} since signal (threshold 3%)",
                    checks_run, checks_failed
                )

        # ── Check 4: Orderbook depth supports size ────────────────────────────
        checks_run.append("orderbook_depth")
        if orderbook and mode == "LIVE":
            asks = orderbook.get("asks", [])
            total_ask_depth = sum(
                float(a.get("price", 0)) * float(a.get("size", 0))
                for a in asks[:10]  # top 10 levels
            )
            if total_ask_depth < size_usdc * 2:  # need 2x depth for our size
                checks_failed.append("orderbook_depth")
                return SafetyCheckResult(
                    False,
                    f"Insufficient orderbook depth: ${total_ask_depth:.0f} < ${size_usdc * 2:.0f}",
                    checks_run, checks_failed
                )

        # ── Check 5: No duplicate order ───────────────────────────────────────
        checks_run.append("no_duplicate")
        market_slug = market.get("slug") or market.get("conditionId", "")
        open_positions = portfolio_state.get("open_positions", [])
        for pos in open_positions:
            if pos.get("market_slug") == market_slug:
                checks_failed.append("no_duplicate")
                return SafetyCheckResult(
                    False, f"Already have open position in {market_slug}",
                    checks_run, checks_failed
                )

        # ── Check 6: Total exposure under max ─────────────────────────────────
        checks_run.append("total_exposure")
        total_budget = portfolio_state.get("total_budget", self.settings.budget)
        current_exposure = portfolio_state.get("current_exposure", 0.0)
        if (current_exposure + size_usdc) > total_budget * self.settings.max_total_exposure_pct:
            checks_failed.append("total_exposure")
            return SafetyCheckResult(
                False,
                f"Would exceed max exposure: {(current_exposure + size_usdc) / total_budget:.1%} > {self.settings.max_total_exposure_pct:.0%}",
                checks_run, checks_failed
            )

        # ── Check 7: Daily loss limit ─────────────────────────────────────────
        checks_run.append("daily_loss_limit")
        if portfolio_state.get("daily_loss_limit_breached", False):
            checks_failed.append("daily_loss_limit")
            return SafetyCheckResult(
                False, "Daily loss limit has been breached",
                checks_run, checks_failed
            )

        # ── Check 8: Max drawdown ─────────────────────────────────────────────
        checks_run.append("max_drawdown")
        if portfolio_state.get("max_drawdown_breached", False):
            checks_failed.append("max_drawdown")
            return SafetyCheckResult(
                False, "Maximum drawdown limit has been reached",
                checks_run, checks_failed
            )

        # ── Check 9: Open positions count ─────────────────────────────────────
        checks_run.append("position_count")
        if len(open_positions) >= self.settings.max_open_positions:
            checks_failed.append("position_count")
            return SafetyCheckResult(
                False, f"Max open positions ({self.settings.max_open_positions}) reached",
                checks_run, checks_failed
            )

        # ── Check 10: API rate limit ──────────────────────────────────────────
        checks_run.append("api_rate_limit")
        if not self._check_rate_limit():
            checks_failed.append("api_rate_limit")
            return SafetyCheckResult(
                False, f"API rate limit: max {self.settings.max_orders_per_minute} orders/min",
                checks_run, checks_failed
            )

        # ── Check 11: Size minimum ────────────────────────────────────────────
        checks_run.append("size_minimum")
        if size_usdc < self.settings.min_position_size_usdc:
            checks_failed.append("size_minimum")
            return SafetyCheckResult(
                False, f"Size ${size_usdc:.2f} below minimum ${self.settings.min_position_size_usdc}",
                checks_run, checks_failed
            )

        return SafetyCheckResult(
            True, "All safety checks passed",
            checks_run, []
        )

    def _check_rate_limit(self) -> bool:
        """Check if we're within the API rate limit."""
        now = now_ts()
        cutoff = now - 60.0
        self._order_timestamps = [t for t in self._order_timestamps if t > cutoff]
        if len(self._order_timestamps) >= self.settings.max_orders_per_minute:
            return False
        return True

    def record_order(self) -> None:
        """Record that an order was placed (for rate limiting)."""
        self._order_timestamps.append(now_ts())

    def orders_in_last_minute(self) -> int:
        now = now_ts()
        cutoff = now - 60.0
        return sum(1 for t in self._order_timestamps if t > cutoff)


class EmergencyStop:
    """
    Monitors for emergency conditions that should halt all trading.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._halted = False
        self._halt_reason = ""
        self._halt_timestamp: Optional[float] = None

    def check_emergency(
        self,
        portfolio_state: dict,
        mode: str,
    ) -> tuple[bool, str]:
        """
        Returns (should_halt, reason).
        """
        if self._halted:
            return True, self._halt_reason

        total_budget = portfolio_state.get("total_budget", self.settings.budget)
        if total_budget <= 0:
            return False, ""

        # Daily P&L check
        daily_pnl = portfolio_state.get("daily_pnl", 0.0)
        daily_loss_threshold = -total_budget * self.settings.daily_loss_limit_pct
        if daily_pnl < daily_loss_threshold:
            reason = f"Daily loss limit: {daily_pnl:+.2f} < {daily_loss_threshold:.2f}"
            self._trigger_halt(reason)
            return True, reason

        # Max drawdown check
        peak_value = portfolio_state.get("peak_value", total_budget)
        current_value = portfolio_state.get("total_portfolio_value", total_budget)
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
            if drawdown > self.settings.max_drawdown_limit_pct:
                reason = f"Max drawdown breached: {drawdown:.1%} > {self.settings.max_drawdown_limit_pct:.0%}"
                self._trigger_halt(reason)
                return True, reason

        return False, ""

    def _trigger_halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason
        self._halt_timestamp = now_ts()
        logger.critical("EMERGENCY STOP TRIGGERED: %s", reason)

    def reset(self) -> None:
        """Allow manual reset of emergency stop (e.g. next trading day)."""
        self._halted = False
        self._halt_reason = ""
        self._halt_timestamp = None
        logger.warning("Emergency stop reset")

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason
