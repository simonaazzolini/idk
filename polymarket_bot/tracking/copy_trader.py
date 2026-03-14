"""
Copy Trading Engine — Module 4, Step E.
Evaluates whale alerts and executes copy trades with comprehensive risk filters,
cascade detection, per-wallet performance tracking, and automatic stop-loss setting.
"""
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings
from data.database import Database
from tracking.wallet_classifier import (
    TIER_LEGENDARY, TIER_INSIDER_ALERT, TIER_SMART_MONEY, TIER_1_WHALE,
    get_tier_weight, should_copy_trade,
)
from tracking.whale_tracker import WhaleAlert
from utils.helpers import clamp, now_ts, safe_div

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CopyTradeDecision:
    """Encapsulates a copy trade decision with full sizing and risk levels."""
    should_execute:  bool
    market_slug:     str
    outcome:         str
    size_usdc:       float
    entry_price:     float
    source_wallet:   str
    source_tier:     list
    insider_score:   float
    reason:          str
    ai_agreed:       bool  = True
    tag:             str   = "WHALE_COPY"
    stop_loss_price: float = 0.0    # exit trigger if position moves against us
    take_profit_price: float = 0.0  # exit trigger if position hits target
    copy_trade_id:   int   = 0      # DB row id once logged


@dataclass
class WalletPerformance:
    """
    Running performance record for a single source wallet.
    Updated every time a copy trade opened from that wallet closes.
    """
    wallet:              str
    total_copies:        int   = 0
    profitable_copies:   int   = 0
    total_pnl:           float = 0.0
    avg_pnl_per_trade:   float = 0.0
    win_rate:            float = 0.0
    avg_hold_hours:      float = 0.0
    last_copy_ts:        float = 0.0
    consecutive_losses:  int   = 0
    consecutive_wins:    int   = 0

    def record_outcome(self, pnl: float, hold_hours: float = 0.0) -> None:
        """Update statistics after a copy trade closes."""
        self.total_copies += 1
        self.total_pnl    += pnl
        if pnl > 0:
            self.profitable_copies  += 1
            self.consecutive_wins   += 1
            self.consecutive_losses  = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins    = 0
        self.win_rate         = safe_div(self.profitable_copies, self.total_copies)
        self.avg_pnl_per_trade = safe_div(self.total_pnl, self.total_copies)
        # Rolling average hold duration
        n = self.total_copies
        self.avg_hold_hours = ((self.avg_hold_hours * (n - 1)) + hold_hours) / n
        self.last_copy_ts   = now_ts()

    @property
    def is_on_losing_streak(self) -> bool:
        return self.consecutive_losses >= 3

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss (> 1 = profitable overall)."""
        if self.total_copies == 0:
            return 1.0
        wins = self.avg_pnl_per_trade * self.profitable_copies
        losses = abs(self.avg_pnl_per_trade * (self.total_copies - self.profitable_copies))
        return safe_div(wins, losses, default=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Copy Trader
# ─────────────────────────────────────────────────────────────────────────────

class CopyTrader:
    """
    Evaluates whale alerts and generates copy trade decisions.

    Key responsibilities:
      - 5 eligibility filters (resolution timing, existing positions, daily
        loss limit, recent reversal, price staleness)
      - Full copy trade size calculation with Kelly + tier + conviction + streak
        multipliers
      - AI disagreement handling: legendary/insider whales still copied at 50%
        size; other tiers rejected when AI opposes
      - Automatic stop-loss (−20%) and take-profit (+60%) price calculation
      - Cascade detection: suppress copy when 3+ wallets are entering the same
        side simultaneously (suggests herding, not information)
      - Per-wallet performance tracking with consecutive-loss safeguard
    """

    def __init__(self, db: Database, settings: Settings):
        self.db       = db
        self.settings = settings
        self._recent_copies:     dict[str, float]          = {}  # slug → ts
        self._wallet_performance: dict[str, WalletPerformance] = {}
        self._pending_alerts_buffer: list[WhaleAlert]      = []  # for cascade detection

    # ── Main evaluation entry point ────────────────────────────────────────────

    async def evaluate_alert(
        self,
        alert: WhaleAlert,
        portfolio_state: dict,
        current_market_data: Optional[dict] = None,
        ai_analysis: Optional[dict] = None,
    ) -> CopyTradeDecision:
        """
        Evaluate a whale alert and decide whether to copy trade.

        Runs 5 eligibility filters in order; returns a rejected decision as
        soon as any filter fails.  If all pass, computes size, applies AI
        disagreement logic, derives risk levels, logs to DB, and returns an
        approved decision.
        """
        tier_tags    = alert.wallet_tier if isinstance(alert.wallet_tier, list) else []
        market_slug  = alert.market_slug
        outcome      = alert.outcome

        def _reject(reason: str) -> CopyTradeDecision:
            return CopyTradeDecision(
                False, market_slug, outcome, 0.0, alert.price,
                alert.wallet_address, tier_tags, alert.insider_score, reason,
            )

        # ── Tier / win-rate eligibility ────────────────────────────────────
        if not should_copy_trade(tier_tags, alert.wallet_win_rate, alert.alert_level):
            return _reject(
                f"Wallet does not meet copy-trade threshold: tier={tier_tags}, "
                f"win_rate={alert.wallet_win_rate:.2f}"
            )

        # ── Filter 1: resolution must be ≥ 3 days away ─────────────────────
        if alert.days_until_resolution < 3:
            return _reject(
                f"Resolution too close: {alert.days_until_resolution:.1f} days"
            )

        # ── Filter 2: no existing position in this market ──────────────────
        mode = portfolio_state.get("mode", "PAPER")
        open_trades = await self.db.get_open_trades(mode)
        for ot in open_trades:
            if ot.get("market_slug") == market_slug:
                return _reject("Already have an open position in this market")

        # ── Filter 3: daily loss limit not breached ────────────────────────
        if portfolio_state.get("daily_loss_limit_breached", False):
            return _reject("Daily loss limit breached — copy trading paused")

        # ── Filter 4: wallet has not reversed position in last 2 hours ─────
        if await self._wallet_recently_reversed(alert.wallet_address, market_slug, outcome):
            return _reject("Wallet reversed its position within the last 2h")

        # ── Filter 5: market price has not moved > 5% since whale trade ────
        if current_market_data:
            current_price = float(current_market_data.get("yes_price", alert.price))
            price_move    = abs(current_price - alert.price) / max(alert.price, 0.01)
            if price_move > 0.05:
                return _reject(
                    f"Market moved {price_move:.1%} since whale trade — signal stale"
                )

        # ── Cascade detection: suppress when many wallets pile in ──────────
        self._pending_alerts_buffer.append(alert)
        if await self._is_cascade_pattern(market_slug, outcome):
            logger.info(
                "CASCADE suppression: 3+ wallets entering %s %s — skipping copy",
                outcome, market_slug,
            )
            return _reject("Cascade pattern detected — herding risk, not copying")

        # ── Per-wallet losing-streak safeguard ─────────────────────────────
        perf = self._wallet_performance.get(alert.wallet_address)
        if perf and perf.is_on_losing_streak:
            logger.warning(
                "Wallet %s is on a %d-trade losing streak — skipping copy",
                alert.wallet_address[:8], perf.consecutive_losses,
            )
            return _reject(
                f"Wallet on {perf.consecutive_losses}-trade losing streak"
            )

        # ── Compute size ───────────────────────────────────────────────────
        bankroll = portfolio_state.get("cash_balance", self.settings.budget)
        size     = self._compute_copy_size(alert, tier_tags, bankroll, current_market_data)

        # Cap to 15% of market liquidity to avoid moving the market
        if current_market_data:
            liquidity = float(current_market_data.get("liquidity", 0))
            if liquidity > 0:
                liq_cap = liquidity * self.settings.copy_trade_max_market_pct
                if size > liq_cap:
                    size = liq_cap
                    logger.info(
                        "Copy trade size capped at 15%% of liquidity: $%.2f", size
                    )

        if size < self.settings.min_position_size_usdc:
            return _reject(
                f"Computed size ${size:.2f} is below minimum "
                f"${self.settings.min_position_size_usdc:.2f}"
            )

        # ── AI disagreement handling ───────────────────────────────────────
        ai_agreed = True
        tag       = "WHALE_COPY"

        if ai_analysis:
            ai_direction = str(ai_analysis.get("recommended_outcome", "SKIP")).upper()
            if ai_direction not in ("SKIP", "UNKNOWN", "") and ai_direction != outcome:
                ai_agreed = False
                is_top_tier = TIER_LEGENDARY in tier_tags or TIER_INSIDER_ALERT in tier_tags
                if is_top_tier:
                    # Legendary / insider: copy at 50% size despite AI disagreement
                    size *= 0.50
                    tag   = "WHALE_COPY_AI_CONFLICT"
                    logger.warning(
                        "AI disagrees with %s whale on %s — halving size to $%.2f, "
                        "still executing (top tier)",
                        tier_tags, market_slug, size,
                    )
                else:
                    return _reject(
                        f"AI direction ({ai_direction}) disagrees with whale "
                        f"direction ({outcome}) — skipping non-top-tier copy"
                    )

        # ── Entry price: slight aggression for faster fill ─────────────────
        if current_market_data:
            mid = float(current_market_data.get("mid_price", alert.price))
        else:
            mid = alert.price
        entry_price = min(0.99, mid * 1.004)

        # ── Risk levels at −20% (stop loss) and configured take profit ─────
        stop_loss_pct   = self.settings.copy_trade_stop_loss_pct    # default 0.20
        take_profit_pct = self.settings.take_profit_pct             # e.g. 0.60

        if outcome == "YES":
            stop_loss_price   = max(0.01, entry_price * (1.0 - stop_loss_pct))
            take_profit_price = min(0.99, entry_price * (1.0 + take_profit_pct))
        else:
            # For NO positions, price must fall (YES price drops)
            stop_loss_price   = min(0.99, entry_price * (1.0 + stop_loss_pct))
            take_profit_price = max(0.01, entry_price * (1.0 - take_profit_pct))

        # ── Log to DB ──────────────────────────────────────────────────────
        ct_id = await self.db.insert_copy_trade({
            "timestamp":         now_ts(),
            "source_wallet":     alert.wallet_address,
            "source_tier":       json.dumps(tier_tags),
            "insider_score":     alert.insider_score,
            "market_slug":       market_slug,
            "outcome":           outcome,
            "our_size":          size,
            "our_entry_price":   entry_price,
            "stop_loss_price":   stop_loss_price,
            "take_profit_price": take_profit_price,
            "exit_price":        None,
            "pnl":               None,
            "was_profitable":    None,
            "ai_agreed":         int(ai_agreed),
            "tag":               tag,
            "conviction_mult":   alert.conviction_multiplier,
            "alert_level":       alert.alert_level,
            "whale_win_rate":    alert.wallet_win_rate,
            "close_timestamp":   None,
        })

        self._recent_copies[market_slug] = now_ts()

        logger.info(
            "COPY TRADE: %s %s @ %.4f  size=$%.2f  sl=%.4f  tp=%.4f  "
            "tier=%s  insider=%.1f  ai_agreed=%s  tag=%s",
            outcome, market_slug, entry_price, size,
            stop_loss_price, take_profit_price,
            tier_tags, alert.insider_score, ai_agreed, tag,
        )

        return CopyTradeDecision(
            should_execute=True,
            market_slug=market_slug,
            outcome=outcome,
            size_usdc=size,
            entry_price=entry_price,
            source_wallet=alert.wallet_address,
            source_tier=tier_tags,
            insider_score=alert.insider_score,
            reason="Copy trade approved",
            ai_agreed=ai_agreed,
            tag=tag,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            copy_trade_id=ct_id,
        )

    # ── Size calculation ───────────────────────────────────────────────────────

    def _compute_copy_size(
        self,
        alert: WhaleAlert,
        tier_tags: list,
        bankroll: float,
        market_data: Optional[dict],
    ) -> float:
        """
        Compute final copy trade position size using a stack of multipliers:

          1. Kelly fraction from whale win rate and current market price.
          2. Quarter-Kelly safety scaling (settings.kelly_multiplier).
          3. Confidence discount (0.7² for copy trades vs own AI trades).
          4. Tier multiplier: legendary/insider 1.5×, smart/whale 1.0×, others 0.7×.
          5. Conviction multiplier from whale trade size vs their average (capped 2×).
          6. Wallet performance multiplier: reduce size when wallet recently struggled.
          7. Hard caps: max_position_size, max_single_market_pct × bankroll.

        Returns 0.0 when Kelly fraction is not positive.
        """
        # ── 1. Kelly fraction ───────────────────────────────────────────────
        win_rate = clamp(float(alert.wallet_win_rate), 0.0, 1.0)
        price    = clamp(float(alert.price), 0.01, 0.99)
        b        = (1.0 - price) / price   # net odds per dollar wagered
        p        = win_rate
        q        = 1.0 - p
        kelly_f  = (b * p - q) / b if b > 1e-8 else 0.0

        if kelly_f <= 0:
            kelly_f = 0.01  # minimum floor: still copy a tiny amount

        # ── 2. Quarter-Kelly scaling ────────────────────────────────────────
        kelly_f *= self.settings.kelly_multiplier   # typically 0.25

        # ── 3. Copy-trade confidence discount (0.7² ≈ 0.49) ────────────────
        # We are following another trader's conviction, not our own analysis,
        # so we apply a conservative discount relative to our own AI signals.
        kelly_f *= 0.49

        # ── 4. Base size from bankroll ──────────────────────────────────────
        base_size = kelly_f * bankroll

        # ── 5. Conviction multiplier (whale trade size / their average) ─────
        # Capped at 2× so a single outsized trade can never double our risk
        conviction_mult = clamp(alert.conviction_multiplier / 3.0, 0.1, 2.0)
        base_size *= conviction_mult

        # ── 6. Tier multiplier ──────────────────────────────────────────────
        if TIER_LEGENDARY in tier_tags or TIER_INSIDER_ALERT in tier_tags:
            tier_mult = 1.5
        elif TIER_SMART_MONEY in tier_tags or TIER_1_WHALE in tier_tags:
            tier_mult = 1.0
        else:
            tier_mult = 0.7
        base_size *= tier_mult

        # ── 7. Per-wallet performance multiplier ────────────────────────────
        perf = self._wallet_performance.get(alert.wallet_address)
        if perf and perf.total_copies >= 5:
            # Scale up for consistently profitable wallets, down for losing ones
            if perf.win_rate >= 0.70 and perf.profit_factor > 1.5:
                base_size *= 1.20   # +20% for proven performers
            elif perf.win_rate < 0.45 or perf.consecutive_losses >= 2:
                base_size *= 0.60   # −40% when wallet is struggling

        # ── 8. Hard caps ────────────────────────────────────────────────────
        size = min(
            base_size,
            self.settings.max_position_size,
            bankroll * self.settings.max_single_market_pct,
        )
        return max(0.0, size)

    # ── Cascade detection ──────────────────────────────────────────────────────

    async def detect_cascade(
        self,
        alerts: list[WhaleAlert],
        window_seconds: float = 3600.0,
    ) -> bool:
        """
        Public method: check a batch of alerts for cascade patterns.

        A cascade is defined as ≥ 3 wallets independently entering the same
        outcome on the same market within `window_seconds`, where at least one
        participant is a top-tier wallet (LEGENDARY, INSIDER, or SMART_MONEY).

        Cascade copiers may be following a leader rather than acting on private
        information — this herd behaviour inflates apparent conviction and should
        suppress copy trades.

        Returns True if a cascade is detected in the alert batch.
        """
        if len(alerts) < 3:
            return False

        now   = now_ts()
        groups: dict[tuple[str, str], list[WhaleAlert]] = defaultdict(list)

        for alert in alerts:
            if now - float(alert.timestamp) <= window_seconds:
                key = (alert.market_slug, str(alert.outcome).upper())
                groups[key].append(alert)

        for (slug, outcome), group in groups.items():
            if len(group) < 3:
                continue
            # Check tier composition
            all_tiers: set[str] = set()
            for a in group:
                tiers = a.wallet_tier if isinstance(a.wallet_tier, list) else []
                all_tiers.update(tiers)
            top_tier_present = any(
                t in all_tiers
                for t in (TIER_LEGENDARY, TIER_INSIDER_ALERT, TIER_SMART_MONEY)
            )
            if top_tier_present:
                logger.info(
                    "CASCADE DETECTED: %d wallets → %s %s (tiers: %s)",
                    len(group), outcome, slug, all_tiers,
                )
                return True

        return False

    async def _is_cascade_pattern(
        self, market_slug: str, outcome: str, window_seconds: float = 3600.0
    ) -> bool:
        """
        Internal helper: check the rolling buffer for cascade involving this
        specific market/outcome.  Prunes alerts older than `window_seconds`.
        """
        now    = now_ts()
        cutoff = now - window_seconds

        # Prune old alerts from buffer
        self._pending_alerts_buffer = [
            a for a in self._pending_alerts_buffer
            if float(a.timestamp) >= cutoff
        ]

        # Count matching alerts for this market+outcome
        matching = [
            a for a in self._pending_alerts_buffer
            if a.market_slug == market_slug
            and str(a.outcome).upper() == str(outcome).upper()
        ]

        if len(matching) < 3:
            return False

        # Top-tier presence check
        all_tiers: set[str] = set()
        for a in matching:
            tiers = a.wallet_tier if isinstance(a.wallet_tier, list) else []
            all_tiers.update(tiers)

        return any(
            t in all_tiers
            for t in (TIER_LEGENDARY, TIER_INSIDER_ALERT, TIER_SMART_MONEY)
        )

    # ── Reversal detection ─────────────────────────────────────────────────────

    async def _wallet_recently_reversed(
        self, wallet: str, market_slug: str, intended_outcome: str
    ) -> bool:
        """
        Return True if the wallet traded the OPPOSITE outcome on this market
        in the last 2 hours — indicating a reversal that should block copying.
        """
        cutoff = now_ts() - 7200.0   # 2 hours
        recent = await self.db.get_recent_whale_trades(since_ts=cutoff)
        for trade in recent:
            if trade.get("wallet") != wallet:
                continue
            if trade.get("market_slug") != market_slug:
                continue
            trade_outcome = str(trade.get("outcome", "")).upper()
            if trade_outcome and trade_outcome != intended_outcome.upper():
                return True
        return False

    # ── Performance tracking ───────────────────────────────────────────────────

    async def update_copy_trade_result(
        self,
        copy_trade_id: int,
        exit_price: float,
        pnl: float,
        hold_hours: float = 0.0,
        source_wallet: Optional[str] = None,
    ) -> None:
        """
        Record the result of a copy trade after it closes.

        Updates both the DB record and the in-memory WalletPerformance tracker
        for the source wallet so future sizing decisions can use the history.
        """
        await self.db.update_copy_trade(copy_trade_id, {
            "exit_price":      exit_price,
            "pnl":             pnl,
            "was_profitable":  int(pnl > 0),
            "close_timestamp": now_ts(),
        })

        if source_wallet:
            perf = self._wallet_performance.setdefault(
                source_wallet, WalletPerformance(wallet=source_wallet)
            )
            perf.record_outcome(pnl, hold_hours)
            logger.info(
                "Copy trade result for wallet %s: pnl=$%.2f  total_copies=%d  "
                "win_rate=%.2f  consecutive_losses=%d",
                source_wallet[:8], pnl, perf.total_copies,
                perf.win_rate, perf.consecutive_losses,
            )

    def get_wallet_performance(self, wallet: str) -> Optional[WalletPerformance]:
        """Return the in-memory performance record for a wallet, or None."""
        return self._wallet_performance.get(wallet)

    def get_all_wallet_performances(self) -> list[WalletPerformance]:
        """Return all wallet performance records sorted by total PnL (descending)."""
        return sorted(
            self._wallet_performance.values(),
            key=lambda p: p.total_pnl,
            reverse=True,
        )

    def performance_summary(self) -> dict:
        """
        Return aggregated copy-trading statistics across all source wallets.
        """
        perfs  = list(self._wallet_performance.values())
        if not perfs:
            return {
                "total_wallets_tracked": 0,
                "total_copy_trades":     0,
                "aggregate_pnl":         0.0,
                "aggregate_win_rate":    0.0,
            }

        total_copies     = sum(p.total_copies for p in perfs)
        profitable_total = sum(p.profitable_copies for p in perfs)
        aggregate_pnl    = sum(p.total_pnl for p in perfs)

        return {
            "total_wallets_tracked": len(perfs),
            "total_copy_trades":     total_copies,
            "aggregate_pnl":         round(aggregate_pnl, 2),
            "aggregate_win_rate":    round(safe_div(profitable_total, total_copies), 4),
            "best_wallet":           max(perfs, key=lambda p: p.total_pnl).wallet if perfs else None,
            "worst_wallet":          min(perfs, key=lambda p: p.total_pnl).wallet if perfs else None,
        }


