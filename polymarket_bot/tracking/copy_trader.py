"""
Copy Trading Engine — Module 4, Step E.
Evaluates whale alerts and executes copy trades with risk filters.
"""
import asyncio
import json
import logging
from typing import Optional

from config.settings import Settings
from data.database import Database
from tracking.wallet_classifier import (
    TIER_LEGENDARY, TIER_INSIDER_ALERT, TIER_SMART_MONEY, TIER_1_WHALE,
    get_tier_weight, should_copy_trade,
)
from tracking.whale_tracker import WhaleAlert
from utils.helpers import now_ts, safe_div

logger = logging.getLogger(__name__)


class CopyTradeDecision:
    """Encapsulates a copy trade decision."""
    def __init__(
        self,
        should_execute: bool,
        market_slug: str,
        outcome: str,
        size_usdc: float,
        entry_price: float,
        source_wallet: str,
        source_tier: list[str],
        insider_score: float,
        reason: str,
        ai_agreed: bool = True,
        tag: str = "WHALE_COPY",
    ):
        self.should_execute = should_execute
        self.market_slug = market_slug
        self.outcome = outcome
        self.size_usdc = size_usdc
        self.entry_price = entry_price
        self.source_wallet = source_wallet
        self.source_tier = source_tier
        self.insider_score = insider_score
        self.reason = reason
        self.ai_agreed = ai_agreed
        self.tag = tag


class CopyTrader:
    """Evaluates whale alerts and generates copy trade decisions."""

    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self._recent_copies: dict[str, float] = {}  # market_slug → timestamp

    async def evaluate_alert(
        self,
        alert: WhaleAlert,
        portfolio_state: dict,
        current_market_data: Optional[dict] = None,
        ai_analysis: Optional[dict] = None,
    ) -> CopyTradeDecision:
        """
        Evaluate a whale alert and decide whether to copy trade.
        Returns a CopyTradeDecision with full reasoning.
        """
        tier_tags = alert.wallet_tier if isinstance(alert.wallet_tier, list) else []
        market_slug = alert.market_slug
        outcome = alert.outcome

        # ── Basic eligibility filters ──────────────────────────────────────
        if not should_copy_trade(tier_tags, alert.wallet_win_rate, alert.alert_level):
            return CopyTradeDecision(
                False, market_slug, outcome, 0, alert.price,
                alert.wallet_address, tier_tags, alert.insider_score,
                f"Wallet tier/win_rate does not qualify: {tier_tags}"
            )

        # Filter 1: resolution must be >= 3 days away
        if alert.days_until_resolution < 3:
            return CopyTradeDecision(
                False, market_slug, outcome, 0, alert.price,
                alert.wallet_address, tier_tags, alert.insider_score,
                f"Resolution too close: {alert.days_until_resolution:.1f} days"
            )

        # Filter 2: check no existing position
        open_trades = await self.db.get_open_trades(portfolio_state.get("mode", "PAPER"))
        for ot in open_trades:
            if ot["market_slug"] == market_slug:
                return CopyTradeDecision(
                    False, market_slug, outcome, 0, alert.price,
                    alert.wallet_address, tier_tags, alert.insider_score,
                    "Already have position in this market"
                )

        # Filter 3: daily loss limit
        if portfolio_state.get("daily_loss_limit_breached", False):
            return CopyTradeDecision(
                False, market_slug, outcome, 0, alert.price,
                alert.wallet_address, tier_tags, alert.insider_score,
                "Daily loss limit breached"
            )

        # Filter 4: wallet hasn't reversed in last 2 hours
        if await self._wallet_recently_reversed(alert.wallet_address, market_slug, outcome):
            return CopyTradeDecision(
                False, market_slug, outcome, 0, alert.price,
                alert.wallet_address, tier_tags, alert.insider_score,
                "Wallet reversed position in last 2h"
            )

        # Filter 5: market price hasn't moved > 5% since their trade
        if current_market_data:
            current_price = current_market_data.get("yes_price", alert.price)
            price_move = abs(current_price - alert.price) / max(alert.price, 0.01)
            if price_move > 0.05:
                return CopyTradeDecision(
                    False, market_slug, outcome, 0, alert.price,
                    alert.wallet_address, tier_tags, alert.insider_score,
                    f"Market moved {price_move:.1%} since whale trade"
                )

        # ── Compute copy trade size ───────────────────────────────────────
        bankroll = portfolio_state.get("cash_balance", self.settings.budget)
        size = self._compute_copy_size(
            alert, tier_tags, bankroll, current_market_data
        )

        # Filter 6: our position would be <= 15% of market liquidity
        if current_market_data:
            liquidity = float(current_market_data.get("liquidity", 0))
            if liquidity > 0 and size > liquidity * self.settings.copy_trade_max_market_pct:
                size = liquidity * self.settings.copy_trade_max_market_pct
                logger.info("Copy trade size capped at 15%% of liquidity: $%.2f", size)

        if size < self.settings.min_position_size_usdc:
            return CopyTradeDecision(
                False, market_slug, outcome, 0, alert.price,
                alert.wallet_address, tier_tags, alert.insider_score,
                f"Computed size ${size:.2f} below minimum"
            )

        # ── AI agreement check ────────────────────────────────────────────
        ai_agreed = True
        tag = "WHALE_COPY"
        if ai_analysis:
            ai_direction = ai_analysis.get("recommended_outcome", "SKIP")
            if ai_direction != "SKIP" and ai_direction != outcome:
                ai_agreed = False
                # If legendary or insider: still copy but reduce size by 50%
                if TIER_LEGENDARY in tier_tags or TIER_INSIDER_ALERT in tier_tags:
                    size *= 0.50
                    tag = "WHALE_COPY_AI_CONFLICT"
                    logger.warning(
                        "AI disagrees with %s whale on %s. Size halved. Still executing.",
                        tier_tags, market_slug
                    )
                else:
                    # Non-legendary: don't copy when AI disagrees
                    return CopyTradeDecision(
                        False, market_slug, outcome, 0, alert.price,
                        alert.wallet_address, tier_tags, alert.insider_score,
                        "AI disagrees with whale direction"
                    )

        # ── Entry price: mid + 0.4% for faster fill ───────────────────────
        if current_market_data:
            mid = float(current_market_data.get("mid_price", alert.price))
        else:
            mid = alert.price
        entry_price = min(0.99, mid * 1.004)

        # Log copy trade to DB
        ct_id = await self.db.insert_copy_trade({
            "timestamp": now_ts(),
            "source_wallet": alert.wallet_address,
            "source_tier": json.dumps(tier_tags),
            "insider_score": alert.insider_score,
            "market_slug": market_slug,
            "outcome": outcome,
            "our_size": size,
            "our_entry_price": entry_price,
            "exit_price": None,
            "pnl": None,
            "was_profitable": None,
            "ai_agreed": int(ai_agreed),
            "close_timestamp": None,
        })

        self._recent_copies[market_slug] = now_ts()
        logger.info(
            "COPY TRADE: %s %s @ %.4f size=$%.2f (tier=%s, insider=%.1f, ai_agreed=%s)",
            outcome, market_slug, entry_price, size, tier_tags, alert.insider_score, ai_agreed
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
        )

    def _compute_copy_size(
        self,
        alert: WhaleAlert,
        tier_tags: list[str],
        bankroll: float,
        market_data: Optional[dict],
    ) -> float:
        """Compute final copy trade size using Kelly + tier + conviction multipliers."""
        # Base Kelly
        wr = alert.wallet_win_rate
        price = alert.price
        if price <= 0 or price >= 1:
            price = 0.5
        b = (1 - price) / price  # net odds
        p = wr
        q = 1 - p
        kelly_f = (b * p - q) / b if b > 0 else 0.0

        if kelly_f <= 0:
            kelly_f = 0.01  # minimum if Kelly is negative (still copy small amount)

        # Scale by quarter Kelly
        kelly_f *= self.settings.kelly_multiplier

        # Confidence scale: use 0.7 as default confidence for copy trades
        kelly_f *= 0.7 ** 2

        # Base size from bankroll
        base_size = kelly_f * bankroll

        # Conviction multiplier (capped at 2x)
        conviction_mult = min(alert.conviction_multiplier / 3.0, 2.0)
        base_size *= conviction_mult

        # Tier multiplier
        if TIER_LEGENDARY in tier_tags or TIER_INSIDER_ALERT in tier_tags:
            tier_mult = 1.5
        elif TIER_SMART_MONEY in tier_tags or TIER_1_WHALE in tier_tags:
            tier_mult = 1.0
        else:
            tier_mult = 0.7
        base_size *= tier_mult

        # Apply caps
        size = min(
            base_size,
            self.settings.max_position_size,
            bankroll * self.settings.max_single_market_pct,
        )
        return max(0.0, size)

    async def _wallet_recently_reversed(
        self, wallet: str, market_slug: str, intended_outcome: str
    ) -> bool:
        """Check if wallet reversed position in last 2 hours."""
        cutoff = now_ts() - 7200  # 2 hours ago
        recent = await self.db.get_recent_whale_trades(since_ts=cutoff)
        for trade in recent:
            if trade.get("wallet") != wallet:
                continue
            if trade.get("market_slug") != market_slug:
                continue
            # If they traded the opposite side, that's a reversal
            trade_outcome = str(trade.get("outcome", "")).upper()
            if trade_outcome and trade_outcome != intended_outcome.upper():
                return True
        return False

    async def update_copy_trade_result(
        self, copy_trade_id: int, exit_price: float, pnl: float
    ) -> None:
        """Record the result of a copy trade after it closes."""
        await self.db.update_copy_trade(copy_trade_id, {
            "exit_price": exit_price,
            "pnl": pnl,
            "was_profitable": int(pnl > 0),
            "close_timestamp": now_ts(),
        })
