"""
Arbitrage Detection Engine — Module 7.
All 5 arbitrage types: YES/NO sum, multi-outcome, implied contradiction,
spread capture, and time decay.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from analysis.orderbook import OrderbookMetrics
from config.settings import Settings
from utils.helpers import now_ts, safe_div

logger = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    arb_type: str           # TYPE_1 through TYPE_5
    market_slug: str
    description: str
    profit_pct: float       # Expected profit as fraction
    profit_usdc: float      # Expected dollar profit
    max_size: float         # Max capital to deploy
    legs: list[dict]        # Each leg: {token_id, outcome, side, size, price}
    urgency: float          # 0-10, higher = execute sooner
    arb_score: float        # Signal score for this arb
    detected_at: float = field(default_factory=now_ts)
    confidence: float = 1.0


class ArbitrageDetector:
    """
    Detects all 5 types of arbitrage opportunities on Polymarket.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    # ── TYPE 1: YES/NO Sum Arbitrage ──────────────────────────────────────────

    def detect_yes_no_arb(
        self,
        market: dict,
        yes_book: Optional[OrderbookMetrics],
        no_book: Optional[OrderbookMetrics],
        paper_mode: bool = False,
    ) -> Optional[ArbOpportunity]:
        """
        Buy both YES and NO simultaneously if yes_ask + no_ask < 1.0.
        Threshold: arb_min_profit_pct (0.5% live, 0.3% paper).
        """
        slug = market.get("slug") or market.get("conditionId", "unknown")

        if not yes_book or not no_book:
            logger.debug("ARB_SKIP %s: missing orderbook (yes=%s no=%s)",
                         slug[:30], bool(yes_book), bool(no_book))
            return None

        yes_ask = yes_book.best_ask
        no_ask = no_book.best_ask

        if yes_ask <= 0 or no_ask <= 0:
            logger.debug("ARB_SKIP %s: zero ask (yes_ask=%.4f no_ask=%.4f)",
                         slug[:30], yes_ask, no_ask)
            return None

        yes_no_sum = yes_ask + no_ask
        arb_gap = 1.0 - yes_no_sum
        min_pct = (getattr(self.settings, 'arb_paper_min_profit_pct', 0.003)
                   if paper_mode else self.settings.arb_min_profit_pct)

        if arb_gap <= min_pct:
            logger.debug("ARB_SKIP %s: gap=%.4f (%.2f%%) below min=%.4f (%.2f%%) | sum=%.4f",
                         slug[:30], arb_gap, arb_gap*100, min_pct, min_pct*100, yes_no_sum)
            return None

        profit_pct = arb_gap
        # Max size limited by depth — use at least $20 minimum per side
        yes_depth = max(yes_book.ask_depth_1pct, 20.0)
        no_depth = max(no_book.ask_depth_1pct, 20.0)
        max_size = min(yes_depth, no_depth, 500.0)
        max_size_usdc = max_size

        profit_usdc = max_size_usdc * profit_pct
        min_profit = (0.10 if paper_mode else self.settings.arb_min_profit_usdc)
        if profit_usdc < min_profit:
            logger.debug("ARB_SKIP %s: profit_usdc=%.2f below min=%.2f (size=%.1f pct=%.4f)",
                         slug[:30], profit_usdc, min_profit, max_size, profit_pct)
            return None

        slug = market.get("slug") or market.get("conditionId", "unknown")
        yes_token = market.get("yes_token_id") or (market.get("clobTokenIds") or [None])[0]
        no_token = market.get("no_token_id") or (market.get("clobTokenIds") or [None, None])[1]

        return ArbOpportunity(
            arb_type="TYPE_1_YES_NO_SUM",
            market_slug=slug,
            description=f"YES({yes_ask:.4f}) + NO({no_ask:.4f}) = {yes_no_sum:.4f} < 1.0 (gap={arb_gap:.4f})",
            profit_pct=profit_pct,
            profit_usdc=profit_usdc,
            max_size=max_size_usdc,
            legs=[
                {"token_id": str(yes_token), "outcome": "YES", "side": "BUY",
                 "price": yes_ask, "size": max_size_usdc / 2},
                {"token_id": str(no_token), "outcome": "NO", "side": "BUY",
                 "price": no_ask, "size": max_size_usdc / 2},
            ],
            urgency=10.0,
            arb_score=10.0,
            confidence=1.0,
        )

    # ── TYPE 2: Multi-outcome Categorical Arbitrage ───────────────────────────

    def detect_categorical_arb(
        self,
        markets_in_group: list[dict],
        orderbooks: dict[str, OrderbookMetrics],
    ) -> Optional[ArbOpportunity]:
        """
        For categorical markets with 3+ outcomes, if sum of best asks < 0.97.
        """
        if len(markets_in_group) < 3:
            return None

        legs = []
        total_ask = 0.0

        for market in markets_in_group:
            slug = market.get("slug") or market.get("conditionId", "")
            yes_token = str((market.get("clobTokenIds") or [None])[0] or "")
            book = orderbooks.get(yes_token)
            if not book or book.best_ask <= 0 or book.best_ask >= 1:
                continue
            total_ask += book.best_ask
            legs.append({
                "token_id": yes_token,
                "outcome": market.get("question", "")[:30],
                "side": "BUY",
                "price": book.best_ask,
                "size": 0,  # TBD
                "market_slug": slug,
            })

        if total_ask <= 0 or total_ask >= 0.97:
            return None

        profit_pct = 1.0 - total_ask
        if profit_pct < 0.015:
            return None

        # Allocate capital proportionally
        max_total = 200.0  # $200 max for categorical arb
        for leg in legs:
            leg["size"] = max_total * (leg["price"] / total_ask)

        profit_usdc = max_total * profit_pct
        if profit_usdc < self.settings.arb_min_profit_usdc:
            return None

        group_slug = markets_in_group[0].get("groupSlug", "categorical_group")
        return ArbOpportunity(
            arb_type="TYPE_2_CATEGORICAL",
            market_slug=group_slug,
            description=f"Categorical arb: sum={total_ask:.4f} < 1.0 ({len(legs)} outcomes)",
            profit_pct=profit_pct,
            profit_usdc=profit_usdc,
            max_size=max_total,
            legs=legs,
            urgency=9.0,
            arb_score=10.0,
        )

    # ── TYPE 3: Implied Probability Contradiction ─────────────────────────────

    def detect_implied_contradiction(
        self,
        market_a: dict,
        market_b: dict,
    ) -> Optional[ArbOpportunity]:
        """
        If P(A) > P(B) but A logically requires B: mispricing exists.
        For example: P(X wins championship) > P(X wins semifinal).
        """
        price_a = float((market_a.get("outcomePrices") or [0.5])[0]
                        if isinstance(market_a.get("outcomePrices"), list) else 0.5)
        price_b = float((market_b.get("outcomePrices") or [0.5])[0]
                        if isinstance(market_b.get("outcomePrices"), list) else 0.5)

        # A implies B → P(A) should be <= P(B)
        # Mispricing = P(A) - P(B) if positive
        mispricing = price_a - price_b
        if mispricing <= 0.03:  # min 3% gap to be interesting
            return None

        slug_a = market_a.get("slug", "market_a")
        slug_b = market_b.get("slug", "market_b")

        return ArbOpportunity(
            arb_type="TYPE_3_IMPLIED_CONTRADICTION",
            market_slug=f"{slug_a}+{slug_b}",
            description=(
                f"Implied contradiction: P({slug_a[:20]})={price_a:.3f} > "
                f"P({slug_b[:20]})={price_b:.3f} but A requires B"
            ),
            profit_pct=mispricing * 0.5,  # conservative estimate
            profit_usdc=mispricing * 100,
            max_size=0,  # Flag only, no auto-execution
            legs=[
                {"market_slug": slug_a, "action": "SELL_YES", "price": price_a},
                {"market_slug": slug_b, "action": "BUY_YES", "price": price_b},
            ],
            urgency=5.0,
            arb_score=7.0,
            confidence=0.7,
        )

    # ── TYPE 4: Spread Capture / Market Making ────────────────────────────────

    def detect_spread_capture(
        self,
        market: dict,
        yes_book: OrderbookMetrics,
    ) -> Optional[ArbOpportunity]:
        """
        Post limit orders just inside the spread and capture the difference.
        Requires spread > 3% and liquidity > $2,000.
        """
        liquidity = float(market.get("liquidity") or 0)

        if (yes_book.spread_pct < self.settings.spread_capture_min_spread_pct * 100
                or liquidity < 2000):
            return None

        bid_to_post = yes_book.best_bid + 0.002
        ask_to_post = yes_book.best_ask - 0.002

        if bid_to_post >= ask_to_post:
            return None  # Can't cross the market

        capture_pct = ask_to_post - bid_to_post
        if capture_pct < 0.002:
            return None

        size_per_side = min(50.0, yes_book.bid_depth_1pct * 0.1)
        profit_usdc = size_per_side * capture_pct

        slug = market.get("slug") or market.get("conditionId", "unknown")
        yes_token = str((market.get("clobTokenIds") or [None])[0] or "")

        return ArbOpportunity(
            arb_type="TYPE_4_SPREAD_CAPTURE",
            market_slug=slug,
            description=(
                f"Spread capture: bid={bid_to_post:.4f} ask={ask_to_post:.4f} "
                f"capture={capture_pct:.4f} ({yes_book.spread_pct:.1f}% spread)"
            ),
            profit_pct=capture_pct,
            profit_usdc=profit_usdc,
            max_size=size_per_side * 2,
            legs=[
                {"token_id": yes_token, "side": "BUY", "price": bid_to_post, "size": size_per_side},
                {"token_id": yes_token, "side": "SELL", "price": ask_to_post, "size": size_per_side},
            ],
            urgency=5.0,
            arb_score=5.0,
            confidence=0.8,
        )

    # ── TYPE 5: Time Decay Arbitrage ──────────────────────────────────────────

    def detect_time_decay_arb(
        self,
        market: dict,
        days_to_resolution: float,
    ) -> Optional[ArbOpportunity]:
        """
        Near-resolution markets with prices still in the middle are likely mispriced.
        """
        current_price = float((market.get("outcomePrices") or [0.5])[0]
                               if isinstance(market.get("outcomePrices"), list) else 0.5)

        if days_to_resolution >= 1.0:
            return None  # Only for markets resolving within 24h
        if not (0.20 < current_price < 0.80):
            return None  # Already near extreme

        # The market should be near 0 or 1 if resolving soon
        # This is a signal to run AI analysis urgently
        slug = market.get("slug") or market.get("conditionId", "unknown")
        estimated_urgency = 10.0 - days_to_resolution * 8  # Higher urgency closer to end

        return ArbOpportunity(
            arb_type="TYPE_5_TIME_DECAY",
            market_slug=slug,
            description=(
                f"Time decay: {days_to_resolution*24:.1f}h to resolution, "
                f"price={current_price:.3f} still in midrange"
            ),
            profit_pct=0.0,  # Depends on AI analysis
            profit_usdc=0.0,
            max_size=100.0,
            legs=[],  # Legs determined after AI analysis
            urgency=min(10.0, estimated_urgency),
            arb_score=8.0,
            confidence=0.6,
        )

    # ── Main scan ─────────────────────────────────────────────────────────────

    def scan_all(
        self,
        markets: list[dict],
        orderbooks: dict[str, dict],  # token_id → raw book
        days_to_resolution_map: dict[str, float],  # slug → days
        paper_mode: bool = False,
    ) -> list[ArbOpportunity]:
        """
        Run all arb type scans and return all detected opportunities.
        """
        from analysis.orderbook import analyze_orderbook
        opportunities: list[ArbOpportunity] = []

        for market in markets:
            slug = market.get("slug") or market.get("conditionId", "")
            try:
                tokens = market.get("clobTokenIds") or []
                if isinstance(tokens, str):
                    try:
                        tokens = json.loads(tokens)
                    except json.JSONDecodeError:
                        tokens = []

                yes_token = str(tokens[0]) if len(tokens) > 0 else ""
                no_token = str(tokens[1]) if len(tokens) > 1 else ""

                yes_raw = orderbooks.get(yes_token)
                no_raw = orderbooks.get(no_token)
                yes_metrics = analyze_orderbook(yes_raw) if yes_raw else None
                no_metrics = analyze_orderbook(no_raw) if no_raw else None

                # Type 1
                if yes_metrics and no_metrics:
                    t1 = self.detect_yes_no_arb(market, yes_metrics, no_metrics, paper_mode=paper_mode)
                    if t1:
                        opportunities.append(t1)
                        logger.info("TYPE_1 ARB FOUND: %s profit=%.2f%% ($%.2f) yes_ask=%.4f no_ask=%.4f",
                                    slug[:30], t1.profit_pct * 100, t1.profit_usdc,
                                    yes_metrics.best_ask, no_metrics.best_ask)

                # Type 4
                if yes_metrics:
                    t4 = self.detect_spread_capture(market, yes_metrics)
                    if t4:
                        opportunities.append(t4)

                # Type 5
                days = days_to_resolution_map.get(slug, 30.0)
                t5 = self.detect_time_decay_arb(market, days)
                if t5:
                    opportunities.append(t5)

            except Exception as exc:
                logger.debug("Arb scan error for market %s: %s", slug, exc)

        # Sort by urgency
        opportunities.sort(key=lambda x: x.urgency, reverse=True)
        if opportunities:
            logger.info("Arb scan: found %d opportunities", len(opportunities))
        return opportunities

    def get_arb_score(self, opportunities: list[ArbOpportunity], market_slug: str) -> float:
        """Get highest arb score for a specific market."""
        market_arbs = [a for a in opportunities if a.market_slug == market_slug]
        if not market_arbs:
            return 0.0
        return max(a.arb_score for a in market_arbs)
