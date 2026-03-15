"""
Composite Signal Aggregator — Module 8.
Combines all signal sources into a final trade decision.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings
from utils.helpers import clamp, now_ts

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    market_slug: str
    question: str
    category: str

    # Individual scores (0-10 each)
    ai_score: float = 5.0
    whale_score: float = 5.0
    news_score: float = 5.0
    technical_score: float = 5.0
    orderbook_score: float = 5.0
    arb_score: float = 0.0

    # Directions for each signal
    ai_direction: str = "NEUTRAL"
    whale_direction: str = "NEUTRAL"
    news_direction: str = "NEUTRAL"
    technical_direction: str = "NEUTRAL"
    orderbook_direction: str = "NEUTRAL"

    # Composite
    composite_score: float = 5.0
    final_direction: str = "NEUTRAL"
    action: str = "SKIP"               # STRONG_BUY | BUY | WEAK_BUY | SKIP

    # Trade details
    current_price: float = 0.5
    ai_probability: float = 0.5
    ai_edge: float = 0.0
    ai_confidence: float = 0.5
    signal_strength: str = "WEAK"

    # Metadata
    computed_at: float = field(default_factory=now_ts)
    reason_skipped: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "market_slug": self.market_slug,
            "question": self.question,
            "category": self.category,
            "ai_score": self.ai_score,
            "whale_score": self.whale_score,
            "news_score": self.news_score,
            "technical_score": self.technical_score,
            "orderbook_score": self.orderbook_score,
            "arb_score": self.arb_score,
            "ai_direction": self.ai_direction,
            "whale_direction": self.whale_direction,
            "news_direction": self.news_direction,
            "technical_direction": self.technical_direction,
            "orderbook_direction": self.orderbook_direction,
            "composite_score": self.composite_score,
            "final_direction": self.final_direction,
            "action": self.action,
            "current_price": self.current_price,
            "ai_probability": self.ai_probability,
            "ai_edge": self.ai_edge,
            "ai_confidence": self.ai_confidence,
            "signal_strength": self.signal_strength,
            "computed_at": self.computed_at,
            "reason_skipped": self.reason_skipped,
        }


class SignalAggregator:
    """
    Combines signals from all analysis modules into a unified trade decision.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def aggregate(
        self,
        market: dict,
        ai_result=None,            # AIAnalysisResult or None
        whale_consensus: Optional[dict] = None,
        news_result: Optional[dict] = None,
        technical_signal=None,     # TechnicalSignal or None
        orderbook_yes=None,        # OrderbookMetrics or None
        arb_score: float = 0.0,
        open_positions: list = None,
    ) -> TradeSignal:
        """
        Aggregate all signals for a market into a single trade decision.
        """
        slug = market.get("slug") or market.get("conditionId", "unknown")
        question = market.get("question", "Unknown")
        category = str(market.get("category") or "unknown").lower()
        current_price = _extract_yes_price(market)

        sig = TradeSignal(
            market_slug=slug,
            question=question,
            category=category,
            current_price=current_price,
        )

        # ── Skip: existing position ───────────────────────────────────────────
        if open_positions:
            for pos in open_positions:
                if pos.get("market_slug") == slug:
                    sig.action = "SKIP"
                    sig.reason_skipped = "Already have open position"
                    return sig

        # ── Extract individual signals ────────────────────────────────────────

        # AI signal
        if ai_result:
            sig.ai_score = float(getattr(ai_result, "ai_score", 5.0) or 5.0)
            sig.ai_direction = str(getattr(ai_result, "recommended_outcome", "SKIP"))
            if sig.ai_direction == "SKIP":
                sig.ai_direction = "NEUTRAL"
            sig.ai_probability = float(getattr(ai_result, "yes_probability", current_price))
            sig.ai_edge = float(getattr(ai_result, "edge", 0.0))
            sig.ai_confidence = float(getattr(ai_result, "confidence", 0.5))
            sig.signal_strength = str(getattr(ai_result, "signal_strength", "WEAK"))
        else:
            sig.reason_skipped = "No AI analysis available"
            sig.action = "SKIP"
            return sig

        # Whale signal
        if whale_consensus:
            conviction = float(whale_consensus.get("smart_money_conviction", 0))
            sig.whale_score = min(10.0, conviction)
            sig.whale_direction = str(whale_consensus.get("smart_money_net_direction", "NEUTRAL"))
            # Boost if insider or legendary confirms
            if whale_consensus.get("insider_direction") == sig.whale_direction != "NEUTRAL":
                sig.whale_score = min(10.0, sig.whale_score + 2.0)
        else:
            sig.whale_score = 5.0
            sig.whale_direction = "NEUTRAL"

        # News signal
        if news_result:
            sig.news_score = float(news_result.get("news_score", 5.0))
            consensus = str(news_result.get("consensus_direction", "UNCERTAIN"))
            sig.news_direction = "YES" if consensus == "YES" else ("NO" if consensus == "NO" else "NEUTRAL")
        else:
            sig.news_score = 5.0
            sig.news_direction = "NEUTRAL"

        # Technical signal
        if technical_signal:
            sig.technical_score = float(getattr(technical_signal, "technical_score", 5.0))
            sig.technical_direction = str(getattr(technical_signal, "technical_direction", "NEUTRAL"))
        else:
            sig.technical_score = 5.0
            sig.technical_direction = "NEUTRAL"

        # Orderbook signal
        if orderbook_yes:
            sig.orderbook_score = float(getattr(orderbook_yes, "orderbook_score", 5.0))
            sig.orderbook_direction = str(getattr(orderbook_yes, "orderbook_direction", "NEUTRAL"))
        else:
            sig.orderbook_score = 5.0
            sig.orderbook_direction = "NEUTRAL"

        # Arb signal
        sig.arb_score = clamp(arb_score, 0.0, 10.0)

        # ── Compute weighted composite score ──────────────────────────────────
        composite = (
            sig.ai_score * self.settings.ai_weight
            + sig.whale_score * self.settings.whale_weight
            + sig.news_score * self.settings.news_weight
            + sig.technical_score * self.settings.technical_weight
            + sig.orderbook_score * self.settings.orderbook_weight
            + sig.arb_score * self.settings.arb_weight
        )

        # ── Direction reconciliation ──────────────────────────────────────────
        directions = [
            (sig.ai_direction, self.settings.ai_weight),
            (sig.whale_direction, self.settings.whale_weight),
            (sig.news_direction, self.settings.news_weight),
            (sig.technical_direction, self.settings.technical_weight),
            (sig.orderbook_direction, self.settings.orderbook_weight),
        ]

        yes_weight = sum(w for d, w in directions if d == "YES")
        no_weight = sum(w for d, w in directions if d == "NO")
        non_neutral_weight = yes_weight + no_weight

        if non_neutral_weight == 0:
            sig.final_direction = "NEUTRAL"
        elif yes_weight > no_weight:
            sig.final_direction = "YES"
        else:
            sig.final_direction = "NO"

        # Conflicting signals penalty: reduce by 30%
        non_neutral_directions = {d for d, w in directions if d in ("YES", "NO")}
        has_conflict = len(non_neutral_directions) > 1
        if has_conflict:
            composite *= 0.70
            logger.debug("Signal conflict penalty applied for %s", slug[:30])

        # Full consensus bonus: +20%
        non_neutral_ds = [d for d, _ in directions if d in ("YES", "NO")]
        all_agree = len(set(non_neutral_ds)) == 1 and len(non_neutral_ds) >= 3
        if all_agree:
            composite *= 1.20
            logger.debug("Full consensus bonus applied for %s", slug[:30])

        sig.composite_score = clamp(composite, 0.0, 10.0)

        # ── Determine trade action ────────────────────────────────────────────
        ai_edge = sig.ai_edge
        abs_edge = abs(ai_edge)
        outcome_direction = sig.final_direction

        # Arb override: if pure arb detected with high score, skip directional check
        if arb_score >= 9.0:
            sig.action = "STRONG_BUY"
            sig.final_direction = "YES"  # Arb specific legs handled elsewhere
        elif sig.composite_score >= 5.5 and abs_edge >= 0.04:
            sig.action = "STRONG_BUY"
            if ai_edge < 0:
                sig.final_direction = "NO"
        elif sig.composite_score >= 4.0 and abs_edge >= 0.02:
            sig.action = "BUY"
            if ai_edge < 0:
                sig.final_direction = "NO"
        elif sig.composite_score >= 3.0 and abs_edge >= 0.01:
            sig.action = "WEAK_BUY"
            if ai_edge < 0:
                sig.final_direction = "NO"
        else:
            sig.action = "SKIP"
            reasons = []
            if abs_edge < 0.01:
                reasons.append(f"edge too small ({ai_edge:+.3f})")
            if sig.composite_score < 3.0:
                reasons.append(f"score too low ({sig.composite_score:.1f})")
            sig.reason_skipped = ", ".join(reasons) or "insufficient signal"

        # ── Confirmation requirements ─────────────────────────────────────────
        if sig.action != "SKIP":
            # At least 2 of 6 signals must agree on direction
            agreeing_signals = sum(
                1 for d, _ in directions if d == sig.final_direction
            )
            if agreeing_signals < 2:
                sig.action = "SKIP"
                sig.reason_skipped = f"Only {agreeing_signals}/6 signals agree on {sig.final_direction}"

            # AI confidence must be >= 0.40 (was 0.55)
            if sig.ai_confidence < 0.40:
                sig.action = "SKIP"
                sig.reason_skipped = f"AI confidence too low ({sig.ai_confidence:.2f})"

        # ── Data flow diagnostic log ──────────────────────────────────────────
        logger.info(
            "SIGNAL %s | score=%.2f | ai_prob=%.3f | edge=%+.3f | conf=%.2f"
            " | whale=%s(%.1f) | action=%s%s",
            slug[:30], sig.composite_score, sig.ai_probability, sig.ai_edge,
            sig.ai_confidence, sig.whale_direction, sig.whale_score, sig.action,
            f" | skip={sig.reason_skipped}" if sig.action == "SKIP" else "",
        )

        return sig

    def aggregate_batch(
        self,
        markets: list[dict],
        ai_results: dict,
        whale_consensuses: dict,
        news_results: dict,
        technical_signals: dict,
        orderbook_yes_metrics: dict,
        arb_scores: dict,
        open_positions: list,
    ) -> list[TradeSignal]:
        """
        Aggregate signals for multiple markets, sorted by composite score.
        """
        signals: list[TradeSignal] = []
        for market in markets:
            slug = market.get("slug") or market.get("conditionId", "unknown")
            signal = self.aggregate(
                market=market,
                ai_result=ai_results.get(slug),
                whale_consensus=whale_consensuses.get(slug),
                news_result=news_results.get(slug),
                technical_signal=technical_signals.get(slug),
                orderbook_yes=orderbook_yes_metrics.get(slug),
                arb_score=arb_scores.get(slug, 0.0),
                open_positions=open_positions,
            )
            signals.append(signal)

        # Sort: actionable first, then by composite score
        signals.sort(
            key=lambda s: (s.action != "SKIP", s.composite_score),
            reverse=True
        )
        return signals

    def freshness_check(self, signal: TradeSignal) -> bool:
        """Check if signal is fresh enough to act on (< 5 min old)."""
        age = now_ts() - signal.computed_at
        return age < 300  # 5 minutes


def _extract_yes_price(market: dict) -> float:
    prices = market.get("outcomePrices")
    if isinstance(prices, list) and len(prices) > 0:
        try:
            return float(prices[0])
        except (ValueError, TypeError):
            pass
    if isinstance(prices, str):
        import json
        try:
            ps = json.loads(prices)
            if ps:
                return float(ps[0])
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
    return float(market.get("yes_price") or market.get("bestAsk") or 0.5)
