"""
Kelly Criterion Position Sizer — Module 9.
Computes optimal position sizes with multiple safety caps.
"""
import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import Settings
from utils.helpers import clamp, safe_div

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    raw_kelly_f: float            # Full Kelly fraction
    applied_kelly_f: float        # After all scaling
    raw_size: float               # Before caps
    final_size: float             # After all caps
    expected_value: float         # In $
    expected_roi_pct: float       # As %
    risk_reward_ratio: float
    breakeven_prob: float         # The current market price
    capped_by: str                # What constraint limited size
    is_viable: bool               # False if size < minimum


def kelly_size(
    estimated_probability: float,
    current_price: float,
    confidence: float,
    composite_score: float,
    available_bankroll: float,
    market_liquidity: float,
    current_exposure: float,
    total_budget: float,
    settings: Settings,
    is_weak_signal: bool = False,
) -> KellyResult:
    """
    Full Kelly formula with all safety layers.

    Args:
        estimated_probability: Our estimated P(YES)
        current_price: Market's current YES price (= cost per share)
        confidence: AI confidence 0-1
        composite_score: Signal score 0-10
        available_bankroll: Cash available to deploy
        market_liquidity: Total market liquidity in USDC
        current_exposure: Current deployed capital in USDC
        total_budget: Total starting budget
        settings: Bot settings
        is_weak_signal: If True, halve the final size
    """
    # Ensure valid inputs
    p = clamp(estimated_probability, 0.01, 0.99)
    price = clamp(current_price, 0.01, 0.99)
    conf = clamp(confidence, 0.0, 1.0)
    score = clamp(composite_score, 0.0, 10.0)

    # Net odds: profit per $1 risked
    b = (1.0 - price) / price

    # Full Kelly: f* = (b*p - q) / b
    q = 1.0 - p
    kelly_f = (b * p - q) / b

    if kelly_f <= 0:
        # Negative Kelly: edge is against us
        return KellyResult(
            raw_kelly_f=kelly_f,
            applied_kelly_f=0.0,
            raw_size=0.0,
            final_size=0.0,
            expected_value=kelly_f * available_bankroll,
            expected_roi_pct=kelly_f * 100,
            risk_reward_ratio=b,
            breakeven_prob=price,
            capped_by="negative_kelly",
            is_viable=False,
        )

    raw_kelly_f = kelly_f

    # Scale 1: Quarter Kelly (hard max)
    kelly_f *= settings.kelly_multiplier

    # Scale 2: Confidence (quadratic penalty)
    kelly_f *= conf ** 2

    # Scale 3: Signal quality
    kelly_f *= score / 10.0

    applied_kelly_f = kelly_f

    # Dollar size from bankroll
    raw_size = applied_kelly_f * available_bankroll

    # Weak signal: halve the size
    if is_weak_signal:
        raw_size *= 0.50

    # ── Apply caps in order of restrictiveness ────────────────────────────────
    cap_size = raw_size
    capped_by = "none"

    # Cap 1: max position size setting
    if cap_size > settings.max_position_size:
        cap_size = settings.max_position_size
        capped_by = "max_position_size"

    # Cap 2: max single market (15% of total budget)
    market_cap = total_budget * settings.max_single_market_pct
    if cap_size > market_cap:
        cap_size = market_cap
        capped_by = "max_single_market_pct"

    # Cap 3: liquidity cap (never be > 10% of market liquidity)
    if market_liquidity > 0:
        liquidity_cap = market_liquidity * 0.10
        if cap_size > liquidity_cap:
            cap_size = liquidity_cap
            capped_by = "liquidity_cap"

    # Cap 4: exposure cap (can't deploy more than remaining capacity)
    total_exposure_max = total_budget * settings.max_total_exposure_pct
    remaining_capacity = max(0.0, total_exposure_max - current_exposure)
    if cap_size > remaining_capacity:
        cap_size = remaining_capacity
        capped_by = "exposure_cap"

    # Cap 5: available cash
    if cap_size > available_bankroll:
        cap_size = available_bankroll
        capped_by = "cash_available"

    final_size = max(0.0, cap_size)
    is_viable = final_size >= settings.min_position_size_usdc

    # ── Analytics ─────────────────────────────────────────────────────────────
    # If we invest `final_size` at price `price`, we get `shares = final_size / price`
    # Win: shares * 1.0 = final_size / price → profit = final_size/price - final_size = final_size * (1-price)/price
    # Loss: lose final_size
    win_profit = final_size * b
    expected_value = p * win_profit - q * final_size
    expected_roi_pct = safe_div(expected_value, final_size) * 100 if final_size > 0 else 0.0

    return KellyResult(
        raw_kelly_f=raw_kelly_f,
        applied_kelly_f=applied_kelly_f,
        raw_size=raw_size,
        final_size=final_size,
        expected_value=expected_value,
        expected_roi_pct=expected_roi_pct,
        risk_reward_ratio=b,
        breakeven_prob=price,
        capped_by=capped_by,
        is_viable=is_viable,
    )


def compute_portfolio_kelly(
    opportunities: list[dict],
    bankroll: float,
    settings: Settings,
) -> list[dict]:
    """
    Compute Kelly sizes for a list of opportunities simultaneously,
    accounting for portfolio-level constraints.
    """
    if not opportunities:
        return []

    # Sort by composite score descending
    sorted_opps = sorted(
        opportunities,
        key=lambda x: x.get("composite_score", 0),
        reverse=True
    )

    results = []
    total_deployed = 0.0
    positions_taken = 0
    exposure_by_category: dict[str, float] = {}

    for opp in sorted_opps:
        if positions_taken >= settings.max_open_positions:
            break

        category = str(opp.get("category", "unknown")).lower()
        category_exposure = exposure_by_category.get(category, 0.0)

        # Category cap
        max_cat = bankroll * settings.max_category_pct
        if category_exposure >= max_cat:
            continue

        # Total exposure cap
        remaining_cap = bankroll * settings.max_total_exposure_pct - total_deployed
        if remaining_cap < settings.min_position_size_usdc:
            break

        kr = kelly_size(
            estimated_probability=opp.get("ai_probability", 0.5),
            current_price=opp.get("current_price", 0.5),
            confidence=opp.get("ai_confidence", 0.5),
            composite_score=opp.get("composite_score", 5.0),
            available_bankroll=min(bankroll - total_deployed, remaining_cap),
            market_liquidity=opp.get("liquidity", 0),
            current_exposure=total_deployed,
            total_budget=bankroll,
            settings=settings,
            is_weak_signal=opp.get("signal_strength") == "WEAK_BUY",
        )

        if not kr.is_viable:
            continue

        # Category limit enforcement
        allowed = min(kr.final_size, max_cat - category_exposure)
        if allowed < settings.min_position_size_usdc:
            continue

        result = {**opp, "kelly_result": kr, "final_size": allowed}
        results.append(result)
        total_deployed += allowed
        exposure_by_category[category] = category_exposure + allowed
        positions_taken += 1

    return results
