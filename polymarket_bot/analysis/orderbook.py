"""
Orderbook analysis module — Module 2.
Computes depth, imbalance, spread, market impact, and shock detection.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.helpers import clamp, safe_div

logger = logging.getLogger(__name__)


@dataclass
class OrderbookMetrics:
    # Token-level
    best_bid: float = 0.0
    best_ask: float = 1.0
    mid_price: float = 0.5
    spread: float = 1.0
    spread_pct: float = 200.0
    bid_depth_1pct: float = 0.0
    ask_depth_1pct: float = 0.0
    total_bid_depth: float = 0.0
    total_ask_depth: float = 0.0
    bid_ask_imbalance: float = 0.0
    orderbook_skew: float = 0.0
    market_impact_100: float = 0.0
    market_impact_500: float = 0.0
    thinness_score: float = 1.0

    # Cross-token (YES + NO)
    yes_no_sum: float = 1.0
    arb_gap: float = 0.0

    # Signal (0-10)
    orderbook_score: float = 5.0
    orderbook_direction: str = "NEUTRAL"


@dataclass
class PriceShock:
    token_id: str
    old_price: float
    new_price: float
    change_pct: float
    direction: str  # "UP" | "DOWN"
    timestamp: float


def analyze_orderbook(book: dict) -> OrderbookMetrics:
    """
    Compute all orderbook metrics from a normalized book dict.
    book: {"bids": [{"price": float, "size": float}, ...], "asks": [...]}
    """
    m = OrderbookMetrics()
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids and not asks:
        return m

    # Filter and sort
    valid_bids = sorted(
        [b for b in bids if b.get("price", 0) > 0 and b.get("size", 0) > 0],
        key=lambda x: x["price"], reverse=True
    )
    valid_asks = sorted(
        [a for a in asks if a.get("price", 0) > 0 and a.get("size", 0) > 0],
        key=lambda x: x["price"]
    )

    if valid_bids:
        m.best_bid = valid_bids[0]["price"]
    if valid_asks:
        m.best_ask = valid_asks[0]["price"]

    m.mid_price = (m.best_bid + m.best_ask) / 2 if (valid_bids and valid_asks) else 0.5
    m.spread = m.best_ask - m.best_bid if (valid_bids and valid_asks) else 1.0
    m.spread_pct = (m.spread / m.mid_price * 100) if m.mid_price > 0 else 200.0

    # Depth within 1% of best bid/ask
    bid_1pct_threshold = m.best_bid * 0.99 if m.best_bid > 0 else 0
    ask_1pct_threshold = m.best_ask * 1.01 if m.best_ask > 0 else 1
    m.bid_depth_1pct = sum(
        b["size"] for b in valid_bids if b["price"] >= bid_1pct_threshold
    )
    m.ask_depth_1pct = sum(
        a["size"] for a in valid_asks if a["price"] <= ask_1pct_threshold
    )

    # Total depth in $ value
    m.total_bid_depth = sum(b["price"] * b["size"] for b in valid_bids)
    m.total_ask_depth = sum(a["price"] * a["size"] for a in valid_asks)

    # Bid-ask imbalance
    total_depth = m.total_bid_depth + m.total_ask_depth
    if total_depth > 0:
        m.bid_ask_imbalance = (m.total_bid_depth - m.total_ask_depth) / total_depth
    else:
        m.bid_ask_imbalance = 0.0

    # Orderbook skew: VWAP of full book vs mid
    all_levels = (
        [(b["price"], b["size"]) for b in valid_bids]
        + [(a["price"], a["size"]) for a in valid_asks]
    )
    if all_levels:
        total_size = sum(s for _, s in all_levels)
        if total_size > 0:
            vwap = sum(p * s for p, s in all_levels) / total_size
            m.orderbook_skew = vwap - m.mid_price

    # Market impact: price after absorbing $X of buy orders
    m.market_impact_100 = _compute_market_impact(valid_asks, 100.0)
    m.market_impact_500 = _compute_market_impact(valid_asks, 500.0)

    # Thinness score: lower is better (thicker book)
    m.thinness_score = 1.0 / (m.bid_depth_1pct + m.ask_depth_1pct + 1.0)

    # Composite score
    m.orderbook_score, m.orderbook_direction = _compute_score(m)
    return m


def analyze_cross_token(yes_metrics: OrderbookMetrics, no_metrics: OrderbookMetrics) -> tuple[float, float]:
    """
    Compute YES/NO sum and arb gap.
    Returns (yes_no_sum, arb_gap).
    """
    yes_ask = yes_metrics.best_ask if yes_metrics.best_ask < 1 else 0.99
    no_ask = no_metrics.best_ask if no_metrics.best_ask < 1 else 0.99
    yes_no_sum = yes_ask + no_ask
    arb_gap = 1.0 - yes_no_sum
    return yes_no_sum, arb_gap


def detect_price_shock(
    token_id: str,
    old_book: Optional[dict],
    new_book: dict,
    shock_threshold_pct: float = 5.0,
) -> Optional[PriceShock]:
    """
    Detect if the mid price moved more than threshold% since last update.
    """
    if old_book is None:
        return None

    old_bids = old_book.get("bids", [])
    old_asks = old_book.get("asks", [])
    new_bids = new_book.get("bids", [])
    new_asks = new_book.get("asks", [])

    def mid(bids, asks):
        best_bid = max((b["price"] for b in bids if b.get("price")), default=0)
        best_ask = min((a["price"] for a in asks if a.get("price")), default=1)
        return (best_bid + best_ask) / 2

    old_mid = mid(old_bids, old_asks)
    new_mid = mid(new_bids, new_asks)

    if old_mid == 0:
        return None

    change_pct = abs(new_mid - old_mid) / old_mid * 100
    if change_pct >= shock_threshold_pct:
        import time
        return PriceShock(
            token_id=token_id,
            old_price=old_mid,
            new_price=new_mid,
            change_pct=change_pct,
            direction="UP" if new_mid > old_mid else "DOWN",
            timestamp=time.time(),
        )
    return None


def _compute_market_impact(asks: list[dict], target_usdc: float) -> float:
    """
    Compute the effective price after absorbing `target_usdc` of buy orders
    walking up the ask side.
    """
    remaining = target_usdc
    cost = 0.0
    shares_bought = 0.0
    for ask in asks:
        price = ask["price"]
        size = ask["size"]
        level_cost = price * size
        if level_cost <= remaining:
            cost += level_cost
            shares_bought += size
            remaining -= level_cost
        else:
            # Partial fill
            shares_this_level = remaining / price
            cost += remaining
            shares_bought += shares_this_level
            remaining = 0
            break
    if shares_bought == 0:
        return asks[0]["price"] if asks else 1.0
    return cost / shares_bought


def _compute_score(m: OrderbookMetrics) -> tuple[float, str]:
    """Compute composite orderbook signal score (0-10) and direction."""
    # Imbalance signal: positive imbalance → buy pressure → bullish
    imbalance_signal = clamp(m.bid_ask_imbalance * 10, -10, 10)

    # Depth signal: deeper book = better signal quality
    depth_signal = min(m.bid_depth_1pct / 500.0, 10.0)

    # Spread signal: tighter spread = better
    spread_signal = max(0.0, 10.0 - m.spread_pct * 2)

    score = (imbalance_signal + depth_signal + spread_signal) / 3.0
    score = clamp((score + 10) / 2, 0.0, 10.0)  # normalize to 0-10

    # Direction from imbalance
    if m.bid_ask_imbalance > 0.1:
        direction = "YES"
    elif m.bid_ask_imbalance < -0.1:
        direction = "NO"
    else:
        direction = "NEUTRAL"

    return score, direction
