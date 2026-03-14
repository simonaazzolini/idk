"""
Orderbook analysis module — Module 2.
Computes depth, imbalance, spread, market impact, thinness, and shock detection.
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.helpers import clamp, safe_div

logger = logging.getLogger(__name__)


@dataclass
class OrderbookMetrics:
    # ── Best levels ───────────────────────────────────────────────────────────
    best_bid: float = 0.0
    best_ask: float = 1.0
    mid_price: float = 0.5
    spread: float = 1.0
    spread_pct: float = 200.0

    # ── Depth bands ───────────────────────────────────────────────────────────
    bid_depth_1pct: float = 0.0     # USDC value of bids within 1% of best bid
    ask_depth_1pct: float = 0.0     # USDC value of asks within 1% of best ask
    bid_depth_2pct: float = 0.0     # within 2%
    ask_depth_2pct: float = 0.0
    bid_depth_5pct: float = 0.0     # within 5%
    ask_depth_5pct: float = 0.0
    total_bid_depth: float = 0.0    # total book-wide USDC value
    total_ask_depth: float = 0.0

    # ── Imbalance ─────────────────────────────────────────────────────────────
    bid_ask_imbalance: float = 0.0      # (bid_depth − ask_depth) / total_depth
    depth_imbalance_1pct: float = 0.0   # imbalance within 1% bands only
    depth_imbalance_5pct: float = 0.0   # imbalance within 5% bands only
    orderbook_skew: float = 0.0         # VWAP of full book vs mid_price

    # ── Market impact ─────────────────────────────────────────────────────────
    market_impact_100: float = 0.0      # effective avg price buying $100 of asks
    market_impact_500: float = 0.0      # effective avg price buying $500 of asks
    market_impact_100_pct: float = 0.0  # above expressed as % above mid
    market_impact_500_pct: float = 0.0

    # ── Thinness ──────────────────────────────────────────────────────────────
    thinness_score: float = 1.0         # 0 = very deep; 1 = empty book

    # ── Cross-token (YES + NO) ────────────────────────────────────────────────
    yes_no_sum: float = 1.0
    arb_gap: float = 0.0                # positive = potential arbitrage

    # ── Signal ────────────────────────────────────────────────────────────────
    orderbook_score: float = 5.0
    orderbook_direction: str = "NEUTRAL"


@dataclass
class PriceShock:
    token_id: str
    old_price: float
    new_price: float
    change_pct: float
    direction: str      # "UP" | "DOWN"
    timestamp: float


def analyze_orderbook(book: dict) -> OrderbookMetrics:
    """
    Compute all orderbook metrics from a normalised book dict.

    Expected input format:
        {
            "bids": [{"price": float, "size": float}, ...],
            "asks": [{"price": float, "size": float}, ...]
        }
    Bids are sorted descending by price; asks ascending.

    Returns OrderbookMetrics with defaults when the book is empty.
    """
    m = OrderbookMetrics()
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids and not asks:
        return m

    # Filter invalid levels and sort
    valid_bids = sorted(
        [b for b in bids if b.get("price", 0) > 0 and b.get("size", 0) > 0],
        key=lambda x: x["price"], reverse=True,
    )
    valid_asks = sorted(
        [a for a in asks if a.get("price", 0) > 0 and a.get("size", 0) > 0],
        key=lambda x: x["price"],
    )

    if not valid_bids and not valid_asks:
        return m

    # ── Best levels and spread ─────────────────────────────────────────────
    if valid_bids:
        m.best_bid = float(valid_bids[0]["price"])
    if valid_asks:
        m.best_ask = float(valid_asks[0]["price"])

    if valid_bids and valid_asks:
        m.mid_price = (m.best_bid + m.best_ask) / 2.0
        m.spread    = m.best_ask - m.best_bid
    elif valid_bids:
        m.mid_price = m.best_bid
        m.spread    = 1.0 - m.best_bid
    else:
        m.mid_price = m.best_ask
        m.spread    = m.best_ask

    m.spread_pct = (m.spread / m.mid_price * 100.0) if m.mid_price > 1e-8 else 200.0

    # ── Depth bands ────────────────────────────────────────────────────────
    m.bid_depth_1pct = _depth_within_pct(valid_bids, m.best_bid, 1.0, "bid")
    m.ask_depth_1pct = _depth_within_pct(valid_asks, m.best_ask, 1.0, "ask")
    m.bid_depth_2pct = _depth_within_pct(valid_bids, m.best_bid, 2.0, "bid")
    m.ask_depth_2pct = _depth_within_pct(valid_asks, m.best_ask, 2.0, "ask")
    m.bid_depth_5pct = _depth_within_pct(valid_bids, m.best_bid, 5.0, "bid")
    m.ask_depth_5pct = _depth_within_pct(valid_asks, m.best_ask, 5.0, "ask")

    m.total_bid_depth = sum(float(b["price"]) * float(b["size"]) for b in valid_bids)
    m.total_ask_depth = sum(float(a["price"]) * float(a["size"]) for a in valid_asks)

    # ── Bid-ask imbalance (full book) ──────────────────────────────────────
    # Formula: (bid_depth − ask_depth) / (bid_depth + ask_depth)
    # Range: −1 (all asks, bearish) to +1 (all bids, bullish)
    total_depth = m.total_bid_depth + m.total_ask_depth
    if total_depth > 1e-8:
        m.bid_ask_imbalance = (m.total_bid_depth - m.total_ask_depth) / total_depth
    else:
        m.bid_ask_imbalance = 0.0

    # Imbalance within tight 1% band (more informative for short-term signals)
    depth1_total = m.bid_depth_1pct + m.ask_depth_1pct
    if depth1_total > 1e-8:
        m.depth_imbalance_1pct = (m.bid_depth_1pct - m.ask_depth_1pct) / depth1_total

    # Imbalance within 5% band
    depth5_total = m.bid_depth_5pct + m.ask_depth_5pct
    if depth5_total > 1e-8:
        m.depth_imbalance_5pct = (m.bid_depth_5pct - m.ask_depth_5pct) / depth5_total

    # ── Orderbook VWAP skew ────────────────────────────────────────────────
    # VWAP of all levels vs mid-price: positive = weight above mid (bullish)
    all_levels = (
        [(float(b["price"]), float(b["size"])) for b in valid_bids]
        + [(float(a["price"]), float(a["size"])) for a in valid_asks]
    )
    if all_levels:
        total_size = sum(s for _, s in all_levels)
        if total_size > 1e-8:
            vwap = sum(p * s for p, s in all_levels) / total_size
            m.orderbook_skew = vwap - m.mid_price

    # ── Market impact: simulate walking the ask side ───────────────────────
    if valid_asks:
        m.market_impact_100, _ = _compute_market_impact(valid_asks, 100.0)
        m.market_impact_500, _ = _compute_market_impact(valid_asks, 500.0)
        if m.mid_price > 1e-8:
            m.market_impact_100_pct = (m.market_impact_100 - m.mid_price) / m.mid_price * 100.0
            m.market_impact_500_pct = (m.market_impact_500 - m.mid_price) / m.mid_price * 100.0

    # ── Thinness score ─────────────────────────────────────────────────────
    # Ranges from ~1.0 (empty book) to near 0 (very deep book).
    # Uses the 1% depth as the primary signal of accessible liquidity.
    accessible_depth = m.bid_depth_1pct + m.ask_depth_1pct
    m.thinness_score = float(1.0 / (1.0 + accessible_depth / 100.0))

    # ── Composite score ────────────────────────────────────────────────────
    m.orderbook_score, m.orderbook_direction = _compute_score(m)

    return m


def analyze_cross_token(
    yes_metrics: OrderbookMetrics, no_metrics: OrderbookMetrics
) -> tuple[float, float]:
    """
    Compute YES/NO sum and arb gap using the cheapest available side for each.

    The theoretical no-arb condition is: YES_ask + NO_ask == 1.0
    When YES_ask + NO_ask < 1.0 → positive arb gap (buy both outcomes cheap).
    When YES_ask + NO_ask > 1.0 → over-priced (sell both, or just avoid).

    Returns:
        (yes_no_sum, arb_gap)
        arb_gap > 0 = profitable if simultaneously fillable
    """
    yes_ask = float(np.clip(yes_metrics.best_ask, 0.01, 0.99))
    no_ask  = float(np.clip(no_metrics.best_ask,  0.01, 0.99))
    yes_no_sum = yes_ask + no_ask
    arb_gap    = 1.0 - yes_no_sum
    return yes_no_sum, arb_gap


def compute_execution_cost(
    metrics: OrderbookMetrics,
    trade_size_usdc: float,
) -> dict:
    """
    Estimate the total cost of executing a trade of `trade_size_usdc`.

    Returns a dict with:
        effective_price:    average fill price (walking the ask)
        slippage_pct:       percentage above mid
        spread_cost_usdc:   half-spread cost for the trade
        is_executable:      False if book has < 10% of needed depth
    """
    if trade_size_usdc <= 0:
        return {
            "effective_price": metrics.mid_price,
            "slippage_pct": 0.0,
            "spread_cost_usdc": 0.0,
            "is_executable": False,
        }

    available_depth = metrics.total_ask_depth
    is_executable   = available_depth >= trade_size_usdc * 0.10

    # Use market impact data when trade size is close to pre-computed levels
    if trade_size_usdc <= 110:
        eff_price = metrics.market_impact_100
    elif trade_size_usdc <= 550:
        # Linear interpolation between 100 and 500 levels
        t = (trade_size_usdc - 100) / 400.0
        eff_price = (1 - t) * metrics.market_impact_100 + t * metrics.market_impact_500
    else:
        eff_price = metrics.market_impact_500

    if metrics.mid_price > 1e-8:
        slippage_pct = (eff_price - metrics.mid_price) / metrics.mid_price * 100.0
    else:
        slippage_pct = 0.0

    spread_cost = metrics.spread / 2.0 * (trade_size_usdc / max(eff_price, 1e-6))

    return {
        "effective_price": float(eff_price),
        "slippage_pct": float(slippage_pct),
        "spread_cost_usdc": float(spread_cost),
        "is_executable": bool(is_executable),
    }


def detect_price_shock(
    token_id: str,
    old_book: Optional[dict],
    new_book: dict,
    shock_threshold_pct: float = 5.0,
    max_age_seconds: float = 60.0,
    old_book_ts: Optional[float] = None,
) -> Optional[PriceShock]:
    """
    Detect if the mid-price moved more than `shock_threshold_pct` percent
    since the previous snapshot.

    The `max_age_seconds` parameter gates the comparison: if the old book is
    older than 60 seconds (default) the comparison is still valid but the
    interpretation changes — the caller should handle very old baselines
    differently.  This function does not filter by age; it always computes.

    Args:
        token_id:           identifier for logging.
        old_book:           previous book snapshot dict (or None on first call).
        new_book:           current book snapshot dict.
        shock_threshold_pct: minimum % move to classify as a shock.
        max_age_seconds:    informational — passed through but not enforced here.
        old_book_ts:        timestamp of old_book snapshot (for change_rate calc).

    Returns:
        PriceShock if threshold crossed, else None.
    """
    if old_book is None:
        return None

    def _mid(book: dict) -> float:
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = max(
            (float(b["price"]) for b in bids if b.get("price", 0) > 0), default=0.0
        )
        best_ask = min(
            (float(a["price"]) for a in asks if a.get("price", 0) > 0), default=1.0
        )
        if best_bid > 0 and best_ask < 1:
            return (best_bid + best_ask) / 2.0
        if best_bid > 0:
            return best_bid
        return best_ask

    old_mid = _mid(old_book)
    new_mid = _mid(new_book)

    if old_mid < 1e-8:
        return None

    change_pct = abs(new_mid - old_mid) / old_mid * 100.0

    if change_pct >= shock_threshold_pct:
        direction = "UP" if new_mid > old_mid else "DOWN"
        logger.info(
            "PRICE SHOCK %s: %.4f → %.4f (%.1f%% %s)",
            token_id, old_mid, new_mid, change_pct, direction,
        )
        return PriceShock(
            token_id=token_id,
            old_price=old_mid,
            new_price=new_mid,
            change_pct=change_pct,
            direction=direction,
            timestamp=time.time(),
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _depth_within_pct(
    levels: list[dict],
    reference_price: float,
    pct: float,
    side: str,
) -> float:
    """
    Sum the USDC value (price × size) of all levels within `pct`% of
    `reference_price` on the given side.

    Args:
        levels:          list of {"price": float, "size": float}, sorted.
        reference_price: best bid (for "bid") or best ask (for "ask").
        pct:             percentage band, e.g. 1.0 = 1%.
        side:            "bid" or "ask" (determines direction of the band).

    Returns 0.0 when reference_price is zero or levels is empty.
    """
    if reference_price < 1e-8 or not levels:
        return 0.0

    band = pct / 100.0

    if side == "bid":
        # Bids: levels at or above (reference × (1 − band))
        threshold = reference_price * (1.0 - band)
        return float(sum(
            float(b["price"]) * float(b["size"])
            for b in levels
            if float(b["price"]) >= threshold
        ))
    else:
        # Asks: levels at or below (reference × (1 + band))
        threshold = reference_price * (1.0 + band)
        return float(sum(
            float(a["price"]) * float(a["size"])
            for a in levels
            if float(a["price"]) <= threshold
        ))


def _compute_market_impact(
    asks: list[dict], target_usdc: float
) -> tuple[float, float]:
    """
    Simulate buying `target_usdc` of shares by walking up the ask side.

    Returns:
        (effective_avg_price, total_shares_bought)

    If the ask side is thinner than the order, the effective price is capped
    at the last available ask level; `shares_bought` will be < target / price.
    """
    if not asks:
        return 1.0, 0.0

    remaining     = float(target_usdc)
    cost          = 0.0
    shares_bought = 0.0

    for ask in asks:
        price = float(ask["price"])
        size  = float(ask["size"])
        if price <= 0 or size <= 0:
            continue

        level_cost = price * size
        if level_cost <= remaining:
            cost          += level_cost
            shares_bought += size
            remaining     -= level_cost
            if remaining < 1e-8:
                break
        else:
            # Partial fill at this level
            shares_this = remaining / price
            cost         += remaining
            shares_bought += shares_this
            remaining     = 0.0
            break

    if shares_bought < 1e-10:
        return float(asks[0]["price"]) if asks else 1.0, 0.0

    return cost / shares_bought, shares_bought


def _compute_score(m: OrderbookMetrics) -> tuple[float, str]:
    """
    Compute composite orderbook signal score (0–10) and direction.

    Components:
        1. Tight-band imbalance (1% depth)     weighted 40%  → bullish when +
        2. Spread quality                        weighted 20%  → tight = better
        3. Thinness penalty                      weighted 20%  → thick = better
        4. Market impact cost                    weighted 10%  → low cost = better
        5. Full-book imbalance confirmation      weighted 10%

    The imbalance signals are directional; the rest are quality modifiers.
    The final score is normalised to 0–10 with a neutral baseline of 5.0.
    """
    # ── 1. Tight-band imbalance (primary directional signal) ──────────────────
    # Ranges: −1 to +1 → mapped to 0–10 contribution
    imbalance_score = float(np.clip((m.depth_imbalance_1pct + 1.0) * 5.0, 0.0, 10.0))

    # ── 2. Spread quality (tighter = more liquid = better execution) ──────────
    # spread_pct < 2% → excellent (10); > 20% → poor (0)
    spread_score = float(np.clip(10.0 - m.spread_pct * 0.5, 0.0, 10.0))

    # ── 3. Book thickness (low thinness = deep book = good) ───────────────────
    thickness_score = float(np.clip((1.0 - m.thinness_score) * 10.0, 0.0, 10.0))

    # ── 4. Market impact cost (low impact = good) ─────────────────────────────
    # impact_500_pct in [0, 10%] → score 10..0
    impact_score = float(np.clip(10.0 - m.market_impact_500_pct * 2.0, 0.0, 10.0))

    # ── 5. Full-book imbalance confirmation ───────────────────────────────────
    full_imbalance_score = float(np.clip((m.bid_ask_imbalance + 1.0) * 5.0, 0.0, 10.0))

    # ── Weighted composite ────────────────────────────────────────────────────
    score = (
        imbalance_score    * 0.40
        + spread_score     * 0.20
        + thickness_score  * 0.20
        + impact_score     * 0.10
        + full_imbalance_score * 0.10
    )
    score = float(np.clip(score, 0.0, 10.0))

    # ── Direction from the tight-band (1%) imbalance ──────────────────────────
    if m.depth_imbalance_1pct > 0.15:
        direction = "YES"
    elif m.depth_imbalance_1pct < -0.15:
        direction = "NO"
    else:
        direction = "NEUTRAL"

    return score, direction
