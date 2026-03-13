"""
Wallet tier classification and insider scoring.
Module 4, Step A (metrics) + Step D (insider detection).
"""
import logging
import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy import stats

from utils.helpers import now_ts, safe_div, sharpe_ratio, win_rate, profit_factor

logger = logging.getLogger(__name__)

# ── Tier Tags ─────────────────────────────────────────────────────────────────

TIER_LEGENDARY = "LEGENDARY"
TIER_1_WHALE = "TIER_1_WHALE"
TIER_2_SHARK = "TIER_2_SHARK"
TIER_3_FISH = "TIER_3_FISH"
TIER_SMART_MONEY = "SMART_MONEY"
TIER_INSIDER_ALERT = "INSIDER_ALERT"
TIER_BOT = "BOT"
TIER_NOISE = "NOISE"

# Tier weights for smart-money conviction calculation
TIER_WEIGHTS: dict[str, float] = {
    TIER_LEGENDARY: 5.0,
    TIER_INSIDER_ALERT: 4.0,
    TIER_SMART_MONEY: 3.0,
    TIER_1_WHALE: 2.0,
    TIER_2_SHARK: 1.5,
    TIER_3_FISH: 1.0,
    TIER_BOT: 0.1,
    TIER_NOISE: 0.1,
}


def compute_wallet_metrics(trades: list[dict], positions: list[dict]) -> dict:
    """
    Compute all metrics for a wallet from its trade and position history.
    Returns a dict matching the whale_wallets schema.
    """
    if not trades:
        return _empty_metrics()

    closed_trades = [t for t in trades if t.get("status") == "closed" or t.get("pnl") is not None]
    open_trades = [t for t in trades if t.get("status") == "open"]

    # ── Basic counts ─────────────────────────────────────────────────────────
    total_trades = len(trades)

    # ── PnL ──────────────────────────────────────────────────────────────────
    pnls = []
    for t in closed_trades:
        raw_pnl = t.get("pnl") or t.get("profit") or 0.0
        try:
            pnls.append(float(raw_pnl))
        except (ValueError, TypeError):
            pass

    total_realized_pnl = sum(pnls)
    total_unrealized_pnl = sum(
        float(p.get("unrealized_pnl", 0)) for p in positions
    )
    total_pnl = total_realized_pnl + total_unrealized_pnl

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = profit_factor(pnls)

    best_trade = max(pnls) if pnls else 0.0
    worst_trade = min(pnls) if pnls else 0.0
    avg_profit_per_trade = safe_div(sum(pnls), len(pnls)) if pnls else 0.0

    # ── Volume ────────────────────────────────────────────────────────────────
    sizes = []
    for t in trades:
        try:
            sizes.append(float(t.get("size") or t.get("size_usdc") or t.get("usdcSize") or 0))
        except (ValueError, TypeError):
            pass
    total_volume = sum(sizes)
    avg_position_size = safe_div(total_volume, len(sizes)) if sizes else 0.0
    median_position_size = float(np.median(sizes)) if sizes else 0.0
    max_position_size = max(sizes) if sizes else 0.0

    # ── Win rate ──────────────────────────────────────────────────────────────
    wr = win_rate(pnls) if pnls else 0.0
    large_trades = [t for t, p in zip(closed_trades, pnls)
                    if float(t.get("size") or t.get("size_usdc") or t.get("usdcSize") or 0) > 500]
    large_pnls = [p for t, p in zip(closed_trades, pnls)
                  if float(t.get("size") or t.get("size_usdc") or t.get("usdcSize") or 0) > 500]
    win_rate_large = win_rate(large_pnls) if large_pnls else 0.0

    # ── Win/loss streaks ──────────────────────────────────────────────────────
    longest_win_streak, longest_loss_streak = _compute_streaks(pnls)

    # ── Sharpe ratio ──────────────────────────────────────────────────────────
    daily_pnl_series = _compute_daily_pnl(closed_trades, pnls)
    sharpe = sharpe_ratio(daily_pnl_series) if len(daily_pnl_series) >= 2 else 0.0

    # ── ROI ───────────────────────────────────────────────────────────────────
    roi_pct = safe_div(total_pnl, total_volume) * 100 if total_volume > 0 else 0.0

    # ── Timing analysis ───────────────────────────────────────────────────────
    avg_holding_hours = _compute_avg_holding_hours(trades)
    avg_days_before_resolution = _compute_avg_days_before_resolution(trades)
    early_bird_score = _compute_early_bird_score(trades)

    # ── Bot detection ─────────────────────────────────────────────────────────
    bot_pattern_score, is_bot = _detect_bot_pattern(trades)

    # ── Category win rates ────────────────────────────────────────────────────
    win_rate_by_category = _compute_category_win_rates(closed_trades, pnls)

    # ── Information edge metrics ─────────────────────────────────────────────
    information_lead_score = _compute_information_lead_score(trades)
    timing_alpha = _compute_timing_alpha(trades)
    news_precession_rate = _compute_news_precession_rate(trades)

    insider_score = _compute_insider_score(
        information_lead_score, timing_alpha, news_precession_rate
    )

    # ── Behavioral flags ──────────────────────────────────────────────────────
    is_contrarian = _detect_contrarian(trades)
    is_market_maker = _detect_market_maker(trades)

    # ── Additional patterns ───────────────────────────────────────────────────
    late_entry_pattern = _detect_late_entry_pattern(trades)
    low_volume_preference = _detect_low_volume_preference(trades)

    # ── Timestamps ───────────────────────────────────────────────────────────
    timestamps = []
    for t in trades:
        ts = t.get("timestamp") or t.get("created_at") or t.get("ts")
        if ts:
            try:
                timestamps.append(float(ts))
            except (ValueError, TypeError):
                pass

    first_seen = min(timestamps) if timestamps else now_ts()
    last_active = max(timestamps) if timestamps else now_ts()

    is_insider_candidate = insider_score >= 7.0 and total_trades >= 10

    import json
    full_stats = {
        "total_trades": total_trades,
        "total_realized_pnl": total_realized_pnl,
        "total_unrealized_pnl": total_unrealized_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "avg_profit_per_trade": avg_profit_per_trade,
        "roi_pct": roi_pct,
        "median_position_size": median_position_size,
        "max_position_size": max_position_size,
        "longest_win_streak": longest_win_streak,
        "longest_loss_streak": longest_loss_streak,
        "avg_holding_hours": avg_holding_hours,
        "avg_days_before_resolution": avg_days_before_resolution,
        "early_bird_score": early_bird_score,
        "bot_pattern_score": bot_pattern_score,
        "is_contrarian": is_contrarian,
        "is_market_maker": is_market_maker,
        "late_entry_pattern": late_entry_pattern,
        "low_volume_preference": low_volume_preference,
        "daily_pnl_series": daily_pnl_series[-30:],  # Keep last 30 days
    }

    return {
        "win_rate": wr,
        "win_rate_large": win_rate_large,
        "win_rate_by_category": json.dumps(win_rate_by_category),
        "total_pnl": total_pnl,
        "total_volume": total_volume,
        "profit_factor": pf,
        "sharpe_ratio": sharpe,
        "total_trades": total_trades,
        "avg_position_size": avg_position_size,
        "information_lead_score": information_lead_score,
        "timing_alpha": timing_alpha,
        "news_precession_rate": news_precession_rate,
        "insider_score": insider_score,
        "is_bot": int(is_bot),
        "is_insider_candidate": int(is_insider_candidate),
        "is_copy_trader": 0,  # set by graph analysis
        "first_seen": first_seen,
        "last_active": last_active,
        "last_refreshed": now_ts(),
        "full_stats_json": json.dumps(full_stats, default=str),
    }


def classify_wallet_tiers(metrics: dict) -> list[str]:
    """Return list of tier tags for a wallet based on its metrics."""
    tags: list[str] = []
    total_pnl = metrics.get("total_pnl", 0)
    total_volume = metrics.get("total_volume", 0)
    wr = metrics.get("win_rate", 0)
    total_trades = metrics.get("total_trades", 0)
    pf = metrics.get("profit_factor", 0)
    insider_score = metrics.get("insider_score", 0)
    is_bot = bool(metrics.get("is_bot", 0))

    # Legendary
    if total_pnl > 1_000_000 or (wr > 0.72 and total_trades > 50):
        tags.append(TIER_LEGENDARY)

    # Tier 1
    if total_volume > 500_000 or total_pnl > 100_000:
        tags.append(TIER_1_WHALE)

    # Tier 2
    if total_volume > 100_000 or total_pnl > 20_000:
        if TIER_1_WHALE not in tags:
            tags.append(TIER_2_SHARK)

    # Tier 3
    if total_volume > 10_000 or total_pnl > 2_000:
        if TIER_1_WHALE not in tags and TIER_2_SHARK not in tags:
            tags.append(TIER_3_FISH)

    # Smart money
    if wr > 0.65 and total_trades >= 20 and pf > 1.5:
        tags.append(TIER_SMART_MONEY)

    # Insider alert
    if insider_score >= 7.0 and total_trades >= 10:
        tags.append(TIER_INSIDER_ALERT)

    # Bot
    if is_bot:
        tags.append(TIER_BOT)

    # Default
    if not tags:
        tags.append(TIER_NOISE)

    return tags


def get_tier_weight(tier_tags: list[str]) -> float:
    """Get max tier weight from a list of tags."""
    return max((TIER_WEIGHTS.get(t, 0.1) for t in tier_tags), default=0.1)


def should_copy_trade(tier_tags: list[str], win_rate: float, alert_level: str) -> bool:
    """Determine if we should copy-trade this wallet's signal."""
    if alert_level == "CRITICAL":
        return True
    if alert_level == "HIGH":
        if TIER_LEGENDARY in tier_tags or TIER_INSIDER_ALERT in tier_tags:
            return True
        if TIER_SMART_MONEY in tier_tags and win_rate > 0.65:
            return True
        if TIER_1_WHALE in tier_tags and win_rate > 0.60:
            return True
    if alert_level == "MEDIUM":
        return TIER_LEGENDARY in tier_tags or TIER_INSIDER_ALERT in tier_tags
    return False


def compute_alert_level(
    tier_tags: list[str],
    size_usdc: float,
    win_rate: float,
    insider_score: float,
) -> str:
    """Determine alert level for a wallet trade."""
    if TIER_LEGENDARY in tier_tags or TIER_INSIDER_ALERT in tier_tags:
        if size_usdc >= 2000:
            return "CRITICAL"
        return "HIGH"
    if win_rate > 0.80 and size_usdc >= 1000:
        return "CRITICAL"
    if TIER_1_WHALE in tier_tags and size_usdc >= 1000:
        return "HIGH"
    if TIER_SMART_MONEY in tier_tags and size_usdc >= 500:
        return "HIGH"
    if TIER_2_SHARK in tier_tags and size_usdc >= 2000:
        return "MEDIUM"
    if TIER_SMART_MONEY in tier_tags:
        return "MEDIUM"
    if TIER_3_FISH in tier_tags:
        return "LOW"
    return "LOW"


# ── Private helpers ───────────────────────────────────────────────────────────

def _empty_metrics() -> dict:
    import json
    return {
        "win_rate": 0.0, "win_rate_large": 0.0, "win_rate_by_category": "{}",
        "total_pnl": 0.0, "total_volume": 0.0, "profit_factor": 0.0,
        "sharpe_ratio": 0.0, "total_trades": 0, "avg_position_size": 0.0,
        "information_lead_score": 0.0, "timing_alpha": 0.0,
        "news_precession_rate": 0.0, "insider_score": 0.0,
        "is_bot": 0, "is_insider_candidate": 0, "is_copy_trader": 0,
        "first_seen": now_ts(), "last_active": now_ts(),
        "last_refreshed": now_ts(), "full_stats_json": "{}",
    }


def _compute_streaks(pnls: list[float]) -> tuple[int, int]:
    if not pnls:
        return 0, 0
    max_win, max_loss = 0, 0
    cur_win, cur_loss = 0, 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def _compute_daily_pnl(trades: list[dict], pnls: list[float]) -> list[float]:
    """Aggregate PnL into daily buckets."""
    from collections import defaultdict
    daily: dict[str, float] = defaultdict(float)
    for t, p in zip(trades, pnls):
        ts = t.get("timestamp") or t.get("created_at") or t.get("ts")
        if ts:
            try:
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                day_key = dt.strftime("%Y-%m-%d")
                daily[day_key] += p
            except (ValueError, TypeError):
                pass
    return list(daily.values())


def _compute_avg_holding_hours(trades: list[dict]) -> float:
    holding_times = []
    for t in trades:
        opened = t.get("created_at") or t.get("timestamp") or t.get("ts")
        closed = t.get("closed_at") or t.get("resolved_at")
        if opened and closed:
            try:
                delta = float(closed) - float(opened)
                holding_times.append(delta / 3600.0)
            except (ValueError, TypeError):
                pass
    return float(np.mean(holding_times)) if holding_times else 0.0


def _compute_avg_days_before_resolution(trades: list[dict]) -> float:
    leads = []
    for t in trades:
        ts = t.get("timestamp") or t.get("created_at") or t.get("ts")
        end_date = t.get("end_date") or t.get("endDate") or t.get("resolution_date")
        if ts and end_date:
            try:
                days = (float(end_date) - float(ts)) / 86400.0
                if 0 < days < 365:
                    leads.append(days)
            except (ValueError, TypeError):
                pass
    return float(np.mean(leads)) if leads else 30.0


def _compute_early_bird_score(trades: list[dict]) -> float:
    """
    Score 0-10: how often the wallet enters before a significant price move.
    Uses price_before vs price_after if available.
    """
    scores = []
    for t in trades:
        before = t.get("price_1h_before") or t.get("entry_price")
        after = t.get("price_24h_after") or t.get("price_after")
        if before and after and before > 0:
            try:
                move = (float(after) - float(before)) / float(before)
                side = t.get("outcome", "YES")
                if side == "YES":
                    scores.append(1.0 if move > 0.02 else 0.0)
                else:
                    scores.append(1.0 if move < -0.02 else 0.0)
            except (ValueError, TypeError):
                pass
    if not scores:
        return 5.0  # neutral default
    return float(np.mean(scores)) * 10.0


def _detect_bot_pattern(trades: list[dict]) -> tuple[float, bool]:
    """Check if trade intervals have very low variance (bot signature)."""
    if len(trades) < 10:
        return 0.0, False
    timestamps = []
    for t in trades:
        ts = t.get("timestamp") or t.get("created_at") or t.get("ts")
        if ts:
            try:
                timestamps.append(float(ts))
            except (ValueError, TypeError):
                pass
    if len(timestamps) < 10:
        return 0.0, False
    timestamps.sort()
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    if not intervals:
        return 0.0, False
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    if mean_interval == 0:
        return 0.0, False
    cv = std_interval / mean_interval  # coefficient of variation
    bot_pattern_score = max(0.0, 1.0 - cv)  # low CV = high bot score

    # Also check active hours (bots are active > 18h/day)
    hours = set()
    for ts in timestamps:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hours.add(dt.hour)
    is_bot = cv < 0.05 and len(hours) > 18
    return bot_pattern_score, is_bot


def _compute_category_win_rates(trades: list[dict], pnls: list[float]) -> dict:
    from collections import defaultdict
    category_results: dict[str, list[float]] = defaultdict(list)
    for t, p in zip(trades, pnls):
        cat = str(t.get("category") or t.get("market_category") or "unknown").lower()
        category_results[cat].append(p)
    return {
        cat: safe_div(sum(1 for p in cat_pnls if p > 0), len(cat_pnls))
        for cat, cat_pnls in category_results.items()
        if len(cat_pnls) >= 3
    }


def _compute_information_lead_score(trades: list[dict]) -> float:
    """
    0-1: how often price moved in wallet's direction after entry.
    Uses price_1h_after, price_6h_after, price_24h_after if available.
    """
    scored = []
    for t in trades:
        entry_price = float(t.get("price") or t.get("entry_price") or 0.5)
        outcome = str(t.get("outcome", "YES")).upper()
        is_yes = outcome == "YES"

        moves = []
        for horizon in ["price_1h_after", "price_6h_after", "price_24h_after"]:
            p_after = t.get(horizon)
            if p_after:
                try:
                    move = float(p_after) - entry_price
                    correct = move > 0 if is_yes else move < 0
                    weight = {"price_1h_after": 0.2, "price_6h_after": 0.3, "price_24h_after": 0.5}[horizon]
                    moves.append((correct, abs(move), weight))
                except (ValueError, TypeError):
                    pass
        if moves:
            total_weight = sum(w for _, _, w in moves)
            weighted_correct = sum(w * abs(m) if c else 0 for c, m, w in moves)
            weighted_total = sum(w * abs(m) for _, m, w in moves)
            scored.append(safe_div(weighted_correct, weighted_total))

    return float(np.mean(scored)) if scored else 0.5


def _compute_timing_alpha(trades: list[dict]) -> float:
    """
    Avg (entry_price - price_24h_before) for YES buys.
    Negative = they buy before price rises = strong signal.
    """
    alphas = []
    for t in trades:
        outcome = str(t.get("outcome", "YES")).upper()
        if outcome != "YES":
            continue
        entry = t.get("price") or t.get("entry_price")
        before = t.get("price_24h_before")
        if entry and before:
            try:
                alphas.append(float(entry) - float(before))
            except (ValueError, TypeError):
                pass
    return float(np.mean(alphas)) if alphas else 0.0


def _compute_news_precession_rate(trades: list[dict]) -> float:
    """
    Fraction of trades where a price-moving event (>8%) happened within 24h after entry.
    """
    if not trades:
        return 0.0
    precession_count = 0
    eligible = 0
    for t in trades:
        after = t.get("price_24h_after")
        entry = t.get("price") or t.get("entry_price")
        if after and entry:
            try:
                eligible += 1
                move = abs(float(after) - float(entry))
                if move > 0.08:
                    precession_count += 1
            except (ValueError, TypeError):
                pass
    return safe_div(precession_count, eligible) if eligible > 0 else 0.0


def _compute_insider_score(
    information_lead_score: float,
    timing_alpha: float,
    news_precession_rate: float,
) -> float:
    """Composite insider score 0-10."""
    lead_component = information_lead_score * 4.0
    timing_component = min(max(0.0, -timing_alpha) * 20, 3.0)
    precession_component = news_precession_rate * 3.0
    raw = lead_component + timing_component + precession_component
    return min(10.0, max(0.0, raw))


def _detect_contrarian(trades: list[dict]) -> bool:
    """True if wallet trades against momentum > 60% of the time."""
    against = 0
    eligible = 0
    for t in trades:
        momentum = t.get("momentum_at_entry")
        outcome = str(t.get("outcome", "YES")).upper()
        if momentum is not None:
            eligible += 1
            try:
                m = float(momentum)
                if (m > 0 and outcome == "NO") or (m < 0 and outcome == "YES"):
                    against += 1
            except (ValueError, TypeError):
                pass
    if eligible < 5:
        return False
    return safe_div(against, eligible) > 0.60


def _detect_market_maker(trades: list[dict]) -> bool:
    """True if wallet places orders on both sides > 30% of markets."""
    from collections import defaultdict
    market_sides: dict[str, set] = defaultdict(set)
    for t in trades:
        slug = t.get("market_slug") or t.get("market") or t.get("condition_id", "")
        outcome = str(t.get("outcome", "YES")).upper()
        market_sides[slug].add(outcome)
    if not market_sides:
        return False
    both_sides = sum(1 for sides in market_sides.values() if len(sides) >= 2)
    return safe_div(both_sides, len(market_sides)) > 0.30


def _detect_late_entry_pattern(trades: list[dict]) -> bool:
    """True if >40% of trades are within 3 days of resolution."""
    late = 0
    eligible = 0
    for t in trades:
        ts = t.get("timestamp") or t.get("created_at") or t.get("ts")
        end_date = t.get("end_date") or t.get("endDate") or t.get("resolution_date")
        if ts and end_date:
            try:
                days_remaining = (float(end_date) - float(ts)) / 86400.0
                eligible += 1
                if 0 < days_remaining < 3:
                    late += 1
            except (ValueError, TypeError):
                pass
    if eligible < 5:
        return False
    return safe_div(late, eligible) > 0.40


def _detect_low_volume_preference(trades: list[dict]) -> bool:
    """True if wallet prefers markets with <$10k daily volume."""
    low_volume = 0
    eligible = 0
    for t in trades:
        vol = t.get("volume_at_entry") or t.get("volume24h")
        if vol:
            try:
                eligible += 1
                if float(vol) < 10_000:
                    low_volume += 1
            except (ValueError, TypeError):
                pass
    if eligible < 5:
        return False
    return safe_div(low_volume, eligible) > 0.50
