"""
Technical analysis module — Module 3.
Computes all price/volume indicators and pattern flags from hourly closes.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    # ── Momentum ──────────────────────────────────────────────────────────────
    price_change_1h: float = 0.0
    price_change_6h: float = 0.0
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0
    volume_vs_7d_avg: float = 1.0

    # ── Volatility ────────────────────────────────────────────────────────────
    volatility_24h: float = 0.0
    volatility_7d: float = 0.0
    atr_14: float = 0.0
    atr_pct: float = 0.0        # ATR expressed as fraction of current price

    # ── Trend ─────────────────────────────────────────────────────────────────
    sma_12h: float = 0.5
    sma_24h: float = 0.5
    ema_12h: float = 0.5
    ema_24h: float = 0.5        # 24-period EMA on hourly closes
    trend_slope_24h: float = 0.0
    trend_slope_7d: float = 0.0
    trend_r2: float = 0.0       # R² for the 24h linear regression
    trend_r2_7d: float = 0.0    # R² for the 7-day linear regression

    # ── Oscillators ───────────────────────────────────────────────────────────
    rsi_14: float = 50.0        # RSI-14 on hourly closes (Wilder's smoothing)
    bb_upper: float = 1.0
    bb_mid: float = 0.5
    bb_lower: float = 0.0
    bb_position: float = 0.5    # 0 = at lower band, 1 = at upper band
    bb_width: float = 1.0       # (upper − lower) / mid — Bollinger bandwidth
    bb_squeeze: bool = False    # True when bandwidth < 10% (compressed)

    # ── Volume / VWAP ─────────────────────────────────────────────────────────
    vwap_24h: float = 0.5
    vwap_deviation: float = 0.0     # (current_price − vwap) / vwap
    volume_spike: bool = False      # volume > 2× 7-day average
    unusual_volume: bool = False    # volume > 3× 7-day average

    # ── Patterns ──────────────────────────────────────────────────────────────
    drift_to_extreme: bool = False
    overreaction_spike: bool = False
    pre_resolution_compression: bool = False
    stagnant_midpoint: bool = False
    late_smart_money: bool = False
    mean_reversion_due: bool = False

    # ── Summary ───────────────────────────────────────────────────────────────
    technical_score: float = 5.0
    technical_direction: str = "NEUTRAL"


def compute_technical_signals(
    price_history: list[dict],
    current_price: float,
    volume_24h: float,
    days_to_resolution: float,
) -> TechnicalSignal:
    """
    Compute all technical indicators from hourly price history.

    Args:
        price_history:      list of {t: unix_ts, p: price, v: volume} dicts.
                            Also accepts {timestamp, price/close/c, volume/vol}.
        current_price:      most recent observed price in [0, 1].
        volume_24h:         total USDC volume traded in the last 24 hours.
        days_to_resolution: calendar days until the market resolves.

    Returns:
        TechnicalSignal with all indicators populated.  Falls back to safe
        neutral defaults when fewer than the required data points exist.
        Handles: empty history, < 3 points, all-constant prices, single point.
    """
    sig = TechnicalSignal()

    # Normalise current price into a valid binary-market range
    current_price = float(np.clip(current_price, 0.001, 0.999))

    if not price_history:
        return sig

    df = _build_df(price_history)
    if df is None or len(df) < 3:
        # Minimal 2-point momentum estimate
        if df is not None and len(df) == 2:
            closes = df["close"].values
            sig.price_change_1h = _pct_change_at(closes, current_price, 1)
        return sig

    closes = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    n = len(closes)

    # Guard: constant price array → nothing meaningful
    if np.ptp(closes) < 1e-8:
        logger.debug("Price history is constant — skipping technical indicators")
        return sig

    # ── Momentum ──────────────────────────────────────────────────────────────
    sig.price_change_1h  = _pct_change_at(closes, current_price, 1)
    sig.price_change_6h  = _pct_change_at(closes, current_price, 6)
    sig.price_change_24h = _pct_change_at(closes, current_price, 24)
    sig.price_change_7d  = _pct_change_at(closes, current_price, 168)

    # ── Volume vs 7-day average ───────────────────────────────────────────────
    window_v = min(168, n)
    avg_vol = float(np.mean(volumes[-window_v:])) if window_v > 0 else 1.0
    sig.volume_vs_7d_avg = float(volume_24h / avg_vol) if avg_vol > 1e-8 else 1.0
    sig.volume_spike   = sig.volume_vs_7d_avg > 2.0
    sig.unusual_volume = sig.volume_vs_7d_avg > 3.0

    # ── Volatility (annualised hourly std of returns) ──────────────────────────
    if n >= 25:
        ret24 = np.diff(closes[-25:]) / np.maximum(closes[-25:-1], 1e-6)
        sig.volatility_24h = float(np.std(ret24) * np.sqrt(24))
    elif n >= 4:
        ret = np.diff(closes) / np.maximum(closes[:-1], 1e-6)
        sig.volatility_24h = float(np.std(ret) * np.sqrt(min(n, 24)))

    if n >= 169:
        ret7d = np.diff(closes[-169:]) / np.maximum(closes[-169:-1], 1e-6)
        sig.volatility_7d = float(np.std(ret7d) * np.sqrt(168))

    # ── ATR-14 (Wilder's smoothed true range on hourly closes) ────────────────
    if n >= 15:
        sig.atr_14, sig.atr_pct = _compute_atr(closes, current_price, period=14)
    elif n >= 4:
        sig.atr_14 = float(np.mean(np.abs(np.diff(closes[-n:]))))
        sig.atr_pct = sig.atr_14 / max(current_price, 1e-6)

    # ── Simple Moving Averages ────────────────────────────────────────────────
    if n >= 12:
        sig.sma_12h = float(np.mean(closes[-12:]))
    if n >= 24:
        sig.sma_24h = float(np.mean(closes[-24:]))

    # ── EMA 12h and 24h ───────────────────────────────────────────────────────
    # Use 3× span as warm-up window so initial value bias is negligible.
    if n >= 12:
        warmup12 = min(n, max(36, 3 * 12))
        sig.ema_12h = float(_ema(closes[-warmup12:], 12)[-1])
    if n >= 24:
        warmup24 = min(n, max(72, 3 * 24))
        sig.ema_24h = float(_ema(closes[-warmup24:], 24)[-1])

    # ── Linear Regression Slope + R² for 24h ─────────────────────────────────
    if n >= 24:
        x24 = np.arange(24, dtype=float)
        y24 = closes[-24:]
        if np.std(y24) > 1e-8:
            slope, _, r, _, _ = scipy_stats.linregress(x24, y24)
            sig.trend_slope_24h = float(slope)
            sig.trend_r2        = float(r ** 2)
    elif n >= 4:
        x = np.arange(n, dtype=float)
        y = closes
        if np.std(y) > 1e-8:
            slope, _, r, _, _ = scipy_stats.linregress(x, y)
            sig.trend_slope_24h = float(slope)
            sig.trend_r2        = float(r ** 2)

    # ── Linear Regression Slope + R² for 7d ──────────────────────────────────
    if n >= 168:
        x7d = np.arange(168, dtype=float)
        y7d = closes[-168:]
        if np.std(y7d) > 1e-8:
            slope7, _, r7, _, _ = scipy_stats.linregress(x7d, y7d)
            sig.trend_slope_7d = float(slope7)
            sig.trend_r2_7d    = float(r7 ** 2)
    else:
        # Fall back to 24h values when insufficient history
        sig.trend_slope_7d = sig.trend_slope_24h
        sig.trend_r2_7d    = sig.trend_r2

    # ── RSI-14 on hourly closes ───────────────────────────────────────────────
    if n >= 15:
        sig.rsi_14 = float(_compute_rsi(closes, period=14))
    elif n >= 4:
        sig.rsi_14 = float(_compute_rsi(closes, period=max(2, n - 1)))

    # ── Bollinger Bands (20-period, 2 std) ────────────────────────────────────
    bb_len = min(20, n)
    if bb_len >= 5:
        window = closes[-bb_len:]
        sig.bb_mid  = float(np.mean(window))
        bb_std = float(np.std(window, ddof=min(1, bb_len - 1)))
        sig.bb_upper = sig.bb_mid + 2.0 * bb_std
        sig.bb_lower = sig.bb_mid - 2.0 * bb_std
        bb_range = sig.bb_upper - sig.bb_lower
        if bb_range > 1e-8:
            raw_pos = (current_price - sig.bb_lower) / bb_range
            sig.bb_position = float(np.clip(raw_pos, -0.5, 1.5))
        else:
            sig.bb_position = 0.5
        sig.bb_width   = bb_range / max(sig.bb_mid, 1e-6)
        sig.bb_squeeze = sig.bb_width < 0.10   # bandwidth < 10% = squeeze

    # ── VWAP 24h with deviation ───────────────────────────────────────────────
    vwap_n   = min(24, n)
    vols_w   = volumes[-vwap_n:]
    prices_w = closes[-vwap_n:]
    total_vol = float(np.sum(vols_w))
    if total_vol > 1e-8:
        sig.vwap_24h = float(np.dot(vols_w, prices_w) / total_vol)
    else:
        sig.vwap_24h = float(np.mean(prices_w))
    sig.vwap_deviation = float(
        (current_price - sig.vwap_24h) / max(sig.vwap_24h, 1e-6)
    )

    # ── Pattern Detection ─────────────────────────────────────────────────────
    recent_24 = closes[-min(24, n):]
    recent_12 = closes[-min(12, n):]
    recent_6  = closes[-min(6,  n):]

    sig.drift_to_extreme = _detect_drift_to_extreme(recent_24)
    sig.overreaction_spike = _detect_overreaction_spike(
        recent_6, current_price, sig.atr_14, sig.volatility_24h
    )
    sig.pre_resolution_compression = _detect_pre_resolution_compression(
        recent_12, current_price, days_to_resolution, sig.volatility_24h
    )
    sig.stagnant_midpoint = _detect_stagnant_midpoint(
        recent_24, current_price, sig.volume_vs_7d_avg
    )
    sig.late_smart_money = _detect_late_smart_money(
        sig.unusual_volume, days_to_resolution,
        sig.volume_vs_7d_avg, sig.price_change_1h
    )
    sig.mean_reversion_due = _detect_mean_reversion_due(
        current_price, sig.vwap_deviation, sig.rsi_14,
        sig.bb_position, sig.trend_r2
    )

    # ── Composite technical score ─────────────────────────────────────────────
    sig.technical_score, sig.technical_direction = _compute_score(sig, current_price)

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Data builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_df(price_history: list[dict]) -> Optional[pd.DataFrame]:
    """
    Build a sorted DataFrame from raw price history dicts.
    Accepts multiple field-name conventions:
      - timestamp: t, timestamp, time, ts
      - price:     p, price, close, c, mid
      - volume:    v, volume, vol  (defaults to 1.0 when absent)
    Prices are clipped to [0.001, 0.999] to stay in binary-market bounds.
    Returns None if no valid rows could be parsed.
    """
    rows = []
    for item in price_history:
        if not isinstance(item, dict):
            continue
        t = (item.get("t") or item.get("timestamp")
             or item.get("time") or item.get("ts"))
        p = (item.get("p") or item.get("price") or item.get("close")
             or item.get("c") or item.get("mid"))
        v = item.get("v") or item.get("volume") or item.get("vol")
        if t is None or p is None:
            continue
        try:
            t_f = float(t)
            p_f = float(np.clip(float(p), 0.001, 0.999))
            v_f = max(float(v), 0.0) if v is not None else 1.0
            rows.append({"timestamp": t_f, "close": p_f, "volume": v_f})
        except (ValueError, TypeError):
            continue

    if not rows:
        return None

    df = (pd.DataFrame(rows)
          .sort_values("timestamp")
          .drop_duplicates("timestamp")
          .reset_index(drop=True))
    df["close"] = df["close"].clip(0.001, 0.999)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Indicator calculations
# ─────────────────────────────────────────────────────────────────────────────

def _pct_change_at(closes: np.ndarray, current: float, hours_back: int) -> float:
    """
    Percentage change from `hours_back` periods ago to `current`.
    Uses the oldest available close when not enough history exists.
    """
    n = len(closes)
    idx = max(0, n - hours_back - 1)
    past = float(closes[idx])
    if past < 1e-8:
        return 0.0
    return (current - past) / past


def _ema(prices: np.ndarray, span: int) -> np.ndarray:
    """
    Compute EMA with the standard multiplier alpha = 2 / (span + 1).
    Initialised at the first price value to avoid look-ahead bias.
    Returns an array of the same length as `prices`.
    """
    if len(prices) == 0:
        return np.array([], dtype=float)
    alpha = 2.0 / (span + 1)
    ema = np.empty(len(prices), dtype=float)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """
    Compute RSI using Wilder's smoothing (standard definition).
      - Returns 50.0 when fewer than period+1 data points are available.
      - Returns 100.0 when average loss is effectively zero (all gains).
      - Returns 50.0 when both average gain and loss are zero (flat market).
    """
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes.astype(float))
    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average over the first `period` bars
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Wilder's smoothing for the remaining bars
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss < 1e-10:
        return 100.0 if avg_gain > 1e-10 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _compute_atr(
    closes: np.ndarray, current_price: float, period: int = 14
) -> tuple[float, float]:
    """
    Compute ATR-14 using Wilder's smoothing.
    For binary prediction-market tokens (single price per period) the True
    Range is simply |close[i] − close[i−1]|.
    Returns:
        (atr_absolute, atr_as_fraction_of_current_price)
    Edge cases:
        < period+1 bars  → (0.0, 0.0)
        current_price ≤ 0 → atr_pct uses 1e-6 floor
    """
    if len(closes) < period + 1:
        return 0.0, 0.0

    true_ranges = np.abs(np.diff(closes.astype(float)))

    # Seed: simple mean of first `period` true ranges
    atr = float(np.mean(true_ranges[:period]))

    # Wilder's smoothing
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period

    atr_pct = atr / max(current_price, 1e-6)
    return float(atr), float(atr_pct)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern detectors
# ─────────────────────────────────────────────────────────────────────────────

def _detect_drift_to_extreme(recent_closes: np.ndarray) -> bool:
    """
    True if price has drifted monotonically toward 0 or 1 for the last
    12 consecutive hourly periods.

    A monotone drift near a binary extreme (> 0.75 or < 0.25) often indicates
    informed participants pricing in an approaching resolution.

    Edge cases:
        < 12 bars → False (insufficient evidence)
        Perfectly flat closes → False (diffs are zero, not monotone to extreme)
    """
    if len(recent_closes) < 12:
        return False

    last_12 = recent_closes[-12:].astype(float)
    diffs   = np.diff(last_12)

    # Allow tiny floating-point noise (1e-6 tolerance)
    monotone_up   = bool(np.all(diffs >= -1e-6)) and float(last_12[-1]) > 0.75
    monotone_down = bool(np.all(diffs <=  1e-6)) and float(last_12[-1]) < 0.25
    return monotone_up or monotone_down


def _detect_overreaction_spike(
    recent_6: np.ndarray,
    current_price: float,
    atr_14: float,
    volatility_24h: float,
) -> bool:
    """
    True when a sharp spike has occurred recently, suggesting an overreaction
    that may be partially or fully reversed.

    Conditions (any one triggers):
      a) Current price moved > 10% from 6 hours ago.
      b) The last 1h move is > 3× ATR-14 (when ATR is available).
      c) A spike-and-partial-revert pattern over 5+ candles: the midpoint
         deviated > 8% from the rolling mean and price has since partially
         retraced (> 35% of the deviation recovered).

    Edge cases:
        < 2 bars          → False
        atr_14 ≈ 0        → condition b skipped
        < 5 bars          → condition c skipped
    """
    if len(recent_6) < 2:
        return False

    # Condition a: large move versus 6h ago
    p0 = float(recent_6[0])
    if p0 > 1e-6 and abs(current_price - p0) / p0 > 0.10:
        return True

    # Condition b: last candle move > 3× ATR
    last_move = abs(float(recent_6[-1]) - float(recent_6[-2]))
    if atr_14 > 1e-8 and last_move > 3.0 * atr_14:
        return True

    # Condition c: spike-and-partial-revert (≥ 5 candles)
    if len(recent_6) >= 5:
        body     = recent_6[:-1].astype(float)
        mid_val  = float(np.mean(body))
        max_dev  = float(np.max(np.abs(body - mid_val)))
        if max_dev > 0.08:
            peak_idx  = int(np.argmax(np.abs(body - mid_val)))
            peak_val  = float(body[peak_idx])
            revert    = abs(current_price - peak_val) / max_dev
            if revert > 0.35:
                return True

    return False


def _detect_pre_resolution_compression(
    recent_12: np.ndarray,
    current_price: float,
    days_to_resolution: float,
    volatility_24h: float,
) -> bool:
    """
    True when price is compressing within 3 days of resolution while
    remaining in the uncertain zone (not near 0 or 1).

    Compression criteria (any one suffices when timing check passes):
      a) Price range over the last 12h is < 5%.
      b) Realised 24h volatility is < 3% AND price is between 0.15–0.85.
      c) Proximity fallback: < 1.5 days to resolution AND price 0.20–0.80.

    Edge cases:
        > 3 days to resolution → False
        price near resolved (< 0.08 or > 0.92) → False
        < 6 bars → falls through to condition c only
    """
    if days_to_resolution > 3.0:
        return False
    if current_price < 0.08 or current_price > 0.92:
        return False

    if len(recent_12) >= 6:
        price_range = float(np.ptp(recent_12.astype(float)))
        if price_range < 0.05:
            return True
        if volatility_24h < 0.03 and 0.15 < current_price < 0.85:
            return True

    return days_to_resolution < 1.5 and 0.20 < current_price < 0.80


def _detect_stagnant_midpoint(
    recent_24: np.ndarray,
    current_price: float,
    volume_vs_7d_avg: float,
) -> bool:
    """
    True when price has been pinned near 0.50 for the majority of the last
    24 hours despite non-depressed volume — indicating persistent uncertainty
    and a possible binary catalyst approaching.

    Criteria (all must pass):
      1. Current price within ±8% of 0.50.
      2. ≥ 60% of the last 24 closes are within ±10% of 0.50.
      3. Volume is not severely depressed (≥ 50% of 7d average).

    Edge cases:
        < 6 bars → condition 2 cannot be evaluated robustly → False
    """
    if abs(current_price - 0.50) > 0.08:
        return False
    if len(recent_24) < 6:
        return False

    near_mid = int(np.sum(np.abs(recent_24.astype(float) - 0.50) < 0.10))
    if near_mid / len(recent_24) < 0.60:
        return False

    return volume_vs_7d_avg >= 0.50


def _detect_late_smart_money(
    unusual_volume: bool,
    days_to_resolution: float,
    volume_vs_7d_avg: float,
    price_change_1h: float,
) -> bool:
    """
    True when unusual volume is occurring within 5 days of resolution
    AND is accompanied by a directional price move (> 1% in the last hour).

    This pattern often indicates informed participants positioning ahead of
    a known event or imminent resolution — a classic pre-resolution tell.

    Conditions (all required):
      1. ≤ 5 days to resolution.
      2. Volume > 3× 7d average (unusual_volume flag) OR ≥ 2.5× average.
      3. Last 1h price change is directional (absolute value ≥ 1%).

    Edge cases handled:
        days_to_resolution == 0 (already resolved) → still returns True
        price_change_1h ≈ 0 (flat despite volume) → False (noise filter)
    """
    if days_to_resolution > 5.0:
        return False
    if not unusual_volume and volume_vs_7d_avg < 2.5:
        return False
    return abs(price_change_1h) >= 0.01


def _detect_mean_reversion_due(
    current_price: float,
    vwap_deviation: float,
    rsi_14: float,
    bb_position: float,
    trend_r2: float,
) -> bool:
    """
    True when multiple independent indicators agree that price is over-extended
    from equilibrium and a mean-reversion move is statistically likely.

    Scoring system (triggers at ≥ 3 points):
      +1  |VWAP deviation| > 8%
      +1  |VWAP deviation| > 15% (extra emphasis)
      +1  RSI > 75 or < 25  (moderately extreme)
      +1  RSI > 85 or < 15  (severely extreme)
      +1  BB position < 0.10 or > 0.90 (at band extreme)
      +1  Trend R² < 0.30  (no clean trend — drift more likely to revert)
      +1  Price at binary extreme (> 0.90 or < 0.10)

    Requiring 3+ signals reduces false positives in trending markets.
    """
    score = 0

    if abs(vwap_deviation) > 0.08:
        score += 1
    if abs(vwap_deviation) > 0.15:
        score += 1

    if rsi_14 > 75 or rsi_14 < 25:
        score += 1
    if rsi_14 > 85 or rsi_14 < 15:
        score += 1

    if bb_position > 0.90 or bb_position < 0.10:
        score += 1

    if trend_r2 < 0.30:
        score += 1

    if current_price > 0.90 or current_price < 0.10:
        score += 1

    return score >= 3


# ─────────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(sig: TechnicalSignal, current_price: float) -> tuple[float, str]:
    """
    Compute composite technical score (0–10) and direction (YES / NO / NEUTRAL).

    Score semantics:
        5.0  = perfectly neutral
        > 7  = strong technical case for YES
        < 3  = strong technical case for NO

    Components and maximum absolute contribution:
        1. Trend slope × R² confidence           ±3.0
        2. Momentum (1h + 24h price changes)     ±2.0
        3. RSI contrarian signal                 ±1.0
        4. Bollinger Band position (contrarian)  ±0.6
        5. VWAP deviation (contrarian)           ±0.5
        6. Volume directional confirmation       ±0.5
        7. EMA-12h / EMA-24h cross               ±0.3
        8. Pattern bonuses / penalties           ±2.0

    Direction is assigned only when the score clears a threshold AND the
    trend slope and RSI are consistent (prevents whipsaw labelling).
    """
    score = 5.0  # neutral baseline

    # ── 1. Trend direction and strength ───────────────────────────────────────
    # slope of 0.005/period ≈ strong trend for a binary market; scaled to ±3
    slope_raw = float(np.clip(sig.trend_slope_24h * 200.0, -3.0, 3.0))
    # R² acts as a confidence multiplier: 1.0 = clean trend, 0 = random walk
    r2_conf = 0.40 + 0.60 * float(np.clip(sig.trend_r2, 0.0, 1.0))
    score += slope_raw * r2_conf  # up to ±3.0

    # ── 2. Momentum (recent price changes) ────────────────────────────────────
    mom_1h  = float(np.clip(sig.price_change_1h  * 30.0, -2.0, 2.0))
    mom_24h = float(np.clip(sig.price_change_24h * 10.0, -1.5, 1.5))
    score  += mom_1h * 0.40 + mom_24h * 0.60   # up to ±2.0

    # ── 3. RSI contrarian signal ───────────────────────────────────────────────
    # High RSI (overbought) → slight negative; low RSI (oversold) → slight positive
    rsi_norm = (sig.rsi_14 - 50.0) / 50.0          # −1 to +1
    rsi_comp = float(np.clip(-rsi_norm * 2.0, -2.0, 2.0))
    score   += rsi_comp * 0.50                       # up to ±1.0

    # ── 4. Bollinger Band position (contrarian) ────────────────────────────────
    # bb_position = 0 (at lower band, oversold) → bullish bias
    # bb_position = 1 (at upper band, overbought) → bearish bias
    bb_dev  = (sig.bb_position - 0.5) * 2.0         # −1 to +1
    bb_comp = float(np.clip(-bb_dev * 1.5, -1.5, 1.5))
    score  += bb_comp * 0.40                         # up to ±0.6

    # ── 5. VWAP deviation (contrarian) ────────────────────────────────────────
    vwap_comp = float(np.clip(-sig.vwap_deviation * 5.0, -1.0, 1.0))
    score    += vwap_comp * 0.50                     # up to ±0.5

    # ── 6. Volume directional confirmation ────────────────────────────────────
    # A volume spike accompanying a directional move is a confirming signal
    if sig.volume_spike and abs(sig.price_change_1h) > 0.02:
        vol_boost = 0.50 * float(np.sign(sig.price_change_1h))
        score += vol_boost  # up to ±0.5

    # ── 7. EMA cross confirmation ──────────────────────────────────────────────
    # Price above both EMAs → slight bullish tilt; below both → bearish tilt
    above_12 = int(current_price > sig.ema_12h) if sig.ema_12h > 0 else 0
    above_24 = int(current_price > sig.ema_24h) if sig.ema_24h > 0 else 0
    ema_cross = float(above_12 + above_24 - 1)     # −1, 0, or +1
    score += ema_cross * 0.30                        # up to ±0.3

    # ── 8. Pattern modifiers ───────────────────────────────────────────────────
    if sig.late_smart_money:
        # Informed participants entering near resolution → follow their direction
        direction_sign = 1.0 if sig.price_change_1h > 0 else -1.0
        score += direction_sign * 1.0

    if sig.drift_to_extreme:
        # Sustained monotone drift → continue in same direction
        score += 0.5 if current_price > 0.5 else -0.5

    if sig.overreaction_spike:
        # Overreaction: fade the spike
        score += -0.5 if sig.price_change_1h > 0 else 0.5

    if sig.mean_reversion_due:
        # Multiple stretched indicators → pull score toward neutral 5.0
        score = score * 0.65 + 5.0 * 0.35

    if sig.stagnant_midpoint:
        # Persistent uncertainty → compress toward neutral
        score = score * 0.75 + 5.0 * 0.25

    if sig.pre_resolution_compression:
        # Resolution imminent with compressed price → high uncertainty, dampen hard
        score = score * 0.50 + 5.0 * 0.50

    # ── BB squeeze: compressed volatility, direction is uncertain ──────────────
    if sig.bb_squeeze:
        score = score * 0.80 + 5.0 * 0.20

    # ── Clamp to [0, 10] ──────────────────────────────────────────────────────
    score = float(np.clip(score, 0.0, 10.0))

    # ── Direction thresholds ───────────────────────────────────────────────────
    trending_up = (
        score > 6.2
        and sig.trend_slope_24h > 0.001
        and sig.rsi_14 < 72.0
        and current_price < 0.85
    )
    trending_down = (
        score < 3.8
        and sig.trend_slope_24h < -0.001
        and sig.rsi_14 > 28.0
        and current_price > 0.15
    )

    if trending_up:
        direction = "YES"
    elif trending_down:
        direction = "NO"
    else:
        direction = "NEUTRAL"

    return score, direction
