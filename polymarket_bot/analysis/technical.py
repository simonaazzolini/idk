"""
Technical analysis module — Module 3.
Computes all price/volume indicators and pattern flags.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    # Momentum
    price_change_1h: float = 0.0
    price_change_6h: float = 0.0
    price_change_24h: float = 0.0
    price_change_7d: float = 0.0
    volume_vs_7d_avg: float = 1.0

    # Volatility
    volatility_24h: float = 0.0
    volatility_7d: float = 0.0
    atr_14: float = 0.0

    # Trend
    sma_12h: float = 0.5
    sma_24h: float = 0.5
    ema_12h: float = 0.5
    trend_slope_24h: float = 0.0
    trend_slope_7d: float = 0.0
    trend_r2: float = 0.0

    # Oscillators
    rsi_14: float = 50.0
    bb_upper: float = 1.0
    bb_mid: float = 0.5
    bb_lower: float = 0.0
    bb_position: float = 0.5

    # Volume
    vwap_24h: float = 0.5
    vwap_deviation: float = 0.0
    volume_spike: bool = False
    unusual_volume: bool = False

    # Patterns
    drift_to_extreme: bool = False
    overreaction_spike: bool = False
    pre_resolution_compression: bool = False
    stagnant_midpoint: bool = False
    late_smart_money: bool = False
    mean_reversion_due: bool = False

    # Summary score 0-10
    technical_score: float = 5.0
    technical_direction: str = "NEUTRAL"


def compute_technical_signals(
    price_history: list[dict],
    current_price: float,
    volume_24h: float,
    days_to_resolution: float,
) -> TechnicalSignal:
    """
    Compute all technical indicators from price history.
    price_history: list of {t: timestamp, p: price} dicts or similar.
    """
    sig = TechnicalSignal()

    if not price_history or len(price_history) < 3:
        sig.technical_score = 5.0
        return sig

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = _build_df(price_history)
    if df is None or len(df) < 3:
        return sig

    closes = df["close"].values
    volumes = df["volume"].values if "volume" in df.columns else np.ones(len(closes))
    n = len(closes)

    # ── Momentum ──────────────────────────────────────────────────────────────
    sig.price_change_1h = _pct_change_at(closes, current_price, 1)
    sig.price_change_6h = _pct_change_at(closes, current_price, 6)
    sig.price_change_24h = _pct_change_at(closes, current_price, 24)
    sig.price_change_7d = _pct_change_at(closes, current_price, 168)

    # Volume vs 7d average
    if n >= 168:
        avg_7d_vol = np.mean(volumes[-168:])
    else:
        avg_7d_vol = np.mean(volumes)
    sig.volume_vs_7d_avg = float(volume_24h / avg_7d_vol) if avg_7d_vol > 0 else 1.0
    sig.volume_spike = sig.volume_vs_7d_avg > 2.0
    sig.unusual_volume = sig.volume_vs_7d_avg > 3.0

    # ── Volatility ────────────────────────────────────────────────────────────
    if n >= 24:
        hourly_returns_24h = np.diff(closes[-25:]) / np.maximum(closes[-25:-1], 1e-6)
        sig.volatility_24h = float(np.std(hourly_returns_24h) * np.sqrt(24))
    if n >= 168:
        hourly_returns_7d = np.diff(closes[-169:]) / np.maximum(closes[-169:-1], 1e-6)
        sig.volatility_7d = float(np.std(hourly_returns_7d) * np.sqrt(168))

    # ATR (using hourly high=close+0.5*vol, low=close-0.5*vol approximation)
    if n >= 14:
        true_ranges = np.abs(np.diff(closes[-15:]))
        sig.atr_14 = float(np.mean(true_ranges))

    # ── Trend ─────────────────────────────────────────────────────────────────
    if n >= 12:
        sig.sma_12h = float(np.mean(closes[-12:]))
    if n >= 24:
        sig.sma_24h = float(np.mean(closes[-24:]))

    # EMA
    if n >= 12:
        span = 12
        ema = _ema(closes[-max(span*3, 50):], span)
        sig.ema_12h = float(ema[-1]) if len(ema) > 0 else current_price

    # Linear regression slope and R²
    if n >= 24:
        x24 = np.arange(24)
        y24 = closes[-24:]
        slope, intercept, r_val, _, _ = scipy_stats.linregress(x24, y24)
        sig.trend_slope_24h = float(slope)
        sig.trend_r2 = float(r_val ** 2)

    if n >= 168:
        x7d = np.arange(168)
        y7d = closes[-168:]
        slope7, _, _, _, _ = scipy_stats.linregress(x7d, y7d)
        sig.trend_slope_7d = float(slope7)

    # ── Oscillators ───────────────────────────────────────────────────────────
    if n >= 14:
        sig.rsi_14 = float(_compute_rsi(closes, 14))

    # Bollinger Bands (20-period, 2 std)
    if n >= 20:
        bb_window = closes[-20:]
        sig.bb_mid = float(np.mean(bb_window))
        bb_std = float(np.std(bb_window, ddof=1))
        sig.bb_upper = sig.bb_mid + 2 * bb_std
        sig.bb_lower = sig.bb_mid - 2 * bb_std
        bb_range = sig.bb_upper - sig.bb_lower
        if bb_range > 0:
            sig.bb_position = (current_price - sig.bb_lower) / bb_range
        else:
            sig.bb_position = 0.5

    # ── VWAP ──────────────────────────────────────────────────────────────────
    if n >= 24 and volumes is not None:
        vols_24h = volumes[-24:]
        prices_24h = closes[-24:]
        total_vol = np.sum(vols_24h)
        if total_vol > 0:
            sig.vwap_24h = float(np.sum(vols_24h * prices_24h) / total_vol)
        else:
            sig.vwap_24h = float(np.mean(prices_24h))
        sig.vwap_deviation = (current_price - sig.vwap_24h) / max(sig.vwap_24h, 1e-6)

    # ── Pattern detection ─────────────────────────────────────────────────────
    sig.drift_to_extreme = _detect_drift_to_extreme(closes[-24:] if n >= 24 else closes)
    sig.overreaction_spike = abs(sig.price_change_1h) > 0.10
    sig.pre_resolution_compression = (
        days_to_resolution < 2 and 0.15 < current_price < 0.85
    )
    sig.stagnant_midpoint = abs(current_price - 0.50) < 0.05 and sig.volume_spike
    sig.late_smart_money = sig.unusual_volume and days_to_resolution < 3
    sig.mean_reversion_due = (
        abs(sig.vwap_deviation) > 0.08
        and (sig.rsi_14 > 70 or sig.rsi_14 < 30)
    )

    # ── Composite technical score ─────────────────────────────────────────────
    sig.technical_score, sig.technical_direction = _compute_score(sig, current_price)

    return sig


def _build_df(price_history: list[dict]) -> Optional[pd.DataFrame]:
    """Build a DataFrame from price history with standardized column names."""
    rows = []
    for item in price_history:
        if isinstance(item, dict):
            t = item.get("t") or item.get("timestamp") or item.get("time")
            p = item.get("p") or item.get("price") or item.get("close") or item.get("c")
            v = item.get("v") or item.get("volume") or 1.0
            if t is not None and p is not None:
                try:
                    rows.append({"timestamp": float(t), "close": float(p), "volume": float(v)})
                except (ValueError, TypeError):
                    pass
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["close"] = df["close"].clip(0.001, 0.999)
    return df


def _pct_change_at(closes: np.ndarray, current: float, hours_back: int) -> float:
    n = len(closes)
    if n <= hours_back:
        past = closes[0]
    else:
        past = closes[-(hours_back + 1)]
    if past == 0:
        return 0.0
    return (current - past) / past


def _ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Compute exponential moving average."""
    alpha = 2.0 / (span + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI using Wilder's smoothing."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _detect_drift_to_extreme(recent_closes: np.ndarray) -> bool:
    """Detect if price has monotonically moved toward 0 or 1 for > 12 periods."""
    if len(recent_closes) < 12:
        return False
    last_12 = recent_closes[-12:]
    diffs = np.diff(last_12)
    # All moving same direction
    monotone_up = np.all(diffs >= 0) and last_12[-1] > 0.75
    monotone_down = np.all(diffs <= 0) and last_12[-1] < 0.25
    return bool(monotone_up or monotone_down)


def _compute_score(sig: TechnicalSignal, current_price: float) -> tuple[float, str]:
    """Compute composite technical score (0-10) and direction (YES/NO/NEUTRAL)."""
    # Trend signal: 0-10
    slope_norm = np.clip(sig.trend_slope_24h * 100, -5, 5)
    trend_signal = slope_norm + 5.0

    # Momentum signal: 0-10
    mom_norm = np.clip(sig.price_change_24h * 20, -5, 5)
    momentum_signal = mom_norm + 5.0

    # RSI signal: peak at 50 (no extreme)
    rsi_signal = 10.0 - abs(sig.rsi_14 - 50.0) / 5.0

    # Volume signal
    volume_signal = min(sig.volume_vs_7d_avg * 2, 10.0)

    # Weighted average
    score = (
        trend_signal * 0.30
        + momentum_signal * 0.30
        + rsi_signal * 0.20
        + volume_signal * 0.20
    )
    score = float(np.clip(score, 0.0, 10.0))

    # Direction
    if sig.trend_slope_24h > 0.002 and sig.rsi_14 < 70 and current_price < 0.80:
        direction = "YES"
    elif sig.trend_slope_24h < -0.002 and sig.rsi_14 > 30 and current_price > 0.20:
        direction = "NO"
    else:
        direction = "NEUTRAL"

    return score, direction
