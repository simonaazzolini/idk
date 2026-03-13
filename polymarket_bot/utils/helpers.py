"""
General-purpose utility helpers used across all modules.
"""
import asyncio
import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Time ─────────────────────────────────────────────────────────────────────

def now_ts() -> float:
    """Current UTC timestamp as float."""
    return datetime.now(timezone.utc).timestamp()


def now_dt() -> datetime:
    """Current UTC datetime."""
    return datetime.now(timezone.utc)


def ts_to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def dt_to_ts(dt: datetime) -> float:
    return dt.timestamp()


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration like '14h 23m'."""
    seconds = int(abs(seconds))
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def days_until(iso_date_str: str) -> float:
    """Compute days from now to an ISO8601 date string."""
    try:
        target = datetime.fromisoformat(iso_date_str.replace("Z", "+00:00"))
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        delta = target - now_dt()
        return delta.total_seconds() / 86400.0
    except (ValueError, AttributeError):
        return 999.0


# ── Math ──────────────────────────────────────────────────────────────────────

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_log10(x: float) -> float:
    return math.log10(max(x, 1.0))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / abs(old)


def sharpe_ratio(returns: list[float], annualize_factor: float = 365.0) -> float:
    """Compute annualized Sharpe ratio from a list of daily returns."""
    if len(returns) < 2:
        return 0.0
    import numpy as np
    arr = np.array(returns, dtype=float)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * math.sqrt(annualize_factor))


def profit_factor(pnls: list[float]) -> float:
    """Gross profit / gross loss. Returns 0 if no losses."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


def max_drawdown(values: list[float]) -> float:
    """Maximum peak-to-trough drawdown fraction."""
    if len(values) < 2:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def win_rate(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


# ── Retry & Backoff ───────────────────────────────────────────────────────────

async def with_retry(
    coro_fn: Callable,
    *args,
    retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    label: str = "operation",
    **kwargs,
) -> Any:
    """
    Run an async coroutine function with exponential backoff retry.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except exceptions as e:
            last_exc = e
            if attempt == retries:
                logger.error("Max retries (%d) reached for %s: %s", retries, label, e)
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning(
                "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                label, attempt + 1, retries, e, delay
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore


def exponential_backoff(attempt: int, base: float = 2.0, max_val: float = 60.0) -> float:
    """Return jittered exponential backoff seconds for attempt N (0-indexed)."""
    raw = base * (2 ** attempt)
    jitter = random.uniform(0, raw * 0.1)
    return min(raw + jitter, max_val)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def safe_json_loads(s: str, default: Any = None) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: Any = None) -> str:
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return json.dumps(default or {})


# ── Number formatting ─────────────────────────────────────────────────────────

def fmt_usdc(amount: float) -> str:
    """Format a USDC amount like '$1,234.56'."""
    return f"${amount:,.2f}"


def fmt_pct(fraction: float, decimals: int = 2) -> str:
    """Format a fraction as a percentage string."""
    return f"{fraction * 100:.{decimals}f}%"


def fmt_score(score: float) -> str:
    """Format a 0-10 score."""
    return f"{score:.1f}/10"


# ── List helpers ──────────────────────────────────────────────────────────────

def chunk_list(lst: list, size: int) -> list[list]:
    """Split list into chunks of at most `size` elements."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def flatten(nested: list[list]) -> list:
    return [item for sublist in nested for item in sublist]


def deduplicate(lst: list, key_fn: Callable = lambda x: x) -> list:
    seen = set()
    result = []
    for item in lst:
        k = key_fn(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


# ── Rate limiting ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            # Remove timestamps outside window
            cutoff = now - self.period
            self._timestamps = [t for t in self._timestamps if t > cutoff]

            if len(self._timestamps) >= self.max_calls:
                # Must wait until oldest timestamp is outside window
                oldest = self._timestamps[0]
                wait_time = oldest + self.period - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                # After sleep, clean up again
                now2 = time.monotonic()
                cutoff2 = now2 - self.period
                self._timestamps = [t for t in self._timestamps if t > cutoff2]

            self._timestamps.append(time.monotonic())

    def calls_in_window(self) -> int:
        now = time.monotonic()
        cutoff = now - self.period
        return sum(1 for t in self._timestamps if t > cutoff)


# ── Market helpers ────────────────────────────────────────────────────────────

def extract_yes_no_token_ids(market: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract YES and NO clobTokenIds from market dict."""
    tokens = market.get("clobTokenIds") or market.get("tokens", [])
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except json.JSONDecodeError:
            return None, None
    if isinstance(tokens, list) and len(tokens) >= 2:
        return str(tokens[0]), str(tokens[1])
    return None, None


def extract_yes_no_prices(market: dict) -> tuple[float, float]:
    """Extract YES and NO prices from market dict."""
    prices = market.get("outcomePrices") or []
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            prices = []
    if isinstance(prices, list) and len(prices) >= 2:
        try:
            return float(prices[0]), float(prices[1])
        except (ValueError, TypeError):
            pass
    # Fallback
    yes = market.get("yes_price") or market.get("bestBid") or 0.5
    no = market.get("no_price") or 1.0 - float(yes)
    return float(yes), float(no)


def get_market_category(market: dict) -> str:
    return str(market.get("category") or market.get("groupItemTitle") or "unknown").lower()
