"""
In-memory TTL cache for market data, orderbooks, and price history.
Thread-safe async implementation.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CacheEntry:
    value: Any
    expires_at: float

    def is_valid(self) -> bool:
        return time.monotonic() < self.expires_at


class TTLCache:
    """Async-safe in-memory cache with TTL per entry."""

    def __init__(self, default_ttl_seconds: float = 600.0):
        self._data: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self.default_ttl = default_ttl_seconds
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._data.get(key)
            if entry and entry.is_valid():
                self.hits += 1
                return entry.value
            if entry:
                del self._data[key]
            self.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.monotonic() + ttl
        async with self._lock:
            self._data[key] = CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()

    async def evict_expired(self) -> int:
        now = time.monotonic()
        async with self._lock:
            expired = [k for k, v in self._data.items() if not v.is_valid()]
            for k in expired:
                del self._data[k]
            return len(expired)

    async def size(self) -> int:
        async with self._lock:
            return len(self._data)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MarketDataCache:
    """Specialized cache for different data types used by the bot."""

    def __init__(self):
        # Different TTLs for different data types
        self.markets = TTLCache(default_ttl_seconds=600)      # 10 min
        self.orderbooks = TTLCache(default_ttl_seconds=30)    # 30 sec
        self.price_history = TTLCache(default_ttl_seconds=300)  # 5 min
        self.news = TTLCache(default_ttl_seconds=1800)        # 30 min
        self.ai_analysis = TTLCache(default_ttl_seconds=900)  # 15 min
        self.whale_positions = TTLCache(default_ttl_seconds=300)  # 5 min
        self.leaderboard = TTLCache(default_ttl_seconds=3600)  # 1 hour

    async def get_market(self, slug: str) -> Optional[dict]:
        return await self.markets.get(f"market:{slug}")

    async def set_market(self, slug: str, data: dict) -> None:
        await self.markets.set(f"market:{slug}", data)

    async def get_orderbook(self, token_id: str) -> Optional[dict]:
        return await self.orderbooks.get(f"ob:{token_id}")

    async def set_orderbook(self, token_id: str, data: dict) -> None:
        await self.orderbooks.set(f"ob:{token_id}", data)

    async def get_price_history(self, token_id: str, interval: str) -> Optional[list]:
        return await self.price_history.get(f"ph:{token_id}:{interval}")

    async def set_price_history(self, token_id: str, interval: str, data: list) -> None:
        await self.price_history.set(f"ph:{token_id}:{interval}", data)

    async def get_news(self, market_slug: str) -> Optional[dict]:
        return await self.news.get(f"news:{market_slug}")

    async def set_news(self, market_slug: str, data: dict) -> None:
        await self.news.set(f"news:{market_slug}", data)

    async def get_ai_analysis(self, market_slug: str) -> Optional[dict]:
        return await self.ai_analysis.get(f"ai:{market_slug}")

    async def set_ai_analysis(self, market_slug: str, data: dict) -> None:
        await self.ai_analysis.set(f"ai:{market_slug}", data)

    async def get_whale_positions(self, market_slug: str) -> Optional[list]:
        return await self.whale_positions.get(f"wp:{market_slug}")

    async def set_whale_positions(self, market_slug: str, data: list) -> None:
        await self.whale_positions.set(f"wp:{market_slug}", data)

    async def invalidate_market(self, slug: str) -> None:
        """Invalidate all cache entries related to a specific market."""
        await self.markets.delete(f"market:{slug}")
        await self.news.delete(f"news:{slug}")
        await self.ai_analysis.delete(f"ai:{slug}")
        await self.whale_positions.delete(f"wp:{slug}")

    async def evict_all_expired(self) -> dict[str, int]:
        results = {}
        results["markets"] = await self.markets.evict_expired()
        results["orderbooks"] = await self.orderbooks.evict_expired()
        results["price_history"] = await self.price_history.evict_expired()
        results["news"] = await self.news.evict_expired()
        results["ai_analysis"] = await self.ai_analysis.evict_expired()
        results["whale_positions"] = await self.whale_positions.evict_expired()
        return results

    def stats(self) -> dict:
        return {
            "markets": {"hits": self.markets.hits, "misses": self.markets.misses,
                        "hit_rate": self.markets.hit_rate()},
            "orderbooks": {"hits": self.orderbooks.hits, "misses": self.orderbooks.misses,
                           "hit_rate": self.orderbooks.hit_rate()},
            "ai_analysis": {"hits": self.ai_analysis.hits, "misses": self.ai_analysis.misses,
                            "hit_rate": self.ai_analysis.hit_rate()},
        }


# Singleton
cache = MarketDataCache()
