"""
evidence_collector.py — Fetches all 5 evidence layers for Bayesian updates.

Evidence layers:
  1. Order Flow (Polymarket order book)
  2. Price Momentum (CoinGecko price data)
  3. Social Buzz (NewsAPI)
  4. On-Chain Signals (mempool.space)
  5. Market Sentiment (Polymarket cross-market confluence)
"""
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"
COINGECKO_URL = "https://api.coingecko.com/api/v3"
MEMPOOL_URL = "https://mempool.space/api"
NEWSAPI_URL = "https://newsapi.org/v2"


@dataclass
class EvidenceData:
    """Container for all collected evidence."""
    order_flow: dict = field(default_factory=dict)
    price_momentum: dict = field(default_factory=dict)
    social_buzz: dict = field(default_factory=dict)
    on_chain: dict = field(default_factory=dict)
    market_sentiment: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    price_history: List[float] = field(default_factory=list)  # for regime detection


class EvidenceCollector:
    """Collects all evidence layers needed for Bayesian analysis."""

    # In-memory cache to avoid hammering APIs
    _cache: Dict[str, Tuple[float, object]] = {}
    CACHE_TTL = 30  # seconds

    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BayesianBot/1.0"})

        # Rolling stats for order flow normalization
        self._of_history: List[float] = []
        self._price_history_btc: List[float] = []
        self._price_history_eth: List[float] = []
        self._avg_news_count: float = 5.0  # baseline articles per 30min

    def _cache_get(self, key: str) -> Optional[object]:
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self.CACHE_TTL:
                return val
        return None

    def _cache_set(self, key: str, val: object) -> None:
        self._cache[key] = (time.time(), val)

    # ── Evidence 1: Order Flow ─────────────────────────────────────────────

    def collect_order_flow(self, token_id: str) -> dict:
        """
        Fetch Polymarket order book and calculate net buy pressure.
        Returns buy_pressure (net buy/sell ratio), std_devs from mean.
        """
        if not token_id:
            return {"buy_pressure": 0.0, "std_devs": 0.0, "error": "no token_id"}

        cached = self._cache_get(f"of_{token_id}")
        if cached:
            return cached

        try:
            resp = self.session.get(
                f"{CLOB_URL}/book",
                params={"token_id": token_id},
                timeout=10
            )
            resp.raise_for_status()
            book = resp.json()

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            # Calculate buy/sell volume at top 5 levels
            buy_vol = sum(
                float(b.get("size", 0)) * float(b.get("price", 0))
                for b in bids[:5]
            )
            sell_vol = sum(
                float(a.get("size", 0)) * float(a.get("price", 0))
                for a in asks[:5]
            )
            total_vol = buy_vol + sell_vol

            if total_vol == 0:
                net_pressure = 0.0
            else:
                net_pressure = (buy_vol - sell_vol) / total_vol

            # Update rolling history for std dev calculation
            self._of_history.append(net_pressure)
            if len(self._of_history) > 100:
                self._of_history = self._of_history[-100:]

            std_devs = 0.0
            if len(self._of_history) >= 5:
                mean = statistics.mean(self._of_history)
                stdev = statistics.stdev(self._of_history)
                if stdev > 0:
                    std_devs = (net_pressure - mean) / stdev

            result = {
                "buy_pressure": net_pressure,
                "std_devs": std_devs,
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "bid_count": len(bids),
                "ask_count": len(asks)
            }
            self._cache_set(f"of_{token_id}", result)
            logger.debug("OrderFlow token=%s pressure=%.3f std=%.2f", token_id[:8], net_pressure, std_devs)
            return result

        except Exception as e:
            logger.debug("Order flow fetch error: %s", e)
            return {"buy_pressure": 0.0, "std_devs": 0.0, "error": str(e)}

    # ── Evidence 2: Price Momentum ─────────────────────────────────────────

    def _fetch_prices_binance(self, asset: str) -> List[float]:
        """
        Fetch recent 1-minute candles from Binance public API.
        No API key required. Returns list of close prices (newest last).
        """
        symbol = "BTCUSDT" if asset.upper() == "BTC" else "ETHUSDT"
        try:
            resp = self.session.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": symbol, "interval": "1m", "limit": 60},
                timeout=10
            )
            resp.raise_for_status()
            candles = resp.json()
            # Each candle: [open_time, open, high, low, close, ...]
            return [float(c[4]) for c in candles]
        except Exception as e:
            logger.warning("Binance price fetch for %s failed: %s", asset, e)
            return []

    def _fetch_prices_coingecko(self, asset: str) -> List[float]:
        """Fetch price history from CoinGecko free API (hourly granularity for 1 day)."""
        asset_id = "bitcoin" if asset.upper() == "BTC" else "ethereum"
        try:
            resp = self.session.get(
                f"{COINGECKO_URL}/coins/{asset_id}/market_chart",
                params={"vs_currency": "usd", "days": "1"},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            prices_raw = data.get("prices", [])
            return [p[1] for p in prices_raw]
        except Exception as e:
            logger.warning("CoinGecko market_chart for %s failed: %s", asset, e)
            return []

    def _fetch_prices_simple(self, asset: str) -> List[float]:
        """
        Last-resort: fetch only current price from CoinGecko simple endpoint.
        Returns a flat list of 60 identical prices — momentum will be 0 but
        price will be correct for display and regime detection.
        """
        coin_id = "bitcoin" if asset.upper() == "BTC" else "ethereum"
        try:
            resp = self.session.get(
                f"{COINGECKO_URL}/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            price = float(data.get(coin_id, {}).get("usd", 0.0))
            if price > 0:
                return [price] * 60
        except Exception as e:
            logger.warning("CoinGecko simple price for %s failed: %s", asset, e)
        return []

    def collect_price_momentum(self, asset: str) -> Tuple[dict, List[float]]:
        """
        Fetch price history and compute momentum signals.
        Sources tried in order: Binance → CoinGecko chart → CoinGecko simple price.
        Returns (momentum_dict, price_history_list)
        """
        cached = self._cache_get(f"momentum_{asset}")
        if cached:
            return cached

        # Source 1: Binance 1-minute candles (best, no key required)
        prices = self._fetch_prices_binance(asset)

        # Source 2: CoinGecko hourly chart
        if not prices:
            logger.info("Binance unavailable for %s, trying CoinGecko chart", asset)
            prices = self._fetch_prices_coingecko(asset)

        # Source 3: CoinGecko simple price (current price only, no momentum)
        if not prices:
            logger.info("CoinGecko chart unavailable for %s, trying simple price", asset)
            prices = self._fetch_prices_simple(asset)

        if not prices:
            logger.error("ALL price sources failed for %s — BTC/ETH price will be $0", asset)
            return self._empty_momentum(), []

        # Update rolling history
        if asset.upper() == "BTC":
            self._price_history_btc = prices
        else:
            self._price_history_eth = prices

        now_price = prices[-1]

        def get_return(minutes_back: int) -> float:
            idx = max(0, len(prices) - minutes_back - 1)
            old = prices[idx]
            if old == 0:
                return 0.0
            return (now_price - old) / old

        ret_1min = get_return(1)
        ret_5min = get_return(5)
        ret_15min = get_return(15)

        # ATR-like volatility
        recent = prices[-20:] if len(prices) >= 20 else prices
        if len(recent) > 1:
            ranges = [abs(recent[i] - recent[i-1]) / recent[i-1] for i in range(1, len(recent))]
            atr = sum(ranges) / len(ranges)
            longer = prices[-60:] if len(prices) >= 60 else prices
            long_ranges = [abs(longer[i] - longer[i-1]) / longer[i-1] for i in range(1, len(longer))]
            long_atr = sum(long_ranges) / len(long_ranges) if long_ranges else atr
            atr_ratio = atr / long_atr if long_atr > 0 else 1.0
        else:
            atr = 0.0
            atr_ratio = 1.0

        result = {
            "current_price": now_price,
            "ret_1min": ret_1min,
            "ret_5min": ret_5min,
            "ret_15min": ret_15min,
            "atr": atr,
            "atr_ratio": atr_ratio,
            "price_count": len(prices)
        }
        self._cache_set(f"momentum_{asset}", (result, prices))
        logger.debug(
            "Momentum %s: price=%.2f ret5m=%.3f%% atr_ratio=%.2fx",
            asset, now_price, ret_5min * 100, atr_ratio
        )
        return result, prices

    def _empty_momentum(self) -> dict:
        return {
            "current_price": 0.0,
            "ret_1min": 0.0,
            "ret_5min": 0.0,
            "ret_15min": 0.0,
            "atr": 0.0,
            "atr_ratio": 1.0,
            "price_count": 0
        }

    def get_current_price(self, asset: str) -> float:
        """Get current price for BTC or ETH."""
        cached = self._cache_get(f"momentum_{asset}")
        if cached:
            result_tuple = cached
            if isinstance(result_tuple, tuple):
                return result_tuple[0].get("current_price", 0.0)
            return result_tuple.get("current_price", 0.0)

        momentum, _ = self.collect_price_momentum(asset)
        return momentum.get("current_price", 0.0)

    # ── Evidence 3: Social Buzz ────────────────────────────────────────────

    def collect_social_buzz(self, asset: str, newsapi_key: str) -> dict:
        """
        Fetch recent news about the asset using NewsAPI.
        Returns volume_ratio and sentiment_score.
        """
        if not newsapi_key:
            return {"volume_ratio": 1.0, "sentiment_score": 0.0, "error": "no API key"}

        cached = self._cache_get(f"buzz_{asset}")
        if cached:
            return cached

        query = "bitcoin" if asset.upper() == "BTC" else "ethereum"
        try:
            resp = self.session.get(
                f"{NEWSAPI_URL}/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "from": self._thirty_min_ago_iso()
                },
                headers={"X-Api-Key": newsapi_key},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            articles = data.get("articles", [])
            count = len(articles)

            volume_ratio = count / max(self._avg_news_count, 1)

            # Update rolling average
            self._avg_news_count = 0.9 * self._avg_news_count + 0.1 * count

            # Sentiment: simple keyword scoring
            sentiment = self._score_sentiment(articles, asset)

            result = {
                "volume_ratio": volume_ratio,
                "sentiment_score": sentiment,
                "article_count": count,
                "avg_baseline": self._avg_news_count
            }
            self._cache_set(f"buzz_{asset}", result)
            logger.debug(
                "SocialBuzz %s: articles=%d ratio=%.2f sentiment=%.2f",
                asset, count, volume_ratio, sentiment
            )
            return result

        except Exception as e:
            logger.debug("Social buzz fetch error: %s", e)
            return {"volume_ratio": 1.0, "sentiment_score": 0.0, "error": str(e)}

    def _thirty_min_ago_iso(self) -> str:
        from datetime import datetime, timezone, timedelta
        dt = datetime.now(timezone.utc) - timedelta(minutes=30)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _score_sentiment(self, articles: list, asset: str) -> float:
        """Score sentiment from article titles/descriptions. Returns -1 to +1."""
        positive_words = [
            "surge", "rally", "bull", "gain", "rise", "high", "up", "pump",
            "break", "record", "positive", "growth", "soar", "spike", "moon",
            "ath", "all-time high", "buy", "strong", "momentum", "green"
        ]
        negative_words = [
            "crash", "drop", "fall", "bear", "loss", "low", "down", "dump",
            "decline", "sell", "weak", "red", "correction", "dip", "plunge",
            "fear", "warning", "risk", "liquidat", "panic"
        ]

        scores = []
        for article in articles:
            text = " ".join([
                article.get("title", ""),
                article.get("description", "") or ""
            ]).lower()

            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)

            if pos + neg == 0:
                scores.append(0.0)
            else:
                scores.append((pos - neg) / (pos + neg))

        return sum(scores) / len(scores) if scores else 0.0

    # ── Evidence 4: On-Chain Signals ───────────────────────────────────────

    def collect_on_chain(self, asset: str) -> dict:
        """
        Fetch mempool stats and on-chain activity signals.
        Only BTC has mempool data (uses mempool.space).
        """
        cached = self._cache_get(f"onchain_{asset}")
        if cached:
            return cached

        mempool_count = 0
        volume_ratio = 1.0

        if asset.upper() == "BTC":
            try:
                resp = self.session.get(
                    f"{MEMPOOL_URL}/mempool",
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                mempool_count = data.get("count", 0)
                logger.debug("Mempool count: %d", mempool_count)
            except Exception as e:
                logger.debug("Mempool fetch error: %s", e)

        # Volume spike proxy: use CoinGecko 24h volume
        try:
            asset_id = "bitcoin" if asset.upper() == "BTC" else "ethereum"
            cached_momentum = self._cache_get(f"momentum_{asset}")

            # If we have recent price data, compute approximate volume spike
            # For now we use a neutral signal unless we see mempool congestion
            volume_ratio = 1.0

        except Exception as e:
            logger.debug("On-chain volume error: %s", e)

        result = {
            "mempool_count": mempool_count,
            "volume_ratio": volume_ratio,
            "is_congested": mempool_count > 50000
        }
        self._cache_set(f"onchain_{asset}", result)
        logger.debug("OnChain %s: mempool=%d vol_ratio=%.2f", asset, mempool_count, volume_ratio)
        return result

    # ── Evidence 5: Market Sentiment (confluence) ──────────────────────────

    def collect_market_sentiment(
        self,
        asset: str,
        direction: str,
        related_markets: list
    ) -> dict:
        """
        Look at related markets (same asset, different strike/timeframe)
        to detect confluence signals.
        """
        if not related_markets:
            return {"confluence_signal": 0.0, "confluence_direction": "neutral"}

        same_direction = 0
        opposite_direction = 0
        total = 0

        for market in related_markets:
            mkt_direction = market.direction
            yes_price = market.current_yes_price

            # If YES price > 0.5, market leans "above" (up)
            # If YES price < 0.5, market leans "below" (down)
            if yes_price > 0.55:
                market_lean = "up"
            elif yes_price < 0.45:
                market_lean = "down"
            else:
                total += 1
                continue

            total += 1
            if market_lean == direction:
                same_direction += 1
            else:
                opposite_direction += 1

        if total == 0:
            return {"confluence_signal": 0.0, "confluence_direction": "neutral"}

        confluence = (same_direction - opposite_direction) / total

        if confluence > 0:
            conf_direction = direction
        elif confluence < 0:
            conf_direction = "up" if direction == "down" else "down"
        else:
            conf_direction = "neutral"

        result = {
            "confluence_signal": confluence,
            "confluence_direction": conf_direction,
            "same_direction": same_direction,
            "opposite_direction": opposite_direction,
            "total_related": total
        }
        logger.debug(
            "MarketSentiment %s %s: confluence=%.2f dir=%s",
            asset, direction, confluence, conf_direction
        )
        return result

    # ── Main collection entry point ────────────────────────────────────────

    def collect_all(
        self,
        asset: str,
        direction: str,
        yes_token_id: str,
        related_markets: list
    ) -> EvidenceData:
        """
        Collect all evidence layers for a given market.
        Gracefully handles failures with neutral evidence.
        """
        evidence = EvidenceData()

        # Layer 1: Order flow
        try:
            evidence.order_flow = self.collect_order_flow(yes_token_id)
        except Exception as e:
            evidence.errors.append(f"order_flow: {e}")
            evidence.order_flow = {"buy_pressure": 0.0, "std_devs": 0.0}

        # Layer 2: Price momentum (also provides price history)
        try:
            momentum, prices = self.collect_price_momentum(asset)
            evidence.price_momentum = momentum
            evidence.price_history = prices
        except Exception as e:
            evidence.errors.append(f"price_momentum: {e}")
            evidence.price_momentum = self._empty_momentum()
            evidence.price_history = []

        # Layer 3: Social buzz
        try:
            evidence.social_buzz = self.collect_social_buzz(
                asset, self.config.newsapi_key
            )
        except Exception as e:
            evidence.errors.append(f"social_buzz: {e}")
            evidence.social_buzz = {"volume_ratio": 1.0, "sentiment_score": 0.0}

        # Layer 4: On-chain
        try:
            evidence.on_chain = self.collect_on_chain(asset)
        except Exception as e:
            evidence.errors.append(f"on_chain: {e}")
            evidence.on_chain = {"mempool_count": 0, "volume_ratio": 1.0}

        # Layer 5: Market sentiment
        try:
            evidence.market_sentiment = self.collect_market_sentiment(
                asset, direction, related_markets
            )
        except Exception as e:
            evidence.errors.append(f"market_sentiment: {e}")
            evidence.market_sentiment = {"confluence_signal": 0.0, "confluence_direction": "neutral"}

        if evidence.errors:
            logger.debug("Evidence collection errors: %s", evidence.errors)

        return evidence
