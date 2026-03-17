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
            logger.info(
                "OrderFlow raw book token=%s... status=%d bids=%d asks=%d",
                token_id[:12], resp.status_code, len(bids), len(asks)
            )

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

    def _fetch_prices_binance(self, asset: str) -> Tuple[List[float], List[int]]:
        """
        Fetch recent 1-minute candles from Binance public API.
        Returns (prices, timestamps_ms) newest last.
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
            prices = [float(c[4]) for c in candles]
            timestamps = [int(c[0]) for c in candles]  # open time ms
            return prices, timestamps
        except Exception as e:
            logger.warning("Binance price fetch for %s failed: %s", asset, e)
            return [], []

    def _fetch_prices_coingecko(self, asset: str) -> Tuple[List[float], List[int]]:
        """
        Fetch price history from CoinGecko free API.
        Returns (prices, timestamps_ms) — granularity varies (5min to hourly).
        """
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
            prices = [p[1] for p in prices_raw]
            timestamps = [int(p[0]) for p in prices_raw]
            return prices, timestamps
        except Exception as e:
            logger.warning("CoinGecko market_chart for %s failed: %s", asset, e)
            return [], []

    def _fetch_prices_simple(self, asset: str) -> Tuple[List[float], List[int]]:
        """
        Last-resort: fetch only current price from CoinGecko simple endpoint.
        Returns flat list (no timestamps) — no momentum signal but price is correct.
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
                return [price] * 60, []
        except Exception as e:
            logger.warning("CoinGecko simple price for %s failed: %s", asset, e)
        return [], []

    def collect_price_momentum(self, asset: str) -> Tuple[dict, List[float]]:
        """
        Fetch price history and compute momentum signals.
        Sources tried in order: Binance → CoinGecko chart → CoinGecko simple price.
        Uses timestamps for time-accurate returns regardless of data granularity.
        Returns (momentum_dict, price_history_list)
        """
        cached = self._cache_get(f"momentum_{asset}")
        if cached:
            return cached

        # Source 1: Binance 1-minute candles (best, no key required)
        prices, timestamps = self._fetch_prices_binance(asset)

        # Source 2: CoinGecko chart (5min or hourly depending on free tier)
        if not prices:
            logger.info("Binance unavailable for %s, trying CoinGecko chart", asset)
            prices, timestamps = self._fetch_prices_coingecko(asset)

        # Source 3: CoinGecko simple price (current price only, no momentum)
        if not prices:
            logger.info("CoinGecko chart unavailable for %s, trying simple price", asset)
            prices, timestamps = self._fetch_prices_simple(asset)

        if not prices:
            logger.error("ALL price sources failed for %s — BTC/ETH price will be $0", asset)
            return self._empty_momentum(), []

        # Update rolling history
        if asset.upper() == "BTC":
            self._price_history_btc = prices
        else:
            self._price_history_eth = prices

        now_price = prices[-1]
        now_ms = timestamps[-1] if timestamps else int(time.time() * 1000)

        def get_return(minutes_back: int) -> float:
            """Time-accurate return: find price closest to minutes_back ago."""
            if timestamps:
                target_ms = now_ms - minutes_back * 60 * 1000
                # Find index of closest timestamp to target
                idx = min(range(len(timestamps)),
                          key=lambda i: abs(timestamps[i] - target_ms))
            else:
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
        Fetch social/news buzz about the asset.
        Primary: NewsAPI with 4-hour window.
        Fallback: CoinGecko market data (price change + volume change as proxy).
        Returns volume_ratio and sentiment_score.
        """
        cached = self._cache_get(f"buzz_{asset}")
        if cached:
            return cached

        result = None

        # ── Primary: NewsAPI ────────────────────────────────────────────────
        if newsapi_key:
            query = "bitcoin" if asset.upper() == "BTC" else "ethereum"
            try:
                resp = self.session.get(
                    f"{NEWSAPI_URL}/everything",
                    params={
                        "q": query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 20,
                        "from": self._hours_ago_iso(4),   # 4h window (was 30min)
                    },
                    headers={"X-Api-Key": newsapi_key},
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                logger.info(
                    "SocialBuzz NewsAPI %s: status=%s total=%s articles_returned=%d",
                    asset, data.get("status"), data.get("totalResults"), len(data.get("articles", []))
                )

                articles = data.get("articles", [])
                count = len(articles)
                volume_ratio = count / max(self._avg_news_count, 1)
                self._avg_news_count = 0.9 * self._avg_news_count + 0.1 * max(count, 1)
                sentiment = self._score_sentiment(articles, asset)

                result = {
                    "volume_ratio": volume_ratio,
                    "sentiment_score": sentiment,
                    "article_count": count,
                    "avg_baseline": self._avg_news_count,
                    "source": "newsapi",
                }
                logger.info(
                    "SocialBuzz NewsAPI %s: articles=%d vol_ratio=%.2f sentiment=%.3f",
                    asset, count, volume_ratio, sentiment
                )
            except Exception as e:
                logger.info("SocialBuzz NewsAPI failed for %s: %s — trying CoinGecko fallback", asset, e)

        # ── Fallback: CoinGecko market data ─────────────────────────────────
        if result is None:
            result = self._collect_social_buzz_coingecko(asset)

        self._cache_set(f"buzz_{asset}", result)
        return result

    def _collect_social_buzz_coingecko(self, asset: str) -> dict:
        """
        CoinGecko fallback for social buzz.
        Uses price_change_24h as sentiment and volume_change_24h as buzz proxy.
        """
        asset_id = "bitcoin" if asset.upper() == "BTC" else "ethereum"
        try:
            resp = self.session.get(
                f"{COINGECKO_URL}/simple/price",
                params={
                    "ids": asset_id,
                    "vs_currencies": "usd",
                    "include_24hr_vol": "true",
                    "include_24hr_change": "true",
                },
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json().get(asset_id, {})

            price_change_24h = float(data.get("usd_24h_change") or 0.0)
            vol_24h = float(data.get("usd_24h_vol") or 0.0)

            # Maintain rolling vol baseline per asset
            hist_attr = f"_buzz_vol_hist_{asset.lower()}"
            vol_hist = getattr(self, hist_attr, [])
            vol_hist.append(vol_24h)
            if len(vol_hist) > 48:
                vol_hist = vol_hist[-48:]
            setattr(self, hist_attr, vol_hist)
            avg_vol = sum(vol_hist) / len(vol_hist) if vol_hist else vol_24h

            volume_ratio = vol_24h / avg_vol if avg_vol > 0 else 1.0

            # Sentiment: map price_change_24h to [-1, +1] with ±3% = full signal
            sentiment = max(-1.0, min(1.0, price_change_24h / 3.0))

            logger.info(
                "SocialBuzz CoinGecko %s: price_24h=%.2f%% vol=$%.0f avg=$%.0f "
                "vol_ratio=%.2fx sentiment=%.3f",
                asset, price_change_24h, vol_24h, avg_vol, volume_ratio, sentiment
            )
            return {
                "volume_ratio": volume_ratio,
                "sentiment_score": sentiment,
                "article_count": 0,
                "source": "coingecko",
                "price_change_24h": price_change_24h,
            }
        except Exception as e:
            logger.warning("SocialBuzz CoinGecko fallback failed for %s: %s", asset, e)
            return {"volume_ratio": 1.0, "sentiment_score": 0.0, "source": "failed"}

    def _hours_ago_iso(self, hours: int) -> str:
        from datetime import datetime, timezone, timedelta
        dt = datetime.now(timezone.utc) - timedelta(hours=hours)
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
        Fetch on-chain activity signals.
        BTC: mempool.space transaction count.
        Both: CoinGecko 24h volume vs rolling average as volume-spike proxy.
        """
        cached = self._cache_get(f"onchain_{asset}")
        if cached:
            return cached

        mempool_count = 0
        volume_ratio = 1.0

        # ── BTC: mempool.space ──────────────────────────────────────────────
        if asset.upper() == "BTC":
            try:
                resp = self.session.get(
                    f"{MEMPOOL_URL}/mempool",
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                mempool_count = data.get("count", 0)
                logger.info(
                    "OnChain mempool.space raw: count=%d vsize=%s total_fee=%s",
                    mempool_count,
                    data.get("vsize", "?"),
                    data.get("total_fee", "?"),
                )
            except Exception as e:
                logger.info("OnChain mempool.space failed: %s — will use CoinGecko volume only", e)

        # ── Volume spike: CoinGecko 24h volume vs rolling average ───────────
        try:
            asset_id = "bitcoin" if asset.upper() == "BTC" else "ethereum"
            resp_v = self.session.get(
                f"{COINGECKO_URL}/simple/price",
                params={
                    "ids": asset_id,
                    "vs_currencies": "usd",
                    "include_24hr_vol": "true",
                },
                timeout=10
            )
            resp_v.raise_for_status()
            vol_24h = float(
                resp_v.json().get(asset_id, {}).get("usd_24h_vol") or 0.0
            )

            hist_attr = f"_onchain_vol_hist_{asset.lower()}"
            vol_hist = getattr(self, hist_attr, [])
            vol_hist.append(vol_24h)
            if len(vol_hist) > 48:
                vol_hist = vol_hist[-48:]
            setattr(self, hist_attr, vol_hist)
            avg_vol = sum(vol_hist) / len(vol_hist) if vol_hist else vol_24h

            volume_ratio = vol_24h / avg_vol if avg_vol > 0 else 1.0
            logger.info(
                "OnChain CoinGecko %s: vol_24h=$%.0f avg=$%.0f vol_ratio=%.2fx",
                asset, vol_24h, avg_vol, volume_ratio
            )
        except Exception as e:
            logger.info("OnChain CoinGecko volume fetch failed for %s: %s", asset, e)

        result = {
            "mempool_count": mempool_count,
            "volume_ratio": volume_ratio,
            "is_congested": mempool_count > 50000,
        }
        self._cache_set(f"onchain_{asset}", result)
        logger.info(
            "OnChain %s: mempool=%d vol_ratio=%.2fx congested=%s",
            asset, mempool_count, volume_ratio, result["is_congested"]
        )
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
        logger.info(
            "MarketSentiment %s %s: checking %d related markets",
            asset, direction, len(related_markets)
        )
        if not related_markets:
            return {"confluence_signal": 0.0, "confluence_direction": "neutral",
                    "total_related": 0}

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
