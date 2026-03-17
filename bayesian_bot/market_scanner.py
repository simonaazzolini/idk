"""
market_scanner.py — Scans Polymarket for eligible BTC/ETH short-term markets.

Targets:
  - "Will BTC/ETH be above $X at [time]?" markets
  - Resolves in 1-30 minutes
  - Volume > $200
  - Not already in positions
"""
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


@dataclass
class MarketInfo:
    """Represents a scannable Polymarket market."""
    condition_id: str
    question: str
    asset: str                      # "BTC" or "ETH"
    direction: str                  # "up" (above) or "down"
    strike_price: Optional[float]   # the price threshold
    yes_token_id: str
    no_token_id: str
    current_yes_price: float        # 0-1
    current_no_price: float         # 0-1
    volume_usd: float
    liquidity_usd: float
    end_time: datetime              # resolution time
    minutes_to_resolution: float
    timeframe_label: str            # "5min", "hourly", etc.
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def mid_yes_price(self) -> float:
        """Best available YES price."""
        return self.current_yes_price

    def is_valid(self, config) -> bool:
        """Check if market passes all filters."""
        if self.minutes_to_resolution < config.min_market_window_minutes:
            logger.debug(
                "FILTER: %s too soon (%.1fmin < %dmin min)",
                self.question[:40], self.minutes_to_resolution, config.min_market_window_minutes
            )
            return False
        if self.minutes_to_resolution > config.max_market_window_minutes:
            logger.debug(
                "FILTER: %s too far (%.0fmin > %dmin max)",
                self.question[:40], self.minutes_to_resolution, config.max_market_window_minutes
            )
            return False
        if self.volume_usd < config.min_market_volume:
            logger.debug(
                "FILTER: %s low volume ($%.0f < $%.0f min)",
                self.question[:40], self.volume_usd, config.min_market_volume
            )
            return False
        return True


class MarketScanner:
    """Scans and filters Polymarket markets for BTC/ETH short-term opportunities."""

    ASSET_PATTERNS = {
        "BTC": [
            r"\bBTC\b", r"\bBitcoin\b", r"\bbitcoin\b",
            r"\bBTCUSD\b", r"\bBTC/USD\b"
        ],
        "ETH": [
            r"\bETH\b", r"\bEthereum\b", r"\bethereum\b",
            r"\bETHUSD\b", r"\bETH/USD\b"
        ]
    }

    PRICE_PATTERN = re.compile(
        r"\$?([\d,]+(?:\.\d+)?)\s*[kK]?",
        re.IGNORECASE
    )

    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BayesianBot/1.0"})
        self._last_scan: List[MarketInfo] = []
        self._open_positions: set = set()  # condition_ids

    def set_open_positions(self, condition_ids: set) -> None:
        """Update set of condition_ids we already hold positions in."""
        self._open_positions = condition_ids

    def _is_crypto_market(self, question: str) -> Optional[str]:
        """Return asset name if question is about BTC or ETH, else None."""
        for asset, patterns in self.ASSET_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return asset
        return None

    def _parse_strike_price(self, question: str) -> Optional[float]:
        """Extract strike price from market question."""
        # Look for patterns like "$105,000" or "$105k" or "$3,500"
        dollar_pattern = re.compile(
            r'\$\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)',
            re.IGNORECASE
        )
        match = dollar_pattern.search(question)
        if match:
            num_str = match.group(1).replace(",", "")
            multiplier = match.group(2).upper()
            try:
                value = float(num_str)
                if multiplier == "K":
                    value *= 1000
                elif multiplier == "M":
                    value *= 1_000_000
                return value
            except ValueError:
                pass
        return None

    def _is_above_market(self, question: str) -> Optional[str]:
        """Determine if market is 'above' (up) or 'below' (down)."""
        q = question.lower()
        if "above" in q or "higher than" in q or "exceed" in q or "over" in q:
            return "up"
        elif "below" in q or "lower than" in q or "under" in q:
            return "down"
        # Default assumption: price prediction = "above"
        return "up"

    def _parse_resolution_time(self, market: dict) -> Optional[datetime]:
        """Parse resolution/end time from market data."""
        for key in ["endDate", "end_date_iso", "endDateIso", "resolutionSource",
                    "gameStartTime", "endTime"]:
            val = market.get(key)
            if val and isinstance(val, str):
                try:
                    # Handle ISO format
                    if "T" in val:
                        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                        return dt
                except (ValueError, TypeError):
                    pass

        # Try unix timestamp
        for key in ["endTime", "end_time"]:
            val = market.get(key)
            if val:
                try:
                    ts = int(float(str(val)))
                    if ts > 1_000_000_000:
                        return datetime.fromtimestamp(ts, tz=timezone.utc)
                except (ValueError, TypeError):
                    pass

        return None

    def _minutes_to_resolution(self, end_time: Optional[datetime]) -> float:
        """Calculate minutes until market resolves."""
        if not end_time:
            return 9999.0
        now = datetime.now(timezone.utc)
        delta = (end_time - now).total_seconds() / 60.0
        return max(0.0, delta)

    def _get_token_price(self, market: dict, outcome_index: int) -> float:
        """Extract current price for YES (0) or NO (1) token."""
        try:
            prices = market.get("outcomePrices", [])
            if prices and len(prices) > outcome_index:
                return float(prices[outcome_index])
        except (ValueError, TypeError):
            pass
        return 0.50

    def _get_token_ids(self, market: dict) -> tuple:
        """Get YES and NO token IDs."""
        clob_ids = market.get("clobTokenIds", [])
        if len(clob_ids) >= 2:
            return str(clob_ids[0]), str(clob_ids[1])

        # Alternative fields
        yes_id = market.get("yes_token_id") or market.get("yesTokenId", "")
        no_id = market.get("no_token_id") or market.get("noTokenId", "")
        return str(yes_id), str(no_id)

    def _fetch_markets(self, limit: int = 200) -> List[dict]:
        """Fetch active markets from Gamma API."""
        all_markets = []

        # Primary fetch — general active markets
        try:
            resp = self.session.get(
                f"{GAMMA_URL}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "order": "volume24hr",
                    "ascending": "false"
                },
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                all_markets.extend(data)
            else:
                all_markets.extend(data.get("markets", data.get("data", [])))
        except Exception as e:
            logger.warning("Gamma API primary fetch failed: %s", e)

        # Secondary fetch — crypto-tagged markets
        try:
            resp2 = self.session.get(
                f"{GAMMA_URL}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 100,
                    "tag_id": "crypto",
                    "order": "volume24hr",
                    "ascending": "false"
                },
                timeout=15
            )
            if resp2.ok:
                data2 = resp2.json()
                extras = data2 if isinstance(data2, list) else data2.get("markets", data2.get("data", []))
                # Deduplicate by id
                existing_ids = {m.get("id") or m.get("conditionId") for m in all_markets}
                for m in extras:
                    mid = m.get("id") or m.get("conditionId")
                    if mid not in existing_ids:
                        all_markets.append(m)
        except Exception as e:
            logger.debug("Gamma API crypto fetch: %s", e)

        if all_markets:
            logger.debug("Gamma API returned %d total markets", len(all_markets))
            # Log first market structure for debugging
            if all_markets:
                sample = all_markets[0]
                logger.debug("Sample market keys: %s", list(sample.keys())[:15])

        return all_markets

    def _fetch_clob_markets(self) -> List[dict]:
        """Fetch markets from CLOB API as fallback/supplement."""
        try:
            resp = self.session.get(
                f"{CLOB_URL}/markets",
                params={"active": "true", "limit": 100},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", []) if isinstance(data, dict) else data
        except Exception as e:
            logger.debug("CLOB markets fetch: %s", e)
            return []

    def parse_market(self, raw: dict) -> Optional[MarketInfo]:
        """Parse raw market data into MarketInfo."""
        question = raw.get("question", raw.get("title", ""))
        if not question:
            return None

        asset = self._is_crypto_market(question)
        if not asset:
            return None

        direction = self._is_above_market(question) or "up"
        strike = self._parse_strike_price(question)
        end_time = self._parse_resolution_time(raw)
        minutes_remaining = self._minutes_to_resolution(end_time)

        yes_token_id, no_token_id = self._get_token_ids(raw)
        yes_price = self._get_token_price(raw, 0)
        no_price = self._get_token_price(raw, 1)

        # Ensure prices are reasonable
        if yes_price == 0 and no_price == 0:
            yes_price = 0.50
            no_price = 0.50

        # Volume
        volume = 0.0
        for v_key in ["volume", "volume24hr", "volumeNum", "liquidity"]:
            val = raw.get(v_key)
            if val:
                try:
                    volume = float(str(val).replace(",", ""))
                    break
                except (ValueError, TypeError):
                    pass

        liquidity = float(str(raw.get("liquidity", raw.get("liquidityNum", 0)) or 0))

        # Timeframe label
        if minutes_remaining <= 1:
            tf_label = "1min"
        elif minutes_remaining <= 5:
            tf_label = "5min"
        elif minutes_remaining <= 15:
            tf_label = "15min"
        elif minutes_remaining <= 60:
            tf_label = f"{int(minutes_remaining)}min"
        else:
            tf_label = "hourly"

        condition_id = raw.get("conditionId", raw.get("id", raw.get("slug", "")))

        return MarketInfo(
            condition_id=str(condition_id),
            question=question,
            asset=asset,
            direction=direction,
            strike_price=strike,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            current_yes_price=yes_price,
            current_no_price=no_price,
            volume_usd=volume,
            liquidity_usd=liquidity,
            end_time=end_time or datetime.now(timezone.utc),
            minutes_to_resolution=minutes_remaining,
            timeframe_label=tf_label,
            raw=raw
        )

    def scan(self, top_n: int = 5) -> List[MarketInfo]:
        """
        Scan for top BTC/ETH markets to analyze.
        Returns top_n markets ranked by volume (most liquid first).
        """
        logger.info("Scanning for BTC/ETH markets...")
        raw_markets = self._fetch_markets(limit=300)

        if not raw_markets:
            raw_markets = self._fetch_clob_markets()

        candidates: List[MarketInfo] = []
        rejected_crypto = []

        for raw in raw_markets:
            market = self.parse_market(raw)
            if market is None:
                continue

            # Skip if already in position
            if market.condition_id in self._open_positions:
                continue

            # Apply filters with debug logging
            if not market.is_valid(self.config):
                rejected_crypto.append(
                    f"{market.asset} | {market.minutes_to_resolution:.0f}min | "
                    f"vol=${market.volume_usd:.0f} | {market.question[:50]}"
                )
                continue

            candidates.append(market)

        if rejected_crypto:
            logger.debug(
                "Rejected %d BTC/ETH markets (didn't pass filters):", len(rejected_crypto)
            )
            for r in rejected_crypto[:5]:
                logger.debug("  ✗ %s", r)

        # Sort by volume (highest first)
        candidates.sort(key=lambda m: m.volume_usd, reverse=True)

        top = candidates[:top_n]
        self._last_scan = top

        logger.info(
            "Market scan complete: %d raw markets, %d BTC/ETH found, %d passed filters",
            len(raw_markets), len(candidates) + len(rejected_crypto), len(candidates)
        )
        for m in top:
            logger.info(
                "  → %s | %s %.0fmin | vol=$%.0f | YES=%.2f | %s",
                m.asset, m.direction, m.minutes_to_resolution,
                m.volume_usd, m.current_yes_price, m.question[:60]
            )

        return top

    def get_current_order_book(self, token_id: str) -> Optional[dict]:
        """Fetch current order book for a specific token."""
        if not token_id:
            return None
        try:
            resp = self.session.get(
                f"{CLOB_URL}/book",
                params={"token_id": token_id},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug("Order book fetch for %s: %s", token_id[:8], e)
            return None

    def get_related_markets(self, asset: str, exclude_condition_id: str) -> List[MarketInfo]:
        """Get related markets for confluence signal."""
        related = []
        for m in self._last_scan:
            if m.asset == asset and m.condition_id != exclude_condition_id:
                related.append(m)
        return related
