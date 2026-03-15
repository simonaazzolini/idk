"""
Unified client for all Polymarket APIs:
  - Gamma API (market discovery)
  - CLOB API (orderbooks, order execution)
  - Data API (trades, leaderboard, positions)
  - WebSocket (real-time orderbook updates)
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import aiohttp
import websockets
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from config.settings import Settings
from utils.helpers import RateLimiter, exponential_backoff, now_ts, with_retry

logger = logging.getLogger(__name__)


class GammaClient:
    """Client for the Gamma API (market metadata)."""

    BASE = "https://gamma-api.polymarket.com"

    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        self._rate_limiter = RateLimiter(max_calls=30, period_seconds=60)

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        enable_order_book: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        await self._rate_limiter.acquire()
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "enableOrderBook": str(enable_order_book).lower(),
            "limit": limit,
            "offset": offset,
        }
        async with self._session.get(f"{self.BASE}/markets", params=params) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)
            if isinstance(data, list):
                return data
            return data.get("data", data.get("markets", []))

    async def get_all_active_markets(self) -> list[dict]:
        """Paginate through all active markets."""
        all_markets: list[dict] = []
        offset = 0
        limit = 100
        while True:
            batch = await self.get_markets(limit=limit, offset=offset)
            if not batch:
                break
            all_markets.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.5)  # gentle rate limiting
        logger.info("Fetched %d total active markets from Gamma", len(all_markets))
        return all_markets

    async def get_market(self, slug: str) -> Optional[dict]:
        await self._rate_limiter.acquire()
        async with self._session.get(f"{self.BASE}/markets/{slug}") as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            return await resp.json(content_type=None)


class DataClient:
    """Client for the Data API (trades, leaderboard, positions)."""

    BASE = "https://data-api.polymarket.com"

    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        self._rate_limiter = RateLimiter(max_calls=60, period_seconds=60)

    async def _get(self, path: str, params: Optional[dict] = None) -> Any:
        await self._rate_limiter.acquire()
        url = f"{self.BASE}{path}"
        async with self._session.get(url, params=params) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            return await resp.json(content_type=None)

    async def get_leaderboard(self, window: str = "all", limit: int = 100) -> list[dict]:
        data = await self._get("/leaderboard", {"window": window, "limit": limit, "sort": "profit"})
        logger.debug("Leaderboard raw response (window=%s): %s", window, str(data)[:500])
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "results", "leaderboard"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []

    async def get_trades(
        self,
        maker_address: Optional[str] = None,
        market: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        min_size: Optional[float] = None,
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if maker_address:
            params["maker_address"] = maker_address
        if market:
            params["market"] = market
        if min_size is not None:
            params["minSize"] = min_size
        data = await self._get("/trades", params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", data.get("trades", []))
        return []

    async def get_all_trades_for_wallet(self, wallet: str) -> list[dict]:
        """Paginate through all trades for a wallet."""
        all_trades: list[dict] = []
        offset = 0
        limit = 500
        while True:
            batch = await self.get_trades(maker_address=wallet, limit=limit, offset=offset)
            if not batch:
                break
            all_trades.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.3)
        return all_trades

    async def get_positions(self, wallet: str, size_threshold: float = 0.01) -> list[dict]:
        data = await self._get("/positions", {"user": wallet, "sizeThreshold": size_threshold})
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", data.get("positions", []))
        return []

    async def get_activity(
        self,
        wallet: str,
        activity_type: str = "trade",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        data = await self._get(
            "/activity",
            {"user": wallet, "type": activity_type, "limit": limit, "offset": offset},
        )
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", data.get("activity", []))
        return []

    async def get_recent_trades_for_market(
        self, market_slug: str, since_ts: float, limit: int = 200
    ) -> list[dict]:
        data = await self._get("/trades", {"market": market_slug, "limit": limit})
        if isinstance(data, list):
            trades = data
        elif isinstance(data, dict):
            trades = data.get("data", data.get("trades", []))
        else:
            trades = []
        return [t for t in trades if float(t.get("timestamp", 0)) >= since_ts]

    async def get_prices_history(
        self,
        token_id: str,
        interval: str = "1h",
        fidelity: int = 60,
    ) -> list[dict]:
        """Fetch price history from the Data API (/prices-history)."""
        data = await self._get(
            "/prices-history",
            {"market": token_id, "interval": interval, "fidelity": fidelity},
        )
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("history", data.get("data", []))
        return []


class ClobApiClient:
    """Wrapper around py-clob-client for orderbook and order management."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[ClobClient] = None
        self._rate_limiter = RateLimiter(
            max_calls=settings.max_orders_per_minute, period_seconds=60
        )
        self._initialized = False

    def initialize(self) -> None:
        """Initialize py-clob-client. Must be called before use."""
        if self._initialized:
            return
        creds = ApiCreds(
            api_key=self._settings.api_key,
            api_secret=self._settings.api_secret,
            api_passphrase=self._settings.api_passphrase,
        )
        self._client = ClobClient(
            host=self._settings.clob_url,
            chain_id=self._settings.chain_id,
            key=self._settings.private_key,
            creds=creds,
            signature_type=self._settings.signature_type,
            funder=self._settings.funder_address,
        )
        self._initialized = True
        logger.info("CLOB client initialized")

    def _require_init(self) -> None:
        if not self._initialized or not self._client:
            raise RuntimeError("ClobApiClient not initialized. Call initialize() first.")

    async def get_order_book(self, token_id: str) -> Optional[dict]:
        """Fetch order book for a single token."""
        self._require_init()
        loop = asyncio.get_event_loop()
        try:
            book = await loop.run_in_executor(None, self._client.get_order_book, token_id)
            return self._normalize_book(book)
        except Exception as e:
            err_str = str(e)
            # 404 means no active orderbook for this token — not an error
            if "404" in err_str or "No orderbook exists" in err_str:
                logger.debug("No orderbook for token %s (404), skipping", token_id)
            else:
                logger.warning("Failed to fetch orderbook for %s: %s", token_id, e)
            return None

    async def get_orderbooks(self, token_ids) -> dict:
        """Fetch multiple order books in a batch using BookParams."""
        if not self._client:
            return {}
        try:
            clean_ids = []
            for t in token_ids:
                if t is None:
                    continue
                if isinstance(t, str) and t and t != 'None':
                    clean_ids.append(t)
            if not clean_ids:
                return {}
            from py_clob_client.clob_types import BookParams
            params = [BookParams(token_id=str(tid)) for tid in clean_ids]
            loop = asyncio.get_event_loop()
            books = await loop.run_in_executor(
                None, self._client.get_order_books, params
            )
            result = {}
            for tid, book in zip(clean_ids, books or []):
                if book:
                    result[tid] = {
                        "bids": [{"price": float(b.price), "size": float(b.size)}
                                 for b in (book.bids or [])],
                        "asks": [{"price": float(a.price), "size": float(a.size)}
                                 for a in (book.asks or [])],
                    }
            return result
        except Exception as e:
            logger.error("Batch orderbook error: %s", e)
            return {}

    # Keep alias so any future callers still work
    async def get_order_books_batch(self, token_ids) -> dict:
        return await self.get_orderbooks(token_ids)

    def _normalize_book(self, raw: Any) -> dict:
        """Normalize a raw order book object into a consistent dict."""
        if hasattr(raw, "__dict__"):
            raw = raw.__dict__
        if not isinstance(raw, dict):
            return {"bids": [], "asks": []}

        def normalize_levels(levels: Any) -> list[dict]:
            if not levels:
                return []
            result = []
            for lvl in levels:
                if hasattr(lvl, "__dict__"):
                    lvl = lvl.__dict__
                if isinstance(lvl, dict):
                    result.append({
                        "price": float(lvl.get("price", 0)),
                        "size": float(lvl.get("size", 0)),
                    })
                elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    result.append({"price": float(lvl[0]), "size": float(lvl[1])})
            return result

        bids = normalize_levels(raw.get("bids", []))
        asks = normalize_levels(raw.get("asks", []))
        return {
            "asset_id": raw.get("asset_id", raw.get("token_id", "")),
            "bids": sorted(bids, key=lambda x: x["price"], reverse=True),
            "asks": sorted(asks, key=lambda x: x["price"]),
        }

    async def get_prices_history(
        self,
        token_id: str,
        interval: str = "1h",
        fidelity: int = 60,
    ) -> list[dict]:
        """Fetch price history from CLOB."""
        self._require_init()
        loop = asyncio.get_event_loop()
        try:
            history = await loop.run_in_executor(
                None,
                lambda: self._client.get_prices_history(
                    token_id=token_id, interval=interval, fidelity=fidelity
                ),
            )
            if isinstance(history, dict):
                return history.get("history", [])
            return history or []
        except Exception as e:
            logger.warning("Failed to fetch price history for %s: %s", token_id, e)
            return []

    async def create_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
    ) -> Optional[dict]:
        """Place a limit order."""
        self._require_init()
        await self._rate_limiter.acquire()
        loop = asyncio.get_event_loop()
        try:
            order_args = OrderArgs(
                price=round(price, 4),
                size=round(size, 2),
                side=BUY if side.upper() == "BUY" else SELL,
                token_id=token_id,
            )
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.create_and_post_order(order_args),
            )
            if hasattr(resp, "__dict__"):
                resp = resp.__dict__
            logger.info(
                "Order placed: %s %s @ %.4f size=%.2f → %s",
                side, token_id[:8], price, size, resp
            )
            return resp
        except Exception as e:
            logger.error("Order creation failed: %s", e)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""
        self._require_init()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._client.cancel, order_id)
            logger.info("Order %s cancelled", order_id)
            return True
        except Exception as e:
            logger.warning("Cancel order %s failed: %s", order_id, e)
            return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        self._require_init()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._client.cancel_all)
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.warning("Cancel all orders failed: %s", e)
            return False

    async def get_open_orders(self) -> list[dict]:
        """Get all open orders for the account."""
        self._require_init()
        loop = asyncio.get_event_loop()
        try:
            orders = await loop.run_in_executor(None, self._client.get_orders)
            if isinstance(orders, list):
                return [o.__dict__ if hasattr(o, "__dict__") else o for o in orders]
            return []
        except Exception as e:
            logger.warning("Failed to fetch open orders: %s", e)
            return []

    async def get_balance(self) -> float:
        """Get USDC balance in the trading wallet."""
        self._require_init()
        loop = asyncio.get_event_loop()
        try:
            balance = await loop.run_in_executor(None, self._client.get_balance)
            return float(balance) if balance is not None else 0.0
        except Exception as e:
            logger.warning("Failed to fetch balance: %s", e)
            return 0.0

    async def redeem_positions(self, condition_id: str) -> bool:
        """Redeem resolved positions for a market."""
        self._require_init()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: self._client.redeem(condition_id))
            return True
        except Exception as e:
            logger.warning("Redeem failed for %s: %s", condition_id, e)
            return False


class PolymarketWebSocket:
    """
    WebSocket client for real-time orderbook and trade updates.
    Auto-reconnects with exponential backoff.
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._ws: Optional[Any] = None
        self._subscribed_tokens: set[str] = set()
        self._orderbook_callbacks: list[Callable] = []
        self._trade_callbacks: list[Callable] = []
        self._connected = False
        self._reconnect_attempt = 0
        self._running = False
        self._last_message_ts: float = 0.0
        # In-memory orderbook state updated by WebSocket
        self.live_books: dict[str, dict] = {}

    def add_orderbook_callback(self, fn: Callable) -> None:
        self._orderbook_callbacks.append(fn)

    def add_trade_callback(self, fn: Callable) -> None:
        self._trade_callbacks.append(fn)

    async def subscribe(self, token_ids: list[str]) -> None:
        """Add token IDs to subscription list. Resubscribes if already connected."""
        new_tokens = set(token_ids) - self._subscribed_tokens
        self._subscribed_tokens.update(token_ids)
        if self._connected and new_tokens and self._ws:
            msg = json.dumps({
                "type": "Market",
                "assets_ids": list(new_tokens),
            })
            try:
                await self._ws.send(msg)
                logger.debug("Subscribed to %d new tokens via WS", len(new_tokens))
            except Exception as e:
                logger.warning("WS subscribe failed: %s", e)

    async def start(self) -> None:
        """Start the WebSocket listener in an infinite reconnect loop."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                delay = exponential_backoff(self._reconnect_attempt, base=2.0, max_val=60.0)
                self._reconnect_attempt += 1
                logger.warning(
                    "WS disconnected (attempt %d): %s. Reconnecting in %.1fs",
                    self._reconnect_attempt, e, delay
                )
                await asyncio.sleep(delay)

    async def _connect_and_listen(self) -> None:
        ws_url = getattr(self._settings, "ws_url", self.WS_URL)
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._connected = True
            self._reconnect_attempt = 0
            logger.info("WebSocket connected to %s", ws_url)

            # Subscribe to all tracked tokens
            if self._subscribed_tokens:
                sub_msg = json.dumps({
                    "type": "Market",
                    "assets_ids": list(self._subscribed_tokens),
                })
                await ws.send(sub_msg)
                logger.info("Subscribed to %d tokens", len(self._subscribed_tokens))

            async for raw_msg in ws:
                self._last_message_ts = now_ts()
                try:
                    data = json.loads(raw_msg)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug("WS message handling error: %s", e)

    async def _handle_message(self, msg: Any) -> None:
        if isinstance(msg, list):
            for item in msg:
                await self._handle_single(item)
        elif isinstance(msg, dict):
            await self._handle_single(msg)

    async def _handle_single(self, event: dict) -> None:
        event_type = event.get("event_type") or event.get("type", "")
        asset_id = event.get("asset_id", "")

        if event_type in ("book", "price_change"):
            bids = event.get("buys", event.get("bids", []))
            asks = event.get("sells", event.get("asks", []))
            # Update live book
            if asset_id:
                self.live_books[asset_id] = {
                    "asset_id": asset_id,
                    "bids": sorted(
                        [{"price": float(b["price"]), "size": float(b["size"])} for b in bids],
                        key=lambda x: x["price"],
                        reverse=True,
                    ),
                    "asks": sorted(
                        [{"price": float(a["price"]), "size": float(a["size"])} for a in asks],
                        key=lambda x: x["price"],
                    ),
                    "updated_at": now_ts(),
                }
            for cb in self._orderbook_callbacks:
                try:
                    await cb(asset_id, self.live_books.get(asset_id, {}))
                except Exception:
                    pass

        elif event_type in ("trade", "fill"):
            for cb in self._trade_callbacks:
                try:
                    await cb(event)
                except Exception:
                    pass

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_message_age_seconds(self) -> float:
        if self._last_message_ts == 0:
            return float("inf")
        return now_ts() - self._last_message_ts

    def get_live_book(self, token_id: str) -> Optional[dict]:
        return self.live_books.get(token_id)


class PolymarketClient:
    """
    Unified client combining all API surfaces.
    Single entry point for the rest of the application.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._session: Optional[aiohttp.ClientSession] = None
        self.gamma: Optional[GammaClient] = None
        self.data: Optional[DataClient] = None
        self.clob: Optional[ClobApiClient] = None
        self.ws: Optional[PolymarketWebSocket] = None
        self._ws_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Open HTTP session and initialize all sub-clients."""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "User-Agent": "PolymarketEliteBot/1.0",
                "Accept": "application/json",
            },
        )
        self.gamma = GammaClient(self._session)
        self.data = DataClient(self._session)
        self.clob = ClobApiClient(self.settings)

        # Don't initialize CLOB in test mode without credentials
        if self.settings.private_key and not self.settings.private_key.startswith("TEST"):
            try:
                self.clob.initialize()
            except Exception as e:
                logger.warning("CLOB client init failed (will retry): %s", e)

        self.ws = PolymarketWebSocket(self.settings)
        logger.info("PolymarketClient initialized")

    async def start_websocket(self) -> None:
        """Start WebSocket listener as background task."""
        if self.ws and not self._ws_task:
            self._ws_task = asyncio.create_task(self.ws.start(), name="ws_listener")
            logger.info("WebSocket task started")

    async def shutdown(self, mode: str = "LIVE") -> None:
        """Gracefully shut down all connections."""
        if self.ws:
            await self.ws.stop()
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self.clob and mode == "LIVE":
            try:
                await self.clob.cancel_all_orders()
            except Exception:
                pass
        elif self.clob:
            logger.info("Paper mode: skipping cancel_all_orders on shutdown")
        if self._session:
            await self._session.close()
        logger.info("PolymarketClient shut down")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()
