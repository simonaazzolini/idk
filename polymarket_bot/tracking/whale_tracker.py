"""
Whale & Smart Money Tracker — Module 4.
Steps A-G: wallet database, position tracking, real-time trade monitoring,
insider detection, and per-market smart money consensus.
"""
import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings
from core.client import PolymarketClient
from data.database import Database
from tracking.wallet_classifier import (
    classify_wallet_tiers,
    compute_alert_level,
    compute_wallet_metrics,
    get_tier_weight,
    TIER_LEGENDARY, TIER_INSIDER_ALERT, TIER_SMART_MONEY,
    TIER_1_WHALE, TIER_2_SHARK, TIER_3_FISH, TIER_BOT,
)
from utils.helpers import now_ts, safe_div

logger = logging.getLogger(__name__)


class WhaleAlert:
    __slots__ = [
        "wallet_address", "wallet_tier", "wallet_win_rate", "wallet_total_pnl",
        "insider_score", "market_slug", "question", "outcome", "price",
        "size_usdc", "conviction_multiplier", "is_adding_to_position",
        "other_smart_money_same_direction", "time_since_market_opened",
        "days_until_resolution", "alert_level", "timestamp",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {k: getattr(self, k, None) for k in self.__slots__}


class WhaleTracker:
    """
    Comprehensive tracker for smart money wallets on Polymarket.
    """

    def __init__(self, client: PolymarketClient, db: Database, settings: Settings):
        self.client = client
        self.db = db
        self.settings = settings
        self._wallet_cache: dict[str, dict] = {}  # address → wallet record
        self._last_full_refresh: float = 0.0
        self._last_trade_ts: float = now_ts() - 3600  # last seen trade timestamp
        self.pending_alerts: asyncio.Queue = asyncio.Queue()

    # ── Step A: Build wallet database ─────────────────────────────────────────

    async def full_refresh(self) -> None:
        """Full sweep: leaderboard → trade history → compute metrics → classify."""
        logger.info("Starting full whale wallet database refresh...")

        # 1. Collect wallets from leaderboard — try windows in order, stop on first hit
        wallets: set[str] = set()
        for window in ["all", "1m", "7d", "1d"]:
            try:
                board = await self.client.data.get_leaderboard(window=window, limit=100)
                logger.debug("Leaderboard raw response (window=%s): %s", window, str(board)[:500])
                for entry in board:
                    addr = (
                        entry.get("proxyWallet")
                        or entry.get("proxy_address")
                        or entry.get("address")
                        or entry.get("user")
                        or entry.get("wallet")
                        or entry.get("account")
                    )
                    if addr:
                        wallets.add(str(addr).lower())
                if wallets:
                    logger.info(
                        "Leaderboard window=%s returned %d entries → %d wallets",
                        window, len(board), len(wallets),
                    )
                    break  # stop at first window that yields results
                logger.debug("Leaderboard window=%s returned 0 usable entries, trying next", window)
            except Exception as e:
                logger.warning("Leaderboard fetch failed for window=%s: %s", window, e)
            await asyncio.sleep(0.5)

        # 2. Fall back to recent large trades if leaderboard yielded nothing
        if not wallets:
            logger.warning(
                "Leaderboard returned 0 wallets across all windows — "
                "seeding from recent large trades (minSize=1000)"
            )
            try:
                trades = await self.client.data.get_trades(limit=500, min_size=1000)
                for trade in trades:
                    for field in ("maker_address", "taker_address", "maker", "taker"):
                        addr = trade.get(field)
                        if addr:
                            wallets.add(str(addr).lower())
                logger.info(
                    "Leaderboard empty, seeding from recent large trades: %d wallets found",
                    len(wallets),
                )
            except Exception as e:
                logger.warning("Trade-based wallet fallback also failed: %s", e)

        logger.info("Collected %d unique wallets from leaderboards", len(wallets))

        # 2. For each wallet, fetch trade history and compute metrics
        for i, wallet in enumerate(wallets):
            try:
                await self._refresh_single_wallet(wallet)
                if i % 10 == 0:
                    logger.info("Wallet refresh progress: %d/%d", i + 1, len(wallets))
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.debug("Wallet refresh failed for %s: %s", wallet[:8], e)

        self._last_full_refresh = now_ts()
        logger.info("Full whale refresh complete. Tracked %d wallets.", len(wallets))

    async def _refresh_single_wallet(self, wallet: str) -> None:
        """Fetch trades + positions for one wallet, compute metrics, save to DB."""
        # Fetch trade history (both maker and taker)
        trades = await self.client.data.get_all_trades_for_wallet(wallet)
        positions = await self.client.data.get_positions(wallet, size_threshold=0.01)

        if not trades and not positions:
            return

        # Compute metrics
        metrics = compute_wallet_metrics(trades, positions)
        tier_tags = classify_wallet_tiers(metrics)

        wallet_record = {
            "address": wallet,
            "tier_tags": json.dumps(tier_tags),
            **metrics,
        }
        await self.db.upsert_whale_wallet(wallet_record)
        self._wallet_cache[wallet] = wallet_record

        # Flag insiders for review
        if metrics.get("insider_score", 0) >= 8.0:
            await self._flag_insider(wallet, metrics)

    async def _flag_insider(self, wallet: str, metrics: dict) -> None:
        """Log insider alert to DB."""
        evidence = {
            "information_lead_score": metrics.get("information_lead_score"),
            "timing_alpha": metrics.get("timing_alpha"),
            "news_precession_rate": metrics.get("news_precession_rate"),
            "win_rate": metrics.get("win_rate"),
            "total_trades": metrics.get("total_trades"),
        }
        await self.db.insert_insider_alert({
            "wallet": wallet,
            "insider_score": metrics["insider_score"],
            "evidence_json": json.dumps(evidence),
            "flagged_at": now_ts(),
            "reviewed": 0,
        })
        logger.warning(
            "INSIDER ALERT: wallet=%s score=%.1f win_rate=%.2f",
            wallet[:8], metrics["insider_score"], metrics.get("win_rate", 0)
        )

    # ── Step B: Current positions tracker ────────────────────────────────────

    async def refresh_positions(self, markets: list[dict]) -> None:
        """Poll positions for all tracked wallets (every 5 min)."""
        tracked = await self.db.get_all_tracked_wallets()
        # Filter to non-noise wallets for efficiency
        active_wallets = [
            w for w in tracked
            if json.loads(w.get("tier_tags", "[]")) != ["NOISE"]
        ]
        logger.debug("Refreshing positions for %d active wallets", len(active_wallets))

        for wallet_rec in active_wallets:
            address = wallet_rec["address"]
            try:
                positions = await self.client.data.get_positions(address)
                avg_size = wallet_rec.get("avg_position_size", 1.0) or 1.0
                for pos in positions:
                    market_slug = (
                        pos.get("market") or pos.get("market_slug")
                        or pos.get("conditionId", "unknown")
                    )
                    outcome = str(pos.get("outcome") or pos.get("title") or "YES").upper()
                    shares = float(pos.get("size") or pos.get("shares") or 0)
                    current_price = float(pos.get("currentPrice") or pos.get("price") or 0.5)
                    avg_entry = float(pos.get("avgPrice") or pos.get("avg_entry_price") or current_price)
                    pos_value = shares * current_price
                    unrealized = (current_price - avg_entry) * shares
                    conviction = safe_div(pos_value, avg_size)

                    await self.db.upsert_whale_position({
                        "wallet": address,
                        "market_slug": str(market_slug),
                        "outcome": outcome,
                        "shares": shares,
                        "avg_entry_price": avg_entry,
                        "current_price": current_price,
                        "unrealized_pnl": unrealized,
                        "opened_at": float(pos.get("created_at") or now_ts()),
                        "last_updated": now_ts(),
                        "conviction_score": conviction,
                    })
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.debug("Position refresh failed for %s: %s", address[:8], e)

    async def get_smart_money_consensus(self, market_slug: str) -> dict:
        """
        Aggregate smart money positions for a market.
        Returns consensus direction, conviction score, and breakdown.
        """
        positions = await self.db.get_whale_positions_for_market(market_slug)
        if not positions:
            return self._empty_consensus()

        yes_usdc = 0.0
        no_usdc = 0.0
        yes_wallets = 0
        no_wallets = 0
        legendary_directions = []
        insider_directions = []
        total_weight = 0.0
        weighted_yes = 0.0
        weighted_no = 0.0
        entries_yes = []
        entries_no = []

        for pos in positions:
            tier_tags = json.loads(pos.get("tier_tags") or "[]")
            # Only consider smart money and above
            relevant_tiers = {TIER_LEGENDARY, TIER_INSIDER_ALERT, TIER_SMART_MONEY,
                               TIER_1_WHALE, TIER_2_SHARK}
            if not any(t in relevant_tiers for t in tier_tags):
                continue

            weight = get_tier_weight(tier_tags)
            size = abs(float(pos.get("shares", 0)) * float(pos.get("current_price", 0.5)))
            outcome = str(pos.get("outcome", "")).upper()
            entry_price = float(pos.get("avg_entry_price", 0.5))

            if outcome == "YES":
                yes_usdc += size
                yes_wallets += 1
                weighted_yes += weight * size
                entries_yes.append(entry_price)
            elif outcome == "NO":
                no_usdc += size
                no_wallets += 1
                weighted_no += weight * size
                entries_no.append(entry_price)

            total_weight += weight

            if TIER_LEGENDARY in tier_tags:
                legendary_directions.append(outcome)
            if TIER_INSIDER_ALERT in tier_tags:
                insider_directions.append(outcome)

        total_usdc = yes_usdc + no_usdc
        if total_usdc == 0:
            return self._empty_consensus()

        net_direction = "NEUTRAL"
        if yes_usdc > no_usdc * 1.2:
            net_direction = "YES"
        elif no_usdc > yes_usdc * 1.2:
            net_direction = "NO"

        # Conviction 0-10
        net_imbalance = abs(yes_usdc - no_usdc) / total_usdc
        size_factor = min(total_usdc / 10_000, 1.0)
        conviction = net_imbalance * 5 + size_factor * 5

        # Dominant directions
        legendary_direction = _majority(legendary_directions) if legendary_directions else "NEUTRAL"
        insider_direction = _majority(insider_directions) if insider_directions else "NEUTRAL"

        avg_entry_yes = safe_div(sum(entries_yes), len(entries_yes)) if entries_yes else 0.5
        avg_entry_no = safe_div(sum(entries_no), len(entries_no)) if entries_no else 0.5

        return {
            "smart_money_yes_usdc": yes_usdc,
            "smart_money_no_usdc": no_usdc,
            "smart_money_yes_wallets": yes_wallets,
            "smart_money_no_wallets": no_wallets,
            "smart_money_net_direction": net_direction,
            "smart_money_conviction": min(10.0, conviction),
            "legendary_direction": legendary_direction,
            "insider_direction": insider_direction,
            "smart_money_avg_entry": avg_entry_yes if net_direction == "YES" else avg_entry_no,
            "total_smart_money_usdc": total_usdc,
        }

    def _empty_consensus(self) -> dict:
        return {
            "smart_money_yes_usdc": 0.0,
            "smart_money_no_usdc": 0.0,
            "smart_money_yes_wallets": 0,
            "smart_money_no_wallets": 0,
            "smart_money_net_direction": "NEUTRAL",
            "smart_money_conviction": 0.0,
            "legendary_direction": "NEUTRAL",
            "insider_direction": "NEUTRAL",
            "smart_money_avg_entry": 0.5,
            "total_smart_money_usdc": 0.0,
        }

    # ── Step C: Real-time trade monitoring ───────────────────────────────────

    async def poll_new_trades(self, monitored_market_slugs: list[str]) -> list[WhaleAlert]:
        """
        Poll for new trades on monitored markets. Cross-reference against
        tracked wallets and generate alerts for significant trades.
        """
        since_ts = self._last_trade_ts
        new_alerts: list[WhaleAlert] = []

        # Load all tracked wallet addresses for fast lookup
        all_wallets = await self.db.get_all_tracked_wallets()
        wallet_lookup: dict[str, dict] = {w["address"]: w for w in all_wallets}

        for market_slug in monitored_market_slugs[:20]:  # limit per cycle
            try:
                trades = await self.client.data.get_recent_trades_for_market(
                    market_slug, since_ts=since_ts
                )
                for trade in trades:
                    maker = str(trade.get("maker") or trade.get("maker_address") or "").lower()
                    taker = str(trade.get("taker") or trade.get("taker_address") or "").lower()
                    # Check both sides
                    for wallet_addr in [maker, taker]:
                        if wallet_addr and wallet_addr in wallet_lookup:
                            wallet_rec = wallet_lookup[wallet_addr]
                            alert = await self._process_tracked_trade(
                                wallet_rec, trade, market_slug
                            )
                            if alert:
                                new_alerts.append(alert)
                                await self.pending_alerts.put(alert)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug("Trade poll failed for %s: %s", market_slug, e)

        if new_alerts:
            self._last_trade_ts = now_ts()
            logger.info("Found %d whale trade alerts", len(new_alerts))

        return new_alerts

    async def _process_tracked_trade(
        self, wallet_rec: dict, trade: dict, market_slug: str
    ) -> Optional[WhaleAlert]:
        """Generate a WhaleAlert for a trade from a tracked wallet."""
        tier_tags = json.loads(wallet_rec.get("tier_tags") or "[]")
        win_rate = float(wallet_rec.get("win_rate") or 0)
        total_pnl = float(wallet_rec.get("total_pnl") or 0)
        insider_score = float(wallet_rec.get("insider_score") or 0)
        avg_size = float(wallet_rec.get("avg_position_size") or 1.0) or 1.0

        size_usdc = float(
            trade.get("size") or trade.get("usdcSize") or trade.get("size_usdc") or 0
        )
        price = float(trade.get("price") or 0.5)
        outcome = str(trade.get("outcome") or trade.get("side") or "YES").upper()
        if outcome not in ("YES", "NO"):
            outcome = "YES"

        alert_level = compute_alert_level(tier_tags, size_usdc, win_rate, insider_score)
        conviction_multiplier = safe_div(size_usdc, avg_size)

        # Log to whale_trades table
        await self.db.insert_whale_trade({
            "wallet": wallet_rec["address"],
            "market_slug": market_slug,
            "outcome": outcome,
            "side": "BUY",
            "price": price,
            "size_usdc": size_usdc,
            "timestamp": float(trade.get("timestamp") or now_ts()),
            "alert_level": alert_level,
            "our_copy_trade_id": None,
            "information_lead_at_time": wallet_rec.get("information_lead_score"),
        })

        return WhaleAlert(
            wallet_address=wallet_rec["address"],
            wallet_tier=tier_tags,
            wallet_win_rate=win_rate,
            wallet_total_pnl=total_pnl,
            insider_score=insider_score,
            market_slug=market_slug,
            question=trade.get("question", ""),
            outcome=outcome,
            price=price,
            size_usdc=size_usdc,
            conviction_multiplier=conviction_multiplier,
            is_adding_to_position=False,  # would check existing positions
            other_smart_money_same_direction=0,
            time_since_market_opened=0.0,
            days_until_resolution=float(trade.get("days_to_resolution", 30)),
            alert_level=alert_level,
            timestamp=now_ts(),
        )

    async def get_recent_alerts(self, max_count: int = 20) -> list[WhaleAlert]:
        """Drain pending alerts queue."""
        alerts = []
        while not self.pending_alerts.empty() and len(alerts) < max_count:
            try:
                alert = self.pending_alerts.get_nowait()
                alerts.append(alert)
            except asyncio.QueueEmpty:
                break
        return alerts

    async def should_do_full_refresh(self) -> bool:
        """Check if it's time for a full wallet refresh."""
        elapsed = now_ts() - self._last_full_refresh
        return elapsed > self.settings.whale_refresh_hours * 3600

    def get_whale_score_for_market(self, consensus: dict) -> tuple[float, str]:
        """
        Convert smart money consensus into a 0-10 signal score and direction.
        """
        conviction = float(consensus.get("smart_money_conviction", 0))
        net_dir = str(consensus.get("smart_money_net_direction", "NEUTRAL"))

        # Boost if insider or legendary agrees
        boost = 0.0
        if consensus.get("insider_direction") != "NEUTRAL":
            if consensus["insider_direction"] == net_dir:
                boost += 2.0
        if consensus.get("legendary_direction") != "NEUTRAL":
            if consensus["legendary_direction"] == net_dir:
                boost += 1.0

        score = min(10.0, conviction + boost)
        direction = net_dir if net_dir != "NEUTRAL" else "NEUTRAL"
        return score, direction


def _majority(directions: list[str]) -> str:
    if not directions:
        return "NEUTRAL"
    counts: dict[str, int] = defaultdict(int)
    for d in directions:
        counts[d] += 1
    best = max(counts, key=lambda k: counts[k])
    if counts[best] > len(directions) / 2:
        return best
    return "NEUTRAL"
