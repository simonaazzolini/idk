"""
Wallet relationship graph analysis.
Module 4, Step F: builds directed influence graph, detects clusters and leaders.
"""
import json
import logging
from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
from scipy.stats import pearsonr

from data.database import Database
from utils.helpers import now_ts, safe_div

logger = logging.getLogger(__name__)


class WalletGraph:
    """
    Builds and maintains a directed influence graph of tracked wallets.
    A directed edge A→B means A consistently trades before B in the same direction.
    """

    def __init__(self, db: Database):
        self.db = db
        self.graph = nx.DiGraph()
        self._last_built: float = 0.0

    async def build(self, whale_trades: list[dict]) -> None:
        """
        Rebuild the wallet relationship graph from trade history.
        whale_trades: list of all trades for all tracked wallets,
                      each with at minimum: wallet, market_slug, outcome, timestamp
        """
        # Group trades by (market_slug, outcome)
        market_outcome_trades: dict[tuple, list[dict]] = defaultdict(list)
        for t in whale_trades:
            key = (t.get("market_slug", ""), t.get("outcome", "").upper())
            market_outcome_trades[key].append(t)

        # Get all unique wallets
        all_wallets = list({t["wallet"] for t in whale_trades if t.get("wallet")})

        # Initialize graph nodes
        self.graph.clear()
        for wallet in all_wallets:
            self.graph.add_node(wallet)

        # Compute pairwise relationships
        edge_candidates: dict[tuple[str, str], list[float]] = defaultdict(list)  # (A,B) → [lags]

        for (market_slug, outcome), trades_in_group in market_outcome_trades.items():
            # Sort by timestamp
            sorted_trades = sorted(
                [t for t in trades_in_group if t.get("timestamp")],
                key=lambda t: float(t.get("timestamp", 0))
            )
            if len(sorted_trades) < 2:
                continue

            # For each pair (A, B) where A traded before B in same direction
            for i, trade_a in enumerate(sorted_trades):
                for trade_b in sorted_trades[i+1:]:
                    wallet_a = trade_a.get("wallet")
                    wallet_b = trade_b.get("wallet")
                    if not wallet_a or not wallet_b or wallet_a == wallet_b:
                        continue
                    ts_a = float(trade_a.get("timestamp", 0))
                    ts_b = float(trade_b.get("timestamp", 0))
                    lag = ts_b - ts_a
                    if 0 < lag <= 4 * 3600:  # within 4 hours
                        edge_candidates[(wallet_a, wallet_b)].append(lag)

        # Build edges for pairs with consistent lead/follow behavior
        for (wallet_a, wallet_b), lags in edge_candidates.items():
            if len(lags) < 3:  # need at least 3 co-occurrences
                continue
            avg_lag = float(np.mean(lags))
            # Compute direction correlation across all markets
            corr = self._compute_direction_correlation(
                whale_trades, wallet_a, wallet_b
            )
            # Compute lead-follow score
            lead_follow_score = self._compute_lead_follow(
                whale_trades, wallet_a, wallet_b
            )
            if corr > 0.6 and avg_lag > 0 and lead_follow_score > 0.5:
                self.graph.add_edge(
                    wallet_a,
                    wallet_b,
                    weight=lead_follow_score,
                    avg_lag_seconds=avg_lag,
                    correlation=corr,
                    co_trades=len(lags),
                )
                logger.debug(
                    "Edge: %s → %s (corr=%.2f, lag=%.0fs, score=%.2f)",
                    wallet_a[:8], wallet_b[:8], corr, avg_lag, lead_follow_score
                )

        # Compute scores for each node
        for wallet in all_wallets:
            leader_score = self.graph.out_degree(wallet)
            follower_score = self.graph.in_degree(wallet)
            self.graph.nodes[wallet]["leader_score"] = leader_score
            self.graph.nodes[wallet]["follower_score"] = follower_score

        # Identify clusters (connected components in undirected version)
        undirected = self.graph.to_undirected()
        clusters = list(nx.connected_components(undirected))
        for cluster_id, members in enumerate(clusters):
            cluster_id_str = f"cluster_{cluster_id}"
            # Find the node with highest leader_score in this cluster
            leader = max(
                members,
                key=lambda w: self.graph.nodes[w].get("leader_score", 0),
                default=next(iter(members)),
            )
            for wallet in members:
                self.graph.nodes[wallet]["cluster_id"] = cluster_id_str
                self.graph.nodes[wallet]["cluster_leader"] = leader

        # Persist to DB
        await self._persist_to_db(clusters)
        self._last_built = now_ts()
        logger.info(
            "Graph built: %d nodes, %d edges, %d clusters",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(clusters),
        )

    def get_leader_score(self, wallet: str) -> int:
        if wallet not in self.graph:
            return 0
        return self.graph.nodes[wallet].get("leader_score", 0)

    def get_follower_score(self, wallet: str) -> int:
        if wallet not in self.graph:
            return 0
        return self.graph.nodes[wallet].get("follower_score", 0)

    def get_cluster_id(self, wallet: str) -> Optional[str]:
        if wallet not in self.graph:
            return None
        return self.graph.nodes[wallet].get("cluster_id")

    def get_signal_leaders(self, min_leader_score: int = 3, min_win_rate: float = 0.60) -> list[str]:
        """Return wallets that are signal leaders."""
        leaders = []
        for wallet, data in self.graph.nodes(data=True):
            if data.get("leader_score", 0) >= min_leader_score:
                # Would need win_rate from DB; caller should filter further
                leaders.append(wallet)
        return leaders

    def detect_cascade(
        self,
        leader_wallet: str,
        leader_trade_ts: float,
        outcome: str,
        market_slug: str,
        recent_trades: list[dict],
        window_hours: float = 4.0,
    ) -> bool:
        """
        Detect if 2+ followers of leader_wallet traded same outcome within window.
        """
        if leader_wallet not in self.graph:
            return False
        followers = list(self.graph.successors(leader_wallet))
        if not followers:
            return False

        cutoff = leader_trade_ts + window_hours * 3600
        follower_trades = 0
        for t in recent_trades:
            t_wallet = t.get("wallet", "")
            t_outcome = str(t.get("outcome", "")).upper()
            t_market = t.get("market_slug", "")
            t_ts = float(t.get("timestamp", 0))
            if (
                t_wallet in followers
                and t_outcome == outcome.upper()
                and t_market == market_slug
                and leader_trade_ts < t_ts <= cutoff
            ):
                follower_trades += 1
                if follower_trades >= 2:
                    logger.info(
                        "CASCADE DETECTED: leader=%s, market=%s, outcome=%s, followers=%d",
                        leader_wallet[:8], market_slug, outcome, follower_trades
                    )
                    return True
        return False

    def _compute_direction_correlation(
        self, all_trades: list[dict], wallet_a: str, wallet_b: str
    ) -> float:
        """Pearson correlation of trade directions per market for wallet A and B."""
        trades_a = {
            t["market_slug"]: (1 if t.get("outcome", "").upper() == "YES" else -1)
            for t in all_trades
            if t.get("wallet") == wallet_a and t.get("market_slug")
        }
        trades_b = {
            t["market_slug"]: (1 if t.get("outcome", "").upper() == "YES" else -1)
            for t in all_trades
            if t.get("wallet") == wallet_b and t.get("market_slug")
        }
        common_markets = set(trades_a.keys()) & set(trades_b.keys())
        if len(common_markets) < 3:
            return 0.0
        vec_a = [trades_a[m] for m in common_markets]
        vec_b = [trades_b[m] for m in common_markets]
        try:
            corr, _ = pearsonr(vec_a, vec_b)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def _compute_lead_follow(
        self, all_trades: list[dict], wallet_a: str, wallet_b: str
    ) -> float:
        """
        Fraction of times A traded same-direction before B in the same market.
        """
        trades_a_by_market: dict[str, list[dict]] = defaultdict(list)
        trades_b_by_market: dict[str, list[dict]] = defaultdict(list)

        for t in all_trades:
            if t.get("wallet") == wallet_a:
                trades_a_by_market[t.get("market_slug", "")].append(t)
            elif t.get("wallet") == wallet_b:
                trades_b_by_market[t.get("market_slug", "")].append(t)

        common = set(trades_a_by_market.keys()) & set(trades_b_by_market.keys())
        if len(common) < 3:
            return 0.0

        lead_count = 0
        total = 0
        for market in common:
            for ta in trades_a_by_market[market]:
                for tb in trades_b_by_market[market]:
                    if ta.get("outcome", "").upper() != tb.get("outcome", "").upper():
                        continue
                    ts_a = float(ta.get("timestamp", 0))
                    ts_b = float(tb.get("timestamp", 0))
                    if 0 < ts_b - ts_a <= 4 * 3600:
                        total += 1
                        lead_count += 1
                    elif 0 < ts_a - ts_b <= 4 * 3600:
                        total += 1

        return safe_div(lead_count, total) if total > 0 else 0.0

    async def _persist_to_db(self, clusters: list[set]) -> None:
        """Save cluster information to the database."""
        for cluster_id, members in enumerate(clusters):
            if len(members) < 2:
                continue
            members_list = list(members)
            # Find leader
            leader = max(
                members_list,
                key=lambda w: self.graph.nodes[w].get("leader_score", 0)
            )
            # Compute cluster stats from graph nodes
            cluster_data = {
                "cluster_id": f"cluster_{cluster_id}",
                "leader_wallet": leader,
                "member_wallets": json.dumps(members_list),
                "correlation_score": float(np.mean([
                    self.graph.edges[u, v].get("correlation", 0)
                    for u, v in self.graph.edges()
                    if u in members and v in members
                ]) if any(
                    u in members and v in members
                    for u, v in self.graph.edges()
                ) else 0),
                "total_cluster_volume": 0.0,  # would require joining with whale_wallets
                "cluster_win_rate": 0.0,
            }
            await self.db.upsert_wallet_cluster(cluster_data)

        # Update leader/follower scores in whale_wallets table
        for wallet in self.graph.nodes():
            node_data = self.graph.nodes[wallet]
            await self.db.execute_raw(
                "UPDATE whale_wallets SET leader_score=?, follower_score=?, cluster_id=? "
                "WHERE address=?",
                (
                    node_data.get("leader_score", 0),
                    node_data.get("follower_score", 0),
                    node_data.get("cluster_id"),
                    wallet,
                )
            )
