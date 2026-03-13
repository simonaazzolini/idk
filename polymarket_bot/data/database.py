"""
Async SQLite ORM using aiosqlite.
All table schemas, initialization, and CRUD helpers.
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

# ── Schema DDL ────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS bot_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_cache (
    slug        TEXT PRIMARY KEY,
    data_json   TEXT NOT NULL,
    fetched_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        REAL NOT NULL,
    mode             TEXT NOT NULL,
    market_slug      TEXT NOT NULL,
    question         TEXT NOT NULL,
    category         TEXT,
    outcome          TEXT NOT NULL,
    side             TEXT NOT NULL,
    price            REAL NOT NULL,
    size_usdc        REAL NOT NULL,
    shares           REAL NOT NULL,
    order_id         TEXT,
    order_type       TEXT,
    fill_price       REAL,
    slippage         REAL,
    status           TEXT NOT NULL DEFAULT 'OPEN',
    pnl              REAL,
    hold_hours       REAL,
    exit_reason      TEXT,
    composite_score  REAL,
    ai_probability   REAL,
    ai_edge          REAL,
    ai_confidence    REAL,
    whale_score      REAL,
    news_score       REAL,
    technical_score  REAL,
    arb_score        REAL,
    signal_strength  TEXT,
    is_whale_copy    INTEGER DEFAULT 0,
    source_wallet    TEXT
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        REAL NOT NULL,
    mode             TEXT NOT NULL DEFAULT 'PAPER',
    market_slug      TEXT NOT NULL,
    question         TEXT NOT NULL,
    category         TEXT,
    outcome          TEXT NOT NULL,
    side             TEXT NOT NULL,
    price            REAL NOT NULL,
    size_usdc        REAL NOT NULL,
    shares           REAL NOT NULL,
    order_id         TEXT,
    order_type       TEXT,
    fill_price       REAL,
    slippage         REAL,
    status           TEXT NOT NULL DEFAULT 'OPEN',
    pnl              REAL,
    hold_hours       REAL,
    exit_reason      TEXT,
    composite_score  REAL,
    ai_probability   REAL,
    ai_edge          REAL,
    ai_confidence    REAL,
    whale_score      REAL,
    news_score       REAL,
    technical_score  REAL,
    arb_score        REAL,
    signal_strength  TEXT,
    is_whale_copy    INTEGER DEFAULT 0,
    source_wallet    TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           REAL NOT NULL,
    market_slug         TEXT NOT NULL,
    question            TEXT NOT NULL,
    ai_probability      REAL,
    market_price        REAL,
    ai_edge             REAL,
    whale_score         REAL,
    whale_direction     TEXT,
    news_score          REAL,
    news_direction      TEXT,
    technical_score     REAL,
    technical_direction TEXT,
    orderbook_score     REAL,
    arb_score           REAL,
    composite_score     REAL,
    final_direction     TEXT,
    action_taken        TEXT,
    reason_skipped      TEXT
);

CREATE TABLE IF NOT EXISTS whale_wallets (
    address                 TEXT PRIMARY KEY,
    tier_tags               TEXT NOT NULL DEFAULT '[]',
    win_rate                REAL DEFAULT 0,
    win_rate_large          REAL DEFAULT 0,
    win_rate_by_category    TEXT DEFAULT '{}',
    total_pnl               REAL DEFAULT 0,
    total_volume            REAL DEFAULT 0,
    profit_factor           REAL DEFAULT 0,
    sharpe_ratio            REAL DEFAULT 0,
    total_trades            INTEGER DEFAULT 0,
    avg_position_size       REAL DEFAULT 0,
    information_lead_score  REAL DEFAULT 0,
    timing_alpha            REAL DEFAULT 0,
    news_precession_rate    REAL DEFAULT 0,
    insider_score           REAL DEFAULT 0,
    is_bot                  INTEGER DEFAULT 0,
    is_insider_candidate    INTEGER DEFAULT 0,
    is_copy_trader          INTEGER DEFAULT 0,
    leader_score            INTEGER DEFAULT 0,
    follower_score          INTEGER DEFAULT 0,
    cluster_id              TEXT,
    first_seen              REAL,
    last_active             REAL,
    last_refreshed          REAL,
    full_stats_json         TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS whale_positions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet           TEXT NOT NULL,
    market_slug      TEXT NOT NULL,
    outcome          TEXT NOT NULL,
    shares           REAL DEFAULT 0,
    avg_entry_price  REAL DEFAULT 0,
    current_price    REAL DEFAULT 0,
    unrealized_pnl   REAL DEFAULT 0,
    opened_at        REAL,
    last_updated     REAL NOT NULL,
    conviction_score REAL DEFAULT 1,
    UNIQUE(wallet, market_slug, outcome)
);

CREATE TABLE IF NOT EXISTS whale_trades (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet                  TEXT NOT NULL,
    market_slug             TEXT NOT NULL,
    outcome                 TEXT NOT NULL,
    side                    TEXT NOT NULL,
    price                   REAL NOT NULL,
    size_usdc               REAL NOT NULL,
    timestamp               REAL NOT NULL,
    alert_level             TEXT,
    our_copy_trade_id       INTEGER,
    information_lead_at_time REAL
);

CREATE TABLE IF NOT EXISTS wallet_clusters (
    cluster_id           TEXT PRIMARY KEY,
    leader_wallet        TEXT,
    member_wallets       TEXT DEFAULT '[]',
    correlation_score    REAL DEFAULT 0,
    total_cluster_volume REAL DEFAULT 0,
    cluster_win_rate     REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS copy_trade_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL NOT NULL,
    source_wallet   TEXT NOT NULL,
    source_tier     TEXT NOT NULL,
    insider_score   REAL DEFAULT 0,
    market_slug     TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    our_size        REAL NOT NULL,
    our_entry_price REAL NOT NULL,
    exit_price      REAL,
    pnl             REAL,
    was_profitable  INTEGER,
    ai_agreed       INTEGER,
    close_timestamp REAL
);

CREATE TABLE IF NOT EXISTS insider_alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet       TEXT NOT NULL,
    insider_score REAL NOT NULL,
    evidence_json TEXT NOT NULL,
    flagged_at   REAL NOT NULL,
    reviewed     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            REAL NOT NULL,
    mode                 TEXT NOT NULL,
    cash_balance         REAL NOT NULL,
    position_value       REAL NOT NULL,
    total_value          REAL NOT NULL,
    realized_pnl         REAL NOT NULL,
    unrealized_pnl       REAL NOT NULL,
    roi_pct              REAL NOT NULL,
    drawdown             REAL NOT NULL,
    open_positions_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS arb_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        REAL NOT NULL,
    arb_type         TEXT NOT NULL,
    market_slug      TEXT NOT NULL,
    legs_json        TEXT NOT NULL,
    profit_usdc      REAL NOT NULL,
    profit_pct       REAL NOT NULL,
    executed         INTEGER DEFAULT 0,
    execution_time_ms REAL
);

CREATE TABLE IF NOT EXISTS performance_daily (
    date          TEXT NOT NULL,
    mode          TEXT NOT NULL,
    starting_value REAL NOT NULL,
    ending_value  REAL NOT NULL,
    pnl           REAL NOT NULL,
    win_rate      REAL NOT NULL,
    trades_count  INTEGER NOT NULL,
    avg_edge      REAL NOT NULL,
    sharpe_daily  REAL NOT NULL,
    max_drawdown  REAL NOT NULL,
    best_trade    REAL NOT NULL,
    worst_trade   REAL NOT NULL,
    whale_copy_pnl REAL NOT NULL DEFAULT 0,
    ai_only_pnl   REAL NOT NULL DEFAULT 0,
    arb_pnl       REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (date, mode)
);

CREATE INDEX IF NOT EXISTS idx_trades_slug ON trades(market_slug);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_paper_trades_slug ON paper_trades(market_slug);
CREATE INDEX IF NOT EXISTS idx_paper_trades_ts ON paper_trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_slug ON signals(market_slug);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_whale_positions_wallet ON whale_positions(wallet);
CREATE INDEX IF NOT EXISTS idx_whale_trades_wallet ON whale_trades(wallet);
CREATE INDEX IF NOT EXISTS idx_whale_trades_ts ON whale_trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_portfolio_ts ON portfolio_snapshots(timestamp);
"""


class Database:
    """Async SQLite database with connection pooling."""

    def __init__(self, db_path: str = "polymarket_bot.db"):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create all tables if they don't exist."""
        async with self._get_conn() as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()
        logger.info("Database initialized at %s", self.db_path)

    @asynccontextmanager
    async def _get_conn(self):
        """Get a database connection. Opens a new one per call for safety."""
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        try:
            yield db
        finally:
            await db.close()

    # ── bot_state ─────────────────────────────────────────────────────────────

    async def get_state(self, key: str) -> Optional[str]:
        async with self._get_conn() as db:
            async with db.execute("SELECT value FROM bot_state WHERE key=?", (key,)) as cur:
                row = await cur.fetchone()
                return row["value"] if row else None

    async def set_state(self, key: str, value: str) -> None:
        async with self._get_conn() as db:
            await db.execute(
                "INSERT OR REPLACE INTO bot_state(key,value) VALUES(?,?)",
                (key, str(value))
            )
            await db.commit()

    async def get_state_float(self, key: str) -> Optional[float]:
        v = await self.get_state(key)
        return float(v) if v is not None else None

    # ── market_cache ──────────────────────────────────────────────────────────

    async def cache_markets(self, markets: list[dict]) -> None:
        now = datetime.now(timezone.utc).timestamp()
        async with self._get_conn() as db:
            await db.execute("DELETE FROM market_cache")
            for m in markets:
                await db.execute(
                    "INSERT OR REPLACE INTO market_cache(slug,data_json,fetched_at) VALUES(?,?,?)",
                    (m.get("slug", m.get("conditionId", "")), json.dumps(m), now)
                )
            await db.commit()

    async def get_cached_markets(self, ttl_minutes: int = 10) -> Optional[list[dict]]:
        cutoff = datetime.now(timezone.utc).timestamp() - ttl_minutes * 60
        async with self._get_conn() as db:
            async with db.execute(
                "SELECT data_json, fetched_at FROM market_cache WHERE fetched_at > ? LIMIT 1",
                (cutoff,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
            async with db.execute(
                "SELECT data_json FROM market_cache WHERE fetched_at > ?",
                (cutoff,)
            ) as cur:
                rows = await cur.fetchall()
                return [json.loads(r["data_json"]) for r in rows]

    # ── trades ────────────────────────────────────────────────────────────────

    async def insert_trade(self, trade: dict) -> int:
        table = "paper_trades" if trade.get("mode") == "PAPER" else "trades"
        cols = list(trade.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        async with self._get_conn() as db:
            async with db.execute(
                f"INSERT INTO {table}({col_str}) VALUES({placeholders})",
                [trade[c] for c in cols]
            ) as cur:
                trade_id = cur.lastrowid
            await db.commit()
        return trade_id

    async def update_trade(self, trade_id: int, mode: str, updates: dict) -> None:
        table = "paper_trades" if mode == "PAPER" else "trades"
        set_parts = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [trade_id]
        async with self._get_conn() as db:
            await db.execute(f"UPDATE {table} SET {set_parts} WHERE id=?", vals)
            await db.commit()

    async def get_open_trades(self, mode: str) -> list[dict]:
        table = "paper_trades" if mode == "PAPER" else "trades"
        async with self._get_conn() as db:
            async with db.execute(
                f"SELECT * FROM {table} WHERE status='OPEN' AND side='BUY'"
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_trades_in_range(self, mode: str, start_ts: float, end_ts: float) -> list[dict]:
        table = "paper_trades" if mode == "PAPER" else "trades"
        async with self._get_conn() as db:
            async with db.execute(
                f"SELECT * FROM {table} WHERE timestamp >= ? AND timestamp <= ?",
                (start_ts, end_ts)
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_all_closed_trades(self, mode: str) -> list[dict]:
        table = "paper_trades" if mode == "PAPER" else "trades"
        async with self._get_conn() as db:
            async with db.execute(
                f"SELECT * FROM {table} WHERE status='CLOSED'"
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    # ── signals ───────────────────────────────────────────────────────────────

    async def insert_signal(self, signal: dict) -> int:
        cols = list(signal.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        async with self._get_conn() as db:
            async with db.execute(
                f"INSERT INTO signals({col_str}) VALUES({placeholders})",
                [signal[c] for c in cols]
            ) as cur:
                sig_id = cur.lastrowid
            await db.commit()
        return sig_id

    # ── whale_wallets ─────────────────────────────────────────────────────────

    async def upsert_whale_wallet(self, wallet: dict) -> None:
        cols = list(wallet.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "address")
        async with self._get_conn() as db:
            await db.execute(
                f"INSERT INTO whale_wallets({col_str}) VALUES({placeholders}) "
                f"ON CONFLICT(address) DO UPDATE SET {updates}",
                [wallet[c] for c in cols]
            )
            await db.commit()

    async def get_whale_wallets(self, min_insider_score: float = 0.0) -> list[dict]:
        async with self._get_conn() as db:
            async with db.execute(
                "SELECT * FROM whale_wallets WHERE insider_score >= ? ORDER BY insider_score DESC",
                (min_insider_score,)
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_all_tracked_wallets(self) -> list[dict]:
        async with self._get_conn() as db:
            async with db.execute("SELECT * FROM whale_wallets") as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    # ── whale_positions ───────────────────────────────────────────────────────

    async def upsert_whale_position(self, pos: dict) -> None:
        cols = list(pos.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        updates = ", ".join(
            f"{c}=excluded.{c}" for c in cols
            if c not in ("wallet", "market_slug", "outcome")
        )
        async with self._get_conn() as db:
            await db.execute(
                f"INSERT INTO whale_positions({col_str}) VALUES({placeholders}) "
                f"ON CONFLICT(wallet,market_slug,outcome) DO UPDATE SET {updates}",
                [pos[c] for c in cols]
            )
            await db.commit()

    async def get_whale_positions_for_market(self, market_slug: str) -> list[dict]:
        async with self._get_conn() as db:
            async with db.execute(
                "SELECT wp.*, ww.tier_tags, ww.win_rate, ww.insider_score, ww.total_pnl "
                "FROM whale_positions wp "
                "JOIN whale_wallets ww ON wp.wallet = ww.address "
                "WHERE wp.market_slug = ?",
                (market_slug,)
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    # ── whale_trades ──────────────────────────────────────────────────────────

    async def insert_whale_trade(self, trade: dict) -> int:
        cols = list(trade.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        async with self._get_conn() as db:
            async with db.execute(
                f"INSERT INTO whale_trades({col_str}) VALUES({placeholders})",
                [trade[c] for c in cols]
            ) as cur:
                tid = cur.lastrowid
            await db.commit()
        return tid

    async def get_recent_whale_trades(self, since_ts: float) -> list[dict]:
        async with self._get_conn() as db:
            async with db.execute(
                "SELECT wt.*, ww.tier_tags, ww.win_rate, ww.insider_score "
                "FROM whale_trades wt "
                "JOIN whale_wallets ww ON wt.wallet = ww.address "
                "WHERE wt.timestamp >= ? ORDER BY wt.timestamp DESC",
                (since_ts,)
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    # ── copy_trade_log ────────────────────────────────────────────────────────

    async def insert_copy_trade(self, ct: dict) -> int:
        cols = list(ct.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        async with self._get_conn() as db:
            async with db.execute(
                f"INSERT INTO copy_trade_log({col_str}) VALUES({placeholders})",
                [ct[c] for c in cols]
            ) as cur:
                ctid = cur.lastrowid
            await db.commit()
        return ctid

    async def update_copy_trade(self, ct_id: int, updates: dict) -> None:
        set_parts = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [ct_id]
        async with self._get_conn() as db:
            await db.execute(f"UPDATE copy_trade_log SET {set_parts} WHERE id=?", vals)
            await db.commit()

    # ── insider_alerts ────────────────────────────────────────────────────────

    async def insert_insider_alert(self, alert: dict) -> None:
        cols = list(alert.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        async with self._get_conn() as db:
            await db.execute(
                f"INSERT INTO insider_alerts({col_str}) VALUES({placeholders})",
                [alert[c] for c in cols]
            )
            await db.commit()

    # ── portfolio_snapshots ───────────────────────────────────────────────────

    async def insert_portfolio_snapshot(self, snap: dict) -> None:
        cols = list(snap.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        async with self._get_conn() as db:
            await db.execute(
                f"INSERT INTO portfolio_snapshots({col_str}) VALUES({placeholders})",
                [snap[c] for c in cols]
            )
            await db.commit()

    async def get_portfolio_snapshots(self, since_ts: float) -> list[dict]:
        async with self._get_conn() as db:
            async with db.execute(
                "SELECT * FROM portfolio_snapshots WHERE timestamp >= ? ORDER BY timestamp",
                (since_ts,)
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    # ── arb_trades ────────────────────────────────────────────────────────────

    async def insert_arb_trade(self, arb: dict) -> int:
        cols = list(arb.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        async with self._get_conn() as db:
            async with db.execute(
                f"INSERT INTO arb_trades({col_str}) VALUES({placeholders})",
                [arb[c] for c in cols]
            ) as cur:
                aid = cur.lastrowid
            await db.commit()
        return aid

    # ── performance_daily ─────────────────────────────────────────────────────

    async def upsert_performance_daily(self, perf: dict) -> None:
        cols = list(perf.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        updates = ", ".join(
            f"{c}=excluded.{c}" for c in cols if c not in ("date", "mode")
        )
        async with self._get_conn() as db:
            await db.execute(
                f"INSERT INTO performance_daily({col_str}) VALUES({placeholders}) "
                f"ON CONFLICT(date,mode) DO UPDATE SET {updates}",
                [perf[c] for c in cols]
            )
            await db.commit()

    async def get_performance_daily(self, mode: str, days: int = 7) -> list[dict]:
        async with self._get_conn() as db:
            async with db.execute(
                "SELECT * FROM performance_daily WHERE mode=? ORDER BY date DESC LIMIT ?",
                (mode, days)
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    # ── wallet_clusters ───────────────────────────────────────────────────────

    async def upsert_wallet_cluster(self, cluster: dict) -> None:
        cols = list(cluster.keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        updates = ", ".join(
            f"{c}=excluded.{c}" for c in cols if c != "cluster_id"
        )
        async with self._get_conn() as db:
            await db.execute(
                f"INSERT INTO wallet_clusters({col_str}) VALUES({placeholders}) "
                f"ON CONFLICT(cluster_id) DO UPDATE SET {updates}",
                [cluster[c] for c in cols]
            )
            await db.commit()

    async def get_wallet_clusters(self) -> list[dict]:
        async with self._get_conn() as db:
            async with db.execute("SELECT * FROM wallet_clusters") as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    # ── utility ───────────────────────────────────────────────────────────────

    async def execute_raw(self, sql: str, params: tuple = ()) -> list[dict]:
        async with self._get_conn() as db:
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]

    async def get_db_size_mb(self) -> float:
        import os
        try:
            return os.path.getsize(self.db_path) / (1024 * 1024)
        except OSError:
            return 0.0
