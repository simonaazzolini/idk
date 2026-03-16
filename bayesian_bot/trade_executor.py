"""
trade_executor.py — Manages trade entry, position tracking, and exit logic.

Handles both paper trading (simulated) and live trading (via py_clob_client).
Paper trading simulates realistic order book impact, slippage, and fill delays.
"""
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PAPER_TRADES_PATH = Path(__file__).parent / "paper_trades.json"
LIVE_TRADES_PATH = Path(__file__).parent / "live_trades.json"


@dataclass
class Position:
    """Represents an open trading position."""
    trade_id: str
    mode: str                    # "paper" or "live"
    condition_id: str
    question: str
    asset: str
    direction: str               # "up" or "down"
    outcome: str                 # "YES" or "NO"
    token_id: str
    size_usd: float
    entry_price: float           # price paid for shares
    current_price: float
    shares: float                # number of shares
    posterior_at_entry: float
    market_price_at_entry: float
    divergence_at_entry: float
    evidence_breakdown: dict
    prior_used: float
    regime_at_entry: str
    session_at_entry: str
    entry_time: datetime
    end_time: datetime
    minutes_to_resolution: float
    status: str = "open"         # open, closed, expired
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    resolution_outcome: Optional[bool] = None  # True=YES, False=NO

    def minutes_held(self) -> float:
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60.0
        return (datetime.now(timezone.utc) - self.entry_time).total_seconds() / 60.0

    def minutes_to_expiry(self) -> float:
        now = datetime.now(timezone.utc)
        return max(0.0, (self.end_time - now).total_seconds() / 60.0)

    def position_value(self) -> float:
        return self.shares * self.current_price

    def unrealized_pnl(self) -> float:
        return self.position_value() - self.size_usd

    def unrealized_pnl_pct(self) -> float:
        if self.size_usd == 0:
            return 0.0
        return self.unrealized_pnl() / self.size_usd

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "mode": self.mode,
            "condition_id": self.condition_id,
            "question": self.question,
            "asset": self.asset,
            "direction": self.direction,
            "outcome": self.outcome,
            "token_id": self.token_id,
            "size_usd": self.size_usd,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "shares": self.shares,
            "posterior_at_entry": self.posterior_at_entry,
            "market_price_at_entry": self.market_price_at_entry,
            "divergence_at_entry": self.divergence_at_entry,
            "evidence_breakdown": self.evidence_breakdown,
            "prior_used": self.prior_used,
            "regime_at_entry": self.regime_at_entry,
            "session_at_entry": self.session_at_entry,
            "entry_time": self.entry_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "minutes_to_resolution": self.minutes_to_resolution,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
            "resolution_outcome": self.resolution_outcome,
        }


class TradeExecutor:
    """
    Manages trade lifecycle: entry, monitoring, and exit.

    Paper mode: simulates fills with realistic slippage/delay.
    Live mode: uses py_clob_client for real order placement.
    """

    def __init__(self, config, performance_tracker=None):
        self.config = config
        self.performance_tracker = performance_tracker
        self.open_positions: Dict[str, Position] = {}   # trade_id → Position
        self._clob_client = None
        self._paper_balance = config.paper_starting_balance
        self._live_balance = config.live_starting_balance

        # Load existing open positions
        self._load_open_positions()

    def _load_open_positions(self) -> None:
        """Reload any previously open positions from disk."""
        path = PAPER_TRADES_PATH if self.config.is_paper() else LIVE_TRADES_PATH
        if not path.exists():
            return
        try:
            with open(path) as f:
                trades = json.load(f)
            for t in trades:
                if t.get("status") == "open":
                    try:
                        pos = self._dict_to_position(t)
                        self.open_positions[pos.trade_id] = pos
                    except Exception:
                        pass
            if self.open_positions:
                logger.info("Restored %d open positions", len(self.open_positions))
        except Exception as e:
            logger.debug("Could not load open positions: %s", e)

    def _dict_to_position(self, d: dict) -> Position:
        return Position(
            trade_id=d["trade_id"],
            mode=d["mode"],
            condition_id=d["condition_id"],
            question=d["question"],
            asset=d["asset"],
            direction=d["direction"],
            outcome=d["outcome"],
            token_id=d["token_id"],
            size_usd=d["size_usd"],
            entry_price=d["entry_price"],
            current_price=d["current_price"],
            shares=d["shares"],
            posterior_at_entry=d["posterior_at_entry"],
            market_price_at_entry=d["market_price_at_entry"],
            divergence_at_entry=d["divergence_at_entry"],
            evidence_breakdown=d.get("evidence_breakdown", {}),
            prior_used=d.get("prior_used", 0.5),
            regime_at_entry=d.get("regime_at_entry", "unknown"),
            session_at_entry=d.get("session_at_entry", "unknown"),
            entry_time=datetime.fromisoformat(d["entry_time"]),
            end_time=datetime.fromisoformat(d["end_time"]),
            minutes_to_resolution=d["minutes_to_resolution"],
            status=d["status"],
        )

    def initialize_live_client(self) -> bool:
        """Attempt to initialize the Polymarket CLOB client for live trading."""
        if not self.config.polymarket_private_key:
            logger.warning("No private key configured — live trading unavailable")
            return False
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self.config.polymarket_api_key,
                api_secret=self.config.polymarket_api_secret,
                api_passphrase=self.config.polymarket_api_passphrase,
            )
            self._clob_client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=137,
                key=self.config.polymarket_private_key,
                creds=creds,
                signature_type=0,
            )
            logger.info("Live CLOB client initialized successfully")
            return True
        except ImportError:
            logger.warning("py_clob_client not installed — live trading unavailable")
            return False
        except Exception as e:
            logger.warning("Failed to initialize CLOB client: %s", e)
            return False

    # ── Position sizing ────────────────────────────────────────────────────

    def calculate_position_size(self, divergence: float) -> float:
        """
        Calculate position size using risk-per-trade approach.
        Risk 1% of current balance per trade.
        Hard cap at max_trade_size_usd, minimum at min_trade_size_usd.
        """
        balance = (
            self._paper_balance if self.config.is_paper() else self._live_balance
        )
        risk_amount = balance * self.config.risk_per_trade_pct

        # Scale slightly with divergence (bigger edge = slightly bigger bet)
        scale = min(1.5, 1.0 + abs(divergence) * 2)
        size = risk_amount * scale

        # Hard caps
        size = min(size, self.config.max_trade_size_usd)
        size = max(size, self.config.min_trade_size_usd)
        return round(size, 2)

    # ── Constraint checks ──────────────────────────────────────────────────

    def can_open_position(self, asset: str, condition_id: str) -> tuple:
        """
        Check if we're allowed to open a new position.
        Returns (allowed: bool, reason: str)
        """
        # Max open positions check
        open_count = sum(1 for p in self.open_positions.values() if p.status == "open")
        if open_count >= self.config.max_open_positions:
            return False, f"Max positions ({self.config.max_open_positions}) reached"

        # No 2 positions on same underlying
        same_asset = [
            p for p in self.open_positions.values()
            if p.status == "open" and p.asset == asset
        ]
        if same_asset:
            return False, f"Already holding position in {asset}"

        # No duplicate market
        same_market = [
            p for p in self.open_positions.values()
            if p.status == "open" and p.condition_id == condition_id
        ]
        if same_market:
            return False, f"Already holding position in this market"

        return True, "ok"

    # ── Paper trading ──────────────────────────────────────────────────────

    def _simulate_slippage(self, size_usd: float) -> float:
        """Calculate realistic slippage based on order size."""
        if size_usd < 10:
            base_slippage = 0.005   # 0.5%
        elif size_usd <= 50:
            base_slippage = 0.010   # 1.0%
        else:
            base_slippage = 0.020   # 2.0%

        # Add random noise
        noise = random.uniform(-0.001, 0.002)
        return base_slippage + noise

    def _simulate_fill_delay(self) -> None:
        """Simulate realistic fill delay of 200-500ms."""
        delay = random.uniform(0.2, 0.5)
        time.sleep(delay)

    def enter_paper_trade(
        self,
        market,
        bayesian_result,
        outcome: str
    ) -> Optional[Position]:
        """
        Simulate entering a paper trade with realistic slippage.
        outcome: "YES" or "NO"
        """
        token_id = market.yes_token_id if outcome == "YES" else market.no_token_id
        base_price = market.current_yes_price if outcome == "YES" else market.current_no_price

        size_usd = self.calculate_position_size(bayesian_result.divergence)

        # Check balance
        if size_usd > self._paper_balance:
            size_usd = max(self.config.min_trade_size_usd,
                          min(size_usd, self._paper_balance * 0.95))

        if size_usd < self.config.min_trade_size_usd:
            logger.warning("Insufficient paper balance for minimum trade")
            return None

        # Simulate fill delay
        self._simulate_fill_delay()

        # Apply slippage (worse for buyer: price goes up for YES, down for NO)
        slippage = self._simulate_slippage(size_usd)
        fill_price = min(0.99, base_price * (1 + slippage))

        # Check max slippage
        slippage_actual = (fill_price - base_price) / base_price
        if slippage_actual > self.config.max_slippage_pct:
            logger.warning(
                "Slippage %.2f%% exceeds max %.2f%% — cancelling",
                slippage_actual * 100, self.config.max_slippage_pct * 100
            )
            return None

        shares = size_usd / fill_price if fill_price > 0 else 0

        # Deduct from paper balance
        self._paper_balance -= size_usd
        self.config.paper_balance = self._paper_balance

        trade_id = str(uuid.uuid4())[:8]
        position = Position(
            trade_id=trade_id,
            mode="paper",
            condition_id=market.condition_id,
            question=market.question,
            asset=market.asset,
            direction=market.direction,
            outcome=outcome,
            token_id=token_id,
            size_usd=size_usd,
            entry_price=fill_price,
            current_price=fill_price,
            shares=shares,
            posterior_at_entry=bayesian_result.posterior,
            market_price_at_entry=bayesian_result.market_price,
            divergence_at_entry=bayesian_result.divergence,
            evidence_breakdown={
                upd.layer: {"delta": upd.delta, "desc": upd.description}
                for upd in bayesian_result.evidence_updates
            },
            prior_used=bayesian_result.prior,
            regime_at_entry=bayesian_result.regime,
            session_at_entry=bayesian_result.session,
            entry_time=datetime.now(timezone.utc),
            end_time=market.end_time,
            minutes_to_resolution=market.minutes_to_resolution,
        )

        self.open_positions[trade_id] = position
        self._save_trade(position)

        logger.info(
            "[PAPER] ENTER %s %s @ %.3f | size=$%.2f | posterior=%.3f | edge=%.3f | %s",
            outcome, market.asset, fill_price, size_usd,
            bayesian_result.posterior, bayesian_result.divergence,
            market.question[:50]
        )
        return position

    def exit_paper_trade(
        self,
        position: Position,
        exit_price: float,
        reason: str,
        resolution_outcome: Optional[bool] = None
    ) -> Position:
        """Close a paper position and calculate P&L."""
        position.exit_price = exit_price
        position.exit_time = datetime.now(timezone.utc)
        position.exit_reason = reason
        position.resolution_outcome = resolution_outcome
        position.status = "closed"

        # P&L calculation
        if resolution_outcome is not None:
            # Market resolved: YES pays 1.0, NO pays 0.0
            payout_price = 1.0 if resolution_outcome else 0.0
            if position.outcome == "NO":
                payout_price = 1.0 - payout_price
            gross_payout = position.shares * payout_price
        else:
            # Early exit: sell at current market price
            gross_payout = position.shares * exit_price

        position.pnl_usd = gross_payout - position.size_usd
        position.pnl_pct = position.pnl_usd / position.size_usd if position.size_usd > 0 else 0.0

        # Return proceeds to paper balance
        self._paper_balance += gross_payout
        self.config.paper_balance = self._paper_balance

        if position.trade_id in self.open_positions:
            del self.open_positions[position.trade_id]

        self._save_trade(position)

        outcome_str = "✓ WIN" if position.pnl_usd > 0 else "✗ LOSS"
        logger.info(
            "[PAPER] EXIT %s %s | P&L=$%.2f (%.1f%%) | reason=%s | balance=$%.2f",
            outcome_str, position.asset,
            position.pnl_usd, position.pnl_pct * 100,
            reason, self._paper_balance
        )

        # Notify performance tracker
        if self.performance_tracker:
            self.performance_tracker.record_trade(position)

        return position

    # ── Live trading ───────────────────────────────────────────────────────

    def enter_live_trade(
        self,
        market,
        bayesian_result,
        outcome: str
    ) -> Optional[Position]:
        """Place a real order via Polymarket CLOB API."""
        if not self._clob_client:
            logger.error("CLOB client not initialized — cannot place live trade")
            return None

        token_id = market.yes_token_id if outcome == "YES" else market.no_token_id
        base_price = market.current_yes_price if outcome == "YES" else market.current_no_price
        size_usd = self.calculate_position_size(bayesian_result.divergence)

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType

            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=size_usd,
            )
            response = self._clob_client.create_and_post_order(order_args)

            fill_price = float(response.get("price", base_price))
            shares = float(response.get("size", size_usd / fill_price))
            trade_id = str(response.get("id", uuid.uuid4()))[:8]

            self._live_balance -= size_usd
            self.config.live_balance = self._live_balance

            position = Position(
                trade_id=trade_id,
                mode="live",
                condition_id=market.condition_id,
                question=market.question,
                asset=market.asset,
                direction=market.direction,
                outcome=outcome,
                token_id=token_id,
                size_usd=size_usd,
                entry_price=fill_price,
                current_price=fill_price,
                shares=shares,
                posterior_at_entry=bayesian_result.posterior,
                market_price_at_entry=bayesian_result.market_price,
                divergence_at_entry=bayesian_result.divergence,
                evidence_breakdown={
                    upd.layer: {"delta": upd.delta, "desc": upd.description}
                    for upd in bayesian_result.evidence_updates
                },
                prior_used=bayesian_result.prior,
                regime_at_entry=bayesian_result.regime,
                session_at_entry=bayesian_result.session,
                entry_time=datetime.now(timezone.utc),
                end_time=market.end_time,
                minutes_to_resolution=market.minutes_to_resolution,
            )

            self.open_positions[trade_id] = position
            self._save_trade(position)

            logger.info(
                "[LIVE] ENTER %s %s @ %.3f | size=$%.2f | edge=%.3f",
                outcome, market.asset, fill_price, size_usd, bayesian_result.divergence
            )
            return position

        except Exception as e:
            logger.error("Live trade entry failed: %s", e)
            return None

    # ── Position monitoring ────────────────────────────────────────────────

    def check_exit_conditions(
        self,
        position: Position,
        current_price: float
    ) -> tuple:
        """
        Check if position should be exited early.
        Returns (should_exit: bool, reason: str)
        """
        position.current_price = current_price

        # Stop loss: position value dropped 40%
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        if position.outcome == "NO":
            pnl_pct = -pnl_pct

        if pnl_pct <= -self.config.stop_loss_pct:
            return True, f"stop_loss_{pnl_pct:.1%}"

        # Early exit: 2 minutes to resolution and losing
        mins_left = position.minutes_to_expiry()
        if mins_left <= self.config.early_exit_minutes and pnl_pct < 0:
            return True, f"early_exit_losing_{mins_left:.1f}min_left"

        # Market expired
        if mins_left <= 0:
            return True, "market_expired"

        return False, ""

    def update_position_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all open positions."""
        for pos in list(self.open_positions.values()):
            token_id = pos.token_id
            if token_id in prices:
                pos.current_price = prices[token_id]

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_trade(self, position: Position) -> None:
        """Append/update trade in JSON file."""
        path = PAPER_TRADES_PATH if position.mode == "paper" else LIVE_TRADES_PATH

        trades = []
        if path.exists():
            try:
                with open(path) as f:
                    trades = json.load(f)
            except Exception:
                trades = []

        # Update existing or append
        updated = False
        for i, t in enumerate(trades):
            if t.get("trade_id") == position.trade_id:
                trades[i] = position.to_dict()
                updated = True
                break
        if not updated:
            trades.append(position.to_dict())

        with open(path, "w") as f:
            json.dump(trades, f, indent=2)

    def get_all_trades(self, mode: str = "paper") -> List[dict]:
        """Load all trades from disk."""
        path = PAPER_TRADES_PATH if mode == "paper" else LIVE_TRADES_PATH
        if not path.exists():
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return []

    def get_open_condition_ids(self) -> set:
        """Return set of condition_ids we currently hold positions in."""
        return {p.condition_id for p in self.open_positions.values() if p.status == "open"}

    def get_open_assets(self) -> set:
        """Return set of asset names (BTC/ETH) with open positions."""
        return {p.asset for p in self.open_positions.values() if p.status == "open"}

    def get_paper_balance(self) -> float:
        return self._paper_balance

    def get_live_balance(self) -> float:
        return self._live_balance

    def set_paper_balance(self, balance: float) -> None:
        self._paper_balance = balance
        self.config.paper_balance = balance

    def set_live_balance(self, balance: float) -> None:
        self._live_balance = balance
        self.config.live_balance = balance
