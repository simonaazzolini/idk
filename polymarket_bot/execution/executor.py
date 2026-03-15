"""
Order Execution Engine — Module 10.
Paper simulation and live order placement with iceberg support.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings
from core.client import PolymarketClient
from data.database import Database
from execution.safety import SafetyChecker
from strategies.arbitrage import ArbOpportunity
from strategies.signals import TradeSignal
from utils.helpers import now_ts

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    order_id: Optional[str]
    fill_price: float
    fill_size: float
    slippage: float
    mode: str
    trade_id: int = 0
    reason: str = ""
    is_paper: bool = True


class OrderExecutor:
    """
    Handles all order placement: paper simulation and live execution.
    """

    def __init__(
        self,
        client: PolymarketClient,
        db: Database,
        settings: Settings,
        safety: SafetyChecker,
    ):
        self.client = client
        self.db = db
        self.settings = settings
        self.safety = safety
        self._open_order_ids: dict[str, float] = {}  # order_id → placed_at

    async def execute_signal(
        self,
        signal: TradeSignal,
        size_usdc: float,
        mode: str,
        portfolio_state: dict,
        market: dict,
        token_id: str,
        orderbook: Optional[dict] = None,
    ) -> ExecutionResult:
        """
        Execute a trade signal. In PAPER mode, simulates fill.
        In LIVE mode, places real orders.
        """
        # Pre-trade safety check
        usdc_balance = portfolio_state.get("cash_balance", size_usdc)
        current_price = signal.current_price

        safety_result = await self.safety.check_all(
            mode=mode,
            market=market,
            size_usdc=size_usdc,
            current_price=current_price,
            signal_price=signal.current_price,
            portfolio_state=portfolio_state,
            orderbook=orderbook,
            usdc_balance=usdc_balance,
        )

        if not safety_result.passed:
            logger.info("Safety check failed for %s: %s", signal.market_slug, safety_result.reason)
            return ExecutionResult(
                success=False,
                order_id=None,
                fill_price=0,
                fill_size=0,
                slippage=0,
                mode=mode,
                reason=safety_result.reason,
                is_paper=(mode == "PAPER"),
            )

        # Freshness check: signal must be < 5 min old
        signal_age = now_ts() - signal.computed_at
        if signal_age > 300:
            return ExecutionResult(
                False, None, 0, 0, 0, mode,
                reason=f"Signal stale ({signal_age:.0f}s old)",
                is_paper=(mode == "PAPER"),
            )

        outcome = signal.final_direction  # "YES" or "NO"
        if outcome not in ("YES", "NO"):
            return ExecutionResult(
                False, None, 0, 0, 0, mode,
                reason="Invalid direction",
                is_paper=(mode == "PAPER"),
            )

        # Determine limit price based on signal strength
        if signal.action == "STRONG_BUY":
            limit_price = min(0.99, current_price * 1.005)  # aggressive: +0.5%
        else:
            limit_price = min(0.99, current_price * 1.002)  # passive: +0.2%

        # For large orders: use iceberg
        if size_usdc > self.settings.iceberg_tranche_size * 2:
            return await self._execute_iceberg(
                token_id=token_id,
                outcome=outcome,
                total_size=size_usdc,
                limit_price=limit_price,
                mode=mode,
                signal=signal,
                portfolio_state=portfolio_state,
            )

        return await self._place_single_order(
            token_id=token_id,
            outcome=outcome,
            size_usdc=size_usdc,
            limit_price=limit_price,
            mode=mode,
            signal=signal,
        )

    async def _place_single_order(
        self,
        token_id: str,
        outcome: str,
        size_usdc: float,
        limit_price: float,
        mode: str,
        signal: TradeSignal,
        order_type: str = "GTC",
    ) -> ExecutionResult:
        """Place a single limit order."""

        if mode == "PAPER":
            return await self._simulate_paper_fill(
                token_id=token_id,
                outcome=outcome,
                size_usdc=size_usdc,
                limit_price=limit_price,
                signal=signal,
            )

        # Live mode
        resp = await self.client.clob.create_limit_order(
            token_id=token_id,
            side="BUY",
            price=limit_price,
            size=size_usdc / limit_price,  # shares = USDC / price
            order_type=order_type,
        )

        if not resp:
            return ExecutionResult(
                False, None, 0, 0, 0, mode,
                reason="Order placement returned None",
                is_paper=False,
            )

        order_id = str(resp.get("id") or resp.get("orderID") or resp.get("order_id") or "")
        fill_price = float(resp.get("price") or limit_price)
        fill_size = float(resp.get("size") or (size_usdc / fill_price))
        slippage = abs(fill_price - signal.current_price) / max(signal.current_price, 0.001)

        self.safety.record_order()

        if order_id:
            self._open_order_ids[order_id] = now_ts()

        # Log trade to DB
        trade_id = await self._log_trade(
            mode=mode,
            signal=signal,
            outcome=outcome,
            size_usdc=size_usdc,
            fill_price=fill_price,
            slippage=slippage,
            order_id=order_id,
            order_type=order_type,
            shares=fill_size,
        )

        logger.info(
            "[%s] ORDER: %s %s @ %.4f size=$%.2f order_id=%s",
            mode, outcome, signal.market_slug[:25], fill_price, size_usdc, order_id[:8] if order_id else "N/A"
        )

        return ExecutionResult(
            success=True,
            order_id=order_id,
            fill_price=fill_price,
            fill_size=fill_size,
            slippage=slippage,
            mode=mode,
            trade_id=trade_id,
            reason="Order placed successfully",
            is_paper=False,
        )

    async def _simulate_paper_fill(
        self,
        token_id: str,
        outcome: str,
        size_usdc: float,
        limit_price: float,
        signal: TradeSignal,
    ) -> ExecutionResult:
        """Simulate a paper trade fill at mid + 0.3% slippage."""
        print(f"EXECUTOR REACHED _simulate_paper_fill slug={signal.market_slug[:30]} outcome={outcome} size=${size_usdc:.2f} price={signal.current_price:.4f}")
        fill_price = min(0.99, signal.current_price * (1 + self.settings.paper_slippage_pct))
        slippage = fill_price - signal.current_price
        shares = size_usdc / fill_price if fill_price > 0 else 0

        import uuid
        paper_order_id = f"PAPER_{uuid.uuid4().hex[:8]}"

        trade_id = await self._log_trade(
            mode="PAPER",
            signal=signal,
            outcome=outcome,
            size_usdc=size_usdc,
            fill_price=fill_price,
            slippage=slippage,
            order_id=paper_order_id,
            order_type="PAPER_LIMIT",
            shares=shares,
        )

        logger.info(
            "[PAPER] ORDER: %s %s @ %.4f (mid=%.4f, slip=%.4f) size=$%.2f",
            outcome, signal.market_slug[:25], fill_price,
            signal.current_price, slippage, size_usdc
        )

        return ExecutionResult(
            success=True,
            order_id=paper_order_id,
            fill_price=fill_price,
            fill_size=shares,
            slippage=slippage,
            mode="PAPER",
            trade_id=trade_id,
            reason="Paper fill simulated",
            is_paper=True,
        )

    async def _execute_iceberg(
        self,
        token_id: str,
        outcome: str,
        total_size: float,
        limit_price: float,
        mode: str,
        signal: TradeSignal,
        portfolio_state: dict,
    ) -> ExecutionResult:
        """Execute a large order in tranches to minimize market impact."""
        tranche_size = self.settings.iceberg_tranche_size
        n_tranches = max(2, int(total_size / tranche_size) + 1)
        actual_tranche = total_size / n_tranches

        logger.info(
            "Iceberg: %d tranches of $%.2f for total $%.2f",
            n_tranches, actual_tranche, total_size
        )

        results = []
        total_filled = 0.0
        initial_price = signal.current_price

        for i in range(n_tranches):
            # Re-check price before each tranche
            if i > 0:
                await asyncio.sleep(self.settings.iceberg_delay_seconds)
                # Note: in production we'd re-fetch current price here
                # For now use signal price + drift estimate

            # Check if price drifted adversely
            current_est_price = signal.current_price  # would be updated from WS
            if abs(current_est_price - initial_price) / max(initial_price, 0.001) > 0.02:
                logger.warning(
                    "Iceberg: price drifted >2%%, cancelling remaining %d tranches",
                    n_tranches - i
                )
                break

            result = await self._place_single_order(
                token_id=token_id,
                outcome=outcome,
                size_usdc=actual_tranche,
                limit_price=limit_price,
                mode=mode,
                signal=signal,
            )
            results.append(result)
            if result.success:
                total_filled += actual_tranche

        if not results or not any(r.success for r in results):
            return ExecutionResult(
                False, None, 0, 0, 0, mode,
                reason="All iceberg tranches failed",
                is_paper=(mode == "PAPER"),
            )

        # Aggregate results
        successful = [r for r in results if r.success]
        avg_fill = sum(r.fill_price * (actual_tranche / total_filled) for r in successful) if total_filled > 0 else limit_price
        avg_slippage = sum(r.slippage for r in successful) / len(successful) if successful else 0

        return ExecutionResult(
            success=True,
            order_id=f"ICEBERG_{len(successful)}_tranches",
            fill_price=avg_fill,
            fill_size=total_filled / avg_fill if avg_fill > 0 else 0,
            slippage=avg_slippage,
            mode=mode,
            trade_id=successful[0].trade_id if successful else 0,
            reason=f"Iceberg: {len(successful)}/{n_tranches} tranches filled",
            is_paper=(mode == "PAPER"),
        )

    async def execute_arb(
        self,
        arb: ArbOpportunity,
        mode: str,
        portfolio_state: dict,
    ) -> ExecutionResult:
        """
        Execute an arbitrage opportunity (FOK market orders for Type 1/2).
        """
        logger.info(
            "ENTERING EXECUTOR: execute_arb type=%s market=%s profit=%.4f (%.2f%%) legs=%d mode=%s",
            arb.arb_type, arb.market_slug[:30], arb.profit_pct, arb.profit_pct * 100,
            len(arb.legs), mode,
        )
        if not arb.legs:
            return ExecutionResult(False, None, 0, 0, 0, mode, reason="No legs defined", is_paper=(mode == "PAPER"))

        if mode == "PAPER":
            # Simulate arb fill
            fill_price = arb.legs[0].get("price", 0.5) if arb.legs else 0.5
            # Weighted avg entry price across all legs
            total_size = sum(leg.get("size", 0) for leg in arb.legs) or 1.0
            avg_price = sum(leg.get("price", 0.5) * leg.get("size", 0) for leg in arb.legs) / total_size

            await self.db.insert_arb_trade({
                "timestamp": now_ts(),
                "arb_type": arb.arb_type,
                "market_slug": arb.market_slug,
                "legs_json": json.dumps(arb.legs),
                "profit_usdc": arb.profit_usdc,
                "profit_pct": arb.profit_pct,
                "executed": 1,
                "execution_time_ms": 0,
            })

            # Also write to main trades table so arb trades appear on the
            # Trades tab and contribute to portfolio P&L tracking.
            # status="FILLED" because arb profit is locked in at entry (instant).
            import uuid as _uuid
            trade_id = await self.db.insert_trade({
                "timestamp": now_ts(),
                "mode": "PAPER",
                "market_slug": arb.market_slug,
                "question": arb.description,
                "category": "ARB",
                "outcome": "ARB",
                "side": "BUY",
                "price": avg_price,
                "size_usdc": arb.max_size,
                "shares": arb.max_size / max(avg_price, 0.001),
                "order_id": f"ARB_{_uuid.uuid4().hex[:8]}",
                "order_type": "ARB",
                "fill_price": avg_price,
                "slippage": 0.001,
                "status": "FILLED",
                "pnl": arb.profit_usdc,
                "hold_hours": 0.0,
                "exit_reason": "Arb locked-in profit",
                "composite_score": 10.0,
                "ai_probability": 1.0,
                "ai_edge": arb.profit_pct,
                "ai_confidence": 1.0,
                "whale_score": 0.0,
                "news_score": 0.0,
                "technical_score": 0.0,
                "arb_score": 10.0,
                "signal_strength": "VERY_STRONG",
                "is_whale_copy": 0,
                "source_wallet": "ARB",
            })

            logger.info(
                "[PAPER] ARB: %s %s profit=$%.2f (%.2f%%) trade_id=%d",
                arb.arb_type, arb.market_slug[:25], arb.profit_usdc, arb.profit_pct * 100, trade_id,
            )
            return ExecutionResult(
                True, f"ARB_{arb.arb_type}", avg_price, arb.max_size, 0.001, mode,
                trade_id=trade_id,
                reason=f"Arb executed: {arb.description}",
                is_paper=True,
            )

        # Live arb execution
        start_ts = now_ts()
        tasks = []
        for leg in arb.legs:
            if leg.get("side") == "BUY":
                task = self.client.clob.create_limit_order(
                    token_id=leg["token_id"],
                    side="BUY",
                    price=leg["price"] * 1.001,  # slight aggression for fill speed
                    size=leg.get("size", 0) / max(leg["price"], 0.001),
                    order_type="FOK",
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_ms = (now_ts() - start_ts) * 1000

        success_count = sum(1 for r in results if isinstance(r, dict) and r)
        executed_ok = success_count == len(tasks)
        fill_price_live = arb.legs[0].get("price", 0) if arb.legs else 0

        await self.db.insert_arb_trade({
            "timestamp": now_ts(),
            "arb_type": arb.arb_type,
            "market_slug": arb.market_slug,
            "legs_json": json.dumps(arb.legs),
            "profit_usdc": arb.profit_usdc,
            "profit_pct": arb.profit_pct,
            "executed": int(executed_ok),
            "execution_time_ms": execution_ms,
        })

        # Also write to main trades table for dashboard visibility
        import uuid as _uuid
        live_total_size = sum(leg.get("size", 0) for leg in arb.legs) or 1.0
        live_avg_price = sum(leg.get("price", 0.5) * leg.get("size", 0) for leg in arb.legs) / live_total_size
        live_trade_id = await self.db.insert_trade({
            "timestamp": now_ts(),
            "mode": "LIVE",
            "market_slug": arb.market_slug,
            "question": arb.description,
            "category": "ARB",
            "outcome": "ARB",
            "side": "BUY",
            "price": live_avg_price,
            "size_usdc": arb.max_size,
            "shares": arb.max_size / max(live_avg_price, 0.001),
            "order_id": f"ARB_{_uuid.uuid4().hex[:8]}",
            "order_type": "ARB",
            "fill_price": live_avg_price,
            "slippage": 0.001,
            "status": "FILLED" if executed_ok else "FAILED",
            "pnl": arb.profit_usdc if executed_ok else 0.0,
            "hold_hours": 0.0,
            "exit_reason": f"Arb {success_count}/{len(tasks)} legs filled",
            "composite_score": 10.0,
            "ai_probability": 1.0,
            "ai_edge": arb.profit_pct,
            "ai_confidence": 1.0,
            "whale_score": 0.0,
            "news_score": 0.0,
            "technical_score": 0.0,
            "arb_score": 10.0,
            "signal_strength": "VERY_STRONG",
            "is_whale_copy": 0,
            "source_wallet": "ARB",
        })

        return ExecutionResult(
            success=executed_ok,
            order_id=f"ARB_{arb.arb_type}",
            fill_price=fill_price_live,
            fill_size=arb.max_size,
            slippage=0.001,
            mode=mode,
            trade_id=live_trade_id,
            reason=f"Arb {success_count}/{len(tasks)} legs filled",
            is_paper=False,
        )

    async def close_position(
        self,
        trade_id: int,
        mode: str,
        token_id: str,
        shares: float,
        current_price: float,
        exit_reason: str,
        entry_price: float,
    ) -> ExecutionResult:
        """
        Close an open position by selling shares.
        """
        sell_price = current_price * 0.998  # small discount for quick fill

        if mode == "PAPER":
            # Simulate sell at current price - slippage
            actual_sell = current_price * (1 - self.settings.paper_slippage_pct)
            pnl = (actual_sell - entry_price) * shares
            await self.db.update_trade(trade_id, mode, {
                "status": "CLOSED",
                "exit_reason": exit_reason,
                "fill_price": actual_sell,
                "pnl": pnl,
                "hold_hours": 0,  # Would compute from opened_at
            })
            logger.info(
                "[PAPER] CLOSE: trade_id=%d @ %.4f (entry=%.4f, pnl=$%.2f, reason=%s)",
                trade_id, actual_sell, entry_price, pnl, exit_reason
            )
            return ExecutionResult(
                True, None, actual_sell, shares, 0.003, mode,
                trade_id=trade_id, reason=exit_reason, is_paper=True
            )

        # Live: place limit sell order
        resp = await self.client.clob.create_limit_order(
            token_id=token_id,
            side="SELL",
            price=sell_price,
            size=shares,
            order_type="GTC",
        )
        if not resp:
            return ExecutionResult(False, None, 0, 0, 0, mode, reason="Sell order failed", is_paper=False)

        order_id = str(resp.get("id") or resp.get("orderID") or "")
        fill_price = float(resp.get("price") or sell_price)
        pnl = (fill_price - entry_price) * shares

        await self.db.update_trade(trade_id, mode, {
            "status": "CLOSED",
            "exit_reason": exit_reason,
            "fill_price": fill_price,
            "pnl": pnl,
        })

        return ExecutionResult(
            True, order_id, fill_price, shares,
            abs(fill_price - current_price) / max(current_price, 0.001),
            mode, trade_id=trade_id, reason=exit_reason, is_paper=False
        )

    async def manage_open_orders(self, mode: str) -> None:
        """
        Cancel and repost GTC orders that haven't filled in 30 minutes.
        """
        if mode == "PAPER":
            return  # Nothing to manage in paper mode

        now = now_ts()
        stale_orders = [
            oid for oid, placed_at in self._open_order_ids.items()
            if now - placed_at > self.settings.order_cancel_timeout_seconds
        ]

        for order_id in stale_orders:
            cancelled = await self.client.clob.cancel_order(order_id)
            if cancelled:
                del self._open_order_ids[order_id]
                logger.info("Cancelled stale GTC order: %s", order_id[:12])

    async def _log_trade(
        self,
        mode: str,
        signal: TradeSignal,
        outcome: str,
        size_usdc: float,
        fill_price: float,
        slippage: float,
        order_id: str,
        order_type: str,
        shares: float,
        is_whale_copy: bool = False,
        source_wallet: Optional[str] = None,
    ) -> int:
        """Log a trade to the database."""
        trade_record = {
            "timestamp": now_ts(),
            "mode": mode,
            "market_slug": signal.market_slug,
            "question": signal.question,
            "category": signal.category,
            "outcome": outcome,
            "side": "BUY",
            "price": signal.current_price,
            "size_usdc": size_usdc,
            "shares": shares,
            "order_id": order_id,
            "order_type": order_type,
            "fill_price": fill_price,
            "slippage": slippage,
            "status": "OPEN",
            "pnl": None,
            "hold_hours": None,
            "exit_reason": None,
            "composite_score": signal.composite_score,
            "ai_probability": signal.ai_probability,
            "ai_edge": signal.ai_edge,
            "ai_confidence": signal.ai_confidence,
            "whale_score": signal.whale_score,
            "news_score": signal.news_score,
            "technical_score": signal.technical_score,
            "arb_score": signal.arb_score,
            "signal_strength": signal.signal_strength,
            "is_whale_copy": int(is_whale_copy),
            "source_wallet": source_wallet,
        }
        return await self.db.insert_trade(trade_record)
