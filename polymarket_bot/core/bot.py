"""
Main Bot Orchestrator — Module 14.
Coordinates all modules in the correct sequence.
Market discovery → Orderbooks → Technicals → Whales → News →
AI → Arb → Signals → Sizing → Execution → Portfolio → Logging → Dashboard.
"""
import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from analysis.ai_analyzer import AIAnalyzer
from analysis.news_analyzer import NewsAnalyzer
from analysis.orderbook import analyze_orderbook, analyze_cross_token
from analysis.technical import compute_technical_signals
from config.settings import Settings
from core.client import PolymarketClient
from core.portfolio import PortfolioManager
from data.cache import MarketDataCache, cache
from data.database import Database
from execution.executor import OrderExecutor
from execution.safety import SafetyChecker
from strategies.arbitrage import ArbitrageDetector
from strategies.kelly import kelly_size
from strategies.signals import SignalAggregator, TradeSignal
from tracking.copy_trader import CopyTrader
from tracking.graph import WalletGraph
from tracking.whale_tracker import WhaleTracker, WhaleAlert
from ui.dashboard import Dashboard
from utils.helpers import (
    days_until, extract_yes_no_prices, extract_yes_no_token_ids,
    get_market_category, now_ts, safe_div, safe_log10,
)

logger = logging.getLogger(__name__)

# Opportunity score weights (from spec)
OPP_SCORE_WEIGHTS = {
    "price_inefficiency": 0.30,
    "volume_score": 0.25,
    "liquidity_score": 0.20,
    "time_score": 0.15,
    "news_velocity": 0.10,
}


class PolymarketBot:
    """
    Main trading bot orchestrator.
    """

    def __init__(
        self,
        client: PolymarketClient,
        db: Database,
        settings: Settings,
        mode: str,
        budget: float,
    ):
        self.client = client
        self.db = db
        self.settings = settings
        self.mode = mode
        self.budget = budget

        # Initialize all modules
        self.cache = cache
        self.portfolio = PortfolioManager(db, settings, mode, budget)
        self.whale_tracker = WhaleTracker(client, db, settings)
        self.wallet_graph = WalletGraph(db)
        self.copy_trader = CopyTrader(db, settings)
        self.news_analyzer = NewsAnalyzer(settings, self.cache)
        self.ai_analyzer = AIAnalyzer(settings, self.cache)
        self.arb_detector = ArbitrageDetector(settings)
        self.signal_aggregator = SignalAggregator(settings)
        self.safety_checker = SafetyChecker(settings)
        self.executor = OrderExecutor(client, db, settings, self.safety_checker)
        self.dashboard = Dashboard(settings)

        # State
        self._running = False
        self._current_markets: list[dict] = []
        self._current_orderbooks: dict[str, dict] = {}
        self._technical_signals: dict[str, object] = {}
        self._whale_consensuses: dict[str, dict] = {}
        self._news_results: dict[str, dict] = {}
        self._ai_results: dict[str, object] = {}
        self._arb_opportunities: list = []
        self._last_full_cycle_ts: float = 0.0
        self._cycle_count: int = 0
        self._api_calls_today: int = 0
        self._paper_start_ts: float = 0.0
        self._paper_evaluated_at: float = 0.0

    async def startup(self) -> None:
        """Initialize everything and prepare for trading."""
        logger.info("Bot starting up in %s mode with budget=$%.2f", self.mode, self.budget)

        # Initialize portfolio
        await self.portfolio.initialize()

        # Load/set paper start time
        if self.mode == "PAPER":
            stored_start = await self.db.get_state_float("paper_start_ts")
            if stored_start:
                self._paper_start_ts = stored_start
            else:
                self._paper_start_ts = now_ts()
                await self.db.set_state("paper_start_ts", str(self._paper_start_ts))

        # Start WebSocket
        await self.client.start_websocket()
        if self.client.ws:
            self.client.ws.add_orderbook_callback(self._on_ws_orderbook_update)

        # Start dashboard
        self.dashboard.mode = self.mode
        self.dashboard.budget = self.budget
        self.dashboard.paper_start_ts = self._paper_start_ts
        self.dashboard.paper_end_ts = self._paper_start_ts + 48 * 3600
        self.dashboard.portfolio_state = self.portfolio.state
        self.dashboard.start()

        # Initial whale database build (background)
        if await self.whale_tracker.should_do_full_refresh():
            asyncio.create_task(self.whale_tracker.full_refresh(), name="whale_refresh")
            logger.info("Whale database refresh started in background")

    async def run(self) -> None:
        """Main bot loop."""
        self._running = True
        await self.startup()

        # Launch fast loop as background task
        fast_loop_task = asyncio.create_task(self._fast_loop(), name="fast_loop")

        try:
            while self._running:
                await self._check_db_commands()
                if not self._running:
                    break
                cycle_start = now_ts()
                try:
                    await self._main_cycle()
                except Exception as e:
                    logger.error("Main cycle error: %s", e, exc_info=True)

                self._last_full_cycle_ts = now_ts()
                self._cycle_count += 1
                self.dashboard.last_cycle_ts = self._last_full_cycle_ts
                self.dashboard.update()

                # Wait until next cycle
                elapsed = now_ts() - cycle_start
                sleep_time = max(0.0, self.settings.cycle_minutes * 60 - elapsed)
                logger.info(
                    "Cycle %d complete in %.1fs. Next in %.0fs.",
                    self._cycle_count, elapsed, sleep_time
                )
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Bot run loop cancelled")
        finally:
            fast_loop_task.cancel()
            try:
                await fast_loop_task
            except asyncio.CancelledError:
                pass
            await self._shutdown()

    async def _main_cycle(self) -> None:
        """
        Execute one full analysis cycle (every ~15 minutes).
        """
        logger.info("=== Main cycle %d starting ===", self._cycle_count + 1)

        # Step 1: Market discovery
        markets = await self._discover_markets()
        if not markets:
            logger.warning("No markets found, skipping cycle")
            return
        self._current_markets = markets
        self.dashboard.markets_tracked = len(markets)

        # Check paper/live timer
        await self._check_paper_timer()

        # Stop if trading halted
        can_trade, halt_reason = self.portfolio.can_trade()

        # Step 2: Fetch orderbooks for all markets
        await self._fetch_orderbooks(markets)

        # Step 3: Compute technical indicators
        await self._compute_technicals(markets)

        # Step 4: Check whale positions
        await self.whale_tracker.refresh_positions(markets)

        # Get whale consensuses for all markets
        for market in markets[:20]:
            slug = market.get("slug") or market.get("conditionId", "")
            consensus = await self.whale_tracker.get_smart_money_consensus(slug)
            self._whale_consensuses[slug] = consensus

        # Step 5: Check whale new trades (alerts)
        market_slugs = [m.get("slug") or m.get("conditionId", "") for m in markets[:20]]
        whale_alerts = await self.whale_tracker.poll_new_trades(market_slugs)
        if whale_alerts:
            logger.info("Processing %d whale alerts", len(whale_alerts))
            for alert in whale_alerts:
                self.dashboard.recent_alerts.append(alert)
                if can_trade:
                    await self._handle_whale_alert(alert, markets)

        # Step 6: News analysis (top 20 markets)
        if can_trade:
            self._news_results = await self.news_analyzer.analyze_batch(markets[:20])

        # Step 7: AI analysis (top 20 markets)
        if can_trade:
            self._ai_results = await self.ai_analyzer.analyze_batch(
                markets=markets[:20],
                technical_signals=self._technical_signals,
                whale_consensuses=self._whale_consensuses,
                news_results=self._news_results,
            )

        # Step 8: Arb scan
        days_map = {
            (m.get("slug") or m.get("conditionId", "")): m.get("days_to_resolution", 30.0)
            for m in markets
        }
        self._arb_opportunities = self.arb_detector.scan_all(
            markets, self._current_orderbooks, days_map
        )

        # Execute immediately-actionable arb opportunities
        if can_trade:
            for arb in self._arb_opportunities[:3]:  # top 3 only
                if arb.arb_type in ("TYPE_1_YES_NO_SUM", "TYPE_2_CATEGORICAL"):
                    await self.executor.execute_arb(
                        arb, self.mode, self.portfolio.state.to_dict()
                    )

        # Step 9-11: Aggregate signals, size, execute
        if can_trade:
            open_positions = await self.db.get_open_trades(self.mode)
            self.dashboard.open_trades = open_positions

            arb_scores_map = {
                m.get("slug") or m.get("conditionId", ""): self.arb_detector.get_arb_score(
                    self._arb_opportunities, m.get("slug") or m.get("conditionId", "")
                )
                for m in markets
            }

            # Build orderbook_yes_metrics map
            ob_yes_map = {}
            for market in markets:
                slug = market.get("slug") or market.get("conditionId", "")
                tokens = market.get("clobTokenIds") or []
                if isinstance(tokens, str):
                    try:
                        tokens = json.loads(tokens)
                    except json.JSONDecodeError:
                        tokens = []
                yes_token = str(tokens[0]) if tokens else ""
                raw_book = self._current_orderbooks.get(yes_token)
                if raw_book:
                    ob_yes_map[slug] = analyze_orderbook(raw_book)

            signals = self.signal_aggregator.aggregate_batch(
                markets=markets[:20],
                ai_results=self._ai_results,
                whale_consensuses=self._whale_consensuses,
                news_results=self._news_results,
                technical_signals=self._technical_signals,
                orderbook_yes_metrics=ob_yes_map,
                arb_scores=arb_scores_map,
                open_positions=open_positions,
            )

            # Update dashboard
            self.dashboard.recent_signals = signals
            self.dashboard.portfolio_state = self.portfolio.state

            # Execute actionable signals
            for signal in signals:
                if signal.action not in ("STRONG_BUY", "BUY", "WEAK_BUY"):
                    continue
                await self._execute_signal(signal, markets)

        # Step 12: Log daily performance
        await self._log_daily_performance()

        # Step 13: Update open position marks
        await self._update_position_prices(markets)

    async def _discover_markets(self) -> list[dict]:
        """Module 1: Fetch and score all active markets."""
        # Check cache
        cached = await self.db.get_cached_markets(self.settings.market_cache_ttl_minutes)
        if cached:
            logger.debug("Using cached market list (%d markets)", len(cached))
            return self._filter_and_score_markets(cached)

        try:
            raw_markets = await self.client.gamma.get_all_active_markets()
        except Exception as e:
            logger.error("Market discovery failed: %s", e)
            return self._current_markets  # use last known

        # Filter and score
        scored = self._filter_and_score_markets(raw_markets)

        # Cache to DB
        await self.db.cache_markets(scored)

        logger.info("Discovered %d scored markets from %d total", len(scored), len(raw_markets))
        return scored[:self.settings.top_markets_count]

    def _filter_and_score_markets(self, raw_markets: list[dict]) -> list[dict]:
        """Apply filters and compute opportunity scores."""
        filtered = []
        now = now_ts()

        for market in raw_markets:
            # Apply filters
            if not market.get("enableOrderBook", True):
                continue
            if market.get("closed") or not market.get("active", True):
                continue

            liquidity = float(market.get("liquidity") or 0)
            volume_24h = float(market.get("volume24hr") or market.get("volume_24h") or 0)
            if liquidity < self.settings.min_liquidity:
                continue
            if volume_24h < self.settings.min_volume:
                continue

            end_date = market.get("endDate") or market.get("end_date") or ""
            days = days_until(str(end_date)) if end_date else 30.0
            if days < self.settings.min_days_to_resolution:
                continue
            if days > self.settings.max_days_to_resolution:
                continue

            # Compute opportunity score
            yes_price, no_price = extract_yes_no_prices(market)
            price_inefficiency = abs(yes_price + no_price - 1.0)
            volume_score = safe_log10(max(volume_24h, 1)) / 6.0
            liquidity_score = safe_log10(max(liquidity, 1)) / 7.0
            time_score = 1.0 - (days / 180.0)
            # news_velocity would come from news cache; use 0 initially
            news_velocity = float(self._news_results.get(
                market.get("slug", ""), {}
            ).get("news_velocity_score", 0.0)) / 10.0 if self._news_results else 0.0

            opportunity_score = (
                price_inefficiency * OPP_SCORE_WEIGHTS["price_inefficiency"]
                + volume_score * OPP_SCORE_WEIGHTS["volume_score"]
                + liquidity_score * OPP_SCORE_WEIGHTS["liquidity_score"]
                + time_score * OPP_SCORE_WEIGHTS["time_score"]
                + news_velocity * OPP_SCORE_WEIGHTS["news_velocity"]
            )

            market["yes_price"] = yes_price
            market["no_price"] = no_price
            market["days_to_resolution"] = days
            market["opportunity_score"] = opportunity_score
            market["category"] = get_market_category(market)
            filtered.append(market)

        # Sort by opportunity score descending
        filtered.sort(key=lambda m: m["opportunity_score"], reverse=True)
        return filtered

    async def _fetch_orderbooks(self, markets: list[dict]) -> None:
        """Module 2: Fetch orderbooks for all candidate markets."""
        token_ids = []
        for market in markets[:50]:
            yes_token, no_token = extract_yes_no_token_ids(market)
            if yes_token:
                token_ids.append(yes_token)
            if no_token:
                token_ids.append(no_token)

        if not token_ids:
            return

        try:
            books = await self.client.clob.get_order_books_batch(token_ids)
            self._current_orderbooks.update(books)
            logger.debug("Fetched orderbooks for %d tokens", len(books))
        except Exception as e:
            logger.warning("Orderbook batch fetch failed: %s", e)

        # Subscribe new tokens to WebSocket
        if self.client.ws:
            await self.client.ws.subscribe(token_ids)

    async def _compute_technicals(self, markets: list[dict]) -> None:
        """Module 3: Compute technical indicators for all markets."""
        for market in markets[:30]:
            slug = market.get("slug") or market.get("conditionId", "")
            yes_token, _ = extract_yes_no_token_ids(market)
            if not yes_token:
                continue

            try:
                # Check cache
                cached = await self.cache.get_price_history(yes_token, "1h")
                if cached is None:
                    history = await self.client.clob.get_prices_history(yes_token, "1h", 60)
                    await self.cache.set_price_history(yes_token, "1h", history)
                else:
                    history = cached

                if not history:
                    continue

                yes_price = float(market.get("yes_price") or 0.5)
                volume_24h = float(market.get("volume24hr") or market.get("volume_24h") or 0)
                days = float(market.get("days_to_resolution") or 30.0)

                tech = compute_technical_signals(history, yes_price, volume_24h, days)
                self._technical_signals[slug] = tech

            except Exception as e:
                logger.debug("Technical analysis failed for %s: %s", slug, e)

    async def _handle_whale_alert(self, alert: WhaleAlert, markets: list[dict]) -> None:
        """Process a whale alert and potentially copy trade."""
        # Find the relevant market
        market = next(
            (m for m in markets if m.get("slug") == alert.market_slug
             or m.get("conditionId") == alert.market_slug),
            None
        )
        if not market:
            return

        # Quick AI analysis for copy trade evaluation
        ai_quick = await self.ai_analyzer.run_quick_analysis(market)

        # Get current market data for copy trade evaluation
        yes_token, _ = extract_yes_no_token_ids(market)
        ob_raw = self._current_orderbooks.get(yes_token or "")
        ob_metrics = analyze_orderbook(ob_raw) if ob_raw else None
        current_market_data = {
            "yes_price": market.get("yes_price", 0.5),
            "mid_price": ob_metrics.mid_price if ob_metrics else market.get("yes_price", 0.5),
            "liquidity": market.get("liquidity", 0),
        }

        decision = await self.copy_trader.evaluate_alert(
            alert=alert,
            portfolio_state=self.portfolio.state.to_dict(),
            current_market_data=current_market_data,
            ai_analysis=ai_quick,
        )

        if not decision.should_execute:
            logger.debug("Copy trade declined for %s: %s", alert.market_slug, decision.reason)
            return

        # Create a minimal signal for execution
        signal = TradeSignal(
            market_slug=alert.market_slug,
            question=market.get("question", ""),
            category=get_market_category(market),
            current_price=decision.entry_price,
            final_direction=decision.outcome,
            action="BUY",
            composite_score=7.0,
            ai_score=5.0,
            whale_score=8.0 if "INSIDER_ALERT" in decision.source_tier else 6.0,
        )

        result = await self.executor.execute_signal(
            signal=signal,
            size_usdc=decision.size_usdc,
            mode=self.mode,
            portfolio_state=self.portfolio.state.to_dict(),
            market=market,
            token_id=yes_token or "",
        )

        if result.success:
            await self.portfolio.on_trade_opened(
                result.trade_id, decision.size_usdc, result.fill_price
            )
            logger.info(
                "COPY TRADE EXECUTED: %s %s $%.2f @ %.4f",
                decision.outcome, alert.market_slug[:25],
                decision.size_usdc, result.fill_price
            )

    async def _execute_signal(self, signal: TradeSignal, markets: list[dict]) -> None:
        """Execute a trade signal after Kelly sizing."""
        market = next(
            (m for m in markets if m.get("slug") == signal.market_slug
             or m.get("conditionId") == signal.market_slug),
            None
        )
        if not market:
            return

        # Kelly sizing
        kr = kelly_size(
            estimated_probability=signal.ai_probability,
            current_price=signal.current_price,
            confidence=signal.ai_confidence,
            composite_score=signal.composite_score,
            available_bankroll=self.portfolio.state.cash_balance,
            market_liquidity=float(market.get("liquidity") or 0),
            current_exposure=self.portfolio.state.current_exposure,
            total_budget=self.portfolio.state.total_budget,
            settings=self.settings,
            is_weak_signal=(signal.action == "WEAK_BUY"),
        )

        if not kr.is_viable:
            logger.debug(
                "Kelly says skip %s: size=$%.2f < min, capped_by=%s",
                signal.market_slug[:25], kr.final_size, kr.capped_by
            )
            return

        yes_token, no_token = extract_yes_no_token_ids(market)
        token_id = yes_token if signal.final_direction == "YES" else no_token
        if not token_id:
            logger.warning("No token ID for %s direction=%s", signal.market_slug, signal.final_direction)
            return

        ob_raw = self._current_orderbooks.get(token_id)
        result = await self.executor.execute_signal(
            signal=signal,
            size_usdc=kr.final_size,
            mode=self.mode,
            portfolio_state=self.portfolio.state.to_dict(),
            market=market,
            token_id=token_id,
            orderbook=ob_raw,
        )

        if result.success:
            await self.portfolio.on_trade_opened(
                result.trade_id, kr.final_size, result.fill_price
            )

        # Log signal regardless
        await self.db.insert_signal({
            "timestamp": now_ts(),
            **signal.to_dict(),
            "action_taken": signal.action if result.success else "SKIPPED",
            "reason_skipped": None if result.success else result.reason,
        })

    async def _update_position_prices(self, markets: list[dict]) -> None:
        """Update open position mark-to-market prices."""
        price_map: dict[str, float] = {}
        for market in markets:
            slug = market.get("slug") or market.get("conditionId", "")
            yes_price = float(market.get("yes_price") or 0.5)
            price_map[slug] = yes_price
            price_map[f"{slug}_YES"] = yes_price
            price_map[f"{slug}_NO"] = 1.0 - yes_price

        positions_to_exit = await self.portfolio.update_mark_to_market(price_map)

        # Execute stop-losses and take-profits
        for pos in positions_to_exit:
            await self.executor.close_position(
                trade_id=pos["trade_id"],
                mode=self.mode,
                token_id=pos.get("token_id", ""),
                shares=pos["shares"],
                current_price=pos["current_price"],
                exit_reason=pos["exit_reason"],
                entry_price=pos["entry_price"],
            )
            pnl = (pos["current_price"] - pos["entry_price"]) * pos["shares"]
            await self.portfolio.on_trade_closed(pos["trade_id"], pnl, pos["size_usdc"])

    async def _log_daily_performance(self) -> None:
        """Snapshot portfolio state and update daily metrics."""
        now = now_ts()
        # Check if we need to save daily snapshot (every 5 min from portfolio)
        await self.portfolio._save_snapshot()

        # Update dashboard state
        self.dashboard.portfolio_state = self.portfolio.state
        self.dashboard.api_calls_today = self._api_calls_today

        # DB size
        self.dashboard.db_size_mb = await self.db.get_db_size_mb()

    # ── Fast loop ─────────────────────────────────────────────────────────────

    async def _fast_loop(self) -> None:
        """
        Fast loop runs every 5 minutes between main cycles.
        Handles: WS health, whale trade poll, stop-loss checks, fast arb, timer check.
        """
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 minutes

                await self._check_db_commands()
                if not self._running:
                    break

                # a. WebSocket health check
                if self.client.ws:
                    self.dashboard.ws_connected = self.client.ws.is_connected
                    if self.client.ws.last_message_age_seconds > 120:
                        logger.warning("WS silent for >2min, checking connection...")

                # b. Whale trade poll
                if self._current_markets:
                    market_slugs = [
                        m.get("slug") or m.get("conditionId", "")
                        for m in self._current_markets[:20]
                    ]
                    alerts = await self.whale_tracker.poll_new_trades(market_slugs)
                    can_trade, _ = self.portfolio.can_trade()
                    if can_trade:
                        for alert in alerts:
                            self.dashboard.recent_alerts.append(alert)
                            await self._handle_whale_alert(alert, self._current_markets)

                # c. Open position stop-loss/take-profit check
                if self._current_markets:
                    await self._update_position_prices(self._current_markets)

                # d. Fast arb scan (Type 1 and 2 only)
                if self._current_markets and self._current_orderbooks:
                    days_map = {
                        (m.get("slug") or m.get("conditionId", "")): m.get("days_to_resolution", 30.0)
                        for m in self._current_markets
                    }
                    arb_ops = self.arb_detector.scan_all(
                        self._current_markets[:20], self._current_orderbooks, days_map
                    )
                    can_trade, _ = self.portfolio.can_trade()
                    if can_trade:
                        for arb in arb_ops[:2]:
                            if arb.arb_type in ("TYPE_1_YES_NO_SUM", "TYPE_2_CATEGORICAL"):
                                await self.executor.execute_arb(
                                    arb, self.mode, self.portfolio.state.to_dict()
                                )

                # e. Paper/live timer check
                await self._check_paper_timer()

                # f. Open order management (cancel stale orders)
                await self.executor.manage_open_orders(self.mode)

                self.dashboard.update()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Fast loop error: %s", e)

    async def _on_ws_orderbook_update(self, token_id: str, book: dict) -> None:
        """Handle real-time orderbook update from WebSocket."""
        if not token_id or not book:
            return
        # Update orderbook in memory
        self._current_orderbooks[token_id] = book

        # Detect price shock
        from analysis.orderbook import detect_price_shock
        old_book = self._current_orderbooks.get(f"_prev_{token_id}")
        shock = detect_price_shock(
            token_id, old_book, book,
            self.settings.ws_price_shock_threshold_pct * 100
        )
        if shock:
            logger.warning(
                "PRICE SHOCK: token=%s change=%.1f%% direction=%s",
                token_id[:12], shock.change_pct, shock.direction
            )
        self._current_orderbooks[f"_prev_{token_id}"] = book

    # ── Paper/Live transition ─────────────────────────────────────────────────

    async def _check_paper_timer(self) -> None:
        """Check if 48h paper period has elapsed and run go/no-go."""
        if self.mode != "PAPER":
            return

        elapsed = now_ts() - self._paper_start_ts
        if elapsed < self.settings.paper_trading_hours * 3600:
            return

        # Re-evaluate every 24h after initial 48h
        since_last_eval = now_ts() - self._paper_evaluated_at
        if self._paper_evaluated_at > 0 and since_last_eval < 24 * 3600:
            return

        self._paper_evaluated_at = now_ts()
        await self._run_go_no_go_evaluation()

    async def _run_go_no_go_evaluation(self) -> None:
        """Evaluate go/no-go criteria after paper trading period."""
        days_total = (now_ts() - self._paper_start_ts) / 86400.0
        logger.info("Running GO/NO-GO evaluation after %.1f days of paper trading", days_total)

        # Gather all paper trades
        all_trades = await self.db.get_all_closed_trades("PAPER")
        pnls = [float(t.get("pnl") or 0) for t in all_trades if t.get("pnl") is not None]
        n_trades = len(all_trades)

        if n_trades < self.settings.go_min_trades:
            logger.info("GO/NO-GO: Not enough trades (%d < %d)", n_trades, self.settings.go_min_trades)

        # Compute metrics
        from utils.helpers import win_rate, profit_factor, max_drawdown
        wr = win_rate(pnls) if pnls else 0.0
        pf = profit_factor(pnls) if pnls else 0.0
        total_pnl = sum(pnls)
        roi_pct = safe_div(total_pnl, self.budget) * 100

        # Daily P&L for Sharpe
        snapshots = await self.db.get_portfolio_snapshots(since_ts=self._paper_start_ts)
        daily_values: dict[str, float] = {}
        for snap in snapshots:
            day = datetime.fromtimestamp(snap["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
            daily_values[day] = float(snap.get("total_value") or self.budget)
        daily_pnls = [
            (list(daily_values.values())[i+1] - list(daily_values.values())[i]) / self.budget
            for i in range(len(daily_values) - 1)
        ]
        from utils.helpers import sharpe_ratio
        sharpe = sharpe_ratio(daily_pnls) if len(daily_pnls) >= 2 else 0.0

        # Max drawdown
        portfolio_values = [snap.get("total_value", self.budget) for snap in snapshots]
        max_dd = max_drawdown(portfolio_values) if len(portfolio_values) > 1 else 0.0

        # Check profitable on both calendar days
        days_with_trades: dict[str, list[float]] = {}
        for t in all_trades:
            if t.get("pnl") is None:
                continue
            ts = float(t.get("timestamp") or now_ts())
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            days_with_trades.setdefault(day, []).append(float(t["pnl"]))
        profitable_days = sum(1 for day_pnls in days_with_trades.values() if sum(day_pnls) > 0)

        # Max single day loss
        max_single_day_loss_pct = max(
            (abs(sum(day_pnls)) / self.budget for day_pnls in days_with_trades.values()
             if sum(day_pnls) < 0),
            default=0.0
        )

        # Max single position loss
        max_single_pos_loss = min(
            (float(t.get("pnl") or 0) / max(float(t.get("size_usdc") or 1), 1)
             for t in all_trades if (t.get("pnl") or 0) < 0),
            default=0.0
        )

        # ── GO criteria ──────────────────────────────────────────────────────
        go_criteria = {
            "win_rate >= 55%": {
                "passed": wr >= self.settings.go_min_win_rate,
                "required": f"{self.settings.go_min_win_rate:.0%}",
                "actual": f"{wr:.1%}",
            },
            "profit_factor >= 1.3": {
                "passed": pf >= self.settings.go_min_profit_factor,
                "required": f"{self.settings.go_min_profit_factor}",
                "actual": f"{pf:.2f}",
            },
            "profitable both days": {
                "passed": profitable_days >= 2,
                "required": "2 days",
                "actual": f"{profitable_days} days",
            },
            "max_single_day_loss < 8%": {
                "passed": max_single_day_loss_pct < self.settings.go_max_single_day_loss_pct,
                "required": f"< {self.settings.go_max_single_day_loss_pct:.0%}",
                "actual": f"{max_single_day_loss_pct:.1%}",
            },
            "min 8 completed trades": {
                "passed": n_trades >= self.settings.go_min_trades,
                "required": f">= {self.settings.go_min_trades}",
                "actual": str(n_trades),
            },
            "sharpe_ratio >= 0.5": {
                "passed": sharpe >= self.settings.go_min_sharpe,
                "required": f"{self.settings.go_min_sharpe}",
                "actual": f"{sharpe:.2f}",
            },
            "no single position > 25% loss": {
                "passed": max_single_pos_loss > -self.settings.go_max_single_position_loss_pct,
                "required": "> -25%",
                "actual": f"{max_single_pos_loss:.1%}",
            },
        }

        # ── NO-GO criteria ────────────────────────────────────────────────────
        nogo_triggered = []
        if wr < 0.50:
            nogo_triggered.append(f"win_rate {wr:.1%} < 50%")
        if total_pnl < 0:
            nogo_triggered.append(f"net P&L negative: ${total_pnl:.2f}")
        if max_dd > self.settings.nogo_max_drawdown_pct:
            nogo_triggered.append(f"max drawdown {max_dd:.1%} > {self.settings.nogo_max_drawdown_pct:.0%}")
        if n_trades < self.settings.go_min_trades:
            nogo_triggered.append(f"only {n_trades} trades < {self.settings.go_min_trades} minimum")

        all_go_pass = all(v["passed"] for v in go_criteria.values())
        verdict = "GO" if all_go_pass and not nogo_triggered else "NO-GO"

        stats = {
            "total_pnl": total_pnl,
            "roi_pct": roi_pct,
            "win_rate": wr,
            "profit_factor": pf,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": n_trades,
            "whale_copy_count": sum(1 for t in all_trades if t.get("is_whale_copy")),
            "arb_count": 0,
        }

        self.dashboard.print_paper_report(stats, go_criteria, verdict)

        if verdict == "GO":
            await self._prompt_for_live_trading()
        else:
            logger.info("NO-GO: %s", ", ".join(nogo_triggered) or "GO criteria not all met")

            # After 7 days, print diagnostic
            if days_total >= 7:
                self.dashboard.print_diagnostic_report(self._generate_diagnostic())

    async def _prompt_for_live_trading(self) -> None:
        """Signal the dashboard that GO criteria are met; wait for UI confirmation.

        The standalone server polls bot_state['paper_eval_go'] and broadcasts
        the paper_evaluation WebSocket event, which auto-opens the confirmation
        modal in the dashboard.  The user types the phrase there; the dashboard
        calls POST /api/control action=activate_live which writes mode=LIVE to
        the DB.  _check_db_commands() picks that up on the next fast-loop tick.
        """
        logger.info(
            "Paper trading COMPLETE — GO criteria met! "
            "Open the dashboard and confirm live trading there."
        )
        await self.db.set_state("paper_eval_go", "1")
        # Continue paper trading until the dashboard writes mode=LIVE to DB.

    async def _check_db_commands(self) -> None:
        """Poll DB for mode changes written by the dashboard server."""
        if not self._running:
            return
        try:
            db_mode = await self.db.get_state("mode")
            if not db_mode or db_mode == self.mode:
                return

            if db_mode == "STOPPED":
                logger.info("Stop command received via dashboard")
                self._running = False

            elif db_mode == "PAUSED" and self.mode not in ("PAUSED", "STOPPED"):
                logger.info("Pause command received via dashboard")
                await self.db.set_state("prev_mode", self.mode)
                self.mode = "PAUSED"
                # Block here until the DB mode changes away from PAUSED
                while self._running:
                    await asyncio.sleep(5)
                    db_mode = await self.db.get_state("mode")
                    if db_mode == "STOPPED":
                        self._running = False
                        break
                    if db_mode != "PAUSED":
                        self.mode = db_mode
                        logger.info("Resumed, mode=%s", self.mode)
                        break

            elif db_mode == "LIVE" and self.mode in ("PAPER", "PAUSED"):
                logger.info("Live mode activated via dashboard")
                self.mode = "LIVE"
                self.portfolio.state.mode = "LIVE"
                await self.db.set_state("live_activated_at", str(now_ts()))

        except Exception as exc:
            logger.debug("_check_db_commands error: %s", exc)

    def _generate_diagnostic(self) -> dict:
        """Generate tuning suggestions after 7-day no-go."""
        return {
            "Module 6 (AI Analyzer)": [
                "Consider lowering min_edge threshold from 0.04 to 0.03",
                "Increase web search depth (current max_uses=6, try 8)",
                "Review confidence calibration - may be too conservative",
            ],
            "Module 4 (Whale Tracker)": [
                "Expand leaderboard lookback windows",
                "Lower smart_money_min_win_rate from 0.65 to 0.60 for more signals",
                "Check if wallet refresh is completing within 12h window",
            ],
            "Module 8 (Signal Aggregator)": [
                "Review signal weight distribution (current: AI=35%, Whale=25%)",
                "Consider reducing conflicting signals penalty from 30% to 20%",
                "Lower BUY composite threshold from 6.5 to 6.0",
            ],
            "Module 1 (Market Discovery)": [
                "Lower MIN_LIQUIDITY from $500 to $250",
                "Lower MIN_VOLUME from $1000 to $500",
                "Consider expanding days_to_resolution range",
            ],
        }

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Bot shutting down...")
        self._running = False

        # Cancel all open orders
        if self.mode == "LIVE":
            await self.client.clob.cancel_all_orders()
        else:
            logger.info("Paper mode: no real orders to cancel")

        # Save final state
        await self.portfolio._save_snapshot()
        await self.portfolio._save_daily_performance()
        await self.db.set_state("last_shutdown_ts", str(now_ts()))

        # Print session summary
        self.dashboard.print_session_summary(self.portfolio.state)
        self.dashboard.stop()

        logger.info("Shutdown complete")
