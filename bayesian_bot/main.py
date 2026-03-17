"""
main.py — Main loop for the Bayesian Prediction Market Trading Bot.

Runs every 60 seconds:
  1. Scan for active BTC/ETH markets
  2. Collect evidence for each candidate market
  3. Run Bayesian analysis
  4. Execute trades when posterior diverges >8% from market price
  5. Monitor open positions for exit conditions
  6. After every 20 trades, run self-calibration
"""
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add bot directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from bayesian_engine import BayesianEngine
from market_scanner import MarketScanner
from evidence_collector import EvidenceCollector
from trade_executor import TradeExecutor
from performance_tracker import PerformanceTracker
from calibrator import Calibrator
import dashboard

# ── Logging setup ──────────────────────────────────────────────────────────

def setup_logging(debug: bool = False) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                Path(__file__).parent / "bayesian_bot.log",
                mode="a", encoding="utf-8"
            )
        ]
    )
    # Quiet noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


logger = logging.getLogger("bayesian_bot")

# ── Bot class ──────────────────────────────────────────────────────────────

class BayesianBot:
    """
    Main bot orchestrator.
    Combines all modules into a continuous scanning and trading loop.
    """

    SCAN_INTERVAL = 60   # seconds between scans

    def __init__(self):
        self.config = load_config()
        self.engine = BayesianEngine(self.config)
        self.scanner = MarketScanner(self.config)
        self.collector = EvidenceCollector(self.config)
        self.tracker = PerformanceTracker(self.config.mode)
        self.executor = TradeExecutor(self.config, self.tracker)
        self.calibrator = Calibrator(self.config)
        self._running = False
        self._scan_count = 0
        self.btc_price = 0.0
        self.eth_price = 0.0
        self._monitored_markets_state = []

    def startup(self) -> None:
        """Initialize bot, test APIs, and log startup summary."""
        logger.info("=" * 60)
        logger.info("BAYESIAN MARKET BOT — Starting up")
        logger.info("=" * 60)
        logger.info("Mode: %s", self.config.mode.upper())
        logger.info("Paper balance: $%.2f", self.config.paper_starting_balance)
        logger.info("Live balance: $%.2f", self.config.live_starting_balance)
        logger.info("Divergence threshold: %.1f%%", self.config.min_divergence_threshold * 100)
        logger.info("Max positions: %d", self.config.max_open_positions)
        logger.info("Max trade size: $%.2f", self.config.max_trade_size_usd)
        logger.info("Evidence weights: %s", json.dumps(self.config.evidence_weights, indent=2))

        # Fetch initial prices
        self._update_prices()
        logger.info("BTC price: $%.2f", self.btc_price)
        logger.info("ETH price: $%.2f", self.eth_price)

        # Initial market scan
        markets = self.scanner.scan(top_n=5)
        logger.info("Initial market scan: %d eligible markets found", len(markets))

        # Test Polymarket authentication
        if self.config.polymarket_private_key:
            success = self.executor.initialize_live_client()
            if success:
                logger.info("Polymarket authentication: OK")
            else:
                logger.warning("Polymarket authentication: FAILED (will continue in paper mode)")
        else:
            logger.info("No Polymarket credentials — paper mode only")

        # Start dashboard
        dashboard.set_bot_ref(self)
        dashboard.run_dashboard(port=8082)
        logger.info("Dashboard: http://localhost:8082")
        logger.info("=" * 60)

    def _update_prices(self) -> None:
        """Fetch current BTC and ETH prices."""
        try:
            btc_momentum, _ = self.collector.collect_price_momentum("BTC")
            self.btc_price = btc_momentum.get("current_price", 0.0)
        except Exception as e:
            logger.debug("BTC price update error: %s", e)

        try:
            eth_momentum, _ = self.collector.collect_price_momentum("ETH")
            self.eth_price = eth_momentum.get("current_price", 0.0)
        except Exception as e:
            logger.debug("ETH price update error: %s", e)

        # Update dashboard state
        dashboard.update_state({
            "btc_price": self.btc_price,
            "eth_price": self.eth_price,
        })

    def _monitor_open_positions(self) -> None:
        """Check exit conditions for all open positions."""
        for trade_id, position in list(self.executor.open_positions.items()):
            if position.status != "open":
                continue

            # Get current market price for the position's token
            # In paper mode, use a proxy from momentum data
            asset_price_data = (
                self.btc_price if position.asset == "BTC" else self.eth_price
            )

            # Estimate current token price from current asset price vs strike
            # This is a proxy — in live mode we'd fetch the actual token price
            current_token_price = self._estimate_token_price(position, asset_price_data)

            should_exit, reason = self.executor.check_exit_conditions(
                position, current_token_price
            )

            if should_exit:
                if self.config.is_paper():
                    self.executor.exit_paper_trade(
                        position,
                        exit_price=current_token_price,
                        reason=reason
                    )
                else:
                    logger.info(
                        "Exit signal for live position %s: %s",
                        trade_id, reason
                    )
                    # Live exit would go here

    def _estimate_token_price(self, position, current_asset_price: float) -> float:
        """
        Estimate current YES token price based on asset price movement.
        Naive approximation: if asset moved toward hypothesis, price moves up.
        """
        if current_asset_price == 0 or position.entry_price <= 0:
            return position.current_price

        # Start from last known price
        last_price = position.current_price

        # Adjust based on time remaining (as time runs out, binary convergence)
        mins_left = position.minutes_to_expiry()
        if mins_left <= 0:
            return position.current_price

        # As market approaches resolution, price moves toward 0 or 1
        # This is a simplistic model — real implementation would fetch live book
        return max(0.01, min(0.99, last_price))

    def _run_analysis_cycle(self) -> None:
        """Main analysis cycle: scan, analyze, trade."""
        self._scan_count += 1

        # Update dashboard
        dashboard.update_state({
            "scan_count": self._scan_count,
            "last_scan_time": datetime.now(timezone.utc).isoformat(),
            "mode": self.config.mode,
            "running": True,
        })

        # Update prices
        self._update_prices()

        # Update open position set for scanner
        self.scanner.set_open_positions(self.executor.get_open_condition_ids())

        # Scan for markets
        candidates = self.scanner.scan(top_n=5)
        if not candidates:
            logger.info("Scan #%d: No eligible markets found", self._scan_count)
            return

        logger.info(
            "Scan #%d: Analyzing %d markets (BTC=$%.0f ETH=$%.0f bal=$%.2f)",
            self._scan_count, len(candidates),
            self.btc_price, self.eth_price,
            self.executor.get_paper_balance() if self.config.is_paper()
            else self.executor.get_live_balance()
        )

        monitored_state = []

        for market in candidates:
            try:
                # Check if we can trade this market
                can_trade, reason = self.executor.can_open_position(
                    market.asset, market.condition_id
                )

                # Always analyze (for transparency), just don't trade if blocked
                related = self.scanner.get_related_markets(
                    market.asset, market.condition_id
                )

                # Collect all evidence
                evidence = self.collector.collect_all(
                    asset=market.asset,
                    direction=market.direction,
                    yes_token_id=market.yes_token_id,
                    related_markets=related
                )

                # Run Bayesian analysis
                result = self.engine.run_full_analysis(
                    asset=market.asset,
                    direction=market.direction,
                    timeframe_minutes=int(market.minutes_to_resolution),
                    market_price=market.current_yes_price,
                    price_data=evidence.price_history,
                    evidence={
                        "order_flow": evidence.order_flow,
                        "price_momentum": evidence.price_momentum,
                        "social_buzz": evidence.social_buzz,
                        "on_chain": evidence.on_chain,
                        "market_sentiment": evidence.market_sentiment,
                    },
                    market_question=market.question
                )

                # Store for dashboard
                monitored_state.append({
                    "condition_id": market.condition_id,
                    "question": market.question,
                    "asset": market.asset,
                    "direction": market.direction,
                    "timeframe": market.timeframe_label,
                    "volume": market.volume_usd,
                    "minutes_left": market.minutes_to_resolution,
                    "market_price": result.market_price,
                    "prior": result.prior,
                    "posterior": result.posterior,
                    "divergence": result.divergence,
                    "signal": result.trade_signal,
                    "regime": result.regime,
                    "session": result.session,
                    "evidence_updates": [
                        {
                            "layer": u.layer,
                            "delta": u.delta,
                            "description": u.description,
                        }
                        for u in result.evidence_updates
                    ],
                    "chain": result.format_chain(),
                    "can_trade": can_trade,
                    "no_trade_reason": reason if not can_trade else "",
                })

                # Execute trade if signal qualifies
                if result.trade_signal == "HOLD":
                    logger.debug(
                        "HOLD %s %s: posterior=%.3f market=%.3f divergence=%.3f",
                        market.asset, market.direction,
                        result.posterior, result.market_price, result.divergence
                    )
                    continue

                if not can_trade:
                    logger.info(
                        "SIGNAL %s on %s but blocked: %s",
                        result.trade_signal, market.asset, reason
                    )
                    continue

                # Determine YES or NO
                if result.trade_signal == "BUY_YES":
                    outcome = "YES"
                else:
                    outcome = "NO"

                logger.info(
                    "TRADE SIGNAL: %s %s | posterior=%.3f | market=%.3f | edge=%.3f",
                    outcome, market.asset,
                    result.posterior, result.market_price, result.divergence
                )

                if self.config.is_paper():
                    position = self.executor.enter_paper_trade(market, result, outcome)
                    if position:
                        logger.info(
                            "Paper trade opened: %s | $%.2f @ %.3f",
                            position.trade_id, position.size_usd, position.entry_price
                        )
                else:
                    position = self.executor.enter_live_trade(market, result, outcome)
                    if position:
                        logger.info("Live trade opened: %s", position.trade_id)

            except Exception as e:
                logger.error("Error analyzing market %s: %s", market.condition_id[:8], e, exc_info=True)

        # Update dashboard with monitored markets
        self._monitored_markets_state = monitored_state
        dashboard.update_state({"monitored_markets": monitored_state})

        # Monitor existing positions
        self._monitor_open_positions()

        # Check if calibration needed
        all_trades = self.executor.get_all_trades(mode=self.config.mode)
        resolved_count = sum(
            1 for t in all_trades
            if t.get("status") == "closed" and t.get("resolution_outcome") is not None
        )

        if self.calibrator.should_calibrate(resolved_count):
            logger.info("Triggering calibration (resolved trades: %d)", resolved_count)
            self.calibrator.run_calibration(all_trades)

    def run(self) -> None:
        """Main blocking loop."""
        self._running = True

        def _signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            self._running = False

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        self.startup()

        logger.info("Starting main loop (interval: %ds)", self.SCAN_INTERVAL)
        dashboard.update_state({"running": True})

        while self._running:
            try:
                start = time.time()
                self._run_analysis_cycle()
                elapsed = time.time() - start
                sleep_time = max(0, self.SCAN_INTERVAL - elapsed)
                logger.debug("Cycle took %.1fs, sleeping %.1fs", elapsed, sleep_time)

                if sleep_time > 0 and self._running:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — shutting down")
                break
            except Exception as e:
                logger.error("Unhandled error in main loop: %s", e, exc_info=True)
                # Brief pause before retrying
                time.sleep(10)

        logger.info("Bot stopped")
        dashboard.update_state({"running": False})


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    debug = "--debug" in sys.argv
    setup_logging(debug=debug)
    bot = BayesianBot()
    bot.run()


if __name__ == "__main__":
    main()
