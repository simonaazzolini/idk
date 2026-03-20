"""
Main entry point — Module 14.
CLI for starting the Polymarket trading bot.
"""
import sys
import os

# Fix bad file descriptor on restart — must happen before any other imports
# that may write to stdout/stderr.  Uses os.fstat() to detect fds that are
# present in sys but point to a closed OS-level descriptor (a common failure
# mode when the process is launched as a subprocess with redirected streams).
for _fd in (0, 1, 2):
    try:
        os.fstat(_fd)
    except OSError:
        open(os.devnull, 'rb' if _fd == 0 else 'wb')

import argparse
import asyncio
import logging
import signal
from pathlib import Path

# Add parent dir to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings, settings as default_settings
from core.bot import PolymarketBot
from core.client import PolymarketClient
from data.database import Database
from utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polymarket Elite Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode paper --budget 1000
  python main.py --mode live --budget 500 --max-position 50
  python main.py --test
  python main.py --mode paper --reset-paper
        """,
    )
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode: paper (simulate) or live (real money). Default: paper"
    )
    parser.add_argument(
        "--budget", type=float, default=1000.0,
        help="Starting budget in USDC. Default: 1000"
    )
    parser.add_argument(
        "--max-position", type=float, default=None,
        help="Max USDC per position. Default: from settings"
    )
    parser.add_argument(
        "--min-edge", type=float, default=None,
        help="Minimum edge (probability minus price) to trade. Default: 0.04"
    )
    parser.add_argument(
        "--cycle", type=int, default=None,
        help="Main cycle interval in minutes. Default: 15"
    )
    parser.add_argument(
        "--categories", nargs="*", default=None,
        help="Filter to specific market categories (space-separated)"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
        help="Logging verbosity. Default: INFO"
    )
    parser.add_argument(
        "--reset-paper", action="store_true",
        help="Reset paper trading period to start fresh"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run module tests with mock data (no credentials required)"
    )
    parser.add_argument(
        "--arb-only", action="store_true",
        help="Skip all signal trades; only execute arbitrage opportunities"
    )
    return parser.parse_args()


def apply_overrides(s: Settings, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to settings."""
    s.budget = args.budget
    if args.max_position is not None:
        s.max_position_size = args.max_position
    if args.min_edge is not None:
        s.min_edge = args.min_edge
    if args.cycle is not None:
        s.cycle_minutes = args.cycle
    if args.categories:
        s.categories = args.categories
    if args.log_level:
        s.log_level = args.log_level
    if args.arb_only:
        s.arb_only = True


async def run_tests() -> int:
    """Run all modules with mock data. Returns 0 on pass, 1 on failure."""
    print("Running module tests with mock data...\n")
    failures = []

    # Test 1: Settings
    try:
        s = Settings()
        assert s.budget == 1000.0
        assert abs(s.ai_weight + s.whale_weight + s.news_weight +
                   s.technical_weight + s.orderbook_weight + s.arb_weight - 1.0) < 0.001
        print("  [PASS] Settings — signal weights sum to 1.0")
    except Exception as e:
        failures.append(f"Settings: {e}")
        print(f"  [FAIL] Settings: {e}")

    # Test 2: Database
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name
        db = Database(tmp_db)
        await db.initialize()
        await db.set_state("test_key", "test_value")
        val = await db.get_state("test_key")
        assert val == "test_value", f"Expected 'test_value', got {val!r}"
        os.unlink(tmp_db)
        print("  [PASS] Database — initialize, set/get state")
    except Exception as e:
        failures.append(f"Database: {e}")
        print(f"  [FAIL] Database: {e}")

    # Test 3: TTL Cache
    try:
        from data.cache import TTLCache
        c = TTLCache(default_ttl_seconds=1)
        await c.set("foo", "bar")
        result = await c.get("foo")
        assert result == "bar"
        print("  [PASS] TTL Cache — set and get")
    except Exception as e:
        failures.append(f"TTL Cache: {e}")
        print(f"  [FAIL] TTL Cache: {e}")

    # Test 4: Technical analysis
    try:
        import numpy as np
        from analysis.technical import compute_technical_signals
        mock_history = [
            {"t": i * 3600, "p": 0.5 + 0.01 * np.sin(i / 5), "v": 1000.0}
            for i in range(50)
        ]
        sig = compute_technical_signals(mock_history, 0.52, 5000.0, 30.0)
        assert 0 <= sig.rsi_14 <= 100
        assert 0 <= sig.technical_score <= 10
        assert sig.technical_direction in ("YES", "NO", "NEUTRAL")
        print(f"  [PASS] Technical — RSI={sig.rsi_14:.1f}, score={sig.technical_score:.1f}")
    except Exception as e:
        failures.append(f"Technical: {e}")
        print(f"  [FAIL] Technical: {e}")

    # Test 5: Orderbook analysis
    try:
        from analysis.orderbook import analyze_orderbook
        mock_book = {
            "bids": [{"price": 0.48, "size": 100}, {"price": 0.47, "size": 200}],
            "asks": [{"price": 0.52, "size": 150}, {"price": 0.53, "size": 100}],
        }
        metrics = analyze_orderbook(mock_book)
        assert metrics.best_bid == 0.48
        assert metrics.best_ask == 0.52
        assert abs(metrics.mid_price - 0.50) < 0.01
        assert 0 <= metrics.orderbook_score <= 10
        print(f"  [PASS] Orderbook — spread={metrics.spread_pct:.1f}%, score={metrics.orderbook_score:.1f}")
    except Exception as e:
        failures.append(f"Orderbook: {e}")
        print(f"  [FAIL] Orderbook: {e}")

    # Test 6: Kelly criterion
    try:
        from strategies.kelly import kelly_size
        s = Settings()
        s.budget = 1000.0
        s.max_position_size = 100.0
        s.min_position_size_usdc = 5.0
        kr = kelly_size(
            estimated_probability=0.65,
            current_price=0.50,
            confidence=0.75,
            composite_score=8.0,
            available_bankroll=1000.0,
            market_liquidity=50_000.0,
            current_exposure=0.0,
            total_budget=1000.0,
            settings=s,
        )
        assert kr.final_size <= s.max_position_size
        assert kr.final_size >= 0
        assert kr.is_viable or kr.final_size < s.min_position_size_usdc
        print(f"  [PASS] Kelly — size=${kr.final_size:.2f}, capped_by={kr.capped_by}")
    except Exception as e:
        failures.append(f"Kelly: {e}")
        print(f"  [FAIL] Kelly: {e}")

    # Test 7: Signal aggregator
    try:
        from strategies.signals import SignalAggregator, TradeSignal
        from analysis.ai_analyzer import AIAnalysisResult
        s = Settings()
        agg = SignalAggregator(s)

        mock_market = {
            "slug": "test-market",
            "question": "Will X happen?",
            "category": "politics",
            "outcomePrices": ["0.45", "0.55"],
            "liquidity": 10000,
        }
        mock_ai_data = {
            "yes_probability": 0.62,
            "confidence": 0.72,
            "recommended_outcome": "YES",
            "edge": 0.17,
            "signal_strength": "STRONG",
            "ai_score": 8.0,
            "reasoning": "Strong evidence for YES.",
            "strongest_yes_evidence": "Evidence A",
            "strongest_no_evidence": "Evidence B",
            "key_facts": [],
            "key_uncertainties": [],
            "resolution_risk": "LOW",
            "aggregator_consensus": None,
            "news_summary": "",
            "base_rate": 0.5,
            "base_rate_source": "historical",
            "evidence_adjustment": 0.12,
            "confidence_interval_low": 0.50,
            "confidence_interval_high": 0.74,
        }
        mock_ai = AIAnalysisResult(mock_ai_data)
        mock_ai.ai_score = 8.0

        signal = agg.aggregate(mock_market, ai_result=mock_ai)
        assert signal.action in ("STRONG_BUY", "BUY", "WEAK_BUY", "SKIP")
        assert signal.composite_score >= 0
        print(f"  [PASS] SignalAggregator — action={signal.action}, composite={signal.composite_score:.1f}")
    except Exception as e:
        failures.append(f"SignalAggregator: {e}")
        print(f"  [FAIL] SignalAggregator: {e}")

    # Test 8: Arbitrage detector
    try:
        from strategies.arbitrage import ArbitrageDetector
        from analysis.orderbook import OrderbookMetrics
        s = Settings()
        s.arb_min_profit_pct = 0.015
        s.arb_min_profit_usdc = 5.0
        detector = ArbitrageDetector(s)

        yes_book = OrderbookMetrics(best_bid=0.45, best_ask=0.48, mid_price=0.465, ask_depth_1pct=500)
        no_book = OrderbookMetrics(best_bid=0.46, best_ask=0.49, mid_price=0.475, ask_depth_1pct=500)
        mock_market = {
            "slug": "test",
            "clobTokenIds": ["yes_tok", "no_tok"],
            "yes_token_id": "yes_tok",
            "no_token_id": "no_tok",
            "liquidity": 10000,
        }
        arb = detector.detect_yes_no_arb(mock_market, yes_book, no_book)
        # yes_ask=0.48 + no_ask=0.49 = 0.97 → gap=0.03 > threshold=0.015 → should detect
        assert arb is not None
        assert arb.profit_pct > 0
        print(f"  [PASS] Arbitrage — detected TYPE_1 arb, profit={arb.profit_pct:.3f}")
    except Exception as e:
        failures.append(f"Arbitrage: {e}")
        print(f"  [FAIL] Arbitrage: {e}")

    # Test 9: Wallet classifier
    try:
        from tracking.wallet_classifier import classify_wallet_tiers
        metrics = {
            "total_pnl": 150_000,
            "total_volume": 600_000,
            "win_rate": 0.70,
            "total_trades": 80,
            "profit_factor": 2.5,
            "insider_score": 7.5,
            "is_bot": 0,
        }
        tiers = classify_wallet_tiers(metrics)
        assert "LEGENDARY" in tiers or "TIER_1_WHALE" in tiers
        assert "SMART_MONEY" in tiers
        assert "INSIDER_ALERT" in tiers
        print(f"  [PASS] WalletClassifier — tiers={tiers}")
    except Exception as e:
        failures.append(f"WalletClassifier: {e}")
        print(f"  [FAIL] WalletClassifier: {e}")

    # Test 10: Safety checker
    try:
        from execution.safety import SafetyChecker
        s = Settings()
        s.max_orders_per_minute = 10
        s.min_position_size_usdc = 5.0
        s.max_open_positions = 10
        s.max_total_exposure_pct = 0.80
        checker = SafetyChecker(s)
        mock_market = {"active": True, "closed": False, "slug": "test-market"}
        portfolio = {
            "open_positions": [],
            "current_exposure": 0.0,
            "total_budget": 1000.0,
            "daily_loss_limit_breached": False,
            "max_drawdown_breached": False,
        }
        result = await checker.check_all(
            mode="PAPER",
            market=mock_market,
            size_usdc=50.0,
            current_price=0.50,
            signal_price=0.50,
            portfolio_state=portfolio,
            usdc_balance=500.0,
        )
        assert result.passed, f"Safety check failed: {result.reason}"
        print(f"  [PASS] SafetyChecker — {len(result.checks_run)} checks passed")
    except Exception as e:
        failures.append(f"SafetyChecker: {e}")
        print(f"  [FAIL] SafetyChecker: {e}")

    # Summary
    print(f"\n{'='*50}")
    total = 10
    passed = total - len(failures)
    if failures:
        print(f"RESULTS: {passed}/{total} tests passed")
        print("\nFailed tests:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print(f"RESULTS: {passed}/{total} tests passed — ALL PASS")
        print("\nBot is ready. Run: python main.py --mode paper --budget 1000")
        return 0


async def main() -> None:
    args = parse_args()

    # Apply settings overrides first so log level is correct
    s = default_settings
    apply_overrides(s, args)

    # Setup logging
    log_level = args.log_level or s.log_level or "INFO"
    setup_logging(log_level)
    _logger = logging.getLogger(__name__)

    # Run tests if requested
    if args.test:
        exit_code = await run_tests()
        sys.exit(exit_code)

    mode = args.mode.upper()
    _logger.info("Starting Polymarket Elite Bot | mode=%s | budget=$%.2f", mode, s.budget)

    # Validate credentials (skip for paper mode with --test)
    if mode == "LIVE":
        errors = s.validate()
        if errors:
            print("\n[ERROR] Missing required credentials:")
            for err in errors:
                print(f"  - {err}")
            print(f"\nCreate config/.env from config/.env.example and fill in your credentials.")
            sys.exit(1)
    else:
        # Paper mode: warn but don't block
        errors = s.validate()
        if errors:
            _logger.warning("Some credentials missing (OK for paper mode): %s", ", ".join(errors))

    # Initialize database
    db = Database(s.db_path)
    await db.initialize()

    # Handle --reset-paper
    if args.reset_paper:
        await db.set_state("paper_start_ts", str(__import__("time").time()))
        _logger.info("Paper trading period reset")

    # Check if mode was persisted from previous session
    stored_mode = await db.get_state("mode")
    if stored_mode and not args.reset_paper:
        if stored_mode == "LIVE" and mode == "PAPER":
            _logger.info("Restoring LIVE mode from previous session")
            mode = "LIVE"

    # Initialize client
    client = PolymarketClient(s)
    await client.initialize()

    # Initialize and run bot
    bot = PolymarketBot(
        client=client,
        db=db,
        settings=s,
        mode=mode,
        budget=s.budget,
    )

    # Cancel this task on SIGTERM so bot._shutdown() runs instead of dying at rc=-15
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, main_task.cancel)

    try:
        await bot.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        _logger.info("Shutdown signal received")
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        await client.shutdown(mode=bot.mode)
        _logger.info("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
