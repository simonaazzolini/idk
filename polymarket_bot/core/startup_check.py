"""
Startup Self-Test — runs automatically when the bot launches.

Verifies every external dependency before the first analysis cycle so
the operator knows exactly what is working.  Failures are logged as
warnings and reported in the startup summary, but they do NOT halt the
bot (except for a failed database, which is fatal).

Output example:
  STARTUP CHECK: Gamma ✓  CLOB-read ✓  Data-API ✓  Anthropic ✓  Database ✓
"""
import asyncio
import logging
from typing import Optional

import anthropic as _anthropic

from config.settings import Settings
from core.client import PolymarketClient
from data.database import Database

logger = logging.getLogger(__name__)

# Symbols used in the printed report
_OK   = "\u2713"   # ✓
_FAIL = "\u2717"   # ✗


async def run_startup_checks(
    client: PolymarketClient,
    db: Database,
    settings: Settings,
) -> dict[str, bool]:
    """
    Run all startup connectivity checks and return a results dict.

    Keys: "Gamma", "CLOB-read", "Data-API", "Anthropic", "Database"
    Values: True = passed, False = failed.

    The function never raises — every check is fully isolated so one
    failure cannot prevent the others from running.
    """
    results: dict[str, bool] = {}

    # Run all checks concurrently except Anthropic (synchronous SDK)
    gamma_ok, clob_ok, data_ok, db_ok = await asyncio.gather(
        _check_gamma(client),
        _check_clob_read(client),
        _check_data_api(client),
        _check_database(db),
    )
    results["Gamma"]    = gamma_ok
    results["CLOB-read"] = clob_ok
    results["Data-API"] = data_ok
    results["Database"] = db_ok

    # Anthropic check runs in a thread pool (blocking SDK call)
    results["Anthropic"] = await asyncio.get_event_loop().run_in_executor(
        None, _check_anthropic_sync, settings
    )

    _print_startup_report(results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

async def _check_gamma(client: PolymarketClient) -> bool:
    """Fetch 1 market from the Gamma API."""
    try:
        markets = await client.gamma.get_markets(limit=1)
        ok = isinstance(markets, list) and len(markets) > 0
        if ok:
            logger.info("Startup | Gamma API: OK (fetched %d market)", len(markets))
        else:
            logger.warning("Startup | Gamma API: returned empty market list")
        return ok
    except Exception as exc:
        logger.warning("Startup | Gamma API: FAILED — %s", exc)
        return False


async def _check_clob_read(client: PolymarketClient) -> bool:
    """
    Verify the CLOB read path by fetching an orderbook for a high-liquidity
    market.  We first fetch 1 market from Gamma to get a real token ID so
    we don't hard-code any specific market.
    """
    try:
        markets = await client.gamma.get_markets(limit=5)
        token_id: Optional[str] = None
        for m in markets or []:
            ids = m.get("clobTokenIds") or []
            if isinstance(ids, list) and ids:
                token_id = str(ids[0])
                break
        if not token_id:
            logger.warning("Startup | CLOB-read: no token_id available from Gamma")
            return False

        book = await client.clob.get_order_book(token_id)
        ok = book is not None
        if ok:
            bid_count = len((book or {}).get("bids", []))
            ask_count = len((book or {}).get("asks", []))
            logger.info(
                "Startup | CLOB-read: OK (token=%s, bids=%d, asks=%d)",
                token_id[:12], bid_count, ask_count,
            )
        else:
            logger.warning("Startup | CLOB-read: get_order_book returned None")
        return ok
    except Exception as exc:
        logger.warning("Startup | CLOB-read: FAILED — %s", exc)
        return False


async def _check_data_api(client: PolymarketClient) -> bool:
    """
    Fetch the leaderboard from the Data API and log the raw response so
    the operator can see exactly what the endpoint returns.
    """
    try:
        board = await client.data.get_leaderboard(window="1d", limit=5)
        # Always log raw response as required — helps diagnose schema changes
        logger.info(
            "Startup | Data-API leaderboard raw (window=1d, limit=5): %s",
            str(board)[:400],
        )
        ok = board is not None   # empty list is still a valid response
        if ok:
            logger.info(
                "Startup | Data-API: OK (leaderboard returned %d entries)",
                len(board) if isinstance(board, list) else "?",
            )
        else:
            logger.warning("Startup | Data-API: returned None")
        return ok
    except Exception as exc:
        logger.warning("Startup | Data-API: FAILED — %s", exc)
        return False


def _check_anthropic_sync(settings: Settings) -> bool:
    """
    Send a minimal 10-token message to verify the Anthropic API key works.
    Uses the fast analysis model (Haiku) and max_tokens=10 to minimise cost.
    """
    if not settings.anthropic_api_key:
        logger.warning("Startup | Anthropic: FAILED — ANTHROPIC_API_KEY not set")
        return False
    try:
        ac = _anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = ac.messages.create(
            model=settings.ai_model_analysis,
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        ok = bool(resp and resp.content)
        if ok:
            logger.info("Startup | Anthropic: OK (model=%s)", settings.ai_model_analysis)
        else:
            logger.warning("Startup | Anthropic: response empty")
        return ok
    except _anthropic.AuthenticationError as exc:
        logger.warning("Startup | Anthropic: FAILED — authentication error: %s", exc)
        return False
    except _anthropic.APIError as exc:
        err_str = str(exc).lower()
        billing_kws = ("credit", "billing", "balance", "payment", "quota", "overdue")
        if any(kw in err_str for kw in billing_kws):
            # Billing issues — API key is valid but account has no credits
            logger.warning(
                "Startup | Anthropic: billing/credit error (key valid but no credits): %s", exc
            )
            return False
        logger.warning("Startup | Anthropic: FAILED — %s", exc)
        return False
    except Exception as exc:
        logger.warning("Startup | Anthropic: FAILED — %s", exc)
        return False


async def _check_database(db: Database) -> bool:
    """Write a test row and immediately read it back to verify DB integrity."""
    try:
        await db.set_state("_startup_test", "ok")
        val = await db.get_state("_startup_test")
        ok = val == "ok"
        if ok:
            logger.info("Startup | Database: OK (write-read round-trip passed)")
        else:
            logger.warning("Startup | Database: write-read mismatch (got %r)", val)
        return ok
    except Exception as exc:
        logger.warning("Startup | Database: FAILED — %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_startup_report(results: dict[str, bool]) -> None:
    """
    Print a clean single-line startup status report to stdout and the log.

    Example:
        STARTUP CHECK: Gamma ✓  CLOB-read ✓  Data-API ✓  Anthropic ✓  Database ✓
    """
    parts = [f"{name} {_OK if ok else _FAIL}" for name, ok in results.items()]
    report = "STARTUP CHECK: " + "  ".join(parts)

    # Print to stdout so it's always visible even when log level is high
    print("\n" + report + "\n")

    # Also log at INFO level for log-file capture
    log_parts = [f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()]
    logger.info("Startup check complete: %s", "  ".join(log_parts))

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.warning(
            "Startup check: %d service(s) unavailable: %s. "
            "Bot will run with degraded functionality.",
            len(failed), ", ".join(failed),
        )
