"""
Settings and configuration management.
Loads from config/.env with sensible defaults.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from config/ directory relative to this file
_ENV_PATH = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    # Try current working directory config/.env
    load_dotenv(Path("config") / ".env")


@dataclass
class Settings:
    # ── Credentials ──────────────────────────────────────────────────────────
    private_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY", ""))
    funder_address: str = field(default_factory=lambda: os.getenv("POLYMARKET_FUNDER_ADDRESS", ""))
    api_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_SECRET", ""))
    api_passphrase: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_PASSPHRASE", ""))
    signature_type: int = field(default_factory=lambda: int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # ── API Endpoints ─────────────────────────────────────────────────────────
    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    data_url: str = "https://data-api.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    chain_id: int = 137

    # ── Trading Parameters ────────────────────────────────────────────────────
    budget: float = 1000.0
    max_position_size: float = 100.0
    min_edge: float = 0.04
    cycle_minutes: int = 15

    # ── Market Filters ────────────────────────────────────────────────────────
    min_liquidity: float = field(default_factory=lambda: float(os.getenv("MIN_LIQUIDITY", "500")))
    min_volume: float = field(default_factory=lambda: float(os.getenv("MIN_VOLUME", "1000")))
    min_days_to_resolution: float = 1.0
    max_days_to_resolution: float = 180.0
    top_markets_count: int = 50

    # ── Signal Weights (must sum to 1.0) ─────────────────────────────────────
    ai_weight: float = 0.35
    whale_weight: float = 0.25
    news_weight: float = 0.18
    technical_weight: float = 0.10
    orderbook_weight: float = 0.07
    arb_weight: float = 0.05

    # ── Risk Limits ───────────────────────────────────────────────────────────
    stop_loss_pct: float = 0.30          # exit if position drops 30%
    take_profit_pct: float = 0.60        # exit if position gains 60%
    daily_loss_limit_pct: float = 0.10   # halt if daily loss > 10% of budget
    max_drawdown_limit_pct: float = 0.20 # halt if drawdown > 20%
    max_single_market_pct: float = 0.15  # max 15% budget in one market
    max_category_pct: float = 0.30       # max 30% in one category
    max_total_exposure_pct: float = 0.80 # max 80% deployed
    max_open_positions: int = 10
    kelly_multiplier: float = 0.25       # quarter Kelly, never increase

    # ── Paper Trading ─────────────────────────────────────────────────────────
    paper_trading_hours: float = 48.0
    paper_slippage_pct: float = 0.003    # simulate fills at mid + 0.3%

    # ── Go/No-Go Criteria ─────────────────────────────────────────────────────
    go_min_win_rate: float = 0.55
    go_min_profit_factor: float = 1.3
    go_max_single_day_loss_pct: float = 0.08
    go_min_trades: int = 8
    go_min_sharpe: float = 0.5
    go_max_single_position_loss_pct: float = 0.25
    nogo_max_drawdown_pct: float = 0.15

    # ── Copy Trading ──────────────────────────────────────────────────────────
    copy_trade_max_market_pct: float = 0.15  # max 15% of liquidity
    copy_trade_stop_loss_pct: float = 0.20   # tighter than own trades

    # ── Arb Settings ──────────────────────────────────────────────────────────
    arb_min_profit_pct: float = 0.015    # min 1.5% gap for Type 1 arb
    arb_min_profit_usdc: float = 5.0
    spread_capture_min_spread_pct: float = 0.03

    # ── Order Execution ───────────────────────────────────────────────────────
    max_orders_per_minute: int = 10
    order_cancel_timeout_seconds: int = 1800   # cancel GTC after 30 min
    iceberg_tranche_size: float = 75.0
    iceberg_delay_seconds: int = 30
    min_position_size_usdc: float = 5.0

    # ── Logging & DB ─────────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    db_path: str = "polymarket_bot.db"
    dashboard_refresh_seconds: int = 5

    # ── Whale Tracking ────────────────────────────────────────────────────────
    whale_refresh_hours: float = 12.0
    whale_position_poll_minutes: float = 5.0
    whale_trade_poll_seconds: int = 60
    insider_score_threshold: float = 7.0
    smart_money_min_win_rate: float = 0.65
    smart_money_min_trades: int = 20
    smart_money_min_profit_factor: float = 1.5

    # ── WebSocket ─────────────────────────────────────────────────────────────
    ws_reconnect_base_seconds: float = 2.0
    ws_reconnect_max_seconds: float = 60.0
    ws_price_shock_threshold_pct: float = 0.05  # 5% move in 60s

    # ── Misc ──────────────────────────────────────────────────────────────────
    market_cache_ttl_minutes: int = 10
    ai_min_call_interval_seconds: float = 3.0
    categories: list = field(default_factory=list)  # empty = all categories

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list means OK."""
        errors = []
        if not self.private_key:
            errors.append("POLYMARKET_PRIVATE_KEY not set")
        if not self.funder_address:
            errors.append("POLYMARKET_FUNDER_ADDRESS not set")
        if not self.api_key:
            errors.append("POLYMARKET_API_KEY not set")
        if not self.api_secret:
            errors.append("POLYMARKET_API_SECRET not set")
        if not self.api_passphrase:
            errors.append("POLYMARKET_API_PASSPHRASE not set")
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY not set")
        weight_sum = (
            self.ai_weight + self.whale_weight + self.news_weight
            + self.technical_weight + self.orderbook_weight + self.arb_weight
        )
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(f"Signal weights sum to {weight_sum:.3f}, must be 1.0")
        if self.budget <= 0:
            errors.append("Budget must be positive")
        if self.max_position_size <= 0 or self.max_position_size > self.budget:
            errors.append("max_position_size must be between 0 and budget")
        return errors

    def validate_for_test(self) -> bool:
        """Less strict validation for --test mode (no credentials required)."""
        return True


# Singleton instance
settings = Settings()
