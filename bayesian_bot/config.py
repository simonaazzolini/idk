"""
config.py — Settings dataclass and loader for the Bayesian Market Bot.
Loads from settings.json in the same directory.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(__file__).parent / "settings.json"
DEFAULTS = {
    "mode": "paper",
    "polymarket_private_key": "",
    "polymarket_api_key": "",
    "polymarket_api_secret": "",
    "polymarket_api_passphrase": "",
    "anthropic_api_key": "",
    "newsapi_key": "",
    "paper_starting_balance": 500,
    "live_starting_balance": 0,
    "risk_per_trade_pct": 0.01,
    "max_trade_size_usd": 10,
    "min_trade_size_usd": 1,
    "min_divergence_threshold": 0.08,
    "max_open_positions": 3,
    "min_market_volume": 200,
    "max_market_window_minutes": 30,
    "min_market_window_minutes": 1,
    "stop_loss_pct": 0.40,
    "early_exit_minutes": 2,
    "max_slippage_pct": 0.02,
    "evidence_weights": {
        "order_flow": 1.0,
        "price_momentum": 0.8,
        "social_buzz": 0.5,
        "on_chain": 0.4,
        "market_sentiment": 0.3
    }
}


@dataclass
class Config:
    mode: str = "paper"
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    anthropic_api_key: str = ""
    newsapi_key: str = ""
    paper_starting_balance: float = 500.0
    live_starting_balance: float = 0.0
    risk_per_trade_pct: float = 0.01
    max_trade_size_usd: float = 10.0
    min_trade_size_usd: float = 1.0
    min_divergence_threshold: float = 0.08
    max_open_positions: int = 3
    min_market_volume: float = 200.0
    max_market_window_minutes: int = 30
    min_market_window_minutes: int = 1
    stop_loss_pct: float = 0.40
    early_exit_minutes: int = 2
    max_slippage_pct: float = 0.02
    evidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "order_flow": 1.0,
        "price_momentum": 0.8,
        "social_buzz": 0.5,
        "on_chain": 0.4,
        "market_sentiment": 0.3
    })

    # Runtime state (not persisted)
    paper_balance: float = 500.0
    live_balance: float = 0.0

    def is_paper(self) -> bool:
        return self.mode.lower() == "paper"

    def current_balance(self) -> float:
        return self.paper_balance if self.is_paper() else self.live_balance

    def update_balance(self, new_balance: float) -> None:
        if self.is_paper():
            self.paper_balance = new_balance
        else:
            self.live_balance = new_balance


def load_config() -> Config:
    """Load config from settings.json, creating defaults if missing."""
    if not SETTINGS_PATH.exists():
        logger.info("settings.json not found, creating with defaults")
        with open(SETTINGS_PATH, "w") as f:
            json.dump(DEFAULTS, f, indent=2)
        data = DEFAULTS.copy()
    else:
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
        # Merge missing keys from defaults
        for k, v in DEFAULTS.items():
            if k not in data:
                data[k] = v

    cfg = Config(
        mode=data.get("mode", "paper"),
        polymarket_private_key=data.get("polymarket_private_key", ""),
        polymarket_api_key=data.get("polymarket_api_key", ""),
        polymarket_api_secret=data.get("polymarket_api_secret", ""),
        polymarket_api_passphrase=data.get("polymarket_api_passphrase", ""),
        anthropic_api_key=data.get("anthropic_api_key", ""),
        newsapi_key=data.get("newsapi_key", ""),
        paper_starting_balance=float(data.get("paper_starting_balance", 500)),
        live_starting_balance=float(data.get("live_starting_balance", 0)),
        risk_per_trade_pct=float(data.get("risk_per_trade_pct", 0.01)),
        max_trade_size_usd=float(data.get("max_trade_size_usd", 10)),
        min_trade_size_usd=float(data.get("min_trade_size_usd", 1)),
        min_divergence_threshold=float(data.get("min_divergence_threshold", 0.08)),
        max_open_positions=int(data.get("max_open_positions", 3)),
        min_market_volume=float(data.get("min_market_volume", 200)),
        max_market_window_minutes=int(data.get("max_market_window_minutes", 30)),
        min_market_window_minutes=int(data.get("min_market_window_minutes", 1)),
        stop_loss_pct=float(data.get("stop_loss_pct", 0.40)),
        early_exit_minutes=int(data.get("early_exit_minutes", 2)),
        max_slippage_pct=float(data.get("max_slippage_pct", 0.02)),
        evidence_weights=data.get("evidence_weights", DEFAULTS["evidence_weights"]),
    )
    cfg.paper_balance = cfg.paper_starting_balance
    cfg.live_balance = cfg.live_starting_balance
    return cfg


def save_settings(updates: dict) -> None:
    """Save partial settings updates to settings.json."""
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
    else:
        data = DEFAULTS.copy()
    data.update(updates)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=2)
