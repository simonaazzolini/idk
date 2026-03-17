"""
dashboard.py — Flask dashboard for the Bayesian Market Bot.
Runs on port 8082. Dark theme, separate from other bots.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request

logger = logging.getLogger(__name__)

BOT_DIR = Path(__file__).parent
app = Flask(__name__, template_folder=str(BOT_DIR / "templates"))
app.config["JSON_SORT_KEYS"] = False

# Shared state (set by main loop)
_bot_state = {
    "mode": "paper",
    "running": False,
    "paper_balance": 500.0,
    "live_balance": 0.0,
    "btc_price": 0.0,
    "eth_price": 0.0,
    "last_scan_time": None,
    "open_positions": [],
    "monitored_markets": [],
    "scan_count": 0,
    "start_time": datetime.now(timezone.utc).isoformat(),
    "errors": [],
}
_bot_ref = None


def set_bot_ref(bot) -> None:
    global _bot_ref
    _bot_ref = bot


def update_state(updates: dict) -> None:
    _bot_state.update(updates)


# ── API Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    state = _bot_state.copy()
    if _bot_ref:
        state["paper_balance"] = round(_bot_ref.executor.get_paper_balance(), 2)
        state["live_balance"] = round(_bot_ref.executor.get_live_balance(), 2)
        state["btc_price"] = round(_bot_ref.btc_price, 2)
        state["eth_price"] = round(_bot_ref.eth_price, 2)
    return jsonify(state)


@app.route("/api/positions")
def api_positions():
    if not _bot_ref:
        return jsonify([])
    positions = []
    for pos in _bot_ref.executor.open_positions.values():
        if pos.status == "open":
            positions.append({
                "trade_id": pos.trade_id,
                "asset": pos.asset,
                "outcome": pos.outcome,
                "question": pos.question[:80],
                "entry_price": round(pos.entry_price, 3),
                "current_price": round(pos.current_price, 3),
                "size_usd": round(pos.size_usd, 2),
                "posterior": round(pos.posterior_at_entry, 3),
                "market_price": round(pos.market_price_at_entry, 3),
                "divergence": round(pos.divergence_at_entry, 3),
                "unrealized_pnl": round(pos.unrealized_pnl(), 2),
                "unrealized_pct": round(pos.unrealized_pnl_pct() * 100, 1),
                "minutes_to_expiry": round(pos.minutes_to_expiry(), 1),
                "regime": pos.regime_at_entry,
                "session": pos.session_at_entry,
            })
    return jsonify(positions)


@app.route("/api/monitored_markets")
def api_monitored_markets():
    if not _bot_ref:
        return jsonify([])
    return jsonify(_bot_state.get("monitored_markets", []))


@app.route("/api/performance")
def api_performance():
    if not _bot_ref:
        return jsonify({})
    try:
        return jsonify(_bot_ref.tracker.get_stats_summary())
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/pnl_curve")
def api_pnl_curve():
    if not _bot_ref:
        return jsonify([])
    try:
        return jsonify(_bot_ref.tracker.get_pnl_curve())
    except Exception as e:
        return jsonify([])


@app.route("/api/calibration")
def api_calibration():
    if not _bot_ref:
        return jsonify({})
    try:
        data = _bot_ref.calibrator.get_calibration_history()
        calibration_data = _bot_ref.tracker.get_calibration_data()
        correlations = _bot_ref.calibrator.get_layer_correlations()
        last_date = _bot_ref.calibrator.get_last_calibration_date()
        weights = _bot_ref.calibrator.get_current_weights()
        return jsonify({
            "history": data[-5:] if data else [],
            "calibration_buckets": calibration_data,
            "layer_correlations": correlations,
            "current_weights": weights,
            "last_calibration": last_date,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/recent_trades")
def api_recent_trades():
    if not _bot_ref:
        return jsonify([])
    try:
        trades = _bot_ref.tracker.get_recent_trades(20)
        return jsonify(trades)
    except Exception as e:
        return jsonify([])


@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    settings_path = BOT_DIR / "settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            data = json.load(f)
        # Mask sensitive keys
        for key in ["polymarket_private_key", "polymarket_api_key", "polymarket_api_secret",
                    "polymarket_api_passphrase", "anthropic_api_key", "newsapi_key"]:
            if data.get(key):
                data[key] = "***configured***"
        return jsonify(data)
    return jsonify({})


@app.route("/api/settings", methods=["POST"])
def api_update_settings():
    """Update non-sensitive settings at runtime."""
    try:
        updates = request.json or {}
        safe_keys = [
            "min_divergence_threshold", "max_open_positions", "min_market_volume",
            "max_market_window_minutes", "min_market_window_minutes",
            "risk_per_trade_pct", "max_trade_size_usd", "stop_loss_pct",
            "evidence_weights"
        ]
        filtered = {k: v for k, v in updates.items() if k in safe_keys}

        if filtered and _bot_ref:
            cfg = _bot_ref.config
            for k, v in filtered.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        from config import save_settings
        save_settings(filtered)
        return jsonify({"success": True, "updated": list(filtered.keys())})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/mode", methods=["POST"])
def api_set_mode():
    """Switch between paper and live mode (requires explicit confirmation)."""
    data = request.json or {}
    new_mode = data.get("mode", "paper").lower()
    confirm = data.get("confirm", False)

    if new_mode == "live" and not confirm:
        return jsonify({
            "success": False,
            "error": "Live mode requires confirm=true in request body"
        })

    if _bot_ref:
        _bot_ref.config.mode = new_mode
        _bot_state["mode"] = new_mode
        from config import save_settings
        save_settings({"mode": new_mode})
        logger.info("Mode switched to: %s", new_mode)

    return jsonify({"success": True, "mode": new_mode})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "bot": "bayesian_market_bot"})


def run_dashboard(host: str = "0.0.0.0", port: int = 8082, debug: bool = False) -> threading.Thread:
    """Start the Flask dashboard in a background thread."""
    def _run():
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=_run, daemon=True, name="dashboard")
    thread.start()
    logger.info("Dashboard started at http://localhost:%d", port)
    return thread
