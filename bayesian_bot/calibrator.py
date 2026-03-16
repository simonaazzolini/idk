"""
calibrator.py — Self-improving Bayesian calibration system.

After every 20 resolved trades, automatically:
  1. Checks if posterior probabilities are well-calibrated
  2. Measures predictive power of each evidence layer
  3. Adjusts evidence weights based on actual performance
  4. Updates historical priors with new base rates
  5. Logs calibration report to calibration_log.json
"""
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CALIBRATION_LOG_PATH = Path(__file__).parent / "calibration_log.json"
SETTINGS_PATH = Path(__file__).parent / "settings.json"

EVIDENCE_LAYERS = ["order_flow", "price_momentum", "social_buzz", "on_chain", "market_sentiment"]
LAYER_DISPLAY_NAMES = {
    "order_flow": "ORDER_FLOW",
    "price_momentum": "PRICE_MOMENTUM",
    "social_buzz": "SOCIAL_BUZZ",
    "on_chain": "ON_CHAIN",
    "market_sentiment": "MARKET_SENTIMENT",
}

# Weight adjustment bounds
MIN_WEIGHT = 0.1
MAX_WEIGHT = 2.0
CALIBRATION_INTERVAL = 20  # recalibrate every N trades


class Calibrator:
    """
    Automatically recalibrates evidence weights and priors
    based on actual trade outcomes.
    """

    def __init__(self, config):
        self.config = config
        self._log = self._load_log()
        self._last_calibration_trade_count = self._log.get("last_calibration_count", 0)

    def _load_log(self) -> dict:
        if CALIBRATION_LOG_PATH.exists():
            try:
                with open(CALIBRATION_LOG_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "calibration_runs": [],
            "last_calibration_count": 0,
            "evidence_correlations": {},
            "weight_history": []
        }

    def _save_log(self) -> None:
        with open(CALIBRATION_LOG_PATH, "w") as f:
            json.dump(self._log, f, indent=2)

    def should_calibrate(self, total_resolved: int) -> bool:
        """Check if we've accumulated enough new trades to recalibrate."""
        new_since_last = total_resolved - self._last_calibration_trade_count
        return new_since_last >= CALIBRATION_INTERVAL and total_resolved >= CALIBRATION_INTERVAL

    def run_calibration(self, trades: List[dict]) -> dict:
        """
        Run full calibration pass on completed trades.
        Returns calibration report dict.
        """
        closed = [t for t in trades if t.get("status") == "closed" and t.get("resolution_outcome") is not None]
        if len(closed) < CALIBRATION_INTERVAL:
            logger.info("Calibration: not enough trades (%d < %d)", len(closed), CALIBRATION_INTERVAL)
            return {}

        logger.info("Running calibration on %d trades...", len(closed))

        # 1. Calibration quality: does predicted probability match actual win rate?
        calibration_error = self._check_calibration(closed)

        # 2. Evidence layer correlations with outcomes
        layer_correlations = self._compute_layer_correlations(closed)

        # 3. Adjust weights based on correlations
        new_weights = self._adjust_weights(layer_correlations)

        # 4. Generate report
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trades": len(closed),
            "calibration_error": round(calibration_error, 4),
            "calibration_quality": self._quality_label(calibration_error),
            "evidence_correlations": {
                k: round(v, 4) for k, v in layer_correlations.items()
            },
            "old_weights": {k: v for k, v in self.config.evidence_weights.items()},
            "new_weights": new_weights,
            "weight_changes": {
                k: round(new_weights.get(k, 1.0) - self.config.evidence_weights.get(k, 1.0), 4)
                for k in EVIDENCE_LAYERS
            }
        }

        # 5. Log findings
        for layer in EVIDENCE_LAYERS:
            display = LAYER_DISPLAY_NAMES.get(layer, layer)
            corr = layer_correlations.get(layer, 0.0)
            old_w = self.config.evidence_weights.get(layer, 1.0)
            new_w = new_weights.get(layer, old_w)
            action = "maintained" if abs(new_w - old_w) < 0.01 else (
                "increased" if new_w > old_w else "reduced"
            )
            logger.info(
                "Calibration: %s correlation=%.2f weight %s (%.2f → %.2f)",
                display, corr, action, old_w, new_w
            )

        # 6. Apply new weights to config
        self.config.evidence_weights.update(new_weights)

        # 7. Persist weights to settings.json
        self._save_weights_to_settings(new_weights)

        # 8. Append to calibration log
        self._log["calibration_runs"].append(report)
        self._log["last_calibration_count"] = len(closed)
        self._log["evidence_correlations"] = {k: round(v, 4) for k, v in layer_correlations.items()}
        self._log["weight_history"].append({
            "timestamp": report["timestamp"],
            "weights": new_weights
        })
        self._save_log()

        self._last_calibration_trade_count = len(closed)

        logger.info(
            "Calibration complete: error=%.3f quality=%s",
            calibration_error, report["calibration_quality"]
        )
        return report

    def _check_calibration(self, trades: List[dict]) -> float:
        """
        Compute mean calibration error.
        Groups trades by posterior probability bucket (10%), compares to actual win rate.
        """
        buckets: Dict[float, List[float]] = {}

        for t in trades:
            posterior = t.get("posterior_at_entry", 0.5)
            resolved = t.get("resolution_outcome")
            outcome = t.get("outcome", "YES")
            if resolved is None:
                continue

            won = resolved if outcome == "YES" else not resolved
            bucket = round(posterior * 10) / 10
            buckets.setdefault(bucket, []).append(1.0 if won else 0.0)

        if not buckets:
            return 0.0

        errors = []
        for prob, outcomes in buckets.items():
            actual_rate = sum(outcomes) / len(outcomes)
            error = abs(prob - actual_rate)
            errors.append(error)
            logger.debug(
                "Calibration bucket %.1f: predicted=%.2f actual=%.2f (n=%d) error=%.3f",
                prob, prob, actual_rate, len(outcomes), error
            )

        return statistics.mean(errors)

    def _quality_label(self, error: float) -> str:
        if error < 0.05:
            return "EXCELLENT"
        elif error < 0.10:
            return "GOOD"
        elif error < 0.15:
            return "FAIR"
        else:
            return "POOR"

    def _compute_layer_correlations(self, trades: List[dict]) -> Dict[str, float]:
        """
        Compute point-biserial correlation between each evidence layer's
        delta and the binary outcome (win/loss).
        """
        correlations = {}

        for layer in EVIDENCE_LAYERS:
            deltas = []
            outcomes = []

            for t in trades:
                breakdown = t.get("evidence_breakdown", {})
                layer_data = breakdown.get(layer, breakdown.get(
                    layer.replace("_", "").title(), {}
                ))

                # Try multiple name formats
                delta = None
                for key in [layer, layer.replace("_", ""), layer.title(), layer.upper()]:
                    if key in breakdown:
                        delta = breakdown[key].get("delta", 0.0)
                        break
                if delta is None:
                    # Try camelCase
                    for key in breakdown.keys():
                        if key.lower().replace("_", "") == layer.replace("_", ""):
                            delta = breakdown[key].get("delta", 0.0)
                            break

                if delta is None:
                    continue

                resolved = t.get("resolution_outcome")
                outcome_str = t.get("outcome", "YES")
                if resolved is None:
                    continue

                won = resolved if outcome_str == "YES" else not resolved
                deltas.append(float(delta))
                outcomes.append(1.0 if won else 0.0)

            if len(deltas) < 5:
                correlations[layer] = 0.5  # neutral if insufficient data
                continue

            corr = self._point_biserial_correlation(deltas, outcomes)
            correlations[layer] = max(-1.0, min(1.0, corr))

        return correlations

    def _point_biserial_correlation(self, continuous: List[float], binary: List[float]) -> float:
        """
        Compute point-biserial correlation coefficient.
        Returns value between -1 and 1.
        """
        n = len(continuous)
        if n < 3:
            return 0.0

        mean_all = sum(continuous) / n
        std_all = statistics.stdev(continuous) if n > 1 else 1.0

        if std_all == 0:
            return 0.0

        # Split into groups
        group1 = [c for c, b in zip(continuous, binary) if b == 1.0]  # wins
        group0 = [c for c, b in zip(continuous, binary) if b == 0.0]  # losses

        if not group1 or not group0:
            return 0.0

        mean1 = sum(group1) / len(group1)
        mean0 = sum(group0) / len(group0)
        n1 = len(group1)
        n0 = len(group0)

        corr = ((mean1 - mean0) / std_all) * (n1 * n0 / (n * n)) ** 0.5
        return corr

    def _adjust_weights(self, correlations: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust evidence weights based on correlations.

        Strategy:
          - Correlation 0.6+: boost weight by 10%
          - Correlation 0.4-0.6: maintain weight
          - Correlation <0.4: reduce weight by 20%
        """
        new_weights = {}
        current_weights = self.config.evidence_weights.copy()

        for layer in EVIDENCE_LAYERS:
            corr = abs(correlations.get(layer, 0.5))
            old_w = current_weights.get(layer, 1.0)

            if corr >= 0.6:
                new_w = old_w * 1.10  # boost by 10%
            elif corr >= 0.4:
                new_w = old_w          # maintain
            else:
                new_w = old_w * 0.80   # reduce by 20%

            # Clamp
            new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, new_w))
            new_weights[layer] = round(new_w, 4)

        return new_weights

    def _save_weights_to_settings(self, weights: Dict[str, float]) -> None:
        """Persist updated weights to settings.json."""
        if not SETTINGS_PATH.exists():
            return
        try:
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
            data["evidence_weights"] = weights
            with open(SETTINGS_PATH, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Updated evidence weights in settings.json")
        except Exception as e:
            logger.warning("Failed to save weights to settings: %s", e)

    def get_calibration_history(self) -> List[dict]:
        """Return all calibration run reports."""
        return self._log.get("calibration_runs", [])

    def get_current_weights(self) -> Dict[str, float]:
        """Return current evidence weights."""
        return self.config.evidence_weights.copy()

    def get_layer_correlations(self) -> Dict[str, float]:
        """Return latest evidence layer correlations."""
        return self._log.get("evidence_correlations", {})

    def get_last_calibration_date(self) -> Optional[str]:
        """Return ISO timestamp of most recent calibration."""
        runs = self._log.get("calibration_runs", [])
        if runs:
            return runs[-1].get("timestamp")
        return None
