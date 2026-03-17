"""
bayesian_engine.py — Core Bayesian probability calculation engine.

Implements full Bayesian update formula:
    P(H|E) = [P(E|H) × P(H)] / P(E)

Sequential evidence updates converge to the posterior probability.
"""
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PRIORS_PATH = Path(__file__).parent / "historical_priors.json"


@dataclass
class EvidenceUpdate:
    """Result of one evidence layer's Bayesian update."""
    layer: str
    prior: float
    posterior: float
    delta: float
    description: str
    raw_signal: float = 0.0   # raw numeric signal before update


@dataclass
class BayesianResult:
    """Full Bayesian analysis result for a market."""
    hypothesis: str           # e.g. "BTC above $105k in 8min"
    asset: str                # "BTC" or "ETH"
    direction: str            # "up" or "down"
    prior: float              # base prior probability
    regime: str               # detected regime
    session: str              # time-of-day session
    evidence_updates: List[EvidenceUpdate] = field(default_factory=list)
    posterior: float = 0.50   # final posterior after all evidence
    market_price: float = 0.50
    divergence: float = 0.0   # posterior - market_price
    trade_signal: str = "HOLD"  # BUY_YES, BUY_NO, HOLD
    confidence_tier: str = "LOW"   # LOW, MEDIUM, HIGH

    def format_chain(self) -> str:
        """Human-readable evidence chain for dashboard."""
        parts = [f"Prior={self.prior:.2f}"]
        for upd in self.evidence_updates:
            sign = "+" if upd.delta >= 0 else ""
            parts.append(f"{upd.layer}{sign}{upd.delta:.2f}")
        parts.append(f"Final={self.posterior:.2f}")
        parts.append(f"Market={self.market_price:.2f}")
        sign = "+" if self.divergence >= 0 else ""
        parts.append(f"EDGE:{sign}{self.divergence:.2f}")
        parts.append(f"→ {self.trade_signal}")
        return " → ".join(parts)


class BayesianEngine:
    """
    Core Bayesian probability engine.

    Workflow:
      1. Calculate prior from historical data + regime + time of day
      2. Sequentially update prior with each evidence layer
      3. Compare posterior to market price
      4. Generate trade signal if divergence > threshold
    """

    def __init__(self, config):
        self.config = config
        self.priors = self._load_priors()

    def _load_priors(self) -> dict:
        if PRIORS_PATH.exists():
            try:
                with open(PRIORS_PATH) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load priors: %s", e)
        return self._default_priors()

    def _default_priors(self) -> dict:
        return {
            "base_rates": {
                "btc": {"1min": 0.50, "5min": 0.50, "15min": 0.50},
                "eth": {"1min": 0.50, "5min": 0.50, "15min": 0.50}
            },
            "regime_priors": {
                "trending_up": 0.62,
                "trending_down": 0.38,
                "ranging": 0.50,
                "high_volatility": 0.50
            },
            "time_of_day_adjustment": {
                "ny_open": {"center": 0.50, "spread_multiplier": 1.3},
                "london_open": {"center": 0.55, "spread_multiplier": 1.1},
                "dead_hours": {"center": 0.50, "spread_multiplier": 0.8},
                "normal": {"center": 0.50, "spread_multiplier": 1.0}
            },
            "trade_history_summary": {
                "total_resolved": 0,
                "last_updated": None
            }
        }

    def save_priors(self) -> None:
        try:
            with open(PRIORS_PATH, "w") as f:
                json.dump(self.priors, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save priors: %s", e)

    def detect_regime(self, price_data: List[float]) -> str:
        """
        Detect market regime from recent price data.
        Returns: trending_up, trending_down, ranging, high_volatility
        """
        if len(price_data) < 5:
            return "ranging"

        prices = price_data[-20:] if len(price_data) >= 20 else price_data
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
            if prices[i-1] != 0
        ]

        if not returns:
            return "ranging"

        # Volatility check
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        # ATR proxy
        if std_dev > 0.008:  # >0.8% std dev per period = high volatility
            return "high_volatility"

        # Trend detection: net return over window
        net_return = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0

        if net_return > 0.005:    # >0.5% net up = trending up
            return "trending_up"
        elif net_return < -0.005: # >0.5% net down = trending down
            return "trending_down"
        else:
            return "ranging"

    def get_session(self) -> str:
        """Determine current trading session based on UTC hour."""
        now_utc = datetime.now(timezone.utc)
        hour_est = (now_utc.hour - 5) % 24  # EST = UTC-5

        if 9 <= hour_est < 11:      # 9:30-11am EST NY open
            return "ny_open"
        elif 3 <= hour_est < 5:     # 3-5am EST London open
            return "london_open"
        elif 0 <= hour_est < 6:     # 12am-6am EST dead hours
            return "dead_hours"
        else:
            return "normal"

    def calculate_prior(
        self,
        asset: str,
        direction: str,
        timeframe_minutes: int,
        regime: str,
        price_data: Optional[List[float]] = None
    ) -> Tuple[float, str, str]:
        """
        Calculate base prior probability.

        Returns (prior, regime, session)
        """
        asset_lower = asset.lower()
        session = self.get_session()

        # Base rate from historical data
        base_rates = self.priors.get("base_rates", {})
        asset_rates = base_rates.get(asset_lower, {"1min": 0.50, "5min": 0.50, "15min": 0.50})

        if timeframe_minutes <= 1:
            base_rate = asset_rates.get("1min", 0.50)
        elif timeframe_minutes <= 5:
            base_rate = asset_rates.get("5min", 0.50)
        else:
            base_rate = asset_rates.get("15min", 0.50)

        # Adjust for regime
        regime_priors = self.priors.get("regime_priors", {})
        regime_prior = regime_priors.get(regime, 0.50)

        # Blend base rate with regime prior (60% base, 40% regime)
        blended = 0.60 * base_rate + 0.40 * regime_prior

        # Adjust for direction (if hypothesis is "down", invert)
        if direction == "down":
            blended = 1.0 - blended

        # Time of day adjustment
        tod_adjustments = self.priors.get("time_of_day_adjustment", {})
        tod = tod_adjustments.get(session, {"center": 0.50, "spread_multiplier": 1.0})
        tod_center = tod.get("center", 0.50)
        spread_mult = tod.get("spread_multiplier", 1.0)

        # Pull prior toward time-of-day center (dead hours → 0.50)
        if session == "dead_hours":
            blended = 0.70 * 0.50 + 0.30 * blended  # strong pull to 0.50
        elif session == "london_open" and direction == "up":
            blended = blended + 0.02 * spread_mult  # slight London momentum boost

        # Clamp to [0.10, 0.90]
        prior = max(0.10, min(0.90, blended))

        logger.debug(
            "Prior for %s %s %dmin: base=%.3f regime=%s(%s=%.3f) session=%s → prior=%.3f",
            asset, direction, timeframe_minutes, base_rate, regime, regime_prior, blended,
            session, prior
        )

        return prior, regime, session

    # ── Bayesian update formula ───────────────────────────────────────────────

    @staticmethod
    def bayesian_update(prior: float, likelihood_ratio: float) -> float:
        """
        Apply Bayes' theorem using likelihood ratio:
            posterior_odds = prior_odds * likelihood_ratio
            posterior = posterior_odds / (1 + posterior_odds)

        likelihood_ratio = P(E|H) / P(E|~H)
        """
        if prior <= 0:
            prior = 0.001
        if prior >= 1:
            prior = 0.999

        prior_odds = prior / (1.0 - prior)
        posterior_odds = prior_odds * max(0.001, likelihood_ratio)
        posterior = posterior_odds / (1.0 + posterior_odds)
        return max(0.01, min(0.99, posterior))

    @staticmethod
    def delta_to_likelihood(delta: float) -> float:
        """
        Convert a probability delta to a likelihood ratio.
        delta = desired change in probability
        """
        # A delta of +0.12 at prior=0.50 means posterior=0.62
        # LR = posterior_odds / prior_odds
        # At prior=0.50: prior_odds=1, posterior_odds = 0.62/0.38 ≈ 1.63 → LR≈1.63
        # We approximate: LR = exp(delta * 3) for small deltas
        return math.exp(delta * 3.0)

    def apply_evidence_update(
        self,
        current_prob: float,
        delta: float,
        weight: float,
        layer_name: str,
        description: str,
        raw_signal: float = 0.0
    ) -> EvidenceUpdate:
        """Apply a weighted evidence update to current probability."""
        weighted_delta = delta * weight
        new_prob = max(0.01, min(0.99, current_prob + weighted_delta))
        actual_delta = new_prob - current_prob

        return EvidenceUpdate(
            layer=layer_name,
            prior=current_prob,
            posterior=new_prob,
            delta=actual_delta,
            description=description,
            raw_signal=raw_signal
        )

    def run_full_analysis(
        self,
        asset: str,
        direction: str,
        timeframe_minutes: int,
        market_price: float,
        price_data: List[float],
        evidence: dict,
        market_question: str = ""
    ) -> BayesianResult:
        """
        Run full Bayesian analysis with all evidence layers.

        evidence dict keys:
          - order_flow: dict with buy_pressure, std_devs
          - price_momentum: dict with ret_1min, ret_5min, ret_15min, atr_ratio
          - social_buzz: dict with volume_ratio, sentiment_score
          - on_chain: dict with mempool_count, volume_ratio
          - market_sentiment: dict with confluence_signal, confluence_direction
        """
        weights = self.config.evidence_weights

        # Step 1: Calculate prior
        regime = self.detect_regime(price_data)
        prior, regime, session = self.calculate_prior(
            asset, direction, timeframe_minutes, regime, price_data
        )

        result = BayesianResult(
            hypothesis=market_question or f"{asset} above target in {timeframe_minutes}min",
            asset=asset,
            direction=direction,
            prior=prior,
            regime=regime,
            session=session,
            market_price=market_price,
        )

        running_prob = prior

        # Step 2: Order flow evidence (highest weight)
        of = evidence.get("order_flow", {})
        of_std = of.get("std_devs", 0.0)
        if of_std > 2.0:
            delta = 0.12 if direction == "up" else -0.12
            if of.get("buy_pressure", 0) < 0:
                delta = -delta
        elif of_std > 1.0:
            delta = 0.06 if direction == "up" else -0.06
            if of.get("buy_pressure", 0) < 0:
                delta = -delta
        elif of_std < -2.0:
            delta = -0.12 if direction == "up" else 0.12
        elif of_std < -1.0:
            delta = -0.06 if direction == "up" else 0.06
        else:
            delta = 0.0

        upd = self.apply_evidence_update(
            running_prob, delta, weights.get("order_flow", 1.0),
            "OrderFlow",
            f"net_std={of_std:.2f}",
            raw_signal=of_std
        )
        result.evidence_updates.append(upd)
        running_prob = upd.posterior

        # Step 3: Price momentum evidence
        pm = evidence.get("price_momentum", {})
        atr_ratio = pm.get("atr_ratio", 1.0)
        volatility_dampen = 0.5 if atr_ratio > 1.5 else 1.0

        ret_5min = pm.get("ret_5min", 0.0)
        if abs(ret_5min) > 0.003:      # >0.3% strong momentum
            delta = 0.08 * (1 if ret_5min > 0 else -1)
        elif abs(ret_5min) > 0.001:    # >0.1% weak momentum
            delta = 0.04 * (1 if ret_5min > 0 else -1)
        else:
            delta = 0.0

        if direction == "down":
            delta = -delta

        delta *= volatility_dampen

        upd = self.apply_evidence_update(
            running_prob, delta, weights.get("price_momentum", 0.8),
            "Momentum",
            f"ret5m={ret_5min*100:.2f}% atr={atr_ratio:.2f}x",
            raw_signal=ret_5min
        )
        result.evidence_updates.append(upd)
        running_prob = upd.posterior

        # Step 4: Social buzz evidence
        sb = evidence.get("social_buzz", {})
        vol_ratio = sb.get("volume_ratio", 1.0)
        sentiment = sb.get("sentiment_score", 0.0)

        # High volume + any sentiment, or strong sentiment on its own
        if vol_ratio > 1.5 and abs(sentiment) > 0.1:
            delta = 0.06 * (1 if sentiment > 0 else -1)
        elif abs(sentiment) > 0.25:   # strong directional signal even without vol spike
            delta = 0.03 * (1 if sentiment > 0 else -1)
        else:
            delta = 0.0

        if direction == "down":
            delta = -delta

        upd = self.apply_evidence_update(
            running_prob, delta, weights.get("social_buzz", 0.5),
            "Social",
            f"vol_ratio={vol_ratio:.1f}x sentiment={sentiment:.2f}",
            raw_signal=vol_ratio * sentiment
        )
        result.evidence_updates.append(upd)
        running_prob = upd.posterior

        # Step 5: On-chain signals
        oc = evidence.get("on_chain", {})
        mempool = oc.get("mempool_count", 0)
        vol_spike = oc.get("volume_ratio", 1.0)

        on_chain_delta = 0.0
        # Normalize mempool against typical baseline (~15K tx) rather than hard 50K threshold
        mempool_pressure = mempool / 15000.0 if mempool > 0 else 0.0
        if mempool_pressure > 3.0:       # very congested: 3× baseline
            on_chain_delta += 0.04
        elif mempool_pressure > 1.5:     # above average: 1.5× baseline
            on_chain_delta += 0.02
        if vol_spike > 1.5:              # volume 1.5× average (was 2.0)
            on_chain_delta += 0.05

        if direction == "down":
            on_chain_delta = -on_chain_delta

        upd = self.apply_evidence_update(
            running_prob, on_chain_delta, weights.get("on_chain", 0.4),
            "OnChain",
            f"mempool={mempool} vol_spike={vol_spike:.1f}x",
            raw_signal=on_chain_delta
        )
        result.evidence_updates.append(upd)
        running_prob = upd.posterior

        # Step 6: Market sentiment / confluence
        ms = evidence.get("market_sentiment", {})
        conf_signal = ms.get("confluence_signal", 0.0)
        conf_dir = ms.get("confluence_direction", "neutral")

        if conf_dir == direction and conf_signal > 0:
            ms_delta = 0.05
        elif conf_dir != direction and conf_dir != "neutral":
            ms_delta = -0.03
        else:
            ms_delta = 0.0

        upd = self.apply_evidence_update(
            running_prob, ms_delta, weights.get("market_sentiment", 0.3),
            "MarketSentiment",
            f"confluence={conf_dir} signal={conf_signal:.2f}",
            raw_signal=conf_signal
        )
        result.evidence_updates.append(upd)
        running_prob = upd.posterior

        # Final posterior
        result.posterior = round(running_prob, 4)
        result.divergence = round(result.posterior - market_price, 4)

        # Trade signal
        threshold = self.config.min_divergence_threshold
        if result.divergence > threshold:
            result.trade_signal = "BUY_YES"
        elif result.divergence < -threshold:
            result.trade_signal = "BUY_NO"
        else:
            result.trade_signal = "HOLD"

        # Confidence tier
        abs_div = abs(result.divergence)
        if abs_div > 0.20:
            result.confidence_tier = "HIGH"
        elif abs_div > 0.12:
            result.confidence_tier = "MEDIUM"
        else:
            result.confidence_tier = "LOW"

        logger.info(
            "Bayesian[%s %s %dmin]: %s",
            asset, direction, timeframe_minutes, result.format_chain()
        )

        return result

    def update_priors_after_trade(
        self,
        asset: str,
        direction: str,
        timeframe_minutes: int,
        posterior_was: float,
        outcome_yes: bool
    ) -> None:
        """
        Update historical base rates after a resolved trade.
        Exponential moving average update: new_rate = 0.95 * old + 0.05 * outcome
        """
        asset_lower = asset.lower()
        if asset_lower not in self.priors["base_rates"]:
            self.priors["base_rates"][asset_lower] = {"1min": 0.50, "5min": 0.50, "15min": 0.50}

        if timeframe_minutes <= 1:
            key = "1min"
        elif timeframe_minutes <= 5:
            key = "5min"
        else:
            key = "15min"

        outcome = 1.0 if outcome_yes else 0.0
        old = self.priors["base_rates"][asset_lower][key]
        new = 0.95 * old + 0.05 * outcome
        self.priors["base_rates"][asset_lower][key] = round(new, 4)

        self.priors["trade_history_summary"]["total_resolved"] = (
            self.priors["trade_history_summary"].get("total_resolved", 0) + 1
        )
        self.priors["trade_history_summary"]["last_updated"] = (
            datetime.now(timezone.utc).isoformat()
        )

        self.save_priors()
        logger.debug(
            "Updated %s %s prior: %.4f → %.4f (outcome=%s)",
            asset, key, old, new, "YES" if outcome_yes else "NO"
        )
