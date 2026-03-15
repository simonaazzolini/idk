"""
AI Probability Engine — Module 6.
Uses Claude with web search to estimate true outcome probabilities.

Rate-limit strategy:
  - Default model: claude-haiku-4-5-20251001  (fast, cheap, all cycle analysis)
  - Deep model:    claude-opus-4-5             (live mode only, composite_score >= 8.0)
  - Minimum 8 s between every call (ai_min_call_interval_seconds)
  - Hard cap: 8 calls per 60-second window (ai_max_calls_per_minute)
  - On 429: wait 60 s, retry up to 2 times, then skip
  - Max 10 markets per cycle
  - max_tokens = 1000
"""
import asyncio
import collections
import json
import logging
import re
import time
from typing import Optional

import anthropic

from config.settings import Settings
from data.cache import MarketDataCache
from utils.helpers import clamp, now_ts

logger = logging.getLogger(__name__)

_RETRY_429_WAIT  = 60    # seconds to wait after a 429
_MAX_429_RETRIES = 2     # max retries before skipping the market
_TOKEN_LIMIT     = 20000 # estimated token threshold for prompt truncation


class _BillingError(Exception):
    """Raised internally when Anthropic returns a billing/credit error."""


# ─────────────────────────────────────────────────────────────────────────────
# Prompts  (trimmed ~30% vs original — all analytical requirements kept)
# ─────────────────────────────────────────────────────────────────────────────

AI_SYSTEM_PROMPT = """Prediction market analyst. Estimate true outcome probabilities as a superforecaster.

Rules:
1. Run 3–6 web searches; find a base rate FIRST, then update with specific evidence
2. Check Metaculus, Manifold, PredictIt for community consensus
3. Consider exact resolution criteria (not just general topic)
4. Calibrate: reduce confidence when uncertain; apply Bayesian updates
5. Return ONLY valid JSON — no markdown, no text outside JSON"""


AI_ANALYSIS_PROMPT = """Analyze this prediction market and estimate the true probability.

QUESTION: {question}
YES PRICE: {current_price:.4f}  |  DAYS TO RESOLUTION: {days_to_resolution:.1f}
CATEGORY: {category}  |  RESOLUTION SOURCE: {resolution_source}
DESCRIPTION: {description}

SIGNALS: vol_24h=${volume_24h:,.0f}  liquidity=${liquidity:,.0f}  \
price_chg={price_change_24h:+.2%}  RSI={rsi_14:.0f}  \
whales={whale_direction}  news={news_sentiment}
NEWS SUMMARY: {news_summary}

SEARCHES TO RUN:
1. Exact question text: "{question}"
2. Base rate for this type of event
3. Recent news and expert forecasts
4. Aggregators: site:metaculus.com OR site:manifold.markets "{topic_keywords}"
5. Resolution source: {resolution_source}
6. Contra-evidence: reasons this does NOT happen

Return ONLY this JSON:
{{
  "yes_probability": <float 0–1>,
  "confidence_interval_low": <float>,
  "confidence_interval_high": <float>,
  "confidence": <float 0–1>,
  "recommended_outcome": <"YES"|"NO"|"SKIP">,
  "base_rate": <float>,
  "base_rate_source": <string>,
  "evidence_adjustment": <float>,
  "reasoning": "<3–5 sentences>",
  "strongest_yes_evidence": "<string>",
  "strongest_no_evidence": "<string>",
  "key_facts": [<strings>],
  "key_uncertainties": [<strings>],
  "resolution_risk": <"LOW"|"MEDIUM"|"HIGH">,
  "aggregator_consensus": <float or null>,
  "news_summary": "<string>",
  "edge": <yes_probability minus current_yes_price>,
  "signal_strength": <"WEAK"|"MODERATE"|"STRONG"|"VERY_STRONG">
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

class AIAnalysisResult:
    """Typed result from the AI probability engine."""

    def __init__(self, data: dict):
        self.yes_probability: float          = float(data.get("yes_probability", 0.5))
        self.confidence_interval_low: float  = float(data.get("confidence_interval_low", 0.3))
        self.confidence_interval_high: float = float(data.get("confidence_interval_high", 0.7))
        self.confidence: float               = float(data.get("confidence", 0.5))
        self.recommended_outcome: str        = data.get("recommended_outcome", "SKIP")
        self.base_rate: float                = float(data.get("base_rate", 0.5))
        self.base_rate_source: str           = data.get("base_rate_source", "unknown")
        self.evidence_adjustment: float      = float(data.get("evidence_adjustment", 0.0))
        self.reasoning: str                  = data.get("reasoning", "")
        self.strongest_yes_evidence: str     = data.get("strongest_yes_evidence", "")
        self.strongest_no_evidence: str      = data.get("strongest_no_evidence", "")
        self.key_facts: list                 = data.get("key_facts", [])
        self.key_uncertainties: list         = data.get("key_uncertainties", [])
        self.resolution_risk: str            = data.get("resolution_risk", "MEDIUM")
        self.aggregator_consensus: Optional[float] = data.get("aggregator_consensus")
        self.news_summary: str               = data.get("news_summary", "")
        self.edge: float                     = float(data.get("edge", 0.0))
        self.signal_strength: str            = data.get("signal_strength", "WEAK")
        self.ai_score: float                 = 5.0   # set after construction
        self.raw_data                        = data

    def to_dict(self) -> dict:
        return {
            "yes_probability":          self.yes_probability,
            "confidence_interval_low":  self.confidence_interval_low,
            "confidence_interval_high": self.confidence_interval_high,
            "confidence":               self.confidence,
            "recommended_outcome":      self.recommended_outcome,
            "base_rate":                self.base_rate,
            "base_rate_source":         self.base_rate_source,
            "evidence_adjustment":      self.evidence_adjustment,
            "reasoning":                self.reasoning,
            "strongest_yes_evidence":   self.strongest_yes_evidence,
            "strongest_no_evidence":    self.strongest_no_evidence,
            "key_facts":                self.key_facts,
            "key_uncertainties":        self.key_uncertainties,
            "resolution_risk":          self.resolution_risk,
            "aggregator_consensus":     self.aggregator_consensus,
            "news_summary":             self.news_summary,
            "edge":                     self.edge,
            "signal_strength":          self.signal_strength,
            "ai_score":                 self.ai_score,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Score helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_signal_strength(edge: float, confidence: float) -> str:
    abs_edge = abs(edge)
    if abs_edge > 0.12 and confidence > 0.75:
        return "VERY_STRONG"
    if abs_edge > 0.08 and confidence > 0.65:
        return "STRONG"
    if abs_edge > 0.04 and confidence > 0.55:
        return "MODERATE"
    return "WEAK"


def compute_ai_score(result: AIAnalysisResult) -> float:
    edge_score       = min(abs(result.edge) / 0.15 * 10.0, 10.0)
    confidence_score = result.confidence * 10.0
    return (edge_score * 0.7) + (confidence_score * 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class AIAnalyzer:
    """
    AI Probability Engine.

    Model routing:
        Regular cycle analysis → settings.ai_model_analysis  (Haiku, fast+cheap)
        Live trade confirmation → settings.ai_model_deep      (Opus, only when
            mode == 'LIVE' AND composite_score >= 8.0)

    Rate limiting (enforced before every call):
        - Minimum gap:  settings.ai_min_call_interval_seconds (8 s default)
        - Window cap:   settings.ai_max_calls_per_minute calls per 60 s window
        - 429 handling: wait 60 s, retry ≤ 2 times, then skip market

    Billing / credit handling:
        - On first billing/credit error: set ai_available=False
        - All subsequent calls return None immediately (no retries)
        - Flag is reset to True at the start of each new cycle
    """

    def __init__(self, settings: Settings, cache: MarketDataCache):
        self.settings  = settings
        self.cache     = cache
        self._client:  Optional[anthropic.Anthropic] = None
        self._last_call_ts: float = 0.0
        self._total_calls:  int   = 0
        self._call_errors:  int   = 0
        # Sliding-window call timestamps for per-minute rate limiting
        self._call_timestamps: collections.deque = collections.deque()
        # Set to False when billing error is detected; reset each cycle
        self.ai_available: bool = True

    def reset_for_cycle(self) -> None:
        """Call at the start of each bot cycle to re-enable AI after a billing error."""
        if not self.ai_available:
            logger.info("AI availability flag reset for new cycle — will attempt one call.")
        self.ai_available = True

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    # ── Public analysis entry points ──────────────────────────────────────────

    async def analyze_market(
        self,
        market: dict,
        technical_signal=None,
        whale_consensus: Optional[dict] = None,
        news_result:     Optional[dict] = None,
        mode:            str   = "PAPER",
        composite_score: float = 0.0,
    ) -> Optional[AIAnalysisResult]:
        """
        Run AI analysis on a single market.

        Model selection:
            - mode == 'LIVE' AND composite_score >= 8.0 → deep model (Opus)
            - all other cases                           → analysis model (Haiku)

        Returns AIAnalysisResult or None when analysis should be skipped.
        """
        slug = market.get("slug") or market.get("conditionId", "unknown")

        # Skip immediately if billing error was already hit this cycle
        if not self.ai_available:
            logger.debug("AI unavailable (billing error this cycle) — skipping %s", slug[:30])
            return None

        # Cache check
        cached = await self.cache.get_ai_analysis(slug)
        if cached:
            result = AIAnalysisResult(cached)
            result.ai_score = compute_ai_score(result)
            return result

        # Model selection
        use_deep_model = (mode.upper() == "LIVE" and composite_score >= 8.0)
        model = (
            self.settings.ai_model_deep
            if use_deep_model
            else self.settings.ai_model_analysis
        )
        if use_deep_model:
            logger.info(
                "Using DEEP model (%s) for %s (live mode, score=%.1f)",
                model, slug[:30], composite_score,
            )

        # Extract market fields
        question = market.get("question", "Unknown")
        raw_prices = market.get("outcomePrices", [0.5])
        current_price = float(
            raw_prices[0] if isinstance(raw_prices, list) and raw_prices else 0.5
        )
        days_to_resolution  = float(market.get("days_to_resolution") or 30.0)
        category            = str(market.get("category") or "general")
        description         = str(market.get("description") or "")[:800]
        resolution_source   = str(
            market.get("resolutionSource") or market.get("resolution_source") or "Not specified"
        )
        volume_24h   = float(market.get("volume24hr") or market.get("volume_24h") or 0)
        liquidity    = float(market.get("liquidity") or 0)
        price_chg_24 = float(technical_signal.price_change_24h if technical_signal else 0.0)
        rsi          = float(technical_signal.rsi_14 if technical_signal else 50.0)
        whale_dir    = (
            whale_consensus.get("smart_money_net_direction", "NEUTRAL")
            if whale_consensus else "NEUTRAL"
        )
        news_sent = str(
            news_result.get("consensus_direction", "UNCERTAIN") if news_result else "UNKNOWN"
        )
        news_summary = str(
            (news_result.get("news_summary") or "")[:300] if news_result else ""
        )
        topic_keywords = " ".join(w for w in question.split() if len(w) > 3)[:50]

        # Rate limit enforcement
        await self._enforce_rate_limit()

        try:
            raw_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._run_analysis(
                    question=question,
                    current_price=current_price,
                    days_to_resolution=days_to_resolution,
                    category=category,
                    description=description,
                    resolution_source=resolution_source,
                    volume_24h=volume_24h,
                    liquidity=liquidity,
                    price_change_24h=price_chg_24,
                    rsi_14=rsi,
                    whale_direction=whale_dir,
                    news_sentiment=news_sent,
                    news_summary=news_summary,
                    topic_keywords=topic_keywords,
                    model=model,
                ),
            )
        except _BillingError:
            self.ai_available = False
            self._call_errors += 1
            logger.error(
                "AI disabled for remainder of this cycle (billing/credit error). "
                "Will retry next cycle."
            )
            return None
        except Exception as e:
            self._call_errors += 1
            logger.warning("AI analysis failed for %s: %s", slug, e)
            return None

        if raw_result is None:
            return None

        # Post-process
        raw_result["edge"] = raw_result.get("yes_probability", current_price) - current_price
        raw_result["signal_strength"] = compute_signal_strength(
            raw_result["edge"], raw_result.get("confidence", 0.5)
        )

        result          = AIAnalysisResult(raw_result)
        result.ai_score = compute_ai_score(result)

        await self.cache.set_ai_analysis(slug, result.to_dict(), ttl_seconds=900)

        tag = "DEEP" if use_deep_model else "HAIKU"
        action_hint = ""
        if abs(result.edge) < 0.01:
            action_hint = " → BELOW_THRESHOLD (edge<0.01)"
        elif abs(result.edge) < 0.02:
            action_hint = " → WEAK_SIGNAL"
        logger.info(
            "AI [%s] %s | market_price=%.3f P(YES)=%.3f edge=%+.3f conf=%.2f "
            "signal=%s outcome=%s%s",
            tag, slug[:30], current_price, result.yes_probability, result.edge,
            result.confidence, result.signal_strength, result.recommended_outcome,
            action_hint,
        )
        if result.reasoning:
            logger.debug("AI reasoning [%s]: %s", slug[:20], result.reasoning[:200])
        return result

    async def analyze_batch(
        self,
        markets: list[dict],
        max_markets: int = 10,           # cap: 10 markets per cycle
        technical_signals:  Optional[dict] = None,
        whale_consensuses:  Optional[dict] = None,
        news_results:       Optional[dict] = None,
        mode:               str   = "PAPER",
        composite_scores:   Optional[dict] = None,
    ) -> dict[str, Optional[AIAnalysisResult]]:
        """
        Analyze up to max_markets markets sequentially (concurrency = 1).
        Per-market errors are isolated — one failure does not stop the batch.
        """
        results: dict[str, Optional[AIAnalysisResult]] = {}
        for market in markets[:max_markets]:
            slug  = market.get("slug") or market.get("conditionId", "unknown")
            tech  = technical_signals.get(slug)  if technical_signals  else None
            whale = whale_consensuses.get(slug)  if whale_consensuses  else None
            news  = news_results.get(slug)        if news_results       else None
            cscore = float(
                composite_scores.get(slug, 0.0) if composite_scores else 0.0
            )
            try:
                result = await self.analyze_market(
                    market, tech, whale, news,
                    mode=mode, composite_score=cscore,
                )
                results[slug] = result
            except Exception as e:
                logger.warning("AI analysis error for %s: %s", slug, e)
                results[slug] = None
        return results

    # ── Synchronous Claude call (runs in thread pool) ─────────────────────────

    def _run_analysis(
        self,
        question: str,
        current_price: float,
        days_to_resolution: float,
        category: str,
        description: str,
        resolution_source: str,
        volume_24h: float,
        liquidity: float,
        price_change_24h: float,
        rsi_14: float,
        whale_direction: str,
        news_sentiment: str,
        news_summary: str,
        topic_keywords: str,
        model: str,
    ) -> Optional[dict]:
        """
        Synchronous Claude API call with web search.

        Token estimation:
            est = len(prompt) / 4
            If est > 20 000: truncate description → 500 chars, news_summary → 300 chars.

        429 handling:
            On RateLimitError: sleep 60 s, retry up to 2 times.
            If still failing after retries: return None (market skipped).
        """
        client = self._get_client()
        self._last_call_ts = now_ts()
        self._total_calls += 1

        def _build_prompt(desc: str, ns: str) -> str:
            return AI_ANALYSIS_PROMPT.format(
                question=question,
                current_price=current_price,
                days_to_resolution=days_to_resolution,
                category=category,
                description=desc,
                resolution_source=resolution_source,
                volume_24h=volume_24h,
                liquidity=liquidity,
                price_change_24h=price_change_24h,
                rsi_14=rsi_14,
                whale_direction=whale_direction,
                news_sentiment=news_sentiment,
                news_summary=ns,
                topic_keywords=topic_keywords,
            )

        # Build initial prompt
        prompt = _build_prompt(description, news_summary)

        # Token estimation — truncate if needed
        est_tokens = len(prompt) // 4
        if est_tokens > _TOKEN_LIMIT:
            logger.debug(
                "Prompt est. %d tokens > %d — truncating description and news_summary",
                est_tokens, _TOKEN_LIMIT,
            )
            prompt = _build_prompt(description[:500], news_summary[:300])

        # API call with 429 retry loop
        response = None
        for attempt in range(_MAX_429_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    system=AI_SYSTEM_PROMPT,
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 6,
                    }],
                    messages=[{"role": "user", "content": prompt}],
                )
                break   # success

            except anthropic.RateLimitError:
                if attempt < _MAX_429_RETRIES:
                    logger.warning(
                        "Rate limited (429) — waiting %ds before retry %d/%d",
                        _RETRY_429_WAIT, attempt + 1, _MAX_429_RETRIES,
                    )
                    time.sleep(_RETRY_429_WAIT)
                else:
                    logger.warning(
                        "Rate limited (429) after %d retries — skipping market",
                        _MAX_429_RETRIES,
                    )
                    self._call_errors += 1
                    return None

            except anthropic.APIError as e:
                err_str = str(e).lower()
                billing_kws = ("credit", "billing", "balance", "payment", "quota",
                               "overdue", "insufficient", "exceeded")
                if any(kw in err_str for kw in billing_kws):
                    logger.error(
                        "Anthropic BILLING/CREDIT error — disabling AI for this cycle: %s", e
                    )
                    # Signal caller to set ai_available=False
                    raise _BillingError(str(e)) from e
                else:
                    logger.error("Anthropic API error: %s", e)
                self._call_errors += 1
                return None

        if response is None:
            return None

        # Collect all text blocks
        text_parts = [
            block.text
            for block in response.content
            if hasattr(block, "text") and block.text
        ]
        full_text = "\n".join(text_parts)
        return self._parse_result(full_text, current_price)

    # ── Quick analysis (copy-trade evaluation) ────────────────────────────────

    async def run_quick_analysis(
        self, market: dict, timeout_seconds: float = 15.0
    ) -> Optional[dict]:
        """
        Fast analysis for copy-trade evaluation.
        Always uses the analysis model (Haiku); 1–2 searches; 300 tokens.
        """
        question      = market.get("question", "")
        raw_prices    = market.get("outcomePrices", [0.5])
        current_price = float(
            raw_prices[0] if isinstance(raw_prices, list) and raw_prices else 0.5
        )
        slug = market.get("slug") or market.get("conditionId", "")

        await self._enforce_rate_limit()

        quick_prompt = (
            f"Quick analysis: {question}\n"
            f"Market price: {current_price:.1%}\n"
            "Search 1–2 times, return JSON only:\n"
            '{"yes_probability": <float>, "confidence": <float>, '
            '"recommended_outcome": "YES"|"NO"|"SKIP", "reasoning": "<1 sentence>"}'
        )

        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._run_quick(quick_prompt),
                ),
                timeout=timeout_seconds,
            )
            if result:
                result["edge"] = result.get("yes_probability", current_price) - current_price
                return result
        except asyncio.TimeoutError:
            logger.warning("Quick AI analysis timed out for %s", slug)
        except Exception as e:
            logger.warning("Quick AI analysis failed for %s: %s", slug, e)
        return None

    def _run_quick(self, prompt: str) -> Optional[dict]:
        client = self._get_client()
        self._last_call_ts = now_ts()
        self._total_calls += 1
        try:
            response = client.messages.create(
                model=self.settings.ai_model_analysis,
                max_tokens=300,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text
                for block in response.content
                if hasattr(block, "text") and block.text
            )
            return self._parse_result(text, 0.5)
        except anthropic.RateLimitError:
            logger.warning("Quick analysis rate limited (429) — skipping")
            return None
        except Exception as e:
            logger.debug("Quick analysis error: %s", e)
            return None

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_result(self, text: str, current_price: float) -> Optional[dict]:
        """Parse JSON from Claude's response with three-level fallback."""
        # Attempt 1: full text is valid JSON
        try:
            return self._validate_result(json.loads(text.strip()), current_price)
        except (json.JSONDecodeError, ValueError):
            pass

        # Attempt 2: brace-depth scan for outermost {...}
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return self._validate_result(
                            json.loads(text[start:i + 1]), current_price
                        )
                    except (json.JSONDecodeError, ValueError):
                        break

        # Attempt 3: regex for nested objects
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return self._validate_result(json.loads(m.group()), current_price)
            except (json.JSONDecodeError, ValueError):
                pass

        # Attempt 4: field-level extraction as last resort
        prob_m = re.search(r'"yes_probability"\s*:\s*([0-9.]+)', text)
        conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if prob_m:
            return self._validate_result({
                "yes_probability":   float(prob_m.group(1)),
                "confidence":        float(conf_m.group(1)) if conf_m else 0.4,
                "recommended_outcome": "SKIP",
                "reasoning":         "Extracted from partial response.",
                "resolution_risk":   "HIGH",
            }, current_price)

        logger.warning("Could not parse AI analysis response")
        return None

    def _validate_result(self, data: dict, current_price: float) -> dict:
        """Validate and clamp all numeric fields; fill required defaults."""
        data["yes_probability"] = clamp(float(data.get("yes_probability", 0.5)), 0.01, 0.99)
        data["confidence"]      = clamp(float(data.get("confidence", 0.5)), 0.0, 1.0)
        data["base_rate"]       = clamp(float(data.get("base_rate", 0.5)), 0.0, 1.0)

        p    = data["yes_probability"]
        low  = data.get("confidence_interval_low")
        high = data.get("confidence_interval_high")
        data["confidence_interval_low"]  = clamp(float(low)  if low  else max(0.01, p - 0.15), 0.01, 0.99)
        data["confidence_interval_high"] = clamp(float(high) if high else min(0.99, p + 0.15), 0.01, 0.99)

        if "recommended_outcome" not in data:
            edge = data["yes_probability"] - current_price
            if abs(edge) < 0.04:
                data["recommended_outcome"] = "SKIP"
            elif edge > 0:
                data["recommended_outcome"] = "YES"
            else:
                data["recommended_outcome"] = "NO"

        data.setdefault("reasoning",              "No reasoning provided.")
        data.setdefault("strongest_yes_evidence", "Not analyzed.")
        data.setdefault("strongest_no_evidence",  "Not analyzed.")
        data.setdefault("key_facts",              [])
        data.setdefault("key_uncertainties",      [])
        data.setdefault("resolution_risk",        "MEDIUM")
        data.setdefault("aggregator_consensus",   None)
        data.setdefault("news_summary",           "")
        data.setdefault("base_rate_source",       "estimated")
        data.setdefault("evidence_adjustment",    0.0)
        data.setdefault("search_queries_used",    [])
        return data

    # ── Rate limiting ─────────────────────────────────────────────────────────

    async def _enforce_rate_limit(self) -> None:
        """
        Two-layer rate limiting before every API call:

        Layer 1 — per-minute window:
            Track call timestamps in a sliding 60-second deque.
            If the window is full (>= ai_max_calls_per_minute), sleep until
            the oldest call expires from the window.

        Layer 2 — minimum interval:
            Ensure at least ai_min_call_interval_seconds have elapsed since
            the last call, regardless of the window.
        """
        window = 60.0
        max_calls = self.settings.ai_max_calls_per_minute

        # Layer 1: per-minute window cap
        while True:
            now = now_ts()
            # Prune timestamps outside the window
            while self._call_timestamps and now - self._call_timestamps[0] > window:
                self._call_timestamps.popleft()

            if len(self._call_timestamps) < max_calls:
                break   # window has capacity

            oldest    = self._call_timestamps[0]
            sleep_secs = window - (now - oldest) + 0.5   # small buffer
            if sleep_secs > 0:
                logger.info(
                    "Rate limit window full (%d calls in 60s) — sleeping %.1fs",
                    len(self._call_timestamps), sleep_secs,
                )
                await asyncio.sleep(sleep_secs)

        # Layer 2: minimum interval between consecutive calls
        elapsed = now_ts() - self._last_call_ts
        gap     = self.settings.ai_min_call_interval_seconds
        if elapsed < gap:
            await asyncio.sleep(gap - elapsed)

        # Record this call in the window
        self._call_timestamps.append(now_ts())

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        now = now_ts()
        recent = sum(1 for ts in self._call_timestamps if now - ts <= 60.0)
        return {
            "total_calls":      self._total_calls,
            "errors":           self._call_errors,
            "success_rate":     (self._total_calls - self._call_errors) / max(self._total_calls, 1),
            "calls_last_60s":   recent,
            "window_remaining": max(0, self.settings.ai_max_calls_per_minute - recent),
        }
