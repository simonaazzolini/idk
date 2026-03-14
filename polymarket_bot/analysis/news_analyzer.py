"""
News & Sentiment Analysis Engine — Module 5.
Uses Claude with web_search to gather and analyze news for each market.
"""
import asyncio
import json
import logging
import re
from typing import Optional

import anthropic

from config.settings import Settings
from data.cache import MarketDataCache
from utils.helpers import clamp, now_ts, safe_div

logger = logging.getLogger(__name__)


NEWS_SYSTEM_PROMPT = """You are an expert news analyst for prediction markets.
Your task is to research and analyze information relevant to a binary prediction
market question and synthesize findings into a precise JSON response.

Research methodology — you MUST execute all 7 of these strategies:
  1. Search for the exact question text verbatim.
  2. Search for the key entities (people, organizations, countries) in the question.
  3. Search for "[topic] latest news" and "[topic] news today" to find breaking stories.
  4. Search aggregators and forecasting platforms: Metaculus, Manifold Markets,
     Polymarket, Good Judgment to get community base-rate estimates.
  5. Search for official statements, government sources, or primary data sources
     that are authoritative for the resolution criteria.
  6. Search specifically for reasons the market might resolve NO — contrary
     evidence, obstacles, historical failures of similar questions.
  7. Search for historical precedents and base rates: "how often does X happen",
     "[similar event] historical outcome", "base rate [topic]".

Only return valid JSON — no markdown code fences, no preamble, no commentary."""


NEWS_ANALYSIS_PROMPT = """Analyze the following prediction market and return a JSON assessment.

=== MARKET DETAILS ===
QUESTION:            {question}
RESOLUTION CRITERIA: {resolution_criteria}
CURRENT YES PRICE:   {current_price:.1%}
DAYS TO RESOLUTION:  {days_to_resolution:.1f}
CATEGORY:            {category}
RESOLUTION SOURCE:   {resolution_source}

=== YOUR TASK ===
Execute all 7 search strategies from your instructions, then return ONLY this
JSON structure with no wrapper text:

{{
  "sentiment_score": <float −1.0 to +1.0; +1.0 = overwhelming YES evidence>,
  "news_recency_score": <float 0–1; 1.0 = breaking news today, 0 = only old articles>,
  "information_quality": <float 0–1; ratio of verifiable hard data vs speculation>,
  "consensus_direction": <"YES" | "NO" | "UNCERTAIN">,
  "surprise_risk": <float 0–1; probability of an unexpected resolution>,
  "key_upcoming_events": [
    "<specific scheduled event/date before resolution that could change outcome>",
    ...
  ],
  "conflicting_signals": [
    "<credible evidence that points toward YES despite overall NO lean, or vice versa>",
    ...
  ],
  "news_velocity_score": <float 0–10; 10 = massive breaking news, 0 = zero coverage>,
  "news_summary": "<3-sentence summary: what is known, what is uncertain, what is next>",
  "search_queries_used": [
    "<exact query string you searched>",
    ...
  ],
  "strongest_yes_evidence": "<single most compelling piece of evidence for YES>",
  "strongest_no_evidence": "<single most compelling piece of evidence for NO>",
  "base_rate_estimate": <float 0–1; historical base rate for this type of question>,
  "days_since_last_relevant_news": <int; 0 = today, 999 = no relevant news found>,
  "market_price_vs_news": <"UNDERPRICED" | "OVERPRICED" | "FAIR" | "UNKNOWN">
}}"""


class NewsAnalyzer:
    """
    Analyzes news and sentiment for prediction markets using Claude with web search.
    Implements all 7 search strategies, computes a full news_score 0–10, and
    handles all failure modes gracefully (no news found, API errors, parse errors).
    """

    def __init__(self, settings: Settings, cache: MarketDataCache):
        self.settings = settings
        self.cache    = cache
        self._client: Optional[anthropic.Anthropic] = None
        self._last_call_ts: float = 0.0
        self._total_calls:  int   = 0
        self._failed_calls: int   = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    async def analyze_market(self, market: dict) -> dict:
        """
        Run full news analysis for a market.

        Checks cache first.  On cache miss, calls Claude with the web_search
        tool executing all 7 research strategies.  Returns a dict with all
        news metrics plus a news_score (0–10).  Falls back to _empty_result()
        on any failure so the bot can continue without news signal.
        """
        slug = market.get("slug") or market.get("conditionId", "unknown")

        # ── Cache check ────────────────────────────────────────────────────
        cached = await self.cache.get_news(slug)
        if cached:
            logger.debug("News cache hit for %s", slug)
            return cached

        # ── Extract market fields ──────────────────────────────────────────
        question         = market.get("question", "Unknown question")
        resolution_source = (
            market.get("resolutionSource")
            or market.get("resolution_source")
            or "Not specified"
        )
        days_to_resolution = float(market.get("days_to_resolution") or 30.0)
        raw_prices = market.get("outcomePrices", [0.5])
        if isinstance(raw_prices, list) and raw_prices:
            current_price = float(raw_prices[0])
        else:
            current_price = float(market.get("yes_price") or 0.5)
        category    = str(market.get("category") or "general")
        description = str(market.get("description") or "")

        # ── Rate-limit enforcement ─────────────────────────────────────────
        await self._enforce_rate_limit()

        # ── Claude call (in thread pool to avoid blocking event loop) ─────
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._run_news_analysis(
                    question, resolution_source, current_price,
                    days_to_resolution, category, description,
                ),
            )
        except Exception as e:
            logger.warning("News analysis failed for %s: %s", slug, e)
            self._failed_calls += 1
            result = self._empty_result()

        # ── Validate, score, annotate ─────────────────────────────────────
        result = _validate_result_fields(result)
        result["news_score"]   = self._compute_news_score(result)
        result["analyzed_at"]  = now_ts()
        result["market_slug"]  = slug

        # ── Cache and return ───────────────────────────────────────────────
        await self.cache.set_news(slug, result, ttl_seconds=1800)

        logger.info(
            "News: %s | score=%.1f | sentiment=%.2f | velocity=%.1f | "
            "direction=%s | quality=%.2f",
            slug[:30], result["news_score"], result["sentiment_score"],
            result["news_velocity_score"], result["consensus_direction"],
            result["information_quality"],
        )
        return result

    async def analyze_batch(
        self, markets: list[dict], max_markets: int = 20
    ) -> dict[str, dict]:
        """
        Analyze news for up to max_markets markets.

        Per-market errors are caught independently so one failure does not
        block the rest.  Markets that fail get a neutral empty result with
        news_score = 5.0 (no signal).

        Returns dict of slug → news_result.
        """
        results: dict[str, dict] = {}
        for market in markets[:max_markets]:
            slug = market.get("slug") or market.get("conditionId", "unknown")
            try:
                result = await self.analyze_market(market)
                results[slug] = result
            except Exception as e:
                logger.warning("News analysis failed for %s: %s", slug, e)
                empty = self._empty_result()
                empty["news_score"]  = 5.0
                empty["analyzed_at"] = now_ts()
                empty["market_slug"] = slug
                results[slug] = empty
        return results

    # ── Core Claude call ───────────────────────────────────────────────────────

    def _run_news_analysis(
        self,
        question: str,
        resolution_source: str,
        current_price: float,
        days_to_resolution: float,
        category: str,
        description: str,
    ) -> dict:
        """
        Synchronous Claude call with the web_search tool.
        All 7 search strategies are baked into the prompt; Claude is
        instructed to execute them before composing the JSON response.
        """
        client = self._get_client()
        self._last_call_ts = now_ts()
        self._total_calls += 1

        prompt = NEWS_ANALYSIS_PROMPT.format(
            question=question,
            resolution_criteria=description[:600] if description else question,
            current_price=current_price,
            days_to_resolution=days_to_resolution,
            category=category,
            resolution_source=resolution_source,
        )

        try:
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=2000,
                system=NEWS_SYSTEM_PROMPT,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 7}],
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as e:
            err_str = str(e).lower()
            billing_kws = ("credit", "billing", "balance", "payment", "quota", "overdue")
            if any(kw in err_str for kw in billing_kws):
                logger.warning("Anthropic billing/credit error — skipping news analysis: %s", e)
            else:
                logger.error("Anthropic API error in news analysis: %s", e)
            return self._empty_result()

        # Collect all text blocks from the (possibly multi-turn) response
        result_text = ""
        for block in response.content:
            if hasattr(block, "text") and block.text:
                result_text += block.text

        return self._parse_json_response(result_text)

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_json_response(self, text: str) -> dict:
        """
        Extract and parse JSON from Claude's response.
        Attempts (in order):
          1. Direct JSON parse of the full stripped text.
          2. Extract the first {...} block with nested-brace handling.
          3. Return _empty_result() as a safe fallback.
        """
        if not text or not text.strip():
            logger.warning("Empty news analysis response from Claude")
            return self._empty_result()

        # Attempt 1: full text is valid JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Attempt 2: find the outermost {...} block (handles leading/trailing text)
        depth   = 0
        start   = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

        # Attempt 3: regex-based extraction (handles markdown-wrapped responses)
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse news analysis JSON — using empty result")
        return self._empty_result()

    # ── Defaults ───────────────────────────────────────────────────────────────

    def _empty_result(self) -> dict:
        """
        Safe neutral result returned when news analysis is unavailable.
        All scores are set to conservative midpoints so they contribute
        minimally to the composite signal.
        """
        return {
            "sentiment_score":            0.0,
            "news_recency_score":         0.5,
            "information_quality":        0.3,
            "consensus_direction":        "UNCERTAIN",
            "surprise_risk":              0.5,
            "key_upcoming_events":        [],
            "conflicting_signals":        [],
            "news_velocity_score":        3.0,
            "news_summary":               "Insufficient information to analyze.",
            "search_queries_used":        [],
            "strongest_yes_evidence":     "Unknown",
            "strongest_no_evidence":      "Unknown",
            "base_rate_estimate":         0.5,
            "days_since_last_relevant_news": 999,
            "market_price_vs_news":       "UNKNOWN",
        }

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _compute_news_score(self, result: dict) -> float:
        """Compute a composite news signal score in [0, 10]. Returns 5.0 on error."""
        try:
            return self._compute_news_score_inner(result)
        except Exception as exc:
            logger.warning("News score computation error — returning neutral 5.0: %s", exc)
            return 5.0

    def _compute_news_score_inner(self, result: dict) -> float:
        """
        Inner scoring logic.

        Components:
          - Sentiment direction:   −1..+1 → core directional signal
          - Information quality:   gates how much the sentiment is trusted
          - Recency:               gates how fresh the information is
          - News velocity:         bonus for high-volume recent coverage
          - Conflicting signals:   penalty when credible contrary evidence exists
          - Surprise risk:         penalty reducing confidence in any direction
          - Base rate alignment:   bonus when sentiment agrees with known base rate
          - Market price vs news:  slight bonus when news identifies mispricing

        Returns a float in [0.0, 10.0].  5.0 = no signal.
        """
        sentiment  = float(result.get("sentiment_score",            0.0))
        quality    = float(result.get("information_quality",        0.3))
        recency    = float(result.get("news_recency_score",         0.5))
        velocity   = float(result.get("news_velocity_score",        3.0))
        surprise   = float(result.get("surprise_risk",              0.5))
        base_rate  = float(result.get("base_rate_estimate",         0.5))
        conflicts  = result.get("conflicting_signals", [])
        consensus  = str(result.get("consensus_direction", "UNCERTAIN"))
        price_vs_n = str(result.get("market_price_vs_news", "UNKNOWN"))

        # ── 1. Sentiment → directional score [0, 10] ──────────────────────
        # sentiment −1..+1 maps to 0..10
        direction_score = (sentiment + 1.0) * 5.0

        # ── 2. Quality gate [0.3..1.0] → trust the signal proportionally ──
        # Clamp quality to [0.1, 1.0] to avoid zeroing out all signal
        quality_gate = clamp(quality, 0.10, 1.00)

        # ── 3. Recency gate [0.3..1.0] → fresh news matters more ──────────
        recency_gate = 0.30 + 0.70 * clamp(recency, 0.0, 1.0)

        # ── 4. Base score from sentiment × quality × recency ──────────────
        base_score = direction_score * quality_gate * recency_gate

        # ── 5. Velocity bonus: high news coverage = more signal [0..+1.5] ─
        # velocity 0..10 → bonus 0..+1.5
        velocity_bonus = clamp(velocity / 10.0, 0.0, 1.0) * 1.5

        # ── 6. Conflicting signals penalty [0..−2.0] ──────────────────────
        # Each credible conflicting signal subtracts 0.4 (max 2.0 penalty)
        conflict_count   = len(conflicts) if isinstance(conflicts, list) else 0
        conflict_penalty = clamp(conflict_count * 0.4, 0.0, 2.0)

        # ── 7. Surprise risk penalty [0..−1.5] ────────────────────────────
        # High surprise risk → reduce confidence in directional score
        # Pull toward neutral: score = base × (1 - surprise*0.6) + 5 × (surprise*0.6)
        surprise_pull = clamp(surprise * 0.60, 0.0, 0.60)
        adjusted      = base_score * (1.0 - surprise_pull) + 5.0 * surprise_pull

        # ── 8. Add velocity bonus and subtract conflict penalty ─────────────
        adjusted = adjusted + velocity_bonus - conflict_penalty

        # ── 9. Base-rate alignment bonus [0..+0.5] ─────────────────────────
        # If base_rate is known and aligns with consensus direction, small boost
        if consensus == "YES" and base_rate > 0.6:
            adjusted += 0.3
        elif consensus == "NO" and base_rate < 0.4:
            adjusted += 0.3

        # ── 10. Market price vs news mispricing bonus [0..+0.5] ───────────
        if price_vs_n == "UNDERPRICED" and consensus == "YES":
            adjusted += 0.5
        elif price_vs_n == "OVERPRICED" and consensus == "NO":
            adjusted += 0.5

        return clamp(adjusted, 0.0, 10.0)

    # ── Utilities ─────────────────────────────────────────────────────────────

    async def _enforce_rate_limit(self) -> None:
        """Ensure minimum spacing between successive Claude API calls."""
        elapsed = now_ts() - self._last_call_ts
        min_gap = self.settings.ai_min_call_interval_seconds
        if elapsed < min_gap:
            await asyncio.sleep(min_gap - elapsed)

    @property
    def call_stats(self) -> dict:
        """Return call statistics for monitoring / dashboard."""
        return {
            "total_calls":  self._total_calls,
            "failed_calls": self._failed_calls,
            "success_rate": safe_div(
                self._total_calls - self._failed_calls, self._total_calls, 1.0
            ),
            "last_call_ts": self._last_call_ts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Field validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_result_fields(result: dict) -> dict:
    """
    Validate and clamp all numeric fields in the news analysis result.
    Replaces missing or out-of-range values with safe defaults so downstream
    consumers never receive None or NaN.

    Also normalises the consensus_direction to one of YES / NO / UNCERTAIN.
    """
    FLOAT_FIELDS = {
        "sentiment_score":               (-1.0,  1.0,  0.0),
        "news_recency_score":            ( 0.0,  1.0,  0.5),
        "information_quality":           ( 0.0,  1.0,  0.3),
        "surprise_risk":                 ( 0.0,  1.0,  0.5),
        "news_velocity_score":           ( 0.0, 10.0,  3.0),
        "base_rate_estimate":            ( 0.0,  1.0,  0.5),
    }

    for field, (lo, hi, default) in FLOAT_FIELDS.items():
        raw = result.get(field)
        try:
            result[field] = float(clamp(float(raw), lo, hi))
        except (TypeError, ValueError):
            result[field] = default

    # Normalise consensus_direction
    raw_dir = str(result.get("consensus_direction", "UNCERTAIN")).upper().strip()
    if raw_dir in ("YES", "NO"):
        result["consensus_direction"] = raw_dir
    else:
        result["consensus_direction"] = "UNCERTAIN"

    # Normalise market_price_vs_news
    raw_mpvn = str(result.get("market_price_vs_news", "UNKNOWN")).upper().strip()
    if raw_mpvn in ("UNDERPRICED", "OVERPRICED", "FAIR"):
        result["market_price_vs_news"] = raw_mpvn
    else:
        result["market_price_vs_news"] = "UNKNOWN"

    # Normalise list fields
    for list_field in ("key_upcoming_events", "conflicting_signals", "search_queries_used"):
        if not isinstance(result.get(list_field), list):
            result[list_field] = []

    # Normalise string fields
    for str_field, default in (
        ("news_summary",          "Insufficient information to analyze."),
        ("strongest_yes_evidence", "Unknown"),
        ("strongest_no_evidence",  "Unknown"),
    ):
        if not isinstance(result.get(str_field), str) or not result[str_field]:
            result[str_field] = default

    # Normalise days_since_last_relevant_news
    try:
        result["days_since_last_relevant_news"] = max(
            0, int(result.get("days_since_last_relevant_news", 999))
        )
    except (TypeError, ValueError):
        result["days_since_last_relevant_news"] = 999

    return result
