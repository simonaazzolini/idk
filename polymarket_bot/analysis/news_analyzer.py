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
Your task is to research and analyze information relevant to a prediction market question.
You will search the web and synthesize findings into a structured JSON response.
Always search from multiple angles: supportive evidence, contrary evidence, and base rates.
Return ONLY valid JSON with no markdown wrapping."""

NEWS_ANALYSIS_PROMPT = """Analyze the following prediction market question and return a JSON assessment.

MARKET QUESTION: {question}
RESOLUTION CRITERIA: {resolution_criteria}
CURRENT YES PRICE: {current_price:.1%}
DAYS TO RESOLUTION: {days_to_resolution:.1f}
CATEGORY: {category}
RESOLUTION SOURCE: {resolution_source}

Please:
1. Search for the exact question text
2. Search for related entities, people, or organizations
3. Search for "[topic] latest news" and "[topic] news today"
4. Search for base rates on prediction aggregators (Metaculus, Manifold)
5. Search for official statements or data sources
6. Search for contradictory evidence or reasons the market might resolve NO

Return ONLY this JSON structure (no markdown, no preamble):
{{
  "sentiment_score": <float -1.0 to 1.0, where +1.0 = strongly YES>,
  "news_recency_score": <float 0-1, based on age of most relevant articles>,
  "information_quality": <float 0-1, ratio of hard data vs speculation>,
  "consensus_direction": <"YES" | "NO" | "UNCERTAIN">,
  "surprise_risk": <float 0-1, probability of unexpected outcome>,
  "key_upcoming_events": [<list of specific scheduled events before resolution>],
  "conflicting_signals": [<list of credible evidence pointing both directions>],
  "news_velocity_score": <float 0-10, volume of recent news>,
  "news_summary": "<3-sentence summary of the information landscape>",
  "search_queries_used": [<list of queries you searched>],
  "strongest_yes_evidence": "<single strongest piece of evidence for YES>",
  "strongest_no_evidence": "<single strongest piece of evidence for NO>"
}}"""


class NewsAnalyzer:
    """
    Analyzes news for prediction markets using Claude with web search.
    """

    def __init__(self, settings: Settings, cache: MarketDataCache):
        self.settings = settings
        self.cache = cache
        self._client: Optional[anthropic.Anthropic] = None
        self._last_call_ts: float = 0.0
        self._total_calls: int = 0

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    async def analyze_market(self, market: dict) -> dict:
        """
        Run full news analysis for a market.
        Returns a dict with all news metrics and a news_score (0-10).
        """
        slug = market.get("slug") or market.get("conditionId", "unknown")

        # Check cache
        cached = await self.cache.get_news(slug)
        if cached:
            logger.debug("News cache hit for %s", slug)
            return cached

        question = market.get("question", "Unknown question")
        resolution_source = market.get("resolutionSource") or market.get("resolution_source") or "Not specified"
        days_to_resolution = float(market.get("days_to_resolution") or 30.0)
        current_price = float(market.get("yes_price") or market.get("outcomePrices", [0.5])[0] if isinstance(market.get("outcomePrices"), list) else 0.5)
        category = str(market.get("category") or "general")
        description = market.get("description") or ""

        # Enforce rate limiting
        await self._enforce_rate_limit()

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._run_news_analysis(
                    question, resolution_source, current_price,
                    days_to_resolution, category, description
                ),
            )
        except Exception as e:
            logger.warning("News analysis failed for %s: %s", slug, e)
            result = self._empty_result()

        # Compute news_score
        result["news_score"] = self._compute_news_score(result)
        result["analyzed_at"] = now_ts()
        result["market_slug"] = slug

        # Cache result
        await self.cache.set_news(slug, result, ttl_seconds=1800)
        return result

    def _run_news_analysis(
        self,
        question: str,
        resolution_source: str,
        current_price: float,
        days_to_resolution: float,
        category: str,
        description: str,
    ) -> dict:
        """Synchronous Claude call with web search tool."""
        client = self._get_client()
        self._last_call_ts = now_ts()
        self._total_calls += 1

        prompt = NEWS_ANALYSIS_PROMPT.format(
            question=question,
            resolution_criteria=description[:500] if description else question,
            current_price=current_price,
            days_to_resolution=days_to_resolution,
            category=category,
            resolution_source=resolution_source,
        )

        messages = [{"role": "user", "content": prompt}]

        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            system=NEWS_SYSTEM_PROMPT,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}],
            messages=messages,
        )

        # Extract JSON from response
        result_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                result_text += block.text
            elif hasattr(block, "type") and block.type == "text":
                result_text += block.text

        # Parse JSON
        result = self._parse_json_response(result_text)
        return result

    def _parse_json_response(self, text: str) -> dict:
        """Extract and parse JSON from Claude's response."""
        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return empty result
        logger.warning("Could not parse news analysis JSON from response")
        return self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "sentiment_score": 0.0,
            "news_recency_score": 0.5,
            "information_quality": 0.3,
            "consensus_direction": "UNCERTAIN",
            "surprise_risk": 0.5,
            "key_upcoming_events": [],
            "conflicting_signals": [],
            "news_velocity_score": 3.0,
            "news_summary": "Insufficient information to analyze.",
            "search_queries_used": [],
            "strongest_yes_evidence": "Unknown",
            "strongest_no_evidence": "Unknown",
        }

    def _compute_news_score(self, result: dict) -> float:
        """Compute news signal score 0-10."""
        sentiment = float(result.get("sentiment_score", 0.0))
        quality = float(result.get("information_quality", 0.3))
        recency = float(result.get("news_recency_score", 0.5))

        # Map sentiment -1..1 to 0..10
        direction_score = (sentiment + 1.0) * 5.0

        # Weight by quality and recency
        news_score = direction_score * quality * recency

        return clamp(news_score, 0.0, 10.0)

    async def _enforce_rate_limit(self) -> None:
        """Enforce minimum 3 seconds between Claude API calls."""
        elapsed = now_ts() - self._last_call_ts
        if elapsed < self.settings.ai_min_call_interval_seconds:
            await asyncio.sleep(self.settings.ai_min_call_interval_seconds - elapsed)

    async def analyze_batch(self, markets: list[dict], max_markets: int = 20) -> dict[str, dict]:
        """
        Analyze news for up to max_markets markets.
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
                results[slug] = self._empty_result()
                results[slug]["news_score"] = 5.0
        return results
