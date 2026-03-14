"""
AI Probability Engine — Module 6.
Uses Claude claude-opus-4-5 with web search to estimate true outcome probabilities.
"""
import asyncio
import json
import logging
import re
from typing import Optional

import anthropic

from config.settings import Settings
from data.cache import MarketDataCache
from utils.helpers import clamp, now_ts

logger = logging.getLogger(__name__)

AI_SYSTEM_PROMPT = """You are an elite prediction market analyst with expertise in probability estimation.
Your job is to estimate the true probability of market outcomes using all available evidence.
You approach this like a superforecaster: you look for base rates, update with specific evidence,
avoid overconfidence, and always consider what could change your estimate.

CRITICAL RULES:
1. Run AT LEAST 3 web searches, up to 6
2. Always find a base rate first before looking at specific evidence
3. Apply Bayesian updates: start from base rate, adjust based on current evidence
4. Check prediction aggregators (Metaculus, Manifold, PredictIt) for market consensus
5. Consider the exact resolution criteria - not just the general topic
6. Calibrate for overconfidence - reduce confidence if you're uncertain
7. Return ONLY valid JSON - no markdown, no preamble, no explanation outside JSON"""

AI_ANALYSIS_PROMPT = """Analyze this prediction market and return a probability estimate.

MARKET QUESTION: {question}
CURRENT YES PRICE: {current_price:.4f} ({current_price_pct:.1%})
DAYS TO RESOLUTION: {days_to_resolution:.1f}
CATEGORY: {category}
DESCRIPTION: {description}
RESOLUTION SOURCE: {resolution_source}

KEY STATISTICS:
- 24h volume: ${volume_24h:,.0f}
- Liquidity: ${liquidity:,.0f}
- Price 24h change: {price_change_24h:+.2%}
- RSI(14): {rsi_14:.0f}
- Whale direction: {whale_direction}
- News sentiment: {news_sentiment}

INSTRUCTIONS:
1. Search for: "{question}"
2. Search for base rates for this type of event
3. Search for recent news and expert forecasts
4. Search prediction aggregators: site:metaculus.com OR site:manifold.markets for "{topic_keywords}"
5. Search for the resolution source: {resolution_source}
6. Search for contra-evidence: reasons this might NOT happen

Return ONLY this JSON (no markdown, no text outside JSON):
{{
  "yes_probability": <float 0.0-1.0>,
  "confidence_interval_low": <float>,
  "confidence_interval_high": <float>,
  "confidence": <float 0.0-1.0>,
  "recommended_outcome": <"YES" | "NO" | "SKIP">,
  "base_rate": <float>,
  "base_rate_source": <string>,
  "evidence_adjustment": <float, how much you shifted from base rate>,
  "reasoning": <string, 3-5 sentences>,
  "strongest_yes_evidence": <string>,
  "strongest_no_evidence": <string>,
  "key_facts": [<list of concrete facts found>],
  "key_uncertainties": [<list of things that could change the outcome>],
  "resolution_risk": <"LOW" | "MEDIUM" | "HIGH">,
  "aggregator_consensus": <float or null>,
  "news_summary": <string>,
  "edge": <float, yes_probability minus current_yes_price>,
  "signal_strength": <"WEAK" | "MODERATE" | "STRONG" | "VERY_STRONG">
}}"""


class AIAnalysisResult:
    """Typed result from the AI probability engine."""

    def __init__(self, data: dict):
        self.yes_probability: float = float(data.get("yes_probability", 0.5))
        self.confidence_interval_low: float = float(data.get("confidence_interval_low", 0.3))
        self.confidence_interval_high: float = float(data.get("confidence_interval_high", 0.7))
        self.confidence: float = float(data.get("confidence", 0.5))
        self.recommended_outcome: str = data.get("recommended_outcome", "SKIP")
        self.base_rate: float = float(data.get("base_rate", 0.5))
        self.base_rate_source: str = data.get("base_rate_source", "unknown")
        self.evidence_adjustment: float = float(data.get("evidence_adjustment", 0.0))
        self.reasoning: str = data.get("reasoning", "")
        self.strongest_yes_evidence: str = data.get("strongest_yes_evidence", "")
        self.strongest_no_evidence: str = data.get("strongest_no_evidence", "")
        self.key_facts: list = data.get("key_facts", [])
        self.key_uncertainties: list = data.get("key_uncertainties", [])
        self.resolution_risk: str = data.get("resolution_risk", "MEDIUM")
        self.aggregator_consensus: Optional[float] = data.get("aggregator_consensus")
        self.news_summary: str = data.get("news_summary", "")
        self.edge: float = float(data.get("edge", 0.0))
        self.signal_strength: str = data.get("signal_strength", "WEAK")
        self.ai_score: float = 5.0  # computed separately
        self.raw_data = data

    def to_dict(self) -> dict:
        return {
            "yes_probability": self.yes_probability,
            "confidence_interval_low": self.confidence_interval_low,
            "confidence_interval_high": self.confidence_interval_high,
            "confidence": self.confidence,
            "recommended_outcome": self.recommended_outcome,
            "base_rate": self.base_rate,
            "base_rate_source": self.base_rate_source,
            "evidence_adjustment": self.evidence_adjustment,
            "reasoning": self.reasoning,
            "strongest_yes_evidence": self.strongest_yes_evidence,
            "strongest_no_evidence": self.strongest_no_evidence,
            "key_facts": self.key_facts,
            "key_uncertainties": self.key_uncertainties,
            "resolution_risk": self.resolution_risk,
            "aggregator_consensus": self.aggregator_consensus,
            "news_summary": self.news_summary,
            "edge": self.edge,
            "signal_strength": self.signal_strength,
            "ai_score": self.ai_score,
        }


def compute_signal_strength(edge: float, confidence: float) -> str:
    """Classify signal strength based on edge and confidence."""
    abs_edge = abs(edge)
    if abs_edge > 0.12 and confidence > 0.75:
        return "VERY_STRONG"
    if abs_edge > 0.08 and confidence > 0.65:
        return "STRONG"
    if abs_edge > 0.04 and confidence > 0.55:
        return "MODERATE"
    return "WEAK"


def compute_ai_score(result: AIAnalysisResult) -> float:
    """Compute AI signal score (0-10) from edge and confidence."""
    edge_score = min(abs(result.edge) / 0.15 * 10.0, 10.0)
    confidence_score = result.confidence * 10.0
    return (edge_score * 0.7) + (confidence_score * 0.3)


class AIAnalyzer:
    """
    AI Probability Engine using Claude claude-opus-4-5 with web search.
    """

    def __init__(self, settings: Settings, cache: MarketDataCache):
        self.settings = settings
        self.cache = cache
        self._client: Optional[anthropic.Anthropic] = None
        self._last_call_ts: float = 0.0
        self._total_calls: int = 0
        self._call_errors: int = 0

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    async def analyze_market(
        self,
        market: dict,
        technical_signal=None,
        whale_consensus: Optional[dict] = None,
        news_result: Optional[dict] = None,
    ) -> Optional[AIAnalysisResult]:
        """
        Run deep AI analysis on a single market.
        Returns AIAnalysisResult or None if analysis should be skipped.
        """
        slug = market.get("slug") or market.get("conditionId", "unknown")

        # Check cache
        cached = await self.cache.get_ai_analysis(slug)
        if cached:
            result = AIAnalysisResult(cached)
            result.ai_score = compute_ai_score(result)
            return result

        # Enforce rate limit
        await self._enforce_rate_limit()

        # Extract market data
        question = market.get("question", "Unknown")
        current_price = float((market.get("outcomePrices") or [0.5])[0] if isinstance(market.get("outcomePrices"), list) else 0.5)
        days_to_resolution = float(market.get("days_to_resolution") or 30.0)
        category = str(market.get("category") or "general")
        description = str(market.get("description") or "")[:800]
        resolution_source = str(market.get("resolutionSource") or market.get("resolution_source") or "Not specified")
        volume_24h = float(market.get("volume24hr") or market.get("volume_24h") or 0)
        liquidity = float(market.get("liquidity") or 0)
        price_change_24h = float(technical_signal.price_change_24h if technical_signal else 0.0)
        rsi = float(technical_signal.rsi_14 if technical_signal else 50.0)
        whale_dir = whale_consensus.get("smart_money_net_direction", "NEUTRAL") if whale_consensus else "NEUTRAL"
        news_sent = str(news_result.get("consensus_direction", "UNCERTAIN")) if news_result else "UNKNOWN"

        # Extract topic keywords for aggregator search
        words = question.split()
        topic_keywords = " ".join(w for w in words if len(w) > 3)[:50]

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
                    price_change_24h=price_change_24h,
                    rsi_14=rsi,
                    whale_direction=whale_dir,
                    news_sentiment=news_sent,
                    topic_keywords=topic_keywords,
                ),
            )
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

        result = AIAnalysisResult(raw_result)
        result.ai_score = compute_ai_score(result)

        # Cache result
        cache_data = result.to_dict()
        await self.cache.set_ai_analysis(slug, cache_data, ttl_seconds=900)

        logger.info(
            "AI: %s | P(YES)=%.3f | edge=%.3f | conf=%.2f | signal=%s",
            slug[:30], result.yes_probability, result.edge,
            result.confidence, result.signal_strength
        )
        return result

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
        topic_keywords: str,
    ) -> Optional[dict]:
        """Synchronous Claude API call with web search."""
        client = self._get_client()
        self._last_call_ts = now_ts()
        self._total_calls += 1

        prompt = AI_ANALYSIS_PROMPT.format(
            question=question,
            current_price=current_price,
            current_price_pct=current_price,
            days_to_resolution=days_to_resolution,
            category=category,
            description=description[:600],
            resolution_source=resolution_source,
            volume_24h=volume_24h,
            liquidity=liquidity,
            price_change_24h=price_change_24h,
            rsi_14=rsi_14,
            whale_direction=whale_direction,
            news_sentiment=news_sentiment,
            topic_keywords=topic_keywords,
        )

        try:
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=3000,
                system=AI_SYSTEM_PROMPT,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}],
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("credit", "billing", "balance", "payment", "quota", "overdue")):
                logger.warning("Anthropic billing/credit error — skipping AI analysis: %s", e)
            else:
                logger.error("Anthropic API error: %s", e)
            return None

        # Extract all text blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                text_parts.append(block.text)
            elif hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)

        full_text = "\n".join(text_parts)
        return self._parse_result(full_text, current_price)

    def _parse_result(self, text: str, current_price: float) -> Optional[dict]:
        """Parse JSON from Claude's response with robust fallback."""
        # Try direct parse
        try:
            data = json.loads(text.strip())
            return self._validate_result(data, current_price)
        except json.JSONDecodeError:
            pass

        # Find JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_result(data, current_price)
            except json.JSONDecodeError:
                pass

        # Last resort: regex extraction
        prob_match = re.search(r'"yes_probability"\s*:\s*([0-9.]+)', text)
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if prob_match:
            prob = float(prob_match.group(1))
            conf = float(conf_match.group(1)) if conf_match else 0.4
            return self._validate_result({
                "yes_probability": prob,
                "confidence": conf,
                "recommended_outcome": "YES" if prob > 0.5 else "NO",
                "reasoning": "Extracted from partial response.",
                "resolution_risk": "HIGH",
            }, current_price)

        logger.warning("Could not parse AI analysis response")
        return None

    def _validate_result(self, data: dict, current_price: float) -> dict:
        """Validate and clamp all numeric fields."""
        data["yes_probability"] = clamp(float(data.get("yes_probability", 0.5)), 0.01, 0.99)
        data["confidence"] = clamp(float(data.get("confidence", 0.5)), 0.0, 1.0)
        data["base_rate"] = clamp(float(data.get("base_rate", 0.5)), 0.0, 1.0)

        low = data.get("confidence_interval_low")
        high = data.get("confidence_interval_high")
        p = data["yes_probability"]
        data["confidence_interval_low"] = clamp(float(low) if low else max(0.01, p - 0.15), 0.01, 0.99)
        data["confidence_interval_high"] = clamp(float(high) if high else min(0.99, p + 0.15), 0.01, 0.99)

        # Ensure recommended_outcome is set
        if "recommended_outcome" not in data:
            edge = data["yes_probability"] - current_price
            if abs(edge) < 0.04:
                data["recommended_outcome"] = "SKIP"
            elif edge > 0:
                data["recommended_outcome"] = "YES"
            else:
                data["recommended_outcome"] = "NO"

        # Ensure required fields
        data.setdefault("reasoning", "No reasoning provided.")
        data.setdefault("strongest_yes_evidence", "Not analyzed.")
        data.setdefault("strongest_no_evidence", "Not analyzed.")
        data.setdefault("key_facts", [])
        data.setdefault("key_uncertainties", [])
        data.setdefault("resolution_risk", "MEDIUM")
        data.setdefault("aggregator_consensus", None)
        data.setdefault("news_summary", "")
        data.setdefault("base_rate_source", "estimated")
        data.setdefault("evidence_adjustment", 0.0)
        data.setdefault("search_queries_used", [])
        return data

    async def _enforce_rate_limit(self) -> None:
        elapsed = now_ts() - self._last_call_ts
        if elapsed < self.settings.ai_min_call_interval_seconds:
            await asyncio.sleep(self.settings.ai_min_call_interval_seconds - elapsed)

    async def run_quick_analysis(self, market: dict, timeout_seconds: float = 15.0) -> Optional[dict]:
        """
        Quick 15-second analysis for copy trade evaluation.
        Uses fewer searches and simplified prompt.
        """
        question = market.get("question", "")
        current_price = float((market.get("outcomePrices") or [0.5])[0]
                               if isinstance(market.get("outcomePrices"), list) else 0.5)
        slug = market.get("slug") or market.get("conditionId", "")

        await self._enforce_rate_limit()

        quick_prompt = f"""Quick analysis for: {question}
Current market price: {current_price:.1%}
Search for current evidence (1-2 searches max) and return JSON:
{{"yes_probability": <float>, "confidence": <float>, "recommended_outcome": "YES"|"NO"|"SKIP", "reasoning": "<1 sentence>"}}"""

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
                model="claude-opus-4-5",
                max_tokens=500,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    text += block.text
            return self._parse_result(text, 0.5)
        except Exception as e:
            logger.debug("Quick analysis error: %s", e)
            return None

    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "errors": self._call_errors,
            "success_rate": (self._total_calls - self._call_errors) / max(self._total_calls, 1),
        }

    async def analyze_batch(
        self,
        markets: list[dict],
        max_markets: int = 20,
        technical_signals: Optional[dict] = None,
        whale_consensuses: Optional[dict] = None,
        news_results: Optional[dict] = None,
    ) -> dict[str, Optional[AIAnalysisResult]]:
        """Analyze multiple markets. Returns slug → AIAnalysisResult."""
        results: dict[str, Optional[AIAnalysisResult]] = {}
        for market in markets[:max_markets]:
            slug = market.get("slug") or market.get("conditionId", "unknown")
            tech = technical_signals.get(slug) if technical_signals else None
            whale = whale_consensuses.get(slug) if whale_consensuses else None
            news = news_results.get(slug) if news_results else None
            try:
                result = await self.analyze_market(market, tech, whale, news)
                results[slug] = result
            except Exception as e:
                logger.warning("AI analysis error for %s: %s", slug, e)
                results[slug] = None
        return results
