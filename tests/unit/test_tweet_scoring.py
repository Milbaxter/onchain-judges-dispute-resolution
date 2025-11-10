"""Unit tests for tweet analysis weighted scoring logic."""

from src.models import TweetLLMResponse, TweetVerdictType
from src.scoring import WeightedScorer


def test_unanimous_credible_verdict():
    """Test scoring when all LLMs say CREDIBLE."""
    scorer = WeightedScorer({"claude": 1.0, "gemini": 1.0, "perplexity": 1.0})

    tweet_url = "https://x.com/cryptonews/status/123"

    responses = [
        TweetLLMResponse(
            provider="claude",
            model="claude-test",
            verdict=TweetVerdictType.CREDIBLE,
            confidence=0.9,
            analysis="Verified through multiple sources",
            identified_claims=["Bitcoin at $150k"],
            red_flags=[],
            raw_response="credible",
            error=None,
        ),
        TweetLLMResponse(
            provider="gemini",
            model="gemini-test",
            verdict=TweetVerdictType.CREDIBLE,
            confidence=0.85,
            analysis="Confirmed by reliable sources",
            identified_claims=["Bitcoin price claim"],
            red_flags=[],
            raw_response="credible",
            error=None,
        ),
        TweetLLMResponse(
            provider="perplexity",
            model="perplexity-test",
            verdict=TweetVerdictType.CREDIBLE,
            confidence=0.95,
            analysis="Accurate reporting",
            identified_claims=["Bitcoin ATH"],
            red_flags=[],
            raw_response="credible",
            error=None,
        ),
    ]

    result = scorer.aggregate_tweet_responses(tweet_url, responses)

    assert result.final_verdict == TweetVerdictType.CREDIBLE
    assert result.final_confidence > 0.8
    assert len(result.llm_responses) == 3
    assert result.tweet.url == tweet_url


def test_weighted_tweet_decision():
    """Test scoring with different weights (higher weight wins)."""
    scorer = WeightedScorer({"claude": 2.0, "gemini": 1.0})

    tweet_url = "https://x.com/gossip/status/456"

    responses = [
        TweetLLMResponse(
            provider="claude",
            model="claude-test",
            verdict=TweetVerdictType.QUESTIONABLE,
            confidence=0.9,
            analysis="No credible sources",
            identified_claims=["Celebrity claim"],
            red_flags=["Unverified source"],
            raw_response="questionable",
            error=None,
        ),
        TweetLLMResponse(
            provider="gemini",
            model="gemini-test",
            verdict=TweetVerdictType.MISLEADING,
            confidence=0.9,
            analysis="Appears to be false",
            identified_claims=["Celebrity claim"],
            red_flags=["Contradicts known facts"],
            raw_response="misleading",
            error=None,
        ),
    ]

    result = scorer.aggregate_tweet_responses(tweet_url, responses)

    # Claude has 2x weight, so QUESTIONABLE should win
    assert result.final_verdict == TweetVerdictType.QUESTIONABLE


def test_opinion_verdict():
    """Test that OPINION verdict is correctly aggregated."""
    scorer = WeightedScorer({"claude": 1.0, "gemini": 1.0})

    tweet_url = "https://x.com/fan/status/789"

    responses = [
        TweetLLMResponse(
            provider="claude",
            model="claude-test",
            verdict=TweetVerdictType.OPINION,
            confidence=1.0,
            analysis="Purely subjective opinion",
            identified_claims=[],
            red_flags=[],
            raw_response="opinion",
            error=None,
        ),
        TweetLLMResponse(
            provider="gemini",
            model="gemini-test",
            verdict=TweetVerdictType.OPINION,
            confidence=1.0,
            analysis="Subjective viewpoint",
            identified_claims=[],
            red_flags=[],
            raw_response="opinion",
            error=None,
        ),
    ]

    result = scorer.aggregate_tweet_responses(tweet_url, responses)

    assert result.final_verdict == TweetVerdictType.OPINION
    assert result.final_confidence == 1.0


def test_handles_tweet_errors_gracefully():
    """Test that scoring handles LLM errors gracefully."""
    scorer = WeightedScorer({"claude": 1.0, "gemini": 1.0})

    tweet_url = "https://x.com/test/status/111"

    responses = [
        TweetLLMResponse(
            provider="claude",
            model="claude-test",
            verdict=TweetVerdictType.CREDIBLE,
            confidence=0.9,
            analysis="Verified",
            identified_claims=["Test claim"],
            red_flags=[],
            raw_response="credible",
            error=None,
        ),
        TweetLLMResponse(
            provider="gemini",
            model="gemini-test",
            verdict=TweetVerdictType.QUESTIONABLE,
            confidence=0.0,
            analysis="Error occurred",
            identified_claims=[],
            red_flags=[],
            raw_response="",
            error="API timeout",
        ),
    ]

    result = scorer.aggregate_tweet_responses(tweet_url, responses)

    # Should still produce a result even with one error
    assert result.final_verdict in [
        TweetVerdictType.CREDIBLE,
        TweetVerdictType.QUESTIONABLE,
        TweetVerdictType.MISLEADING,
        TweetVerdictType.OPINION,
    ]
    assert len(result.llm_responses) == 2


def test_all_providers_failed():
    """Test behavior when all providers fail."""
    scorer = WeightedScorer({"claude": 1.0, "gemini": 1.0})

    tweet_url = "https://x.com/test/status/222"

    responses = [
        TweetLLMResponse(
            provider="claude",
            model="claude-test",
            verdict=TweetVerdictType.QUESTIONABLE,
            confidence=0.0,
            analysis="Error",
            identified_claims=[],
            red_flags=[],
            raw_response="",
            error="Network error",
        ),
        TweetLLMResponse(
            provider="gemini",
            model="gemini-test",
            verdict=TweetVerdictType.QUESTIONABLE,
            confidence=0.0,
            analysis="Error",
            identified_claims=[],
            red_flags=[],
            raw_response="",
            error="Network error",
        ),
    ]

    result = scorer.aggregate_tweet_responses(tweet_url, responses)

    # Should default to QUESTIONABLE with 0 confidence when all fail
    assert result.final_verdict == TweetVerdictType.QUESTIONABLE
    assert result.final_confidence == 0.0


def test_mixed_verdicts():
    """Test aggregation with mixed verdicts."""
    scorer = WeightedScorer({"claude": 1.0, "gemini": 1.0, "openai": 1.0})

    tweet_url = "https://x.com/news/status/333"

    responses = [
        TweetLLMResponse(
            provider="claude",
            model="claude-test",
            verdict=TweetVerdictType.QUESTIONABLE,
            confidence=0.7,
            analysis="Lacks verification",
            identified_claims=["Political claim"],
            red_flags=["No sources cited"],
            raw_response="questionable",
            error=None,
        ),
        TweetLLMResponse(
            provider="gemini",
            model="gemini-test",
            verdict=TweetVerdictType.MISLEADING,
            confidence=0.8,
            analysis="Contains false information",
            identified_claims=["Political claim"],
            red_flags=["Contradicts fact-checkers"],
            raw_response="misleading",
            error=None,
        ),
        TweetLLMResponse(
            provider="openai",
            model="openai-test",
            verdict=TweetVerdictType.QUESTIONABLE,
            confidence=0.6,
            analysis="Dubious sourcing",
            identified_claims=["Political claim"],
            red_flags=["Questionable source"],
            raw_response="questionable",
            error=None,
        ),
    ]

    result = scorer.aggregate_tweet_responses(tweet_url, responses)

    # QUESTIONABLE has 2 votes vs MISLEADING with 1, so should win
    assert result.final_verdict == TweetVerdictType.QUESTIONABLE


def test_claims_and_red_flags_aggregation():
    """Test that claims and red flags are collected across providers."""
    scorer = WeightedScorer({"claude": 1.0, "gemini": 1.0})

    tweet_url = "https://x.com/test/status/444"

    responses = [
        TweetLLMResponse(
            provider="claude",
            model="claude-test",
            verdict=TweetVerdictType.QUESTIONABLE,
            confidence=0.7,
            analysis="Mixed credibility",
            identified_claims=["Claim A", "Claim B"],
            red_flags=["Red flag 1"],
            raw_response="questionable",
            error=None,
        ),
        TweetLLMResponse(
            provider="gemini",
            model="gemini-test",
            verdict=TweetVerdictType.QUESTIONABLE,
            confidence=0.8,
            analysis="Needs verification",
            identified_claims=["Claim B", "Claim C"],
            red_flags=["Red flag 2", "Red flag 3"],
            raw_response="questionable",
            error=None,
        ),
    ]

    result = scorer.aggregate_tweet_responses(tweet_url, responses)

    # Summary should mention identified claims and red flags
    assert "Identified Claims" in result.analysis_summary or "Red Flags" in result.analysis_summary
