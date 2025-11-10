"""Unit tests for Tweet Analysis model validation."""

import pytest
from pydantic import ValidationError

from src.models import TweetAnalysisQuery, TweetVerdictType


def test_valid_twitter_url():
    """Test that valid Twitter URLs pass validation."""
    valid_urls = [
        "https://twitter.com/user/status/1234567890",
        "https://x.com/user/status/1234567890",
        "http://twitter.com/username/status/9876543210",
        "http://x.com/test_user/status/111222333444",
        "https://twitter.com/user123/status/1234567890?s=20",
        "https://x.com/user_name/status/1234567890#reply",
    ]

    for url in valid_urls:
        query = TweetAnalysisQuery(tweet_url=url)
        assert query.tweet_url == url


def test_query_too_short():
    """Test that URLs under 28 characters are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        TweetAnalysisQuery(tweet_url="short")

    assert "at least 28 characters" in str(exc_info.value)


def test_query_too_long():
    """Test that URLs over 200 characters are rejected."""
    long_url = "https://twitter.com/user/status/" + "a" * 200
    with pytest.raises(ValidationError) as exc_info:
        TweetAnalysisQuery(tweet_url=long_url)

    assert "at most 200 characters" in str(exc_info.value)


def test_invalid_url_pattern():
    """Test that non-Twitter URLs are rejected."""
    invalid_urls = [
        "https://facebook.com/post/123",
        "https://reddit.com/r/test",
        "https://twitter.com/user",  # Missing /status/
        "https://x.com/user/post/123",  # Wrong path (post instead of status)
        "not a url at all",
        "twitter.com/user/status/123",  # Missing https://
    ]

    for url in invalid_urls:
        with pytest.raises(ValidationError):
            TweetAnalysisQuery(tweet_url=url)


def test_verdict_types():
    """Test all valid tweet verdict types."""
    assert TweetVerdictType.CREDIBLE.value == "credible"
    assert TweetVerdictType.QUESTIONABLE.value == "questionable"
    assert TweetVerdictType.MISLEADING.value == "misleading"
    assert TweetVerdictType.OPINION.value == "opinion"


def test_url_minimum_length_boundary():
    """Test exact minimum length boundary."""
    # Minimum 28 characters.
    # https://x.com/u/status/12345 is exactly 28 characters.
    valid_short = "https://x.com/u/status/12345"
    query = TweetAnalysisQuery(tweet_url=valid_short)
    assert len(query.tweet_url) == 28


def test_url_maximum_length_boundary():
    """Test exact maximum length boundary."""
    # Create URL exactly at 200 chars
    base = "https://twitter.com/user/status/1234567890"
    padding = "?" + "a" * (200 - len(base) - 1)
    url_200 = base + padding

    query = TweetAnalysisQuery(tweet_url=url_200)
    assert len(query.tweet_url) == 200

    # 201 characters should fail
    url_201 = url_200 + "x"
    with pytest.raises(ValidationError):
        TweetAnalysisQuery(tweet_url=url_201)


def test_both_twitter_and_x_domains():
    """Test that both twitter.com and x.com domains are accepted."""
    twitter_url = "https://twitter.com/elonmusk/status/1234567890"
    x_url = "https://x.com/elonmusk/status/1234567890"

    twitter_query = TweetAnalysisQuery(tweet_url=twitter_url)
    x_query = TweetAnalysisQuery(tweet_url=x_url)

    assert "twitter.com" in twitter_query.tweet_url
    assert "x.com" in x_query.tweet_url


def test_url_with_query_params():
    """Test URLs with query parameters."""
    url_with_params = (
        "https://twitter.com/user/status/123?ref_src=twsrc&ref_url=https://example.com"
    )
    query = TweetAnalysisQuery(tweet_url=url_with_params)
    assert query.tweet_url == url_with_params


def test_url_with_fragment():
    """Test URLs with fragments."""
    url_with_fragment = "https://x.com/user/status/123#reply-456"
    query = TweetAnalysisQuery(tweet_url=url_with_fragment)
    assert query.tweet_url == url_with_fragment
