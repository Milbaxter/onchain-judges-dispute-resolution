"""Unit tests for Tweet LLM response parsing."""

import pytest

from src.llm_clients.base import BaseLLMClient
from src.models import TweetVerdictType


class DummyTweetClient(BaseLLMClient):
    """Dummy client for testing tweet parsing logic."""

    async def query(self, prompt: str):
        """Not used in these tests."""
        pass

    async def analyze_tweet(self, tweet_url: str):
        """Not used in these tests."""
        pass


@pytest.fixture
def client():
    """Create a dummy client for testing."""
    return DummyTweetClient(api_key="dummy", provider_name="test", model_name="test-model")


class TestTweetJSONParsing:
    """Test JSON parsing from tweet analysis responses."""

    def test_simple_tweet_json(self, client):
        """Test parsing plain JSON response for tweet analysis."""
        response = """{
    "verdict": "credible",
    "confidence": 0.95,
    "analysis": "This tweet contains verifiable facts",
    "identified_claims": ["Fact 1", "Fact 2"],
    "red_flags": []
}"""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == "credible"
        assert confidence == 0.95
        assert analysis == "This tweet contains verifiable facts"
        assert claims == ["Fact 1", "Fact 2"]
        assert flags == []

    def test_tweet_json_with_code_fence(self, client):
        """Test parsing JSON wrapped in code fence."""
        response = """```json
{
    "verdict": "misleading",
    "confidence": 0.85,
    "analysis": "The tweet contains out-of-context information",
    "identified_claims": ["Misleading stat"],
    "red_flags": ["Missing context", "Cherry-picked data"]
}
```"""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == "misleading"
        assert confidence == 0.85
        assert "out-of-context" in analysis
        assert len(claims) == 1
        assert len(flags) == 2

    def test_tweet_json_with_text_before(self, client):
        """Test parsing JSON with text before code fence."""
        response = """I'll analyze this tweet for credibility.```json
{
    "verdict": "questionable",
    "confidence": 0.70,
    "analysis": "Cannot verify the claims made in this tweet",
    "identified_claims": ["Unverified claim"],
    "red_flags": ["No sources provided"]
}
```"""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == "questionable"
        assert confidence == 0.70
        assert "Cannot verify" in analysis

    def test_all_verdict_types(self, client):
        """Test all valid tweet verdict values."""
        for verdict_value in ["credible", "questionable", "misleading", "opinion"]:
            response = f"""{{
    "verdict": "{verdict_value}",
    "confidence": 0.8,
    "analysis": "Test analysis",
    "identified_claims": [],
    "red_flags": []
}}"""
            verdict, _, _, _, _ = client._parse_tweet_response(response)
            assert verdict == verdict_value

    def test_verdict_case_insensitive(self, client):
        """Test verdict values are case-insensitive."""
        response = '{"verdict": "CREDIBLE", "confidence": 0.9, "analysis": "test", "identified_claims": [], "red_flags": []}'
        verdict, _, _, _, _ = client._parse_tweet_response(response)
        assert verdict == "credible"

    def test_invalid_verdict_defaults_to_questionable(self, client):
        """Test invalid verdict values default to questionable."""
        response = '{"verdict": "unknown", "confidence": 0.5, "analysis": "test", "identified_claims": [], "red_flags": []}'
        verdict, _, _, _, _ = client._parse_tweet_response(response)
        assert verdict == TweetVerdictType.QUESTIONABLE.value

    def test_confidence_clamping(self, client):
        """Test confidence values are clamped to [0.0, 1.0]."""
        # Above max
        response = '{"verdict": "credible", "confidence": 1.5, "analysis": "test", "identified_claims": [], "red_flags": []}'
        _, confidence, _, _, _ = client._parse_tweet_response(response)
        assert confidence == 1.0

        # Below min
        response = '{"verdict": "credible", "confidence": -0.5, "analysis": "test", "identified_claims": [], "red_flags": []}'
        _, confidence, _, _, _ = client._parse_tweet_response(response)
        assert confidence == 0.0

    def test_missing_fields_defaults(self, client):
        """Test missing fields use appropriate defaults."""
        response = '{"verdict": "credible"}'
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == "credible"
        assert confidence == 0.5  # default
        assert analysis == ""
        assert claims == []
        assert flags == []

    def test_claims_and_flags_parsing(self, client):
        """Test that claims and red flags arrays are parsed correctly."""
        response = """{
    "verdict": "questionable",
    "confidence": 0.65,
    "analysis": "Mixed credibility",
    "identified_claims": ["Claim A", "Claim B", "Claim C"],
    "red_flags": ["Flag 1", "Flag 2"]
}"""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert len(claims) == 3
        assert "Claim A" in claims
        assert "Claim B" in claims
        assert "Claim C" in claims
        assert len(flags) == 2
        assert "Flag 1" in flags
        assert "Flag 2" in flags

    def test_empty_claims_and_flags(self, client):
        """Test handling of empty claims and flags arrays."""
        response = """{
    "verdict": "opinion",
    "confidence": 1.0,
    "analysis": "Purely subjective content",
    "identified_claims": [],
    "red_flags": []
}"""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert claims == []
        assert flags == []

    def test_non_array_claims_defaults_to_empty(self, client):
        """Test that non-array claims/flags default to empty lists."""
        response = """{
    "verdict": "credible",
    "confidence": 0.9,
    "analysis": "Good",
    "identified_claims": "not an array",
    "red_flags": "also not an array"
}"""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert claims == []
        assert flags == []


class TestTweetEdgeCases:
    """Test edge cases for tweet parsing."""

    def test_empty_response(self, client):
        """Test empty response defaults to questionable."""
        response = ""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == TweetVerdictType.QUESTIONABLE.value
        assert confidence == 0.5
        assert analysis == ""

    def test_invalid_json(self, client):
        """Test invalid JSON falls back gracefully."""
        response = "This is not JSON at all"
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == TweetVerdictType.QUESTIONABLE.value
        assert confidence == 0.5
        assert analysis == "This is not JSON at all"

    def test_json_with_extra_fields(self, client):
        """Test JSON with extra fields is parsed correctly."""
        response = """{
    "verdict": "credible",
    "confidence": 0.9,
    "analysis": "Valid",
    "identified_claims": ["Claim"],
    "red_flags": [],
    "extra_field": "ignored",
    "another_field": 123
}"""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == "credible"
        assert confidence == 0.9
        assert analysis == "Valid"
        assert claims == ["Claim"]

    def test_analysis_is_none(self, client):
        """Test when analysis field is null."""
        response = '{"verdict": "credible", "confidence": 0.8, "analysis": null, "identified_claims": [], "red_flags": []}'
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == "credible"
        assert confidence == 0.8
        assert analysis == ""

    def test_analysis_whitespace_only(self, client):
        """Test when analysis is whitespace only."""
        response = '{"verdict": "credible", "confidence": 0.8, "analysis": "   ", "identified_claims": [], "red_flags": []}'
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == "credible"
        assert confidence == 0.8
        assert analysis == ""

    def test_multiline_text_before_code_fence(self, client):
        """Test parsing JSON with multiple lines of text before code fence."""
        response = """Let me analyze this tweet for credibility and misinformation.
First, I'll extract the main claims.
Then I'll verify each claim against reliable sources.```json
{
    "verdict": "misleading",
    "confidence": 0.88,
    "analysis": "The tweet contains partially true information presented in a misleading way",
    "identified_claims": ["Partial truth"],
    "red_flags": ["Missing context"]
}
```"""
        verdict, confidence, analysis, claims, flags = client._parse_tweet_response(response)
        assert verdict == "misleading"
        assert confidence == 0.88
        assert "partially true" in analysis.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
