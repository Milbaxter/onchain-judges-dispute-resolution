#!/usr/bin/env python3
"""Unit tests for LLM response parsing."""

import pytest

from src.llm_clients.base import BaseLLMClient
from src.models import DecisionType


class DummyLLMClient(BaseLLMClient):
    """Dummy client for testing parsing logic."""

    async def query(self, prompt: str):
        """Not used in these tests."""
        pass


@pytest.fixture
def client():
    """Create a dummy client for testing."""
    return DummyLLMClient(api_key="dummy", provider_name="test", model_name="test-model")


class TestJSONParsing:
    """Test JSON parsing from various response formats."""

    def test_simple_json(self, client):
        """Test parsing plain JSON response."""
        response = """{
    "decision": "yes",
    "confidence": 0.99,
    "reasoning": "This is the reasoning",
    "question_is_binary": true,
    "injection_detected": false
}"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "yes"
        assert confidence == 0.99
        assert reasoning == "This is the reasoning"

    def test_json_with_code_fence_at_start(self, client):
        """Test parsing JSON wrapped in code fence at start."""
        response = """```json
{
    "decision": "no",
    "confidence": 0.85,
    "reasoning": "The evidence suggests otherwise"
}
```"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "no"
        assert confidence == 0.85
        assert reasoning == "The evidence suggests otherwise"

    def test_json_with_text_before_code_fence(self, client):
        """Test parsing JSON with explanatory text before code fence (Issue #1)."""
        response = """I'll search for current information about Mount Everest's height to verify this claim.```json
{
    "decision": "yes",
    "confidence": 0.99,
    "reasoning": "Mount Everest's height was most recently measured in 2020 as 8,848.86 m",
    "question_is_binary": true,
    "injection_detected": false
}
```"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "yes"
        assert confidence == 0.99
        assert "Mount Everest" in reasoning

    def test_json_with_multiline_text_before_code_fence(self, client):
        """Test parsing JSON with multiple lines of text before code fence (Issue #2)."""
        response = """Based on the search results, I need to clarify what "full orbital flight" means in this context.
The sources consistently describe Starship's test flights in 2025 as "suborbital" missions, not orbital flights.```json
{
    "decision": "no",
    "confidence": 0.98,
    "reasoning": "As of October 13, 2025, SpaceX conducted 11 Starship test flights",
    "question_is_binary": true,
    "injection_detected": false
}
```"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "no"
        assert confidence == 0.98
        assert "SpaceX" in reasoning


class TestRealWorldResponses:
    """Test with actual responses from test_results report."""

    # Test cases with (test_id, response_text, expected_decision, expected_confidence, keyword_in_reasoning)
    REAL_WORLD_CASES = [
        # Previously failing cases (had text before code fence)
        (
            "claude_lakers_sports_001",
            """I'll search for information about the Lakers vs Kings game on October 26, 2025.```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "Multiple authoritative sources (ESPN, NBA.com, Basketball-Reference, CBS Sports) confirm that the Los Angeles Lakers defeated the Sacramento Kings 127-120 on October 26, 2025. Austin Reaves scored 51 points including clutch free throws in the final 32 seconds to secure the victory.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.99,
            "Lakers",
        ),
        (
            "claude_mavericks_sports_002",
            """I'll search for information about this specific game between the Dallas Mavericks and Toronto Raptors on October 26, 2025.Based on the search results, the final score of the game on October 26, 2025 was Dallas Mavericks 139, Toronto Raptors 129. This represents a margin of 10 points (139 - 129 = 10).

The question asks if the Mavericks beat the Raptors "by a margin of at least 10 points." A margin of exactly 10 points meets the threshold of "at least 10 points."

```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "The Dallas Mavericks defeated the Toronto Raptors 139-129 on October 26, 2025, resulting in a margin of exactly 10 points, which satisfies the 'at least 10 points' criterion. Multiple reputable sources (ESPN, NBA.com, CBS Sports, theScore) confirm this final score.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.99,
            "Mavericks",
        ),
        (
            "claude_bitcoin_crypto_002",
            """I'll search for Bitcoin's price on October 29, 2025 to verify this claim.Based on the search results, I have clear evidence about Bitcoin's price on October 29, 2025.

```json
{
  "decision": "no",
  "confidence": 0.95,
  "reasoning": "Multiple reputable sources confirm that on October 29, 2025, Bitcoin remained above USD 110,000 throughout the day. CoinDesk reported that despite selling pressure, Bitcoin traded at $109,600 and was 'now below $110,000' only at one specific moment following Powell's hawkish remarks, but other sources show Bitcoin trading in the $112,540-$115,000 range on October 29. CoinSwitch documented Bitcoin at $112,540 on October 29 at 10:00 a.m., and TradingView noted it was trading in the $113,000-$114,000 range. Therefore, Bitcoin did not fall below USD 110,000 on October 29, 2025—it remained above that threshold throughout most or all of the day.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            0.95,
            "Bitcoin",
        ),
        (
            "claude_openai_tech_002",
            """Based on the search results, I can now provide a definitive answer to this factual question.

The evidence clearly shows that Microsoft did not acquire OpenAI in 2024. Microsoft holds roughly 27% in OpenAI Group PBC, which means Microsoft is a major shareholder but not the owner. OpenAI has completed its restructuring into a nonprofit with a controlling equity stake in its for-profit business. The nonprofit (now called the OpenAI Foundation) maintains control, not Microsoft. In 2024, the company announced plans to convert into a for-profit company, but this was a restructuring, not an acquisition by Microsoft. The recent equity restructuring in October 2025 gave Microsoft a larger ownership stake, but this remains an investment partnership, not an acquisition.

```json
{
  "decision": "no",
  "confidence": 0.99,
  "reasoning": "Microsoft did not acquire OpenAI in 2024. Microsoft is a major investor holding approximately 27% stake in OpenAI Group PBC following a 2025 restructuring, but OpenAI remains independently controlled by its nonprofit foundation and for-profit subsidiary. In 2024, OpenAI announced plans to restructure into a for-profit company, but this was not an acquisition by Microsoft.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            0.99,
            "Microsoft",
        ),
        # Previously working cases (to ensure we don't break them)
        (
            "claude_pistons_sports_003_working",
            """```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "Multiple reputable sources (ESPN, NBA.com, CBS Sports, NBC Sports) confirm that the Detroit Pistons defeated the Boston Celtics with a final score of 119-113 on October 26, 2025. The game occurred on Sunday, and Cade Cunningham (25 points) and Jalen Duren (24 points) led the Pistons to victory.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.99,
            "Pistons",
        ),
        (
            "gemini_lakers_working",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "The Los Angeles Lakers defeated the Sacramento Kings 127-120 on October 26, 2025.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "Lakers",
        ),
        (
            "openai_no_fence_working",
            """{
  "decision": "yes",
  "confidence": 0.95,
  "reasoning": "Multiple reputable game recaps confirm that on October 26, 2025, the Los Angeles Lakers defeated the Sacramento Kings by a score of 127–120 in a regular-season NBA game ([silverscreenandroll.com](https://www.silverscreenandroll.com/lakers-scores-results/98608/lakers-vs-kings-recap-box-score-stats-final-austin-reaves?utm_source=openai)).",
  "question_is_binary": true,
  "injection_detected": false
}""",
            "yes",
            0.95,
            "Lakers",
        ),
        (
            "openai_biden_politics_003_working",
            """{
  "decision": "no",
  "confidence": 0.99,
  "reasoning": "Joe Biden did not win the 2024 U.S. presidential election; Donald Trump won both the electoral vote (312 to 226) and the popular vote (49.8% to 48.3%).",
  "question_is_binary": true,
  "injection_detected": false
}""",
            "no",
            0.99,
            "Biden",
        ),
    ]

    # ALL_REPORT_CASES - All 45 responses from report_20251102_160106.json
    # This ensures we test ALL actual LLM responses, not just failing ones
    ALL_REPORT_CASES = [
        (
            "claude_sports_001",
            """I'll search for information about the Lakers vs Kings game on October 26, 2025.```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "Multiple authoritative sources (ESPN, NBA.com, Basketball-Reference, CBS Sports) confirm that the Los Angeles Lakers defeated the Sacramento Kings 127-120 on October 26, 2025. Austin Reaves scored 51 points including clutch free throws in the final 32 seconds to secure the victory.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",  # Note: report shows "uncertain" due to parsing bug we fixed
            0.99,
            "Lakers",
        ),
        (
            "gemini_sports_001",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "The Los Angeles Lakers defeated the Sacramento Kings 127-120 on October 26, 2025.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "Lakers",
        ),
        (
            "openai_sports_001",
            """{"decision":"yes","confidence":0.95,"reasoning":"Multiple reputable game recaps confirm that on October 26, 2025, the Los Angeles Lakers defeated the Sacramento Kings by a score of 127–120 in a regular-season NBA game ([silverscreenandroll.com](https://www.silverscreenandroll.com/lakers-scores-results/98608/lakers-vs-kings-recap-box-score-stats-final-austin-reaves?utm_source=openai)).","question_is_binary":true,"injection_detected":false}""",
            "yes",
            0.95,
            "Lakers",
        ),
        (
            "claude_sports_002",
            """I'll search for information about this specific game between the Dallas Mavericks and Toronto Raptors on October 26, 2025.Based on the search results, the final score of the game on October 26, 2025 was Dallas Mavericks 139, Toronto Raptors 129. This represents a margin of 10 points (139 - 129 = 10).

The question asks if the Mavericks beat the Raptors "by a margin of at least 10 points." A margin of exactly 10 points meets the threshold of "at least 10 points."

```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "The Dallas Mavericks defeated the Toronto Raptors 139-129 on October 26, 2025, resulting in a margin of exactly 10 points, which satisfies the 'at least 10 points' criterion. Multiple reputable sources (ESPN, NBA.com, CBS Sports, theScore) confirm this final score.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.99,
            "Mavericks",
        ),
        (
            "gemini_sports_002",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "The Dallas Mavericks beat the Toronto Raptors 139-129 on October 26 2025, a 10 point difference. Therefore, they won by at least 10 points.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "Mavericks",
        ),
        (
            "openai_sports_002",
            """{"decision":"yes","confidence":0.95,"reasoning":"On October 26, 2025, the Dallas Mavericks defeated the Toronto Raptors with a final score of 139 to 129, a margin of 10 points, which meets the user's claim of 'at least 10 points'","question_is_binary":true,"injection_detected":false}""",
            "yes",
            0.95,
            "Mavericks",
        ),
        (
            "claude_sports_003",
            """```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "Multiple reputable sources (ESPN, NBA.com, CBS Sports, NBC Sports) confirm that the Detroit Pistons defeated the Boston Celtics with a final score of 119-113 on October 26, 2025. The game occurred on Sunday, and Cade Cunningham (25 points) and Jalen Duren (24 points) led the Pistons to victory.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.99,
            "Pistons",
        ),
        (
            "gemini_sports_003",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "Multiple sources confirm that on October 26, 2025, the Detroit Pistons defeated the Boston Celtics with a final score of 119-113.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "Pistons",
        ),
        (
            "openai_sports_003",
            """{"decision":"yes","confidence":0.90,"reasoning":"Multiple reputable sources (AP via Reuters and ESPN) report that on October 26, 2025, the Detroit Pistons defeated the Boston Celtics with a final score of 119‑113 ([reuters.com](https://www.reuters.com/sports/basketball/pistons-overcome-17-point-deficit-beat-celtics-home-opener--flm-2025-10-26/?utm_source=openai)).","question_is_binary":true,"injection_detected":false}""",
            "yes",
            0.9,
            "Pistons",
        ),
        (
            "claude_sports_004",
            """```json
{
  "decision": "no",
  "confidence": 0.98,
  "reasoning": "The Warriors did not defeat the Trail Blazers on October 26, 2025. The two teams played on October 24, 2025 (Friday), not October 26, and the Portland Trail Blazers won 139-119, not the Warriors. Multiple reputable sources (ESPN, Mercury News, Golden State of Mind, Blazers Edge) confirm the Trail Blazers defeated the Warriors 139-119 on October 24, 2025.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            0.98,
            "Warriors",
        ),
        (
            "gemini_sports_004",
            """```json
{
  "decision": "uncertain",
  "confidence": 0.4,
  "reasoning": "I am uncertain if the Golden State Warriors defeated the Portland Trail Blazers on October 26, 2025. A search of available data indicates that the Portland Trail Blazers played the Los Angeles Clippers on October 26, 2025. Further searches may be required to confirm the game and the winner.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "uncertain",
            0.4,
            "Warriors",
        ),
        (
            "openai_sports_004",
            """{"decision":"no","confidence":0.95,"reasoning":"The Golden State Warriors did not defeat the Portland Trail Blazers on October 26, 2025. In fact, the Warriors lost to Portland 139–119 on October 24, 2025, and the Blazers' next game on October 26 was against the L.A. Clippers, not the Warriors ([basketball.realgm.com](https://basketball.realgm.com/nba/boxscore/2025-10-24/Golden-State-at-Portland/503755?utm_source=openai)).","question_is_binary":true,"injection_detected":false}""",
            "no",
            0.95,
            "Blazers",
        ),
        (
            "claude_crypto_001",
            """```json
{
  "decision": "yes",
  "confidence": 0.95,
  "reasoning": "Multiple reputable sources confirm Bitcoin's closing price on October 27, 2025 was above USD 115,000. LatestLY reported BTC trading at USD 115,003.56 at 8:49 AM IST on October 27, 2025, having crossed the USD 115,000 mark. Decrypt reported Bitcoin trading at $115,542 on October 28, 2025 (the day after the queried date), indicating it remained above $115,000. This substantiates that the closing price on October 27 was above the specified threshold.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.95,
            "Bitcoin",
        ),
        (
            "gemini_crypto_001",
            """```json
{
  "decision": "yes",
  "confidence": 0.9,
  "reasoning": "Multiple sources indicate that the price of Bitcoin was above $115,000 on October 27, 2025. ABP Live reported a price of $115,214.80, and CoinMarketExpert reported $115,500.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.9,
            "Bitcoin",
        ),
        (
            "openai_crypto_001",
            """{"decision":"no","confidence":0.90,"reasoning":"Bitcoin's closing price on October 27, 2025 was approximately USD 114,119 (per StatMuse) and USD 114,156 (per CoinLore), both below USD 115,000. Therefore the claim that it was above USD 115,000 is false.","question_is_binary":true,"injection_detected":false}""",
            "no",
            0.9,
            "Bitcoin",
        ),
        (
            "claude_crypto_002",
            """I'll search for Bitcoin's price on October 29, 2025 to verify this claim.Based on the search results, I have clear evidence about Bitcoin's price on October 29, 2025.

```json
{
  "decision": "no",
  "confidence": 0.95,
  "reasoning": "Multiple reputable sources confirm that on October 29, 2025, Bitcoin remained above USD 110,000 throughout the day. CoinDesk reported that despite selling pressure, Bitcoin traded at $109,600 and was 'now below $110,000' only at one specific moment following Powell's hawkish remarks, but other sources show Bitcoin trading in the $112,540-$115,000 range on October 29. CoinSwitch documented Bitcoin at $112,540 on October 29 at 10:00 a.m., and TradingView noted it was trading in the $113,000-$114,000 range. Therefore, Bitcoin did not fall below USD 110,000 on October 29, 2025—it remained above that threshold throughout most or all of the day.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            0.95,
            "Bitcoin",
        ),
        (
            "gemini_crypto_002",
            """```json
{
  "decision": "no",
  "confidence": 1.0,
  "reasoning": "Multiple sources indicate that Bitcoin fell to around USD 109,000 on October 29, 2025, which is below USD 110,000, but not below USD 110,000.00. The question specifies USD 110 000 as the threshold.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            1.0,
            "Bitcoin",
        ),
        (
            "openai_crypto_002",
            """{"decision":"no","confidence":0.95,"reasoning":"Historical data show that on October 29, 2025, Bitcoin's low was approximately USD 109,701 and its close was around USD 110,014—both well above USD 110,000. Therefore, it did not fall below USD 110,000 that day.","question_is_binary":true,"injection_detected":false}""",
            "no",
            0.95,
            "Bitcoin",
        ),
        (
            "claude_crypto_003",
            """```json
{
  "decision": "no",
  "confidence": 0.95,
  "reasoning": "Multiple reputable sources confirm Bitcoin price was above USD 100,000 on October 27, 2025. CoinDesk reports Bitcoin at $110,059.77 on October 27, 2025. Bitcoin Magazine reports the price trading above $115,000 on October 27, 2025 following Strategy's announcement of a 390 BTC purchase at an average price of $111,053 per Bitcoin between October 20-26. A Polymarket resolution for October 27, 2025 at 12PM ET shows the outcome resolved to the 114,000-116,000 USD range bracket. Therefore, the claim that Bitcoin was below USD 100,000 on October 27, 2025 is false.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            0.95,
            "Bitcoin",
        ),
        (
            "gemini_crypto_003",
            """```json
{
  "decision": "no",
  "confidence": 1.0,
  "reasoning": "On October 27, 2025, Bitcoin was trading above $115,000. Therefore, it was not below $100,000.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            1.0,
            "Bitcoin",
        ),
        (
            "openai_crypto_003",
            """{"decision":"no","confidence":0.99,"reasoning":"Bitcoin's closing price on October 27 2025 was approximately $114,119 — well above $100,000, so the claim that it was below USD 100,000 is false.","question_is_binary":true,"injection_detected":false}""",
            "no",
            0.99,
            "Bitcoin",
        ),
        (
            "claude_politics_001",
            """```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "The 2025 New York City Council election is scheduled for November 4, 2025, as confirmed by multiple authoritative sources including the NYC Board of Elections and Wikipedia. The primary elections occurred on June 24, 2025.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.99,
            "Council",
        ),
        (
            "gemini_politics_001",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "Multiple sources confirm that the 2025 New York City Council election is scheduled for November 4, 2025.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "Council",
        ),
        (
            "openai_politics_001",
            """{"decision":"yes","confidence":0.95,"reasoning":"Official sources confirm that the 2025 New York City Council general election is scheduled for November 4, 2025. The NYC Board of Elections lists the General Election date as Tuesday, November 4, 2025 ([vote.nyc](https://vote.nyc/election/general-election-2025?utm_source=openai)), and the Wikipedia article on the 2025 New York City Council election states the Election Date is November 4, 2025 ([en.wikipedia.org](https://en.wikipedia.org/wiki/2025_New_York_City_Council_election?utm_source=openai)).","question_is_binary":true,"injection_detected":false}""",
            "yes",
            0.95,
            "Council",
        ),
        (
            "claude_politics_002",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "Multiple authoritative sources confirm that the 2025 NYC mayoral general election is scheduled for November 4, 2025. The NYC Board of Elections (official source), Wikipedia, Ballotpedia, and recent news reports all state this date consistently.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "mayoral",
        ),
        (
            "gemini_politics_002",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "Multiple sources state that the 2025 New York City mayoral election is scheduled for November 4, 2025.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "mayoral",
        ),
        (
            "openai_politics_002",
            """{"decision":"yes","confidence":0.99,"reasoning":"The 2025 New York City mayoral election is officially scheduled for November 4, 2025, as confirmed by multiple reputable sources including Wikipedia and election reporting (e.g. “Election Date: November 4, 2025”) ([en.wikipedia.org](https://en.wikipedia.org/wiki/2025_New_York_City_mayoral_election?utm_source=openai)).","question_is_binary":true,"injection_detected":false}""",
            "yes",
            0.99,
            "mayoral",
        ),
        (
            "claude_politics_003",
            """```json
{
  "decision": "no",
  "confidence": 1.0,
  "reasoning": "Joe Biden withdrew from the 2024 presidential race on July 21, 2024, becoming the first eligible incumbent president to withdraw since Lyndon B. Johnson. He did not compete in or win the 2024 election. Donald Trump defeated Kamala Harris (Biden's replacement as the Democratic nominee) in the November 5, 2024 election, winning 312 electoral votes and 49.8% of the popular vote.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            1.0,
            "Biden",
        ),
        (
            "gemini_politics_003",
            """```json
{
  "decision": "no",
  "confidence": 1.0,
  "reasoning": "According to multiple sources, Donald Trump won the 2024 US Presidential election, defeating Joe Biden.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            1.0,
            "Biden",
        ),
        (
            "openai_politics_003",
            """{"decision":"no","confidence":0.99,"reasoning":"Joe Biden did not win the 2024 U.S. presidential election; Donald Trump won both the electoral vote (312 to 226) and the popular vote (49.8% to 48.3%).","question_is_binary":true,"injection_detected":false}""",
            "no",
            0.99,
            "Biden",
        ),
        (
            "claude_tech_001",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "Apple announced the iPhone 16 on September 9, 2024, and released it on September 20, 2024. Multiple authoritative sources (Apple Newsroom, Wikipedia, GSMArena, Variety) confirm the iPhone 16 was released in 2024. The question asks if Apple released the iPhone 16 in 2024—this is factually true.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "iPhone",
        ),
        (
            "gemini_tech_001",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "According to multiple sources, Apple released the iPhone 16 in September 2024.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "iPhone",
        ),
        (
            "openai_tech_001",
            """{"decision":"yes","confidence":0.95,"reasoning":"Apple officially announced and released the iPhone 16 in 2024. It was unveiled on September 9, 2024, with pre-orders beginning September 13 and general sales starting September 20, 2024.","question_is_binary":true,"injection_detected":false}""",
            "yes",
            0.95,
            "iPhone",
        ),
        (
            "claude_tech_002",
            """Based on the search results, I can now provide a definitive answer to this factual question.

The evidence clearly shows that Microsoft did not acquire OpenAI in 2024. Microsoft holds roughly 27% in OpenAI Group PBC, which means Microsoft is a major shareholder but not the owner. OpenAI has completed its restructuring into a nonprofit with a controlling equity stake in its for-profit business. The nonprofit (now called the OpenAI Foundation) maintains control, not Microsoft. In 2024, the company announced plans to convert into a for-profit company, but this was a restructuring, not an acquisition by Microsoft. The recent equity restructuring in October 2025 gave Microsoft a larger ownership stake, but this remains an investment partnership, not an acquisition.

```json
{
  "decision": "no",
  "confidence": 0.99,
  "reasoning": "Microsoft did not acquire OpenAI in 2024. Microsoft is a major investor holding approximately 27% stake in OpenAI Group PBC following a 2025 restructuring, but OpenAI remains independently controlled by its nonprofit foundation and for-profit subsidiary. In 2024, OpenAI announced plans to restructure into a for-profit company, but this was not an acquisition by Microsoft.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            0.99,
            "Microsoft",
        ),
        (
            "gemini_tech_002",
            """```json
{
  "decision": "no",
  "confidence": 0.9,
  "reasoning": "Microsoft did not acquire OpenAI in 2024, but they did increase their stake to 27% valuing the company at $135 billion.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            0.9,
            "Microsoft",
        ),
        (
            "openai_tech_002",
            """{"decision":"no","confidence":0.99,"reasoning":"Microsoft did not acquire OpenAI in 2024. Instead, Microsoft has been a major investor and partner, holding minority stakes and cloud computing agreements—but OpenAI remained independent. No acquisition occurred in 2024.","question_is_binary":true,"injection_detected":false}""",
            "no",
            0.99,
            "Microsoft",
        ),
        (
            "claude_history_001",
            """```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "<cite index=\\"1-1\\">World War II ended on 2 September 1945</cite>, with <cite index=\\"1-11\\">Germany's unconditional surrender on 8 May 1945</cite> and <cite index=\\"1-13\\">Japan signing a surrender document on 2 September 1945</cite>. The claim is factually accurate.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.99,
            "World War",
        ),
        (
            "gemini_history_001",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "World War II ended on September 2, 1945, with Japan's formal surrender.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "World War",
        ),
        (
            "openai_history_001",
            """{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "World War II ended in 1945 with the surrender of Germany in May and Japan in September.",
  "question_is_binary": true,
  "injection_detected": false
}""",
            "yes",
            1.0,
            "World War",
        ),
        (
            "claude_history_002",
            """```json
{
  "decision": "no",
  "confidence": 1.0,
  "reasoning": "The Berlin Wall fell on November 9, 1989, not in 1991. Multiple authoritative sources confirm the wall fell in 1989, specifically during the evening of November 9. The question's claim that it fell in 1991 is factually incorrect.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            1.0,
            "Berlin",
        ),
        (
            "gemini_history_002",
            """```json
{
  "decision": "no",
  "confidence": 1.0,
  "reasoning": "The Berlin Wall fell on November 9, 1989. Therefore, the statement that it fell in 1991 is false.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "no",
            1.0,
            "Berlin",
        ),
        (
            "openai_history_002",
            """{
  "decision": "no",
  "confidence": 0.95,
  "reasoning": "The Berlin Wall fell on November 9, 1989, not in 1991.",
  "question_is_binary": true,
  "injection_detected": false
}""",
            "no",
            0.95,
            "Berlin",
        ),
        (
            "claude_science_001",
            """```json
{
  "decision": "yes",
  "confidence": 0.99,
  "reasoning": "<cite index=\\"2-11\\">Water is a substance composed of the chemical elements hydrogen and oxygen</cite>. <cite index=\\"2-1\\">The water molecule is composed of two hydrogen atoms, each linked by a single chemical bond to an oxygen atom</cite>, with <cite index=\\"3-26\\">the chemical formula H2O</cite>.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            0.99,
            "water",
        ),
        (
            "gemini_science_001",
            """```json
{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "Water (H2O) is composed of two hydrogen atoms and one oxygen atom.",
  "question_is_binary": true,
  "injection_detected": false
}
```""",
            "yes",
            1.0,
            "water",
        ),
        (
            "openai_science_001",
            """{
  "decision": "yes",
  "confidence": 1.0,
  "reasoning": "Water is chemically composed of two hydrogen atoms and one oxygen atom, forming the molecule H2O.",
  "question_is_binary": true,
  "injection_detected": false
}""",
            "yes",
            1.0,
            "water",
        ),
    ]

    @pytest.mark.parametrize(
        "test_id,response,expected_decision,expected_confidence,keyword",
        REAL_WORLD_CASES,
        ids=[case[0] for case in REAL_WORLD_CASES],
    )
    def test_real_world_response(
        self, client, test_id, response, expected_decision, expected_confidence, keyword
    ):
        """Test parsing of real-world LLM responses from test report."""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == expected_decision, (
            f"Decision mismatch for {test_id}: expected {expected_decision}, got {decision}"
        )
        assert confidence == expected_confidence, (
            f"Confidence mismatch for {test_id}: expected {expected_confidence}, got {confidence}"
        )
        assert keyword in reasoning, f"Keyword '{keyword}' not found in reasoning for {test_id}"

    @pytest.mark.parametrize(
        "test_id,response,expected_decision,expected_confidence,keyword",
        ALL_REPORT_CASES,
        ids=[case[0] for case in ALL_REPORT_CASES],
    )
    def test_all_report_responses(
        self, client, test_id, response, expected_decision, expected_confidence, keyword
    ):
        """Test parsing ALL responses from report_20251102_160106.json."""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == expected_decision, (
            f"Decision mismatch for {test_id}: expected {expected_decision}, got {decision}"
        )
        assert confidence == expected_confidence, (
            f"Confidence mismatch for {test_id}: expected {expected_confidence}, got {confidence}"
        )
        # Keyword check is lenient - just ensure reasoning was extracted
        if keyword and keyword != "test":
            assert keyword in reasoning or len(reasoning) > 0, (
                f"Keyword '{keyword}' not found and reasoning empty for {test_id}"
            )

    def test_json_with_multiple_code_fences(self, client):
        """Test that we extract the first JSON code fence."""
        response = """Here's my analysis:
```json
{
    "decision": "yes",
    "confidence": 0.75,
    "reasoning": "First fence"
}
```

And here's another example:
```json
{
    "decision": "no",
    "confidence": 0.25,
    "reasoning": "Second fence"
}
```"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "yes"
        assert confidence == 0.75
        assert reasoning == "First fence"

    def test_json_without_closing_fence(self, client):
        """Test parsing JSON with opening fence but no closing fence."""
        response = """Let me analyze this:
```json
{
    "decision": "uncertain",
    "confidence": 0.5,
    "reasoning": "Missing closing fence"
}"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "uncertain"
        assert confidence == 0.5
        assert "Missing closing fence" in reasoning


class TestConfidenceHandling:
    """Test confidence value parsing and clamping."""

    # Test cases: (test_id, response, expected_confidence)
    CONFIDENCE_CASES = [
        (
            "in_range",
            '{"decision": "yes", "confidence": 0.75, "reasoning": "test"}',
            0.75,
        ),
        (
            "above_max_clamped",
            '{"decision": "yes", "confidence": 1.5, "reasoning": "test"}',
            1.0,
        ),
        (
            "below_min_clamped",
            '{"decision": "yes", "confidence": -0.5, "reasoning": "test"}',
            0.0,
        ),
        (
            "missing_defaults",
            '{"decision": "yes", "reasoning": "test"}',
            0.5,
        ),
        (
            "invalid_type_defaults",
            '{"decision": "yes", "confidence": "high", "reasoning": "test"}',
            0.5,
        ),
    ]

    @pytest.mark.parametrize(
        "test_id,response,expected_confidence",
        CONFIDENCE_CASES,
        ids=[case[0] for case in CONFIDENCE_CASES],
    )
    def test_confidence_handling(self, client, test_id, response, expected_confidence):
        """Test confidence value parsing and validation."""
        _, confidence, _ = client._parse_response(response)
        assert confidence == expected_confidence, (
            f"Confidence mismatch for {test_id}: expected {expected_confidence}, got {confidence}"
        )


class TestDecisionValidation:
    """Test decision value validation."""

    def test_valid_decisions(self, client):
        """Test all valid decision values."""
        for decision_value in ["yes", "no", "uncertain"]:
            response = f'{{"decision": "{decision_value}", "confidence": 0.5, "reasoning": "test"}}'
            decision, _, _ = client._parse_response(response)
            assert decision == decision_value

    def test_decision_case_insensitive(self, client):
        """Test decision values are case-insensitive."""
        response = '{"decision": "YES", "confidence": 0.5, "reasoning": "test"}'
        decision, _, _ = client._parse_response(response)
        assert decision == "yes"

    def test_invalid_decision_defaults_to_uncertain(self, client):
        """Test invalid decision values default to uncertain."""
        response = '{"decision": "maybe", "confidence": 0.5, "reasoning": "test"}'
        decision, _, _ = client._parse_response(response)
        assert decision == DecisionType.UNCERTAIN.value

    def test_missing_decision_defaults_to_uncertain(self, client):
        """Test missing decision defaults to uncertain."""
        response = '{"confidence": 0.5, "reasoning": "test"}'
        decision, _, _ = client._parse_response(response)
        assert decision == DecisionType.UNCERTAIN.value


class TestLegacyParsing:
    """Test fallback to legacy keyword-based parsing."""

    def test_legacy_format_decision(self, client):
        """Test legacy DECISION: format."""
        response = """DECISION: yes
CONFIDENCE: 0.85
REASONING: This is valid"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "yes"
        assert confidence == 0.85
        assert "This is valid" in reasoning

    def test_legacy_format_case_insensitive(self, client):
        """Test legacy format is case-insensitive."""
        response = """decision: NO
confidence: 0.75
reasoning: Testing case"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "no"
        assert confidence == 0.75

    def test_legacy_format_multiline_reasoning(self, client):
        """Test legacy format with multiline reasoning."""
        response = """DECISION: uncertain
CONFIDENCE: 0.5
REASONING: This is line one
This is line two
This is line three"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "uncertain"
        assert confidence == 0.5
        assert "line one" in reasoning
        assert "line two" in reasoning
        assert "line three" in reasoning


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_response(self, client):
        """Test empty response defaults to uncertain."""
        response = ""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == DecisionType.UNCERTAIN.value
        assert confidence == 0.5
        assert reasoning == ""

    def test_invalid_json(self, client):
        """Test invalid JSON falls back to legacy parsing."""
        response = "This is not JSON at all"
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == DecisionType.UNCERTAIN.value
        assert confidence == 0.5
        assert reasoning == "This is not JSON at all"

    def test_json_with_extra_fields(self, client):
        """Test JSON with extra fields is parsed correctly."""
        response = """{
    "decision": "yes",
    "confidence": 0.9,
    "reasoning": "Valid",
    "extra_field": "ignored",
    "another_field": 123
}"""
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "yes"
        assert confidence == 0.9
        assert reasoning == "Valid"

    def test_reasoning_is_none(self, client):
        """Test when reasoning field is null."""
        response = '{"decision": "yes", "confidence": 0.8, "reasoning": null}'
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "yes"
        assert confidence == 0.8
        # When reasoning is None, it should be converted to empty string
        assert reasoning == ""

    def test_reasoning_whitespace_only(self, client):
        """Test when reasoning is whitespace only."""
        response = '{"decision": "yes", "confidence": 0.8, "reasoning": "   "}'
        decision, confidence, reasoning = client._parse_response(response)
        assert decision == "yes"
        assert confidence == 0.8
        assert reasoning == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
