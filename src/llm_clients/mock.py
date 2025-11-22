"""Mock LLM client for testing and development."""

import asyncio

from src.llm_clients.base import BaseLLMClient
from src.models import DecisionType, DisputeDecisionType, LLMResponse, TweetLLMResponse, TweetVerdictType


class MockLLMClient(BaseLLMClient):
    """Mock LLM client that returns fixed responses after a delay."""

    def __init__(self, provider_name: str = "mock", sleep_duration: float = 5.0):
        """Initialize mock client.

        Args:
            provider_name: Name to identify this mock provider
            sleep_duration: How long to sleep before returning (simulates API latency)
        """
        # Don't call super().__init__ since we don't need an API key.
        self.provider_name = provider_name
        self.sleep_duration = sleep_duration

    async def query(self, prompt: str) -> LLMResponse:
        """Return a mock response after sleeping.

        Args:
            prompt: The query (checked for dispute resolution format)

        Returns:
            LLMResponse with fixed mock data
        """
        # Simulate API latency.
        await asyncio.sleep(self.sleep_duration)

        # Check if this is a dispute resolution query (contains "Party A" and "Party B")
        is_dispute = "Party A" in prompt and "Party B" in prompt
        
        if is_dispute:
            # Return dispute resolution format (A/B)
            # Vary responses by provider to simulate consensus
            if "claude" in self.provider_name.lower():
                winning_party = DisputeDecisionType.A
                decision = DecisionType.YES
            elif "gemini" in self.provider_name.lower():
                winning_party = DisputeDecisionType.A
                decision = DecisionType.YES
            elif "perplexity" in self.provider_name.lower():
                winning_party = DisputeDecisionType.B
                decision = DecisionType.YES
            elif "openai" in self.provider_name.lower():
                winning_party = DisputeDecisionType.A
                decision = DecisionType.YES
            else:  # grok or other
                winning_party = DisputeDecisionType.A
                decision = DecisionType.YES
            
            raw_response = f"""{{
  "winning_party": "{winning_party.value}",
  "confidence": 0.85,
  "reasoning": "Party A (the Freelancer) delivered code per contract. Party B (the Client) rejected but used the code, which indicates acceptance. Therefore Party A wins.",
  "contract_validity": "valid",
  "injection_detected": false
}}"""
            
            return LLMResponse(
                provider=self.provider_name,
                model="mock",
                decision=decision,
                winning_party=winning_party,
                confidence=0.85,
                reasoning="Party A (the Freelancer) delivered code per contract. Party B (the Client) rejected but used the code, which indicates acceptance. Therefore Party A wins.",
                raw_response=raw_response,
                error=None,
            )
        else:
            # Return legacy yes/no format
            raw_response = """DECISION: YES
CONFIDENCE: 0.85
REASONING: This is a mock response for testing purposes. The mock oracle always returns YES with 85% confidence after a 5-second delay to simulate real API behavior."""

            return LLMResponse(
                provider=self.provider_name,
                model="mock",
                decision=DecisionType.YES,
                winning_party=None,
                confidence=0.85,
                reasoning="This is a mock response for testing purposes. The mock oracle always returns YES with 85% confidence after a 5-second delay to simulate real API behavior.",
                raw_response=raw_response,
                error=None,
            )

    async def analyze_tweet(self, tweet_url: str) -> TweetLLMResponse:
        """Return a mock tweet analysis response after sleeping.

        Args:
            tweet_url: The tweet URL (ignored in mock mode)

        Returns:
            TweetLLMResponse with varied mock data based on provider
        """
        # Simulate API latency.
        await asyncio.sleep(self.sleep_duration)

        # Vary responses based on provider to simulate real multi-LLM consensus.
        if "claude" in self.provider_name.lower():
            verdict = TweetVerdictType.CREDIBLE
            confidence = 0.90
            analysis = "The post contains verifiable factual claims that align with public records and reputable sources. No significant red flags detected."
            claims = [
                "Factual claim verified through primary sources",
                "Dates and figures match official records",
            ]
            flags = []
        elif "gemini" in self.provider_name.lower():
            verdict = TweetVerdictType.QUESTIONABLE
            confidence = 0.75
            analysis = "While some facts check out, the post uses emotionally charged language and selective framing that may mislead readers."
            claims = ["Core fact is accurate but lacks context"]
            flags = ["Emotionally charged language", "Selective presentation of facts"]
        elif "perplexity" in self.provider_name.lower():
            verdict = TweetVerdictType.CREDIBLE
            confidence = 0.85
            analysis = "Cross-referenced with multiple reliable sources. Claims are substantiated and presented with appropriate context."
            claims = ["Primary claim verified across multiple sources", "Timeline is accurate"]
            flags = []
        else:  # openai or other
            verdict = TweetVerdictType.OPINION
            confidence = 0.95
            analysis = "The post expresses subjective viewpoints rather than objective claims. While factually grounded, it's primarily an opinion piece."
            claims = []
            flags = ["Subjective interpretation of events"]

        return TweetLLMResponse(
            provider=self.provider_name,
            model="mock",
            verdict=verdict,
            confidence=confidence,
            analysis=analysis,
            identified_claims=claims,
            red_flags=flags,
            raw_response=f'{{"verdict": "{verdict.value}", "confidence": {confidence}, "analysis": "{analysis}"}}',
            error=None,
        )
