"""Grok (xAI) LLM client implementation using xAI Python SDK."""

from openai import AsyncOpenAI

from src.llm_clients.base import BaseLLMClient
from src.models import LLMResponse, TweetLLMResponse, TweetVerdictType


class GrokClient(BaseLLMClient):
    """Client for xAI's Grok API using OpenAI-compatible SDK."""

    def __init__(self, api_key: str, model: str = "grok-4-fast"):
        """Initialize Grok client.

        Args:
            api_key: xAI API key
            model: Grok model name
        """
        super().__init__(api_key, "grok", model)
        # xAI uses OpenAI-compatible API with custom base URL.
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model = model

    async def query(self, prompt: str) -> LLMResponse:
        """Send a query to Grok and return structured response.

        Args:
            prompt: The question/prompt to send to Grok

        Returns:
            LLMResponse with decision, confidence, reasoning, and raw response
        """
        try:
            # TODO: Update API to properly separate contract and dispute_details
            # For now, passing query as both parameters
            dispute_prompt = self._create_dispute_prompt(prompt, prompt)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": dispute_prompt},
                ],
                temperature=0.0,  # Deterministic for fact-checking
                max_tokens=1024,
            )

            raw_response = response.choices[0].message.content

            if not raw_response:
                return LLMResponse(
                    provider=self.provider_name,
                    model=self.model,
                    decision="uncertain",
                    winning_party=None,
                    confidence=0.0,
                    reasoning="No response received from Grok",
                    raw_response="",
                    error="Empty response",
                )

            decision, confidence, reasoning, winning_party = self._parse_response(raw_response)

            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                decision=decision,
                winning_party=winning_party,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw_response,
                error=None,
            )

        except Exception as e:
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                decision="uncertain",
                winning_party=None,
                confidence=0.0,
                reasoning=f"Error querying Grok: {str(e)}",
                raw_response="",
                error=str(e),
            )

    async def analyze_tweet(self, tweet_url: str) -> TweetLLMResponse:
        """Analyze a tweet for credibility using Grok with web search enabled.

        Args:
            tweet_url: Twitter/X URL to analyze

        Returns:
            TweetLLMResponse with verdict, confidence, analysis, and identified issues
        """
        try:
            tweet_prompt = self._create_tweet_analysis_prompt(tweet_url)
            system_prompt = self._system_prompt_tweet()

            # Use the new responses.create API that supports x_search and web_search tools.
            response = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": tweet_prompt},
                ],
                temperature=0.0,  # Deterministic for credibility analysis
                tools=[
                    {"type": "x_search"},  # Enable X/Twitter search
                    {"type": "web_search"},  # Enable web search
                ],
            )

            # Extract response content from the last message in output.
            # The response.output contains tool calls and the final message.
            raw_response = None
            if response.output:
                # Find the last message type item (after all tool calls).
                for item in reversed(response.output):
                    if (
                        hasattr(item, "type")
                        and item.type == "message"
                        and hasattr(item, "content")
                    ):
                        if item.content and len(item.content) > 0:
                            raw_response = item.content[0].text
                            break

            if not raw_response:
                return TweetLLMResponse(
                    provider=self.provider_name,
                    model=self.model,
                    verdict=TweetVerdictType.QUESTIONABLE,
                    confidence=0.0,
                    analysis="No response received from Grok",
                    identified_claims=[],
                    red_flags=[],
                    raw_response="",
                    error="Empty response",
                )

            verdict, confidence, analysis, claims, flags = self._parse_tweet_response(raw_response)

            return TweetLLMResponse(
                provider=self.provider_name,
                model=self.model,
                verdict=verdict,
                confidence=confidence,
                analysis=analysis,
                identified_claims=claims,
                red_flags=flags,
                raw_response=raw_response,
                error=None,
            )

        except Exception as e:
            return TweetLLMResponse(
                provider=self.provider_name,
                model=self.model,
                verdict=TweetVerdictType.QUESTIONABLE,
                confidence=0.0,
                analysis=f"Error analyzing tweet with Grok: {str(e)}",
                identified_claims=[],
                red_flags=[],
                raw_response="",
                error=str(e),
            )
