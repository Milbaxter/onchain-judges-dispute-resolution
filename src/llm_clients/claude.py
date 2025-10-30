"""Claude (Anthropic) LLM client implementation."""

from anthropic import AsyncAnthropic

from src.llm_clients.base import BaseLLMClient
from src.models import LLMResponse


class ClaudeClient(BaseLLMClient):
    """Client for Anthropic's Claude API."""

    def __init__(self, api_key: str):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key
        """
        super().__init__(api_key, "claude")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"

    async def query(self, prompt: str) -> LLMResponse:
        """Send a query to Claude and return structured response.

        Args:
            prompt: The question/prompt to send to Claude

        Returns:
            LLMResponse with decision, confidence, reasoning, and raw response
        """
        try:
            dispute_prompt = self._create_dispute_prompt(prompt)

            message = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": dispute_prompt}],
            )

            raw_response = message.content[0].text
            decision, confidence, reasoning = self._parse_response(raw_response)

            return LLMResponse(
                provider=self.provider_name,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw_response,
                error=None,
            )

        except Exception as e:
            return LLMResponse(
                provider=self.provider_name,
                decision="uncertain",
                confidence=0.0,
                reasoning=f"Error querying Claude: {str(e)}",
                raw_response="",
                error=str(e),
            )
