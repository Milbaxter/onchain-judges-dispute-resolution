"""Perplexity AI LLM client implementation."""

import httpx

from src.llm_clients.base import BaseLLMClient
from src.models import LLMResponse


class PerplexityClient(BaseLLMClient):
    """Client for Perplexity AI API."""

    def __init__(self, api_key: str):
        """Initialize Perplexity client.

        Args:
            api_key: Perplexity API key
        """
        super().__init__(api_key, "perplexity")
        self.api_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-small-128k-online"

    async def query(self, prompt: str) -> LLMResponse:
        """Send a query to Perplexity and return structured response.

        Args:
            prompt: The question/prompt to send to Perplexity

        Returns:
            LLMResponse with decision, confidence, reasoning, and raw response
        """
        try:
            dispute_prompt = self._create_dispute_prompt(prompt)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": dispute_prompt}],
                "max_tokens": 1024,
                "temperature": 0.2,
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            raw_response = data["choices"][0]["message"]["content"]
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
                reasoning=f"Error querying Perplexity: {str(e)}",
                raw_response="",
                error=str(e),
            )
