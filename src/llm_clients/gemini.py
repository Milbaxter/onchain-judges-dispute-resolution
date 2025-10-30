"""Google Gemini LLM client implementation."""

import google.generativeai as genai

from src.llm_clients.base import BaseLLMClient
from src.models import LLMResponse


class GeminiClient(BaseLLMClient):
    """Client for Google's Gemini API."""

    def __init__(self, api_key: str):
        """Initialize Gemini client.

        Args:
            api_key: Google API key
        """
        super().__init__(api_key, "gemini")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def query(self, prompt: str) -> LLMResponse:
        """Send a query to Gemini and return structured response.

        Args:
            prompt: The question/prompt to send to Gemini

        Returns:
            LLMResponse with decision, confidence, reasoning, and raw response
        """
        try:
            dispute_prompt = self._create_dispute_prompt(prompt)

            # Gemini's generate_content is sync by default, but we can await it
            response = await self.model.generate_content_async(dispute_prompt)

            raw_response = response.text
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
                reasoning=f"Error querying Gemini: {str(e)}",
                raw_response="",
                error=str(e),
            )
