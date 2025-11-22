"""OpenAI LLM client implementation."""

import httpx

from src.llm_clients.base import BaseLLMClient
from src.models import LLMResponse


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: OpenAI model name (gpt-4o uses chat completions, gpt-5 uses responses API)
        """
        super().__init__(api_key, "openai", model)
        self.model = model
        # Use Responses API for newer models like gpt-5, Chat Completions for gpt-4o
        self.use_responses_api = model.startswith("gpt-5") or model.startswith("o1")
        if self.use_responses_api:
            self.api_url = "https://api.openai.com/v1/responses"
            self.tools = [{"type": "web_search"}]
        else:
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.tools = None  # Chat completions API doesn't use tools parameter like this

    async def query(self, prompt: str) -> LLMResponse:
        """Send a query to OpenAI and return structured response.

        Args:
            prompt: The question/prompt to send to OpenAI

        Returns:
            LLMResponse with decision, confidence, reasoning, and raw response
        """
        try:
            # TODO: Update API to properly separate contract and dispute_details
            # For now, passing query as both parameters
            dispute_prompt = self._create_dispute_prompt(prompt, prompt)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            if self.use_responses_api:
                # Responses API uses "instructions" and "input" instead of "messages".
                payload = {
                    "model": self.model,
                    "tools": self.tools,
                    "instructions": self._system_prompt(),
                    "input": dispute_prompt,
                }
            else:
                # Standard Chat Completions API for gpt-4o and other models
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": dispute_prompt},
                    ],
                    "temperature": 0.0,  # Deterministic for dispute resolution
                    "max_tokens": 1024,
                }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

            # Parse response based on API type
            if self.use_responses_api:
                # Responses API returns content in output array.
                raw_response = ""
                for item in data.get("output", []):
                    if item.get("type") == "message" and item.get("status") == "completed":
                        content = item.get("content", [])
                        if content and content[0].get("type") == "output_text":
                            raw_response = content[0].get("text", "")
                            break
            else:
                # Chat Completions API returns content in choices[0].message.content
                raw_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not raw_response:
                return LLMResponse(
                    provider=self.provider_name,
                    model=self.model,
                    decision="uncertain",
                    winning_party=None,
                    confidence=0.0,
                    reasoning="No response content received from OpenAI",
                    raw_response=str(data),
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

        except httpx.HTTPStatusError as e:
            # Get the error details from the response.
            error_detail = e.response.text if hasattr(e, "response") else str(e)
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                decision="uncertain",
                winning_party=None,
                confidence=0.0,
                reasoning=f"Error querying OpenAI: {str(e)} - {error_detail}",
                raw_response="",
                error=f"{str(e)} - {error_detail}",
            )
        except Exception as e:
            return LLMResponse(
                provider=self.provider_name,
                model=self.model,
                decision="uncertain",
                winning_party=None,
                confidence=0.0,
                reasoning=f"Error querying OpenAI: {str(e)}",
                raw_response="",
                error=str(e),
            )
