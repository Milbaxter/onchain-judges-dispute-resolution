"""Base abstract class for LLM clients."""

from abc import ABC, abstractmethod
from typing import Optional

from src.models import LLMResponse


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, api_key: str, provider_name: str):
        """Initialize the LLM client.

        Args:
            api_key: API key for the LLM service
            provider_name: Name of the provider (e.g., 'claude', 'gemini')
        """
        self.api_key = api_key
        self.provider_name = provider_name

    @abstractmethod
    async def query(self, prompt: str) -> LLMResponse:
        """Send a query to the LLM and return structured response.

        Args:
            prompt: The question/prompt to send to the LLM

        Returns:
            LLMResponse with decision, confidence, reasoning, and raw response
        """
        pass

    def _create_dispute_prompt(self, query: str) -> str:
        """Create a standardized prompt for dispute resolution.

        Args:
            query: The user's dispute query

        Returns:
            Formatted prompt string
        """
        return f"""You are an oracle for dispute resolution. Your task is to determine if the described event actually happened based on your knowledge.

Query: {query}

Please analyze this query and provide your assessment in the following format:

DECISION: [YES/NO/UNCERTAIN]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Your detailed explanation]

Guidelines:
- YES: The event definitely happened based on reliable information
- NO: The event definitely did not happen or is factually incorrect
- UNCERTAIN: Insufficient information or conflicting sources

Be objective and base your decision on verifiable facts."""

    def _parse_response(self, raw_response: str) -> tuple[str, float, str]:
        """Parse LLM response to extract decision, confidence, and reasoning.

        Args:
            raw_response: The raw text response from the LLM

        Returns:
            Tuple of (decision, confidence, reasoning)
        """
        decision = "uncertain"
        confidence = 0.5
        reasoning = ""

        lines = raw_response.split("\n")
        for i, line in enumerate(lines):
            line_upper = line.upper()
            if "DECISION:" in line_upper:
                decision_text = line.split(":", 1)[1].strip().lower()
                if "yes" in decision_text:
                    decision = "yes"
                elif "no" in decision_text:
                    decision = "no"
                else:
                    decision = "uncertain"

            elif "CONFIDENCE:" in line_upper:
                try:
                    conf_text = line.split(":", 1)[1].strip()
                    confidence = float(conf_text)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    confidence = 0.5

            elif "REASONING:" in line_upper:
                reasoning = line.split(":", 1)[1].strip()
                # Collect all subsequent lines as part of reasoning
                if i + 1 < len(lines):
                    reasoning += "\n" + "\n".join(lines[i + 1 :])
                break

        if not reasoning:
            reasoning = raw_response

        return decision, confidence, reasoning
