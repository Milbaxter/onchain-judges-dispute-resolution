"""Oracle orchestrator for multi-LLM dispute resolution."""

import asyncio
from typing import Dict

from src.config import settings
from src.llm_clients.claude import ClaudeClient
from src.llm_clients.gemini import GeminiClient
from src.llm_clients.perplexity import PerplexityClient
from src.models import LLMResponse, OracleResult
from src.scoring import WeightedScorer


class Oracle:
    """Orchestrates queries to multiple LLM backends and aggregates results."""

    def __init__(self):
        """Initialize the Oracle with all LLM clients and scoring system."""
        # Initialize LLM clients
        self.clients = {
            "claude": ClaudeClient(settings.claude_api_key),
            "gemini": GeminiClient(settings.gemini_api_key),
            "perplexity": PerplexityClient(settings.perplexity_api_key),
        }

        # Initialize weighted scorer
        self.weights = {
            "claude": settings.claude_weight,
            "gemini": settings.gemini_weight,
            "perplexity": settings.perplexity_weight,
        }
        self.scorer = WeightedScorer(self.weights)

    async def resolve_dispute(self, query: str) -> OracleResult:
        """Query all LLM backends and aggregate their responses.

        Args:
            query: The dispute question to resolve

        Returns:
            OracleResult with aggregated decision and individual responses
        """
        # Query all LLMs in parallel
        tasks = [client.query(query) for client in self.clients.values()]
        responses: list[LLMResponse] = await asyncio.gather(*tasks, return_exceptions=False)

        # Aggregate responses using weighted scoring
        result = self.scorer.aggregate_responses(query, responses)

        return result


# Global oracle instance
oracle = Oracle()
