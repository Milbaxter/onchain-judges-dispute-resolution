"""Pydantic models for request/response validation."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Possible decision types from LLM."""

    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class LLMResponse(BaseModel):
    """Response from a single LLM backend."""

    provider: str = Field(..., description="LLM provider name (claude, gemini, perplexity)")
    decision: DecisionType = Field(..., description="The LLM's decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    reasoning: str = Field(..., description="Explanation for the decision")
    raw_response: str = Field(..., description="Raw text response from LLM")
    error: Optional[str] = Field(None, description="Error message if request failed")


class OracleQuery(BaseModel):
    """Input query for the oracle."""

    query: str = Field(..., min_length=1, max_length=512, description="The dispute question to resolve")


class OracleResult(BaseModel):
    """Aggregated result from all LLM backends."""

    query: str = Field(..., description="The original query")
    final_decision: DecisionType = Field(..., description="Weighted final decision")
    final_confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregated confidence score")
    explanation: str = Field(..., description="Summary of how decision was reached")
    llm_responses: list[LLMResponse] = Field(..., description="Individual LLM responses")
    total_weight: float = Field(..., description="Total weight used in calculation")
