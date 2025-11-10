"""Pydantic models for request/response validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Possible decision types from LLM."""

    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


class TweetVerdictType(str, Enum):
    """Possible verdict types for tweet analysis."""

    CREDIBLE = "credible"
    QUESTIONABLE = "questionable"
    MISLEADING = "misleading"
    OPINION = "opinion"


class QueryType(str, Enum):
    """Type of query being processed."""

    FACTUAL = "factual"
    TWEET = "tweet"


class LLMResponse(BaseModel):
    """Response from a single LLM backend."""

    provider: str = Field(..., description="LLM provider name (claude, gemini, perplexity)")
    model: str = Field(..., description="Model name used (e.g., claude-haiku-4-5-20251001, gpt-4o)")
    decision: DecisionType = Field(..., description="The LLM's decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    reasoning: str = Field(..., description="Explanation for the decision")
    raw_response: str = Field(..., description="Raw text response from LLM")
    error: str | None = Field(None, description="Error message if request failed")


class TweetLLMResponse(BaseModel):
    """Response from a single LLM backend for tweet analysis."""

    provider: str = Field(..., description="LLM provider name (claude, gemini, perplexity)")
    model: str = Field(..., description="Model name used (e.g., claude-haiku-4-5-20251001, gpt-4o)")
    verdict: TweetVerdictType = Field(..., description="The LLM's credibility verdict")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    analysis: str = Field(..., description="Detailed credibility analysis")
    identified_claims: list[str] = Field(
        default_factory=list, description="Factual claims identified in the tweet"
    )
    red_flags: list[str] = Field(
        default_factory=list, description="Red flags or concerns identified"
    )
    raw_response: str = Field(..., description="Raw text response from LLM")
    error: str | None = Field(None, description="Error message if request failed")


class OracleQuery(BaseModel):
    """Input query for the oracle."""

    query: str = Field(
        ...,
        min_length=10,
        max_length=256,
        pattern=r'^[a-zA-Z0-9\s.,?!\-\'"":;()/@#$%&+=]+$',
        description="Question to verify (should be answerable with YES/NO, alphanumeric and common punctuation only)",
    )


class OracleResult(BaseModel):
    """Aggregated result from all LLM backends."""

    query: str = Field(..., description="The original query")
    final_decision: DecisionType = Field(..., description="Weighted final decision")
    final_confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregated confidence score")
    explanation: str = Field(..., description="Summary of how decision was reached")
    llm_responses: list[LLMResponse] = Field(..., description="Individual LLM responses")
    total_weight: float = Field(..., description="Total weight used in calculation")
    timestamp: datetime = Field(..., description="UTC timestamp when the result was generated")
    signature: str | None = Field(
        None,
        description="Recoverable ECDSA signature (hex) over the canonical JSON representation of the result. Generated inside the ROFL TEE using a SECP256K1 key.",
    )
    public_key: str | None = Field(
        None,
        description="Compressed SECP256K1 public key (hex) used for signing. Can be verified against the on-chain attested state in the Oasis ROFL registry (https://github.com/ptrus/rofl-registry).",
    )


class JobStatus(str, Enum):
    """Job processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Response when creating a new job."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    query: str = Field(..., description="The submitted query")
    created_at: datetime = Field(..., description="Job creation timestamp")


class JobResultResponse(BaseModel):
    """Response when polling for job results."""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    query: str = Field(..., description="The original query")
    query_type: str = Field(..., description="Type of query: 'fact' or 'tweet'")
    result: OracleResult | TweetAnalysisResult | None = Field(
        None,
        description="Oracle result if completed (OracleResult for facts, TweetAnalysisResult for tweets)",
    )
    error: str | None = Field(None, description="Error message if job failed")
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: datetime | None = Field(None, description="Job completion timestamp")
    payer_address: str | None = Field(None, description="Address of the payer")
    tx_hash: str | None = Field(None, description="Transaction hash of the payment")
    network: str | None = Field(None, description="Network the payment was made on")


class TweetAnalysisQuery(BaseModel):
    """Input query for tweet credibility analysis."""

    tweet_url: str = Field(
        ...,
        min_length=28,
        max_length=200,
        pattern=r"^https?://(twitter\.com|x\.com)/[a-zA-Z0-9_]+/status/[0-9]+.*$",
        description="Twitter/X URL (e.g., https://twitter.com/user/status/123456789)",
    )


class TweetData(BaseModel):
    """Tweet URL (Grok fetches content directly)."""

    url: str = Field(..., description="Tweet URL")


class TweetAnalysisResult(BaseModel):
    """Aggregated result from tweet credibility analysis."""

    tweet: TweetData = Field(..., description="The analyzed tweet data")
    final_verdict: TweetVerdictType = Field(..., description="Weighted final credibility verdict")
    final_confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregated confidence score")
    analysis_summary: str = Field(..., description="Summary of credibility analysis")
    llm_responses: list[TweetLLMResponse] = Field(..., description="Individual LLM analyses")
    total_weight: float = Field(..., description="Total weight used in calculation")
    timestamp: datetime = Field(..., description="UTC timestamp when the analysis was generated")
    signature: str | None = Field(
        None,
        description="Recoverable ECDSA signature (hex) over the canonical JSON representation of the result. Generated inside the ROFL TEE using a SECP256K1 key.",
    )
    public_key: str | None = Field(
        None,
        description="Compressed SECP256K1 public key (hex) used for signing. Can be verified against the on-chain attested state in the Oasis ROFL registry (https://github.com/ptrus/rofl-registry).",
    )
