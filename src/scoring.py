"""Weighted scoring system for aggregating LLM responses."""

from datetime import UTC, datetime

from src.models import (
    DecisionType,
    DisputeDecisionType,
    LLMResponse,
    OracleResult,
    TweetAnalysisResult,
    TweetData,
    TweetLLMResponse,
    TweetVerdictType,
)


class WeightedScorer:
    """Aggregates multiple LLM responses using weighted scoring."""

    def __init__(self, weights: dict[str, float]):
        """Initialize the scorer with provider weights.

        Args:
            weights: Dictionary mapping provider names to their weights
        """
        self.weights = weights

    def aggregate_responses(self, query: str, responses: list[LLMResponse]) -> OracleResult:
        """Aggregate multiple LLM responses into a single decision.

        Args:
            query: The original query
            responses: List of LLM responses

        Returns:
            OracleResult with final decision and explanation
        """
        # Check if we have dispute resolution responses (with winning_party)
        has_dispute_responses = any(r.winning_party is not None for r in responses if r.error is None)
        
        if has_dispute_responses:
            # Use dispute resolution scoring (A/B/uncertain)
            return self._aggregate_dispute_responses(query, responses)
        else:
            # Use legacy yes/no/uncertain scoring
            return self._aggregate_legacy_responses(query, responses)
    
    def _aggregate_dispute_responses(self, query: str, responses: list[LLMResponse]) -> OracleResult:
        """Aggregate dispute resolution responses (A/B/uncertain)."""
        # Calculate weighted scores for each dispute decision type.
        decision_scores = {
            DisputeDecisionType.A: 0.0,
            DisputeDecisionType.B: 0.0,
            DisputeDecisionType.UNCERTAIN: 0.0,
        }

        total_weight = 0.0
        valid_responses = [r for r in responses if r.error is None]

        # Accumulate weighted scores.
        for response in valid_responses:
            weight = self.weights.get(response.provider, 1.0)
            total_weight += weight

            # Use winning_party if available, otherwise map decision to DisputeDecisionType
            if response.winning_party is not None:
                decision_scores[response.winning_party] += weight * response.confidence
            else:
                # Fallback: map decision to DisputeDecisionType
                if response.decision == DecisionType.YES:
                    decision_scores[DisputeDecisionType.A] += weight * response.confidence
                elif response.decision == DecisionType.NO:
                    decision_scores[DisputeDecisionType.B] += weight * response.confidence
                else:
                    decision_scores[DisputeDecisionType.UNCERTAIN] += weight * response.confidence

        # Determine final decision.
        if total_weight == 0:
            # All providers failed.
            final_decision = DisputeDecisionType.UNCERTAIN
            final_confidence = 0.0
        else:
            # Normalize scores by total weight.
            normalized_scores = {
                decision: score / total_weight for decision, score in decision_scores.items()
            }

            # Find decision with highest normalized score.
            winning_decision, winning_score = max(
                normalized_scores.items(), key=lambda item: item[1]
            )

            # Check for ties (excluding UNCERTAIN from tie check).
            a_score = normalized_scores[DisputeDecisionType.A]
            b_score = normalized_scores[DisputeDecisionType.B]

            if winning_score <= 0.0:
                # No positive scores.
                final_decision = DisputeDecisionType.UNCERTAIN
                final_confidence = 0.0
            elif abs(a_score - b_score) < 0.01 and a_score > 0:
                # A and B are tied, return UNCERTAIN.
                final_decision = DisputeDecisionType.UNCERTAIN
                final_confidence = winning_score
            else:
                final_decision = winning_decision
                final_confidence = winning_score

        # Generate explanation.
        explanation = self._generate_dispute_explanation(
            valid_responses, final_decision, final_confidence, total_weight
        )

        return OracleResult(
            query=query,
            final_decision=final_decision,
            final_confidence=final_confidence,
            explanation=explanation,
            llm_responses=responses,
            total_weight=total_weight,
            timestamp=datetime.now(UTC),
        )
    
    def _aggregate_legacy_responses(self, query: str, responses: list[LLMResponse]) -> OracleResult:
        """Aggregate legacy yes/no/uncertain responses."""
        # Calculate weighted scores for each decision type.
        decision_scores = {
            DecisionType.YES: 0.0,
            DecisionType.NO: 0.0,
            DecisionType.UNCERTAIN: 0.0,
        }

        total_weight = 0.0
        valid_responses = [r for r in responses if r.error is None]

        # Accumulate weighted scores.
        for response in valid_responses:
            weight = self.weights.get(response.provider, 1.0)
            total_weight += weight

            # Add weighted confidence to the decision score.
            decision_scores[response.decision] += weight * response.confidence

        # Determine final decision.
        if total_weight == 0:
            # All providers failed.
            final_decision_legacy = DecisionType.UNCERTAIN
            final_confidence = 0.0
        else:
            # Normalize scores by total weight.
            normalized_scores = {
                decision: score / total_weight for decision, score in decision_scores.items()
            }

            # Find decision with highest normalized score.
            winning_decision, winning_score = max(
                normalized_scores.items(), key=lambda item: item[1]
            )

            # Check for ties (excluding UNCERTAIN from tie check).
            yes_score = normalized_scores[DecisionType.YES]
            no_score = normalized_scores[DecisionType.NO]

            if winning_score <= 0.0:
                # No positive scores.
                final_decision_legacy = DecisionType.UNCERTAIN
                final_confidence = 0.0
            elif abs(yes_score - no_score) < 0.01 and yes_score > 0:
                # YES and NO are tied, return UNCERTAIN.
                final_decision_legacy = DecisionType.UNCERTAIN
                final_confidence = winning_score
            else:
                final_decision_legacy = winning_decision
                final_confidence = winning_score

        # Map legacy decision to DisputeDecisionType for OracleResult
        if final_decision_legacy == DecisionType.YES:
            final_decision = DisputeDecisionType.A
        elif final_decision_legacy == DecisionType.NO:
            final_decision = DisputeDecisionType.B
        else:
            final_decision = DisputeDecisionType.UNCERTAIN

        # Generate explanation.
        explanation = self._generate_explanation(
            valid_responses, final_decision_legacy, final_confidence, total_weight
        )

        return OracleResult(
            query=query,
            final_decision=final_decision,
            final_confidence=final_confidence,
            explanation=explanation,
            llm_responses=responses,
            total_weight=total_weight,
            timestamp=datetime.now(UTC),
        )

    def _generate_dispute_explanation(
        self,
        responses: list[LLMResponse],
        final_decision: DisputeDecisionType,
        final_confidence: float,
        total_weight: float,
    ) -> str:
        """Generate human-readable explanation of the dispute decision.

        Args:
            responses: Valid LLM responses
            final_decision: The final aggregated decision (A/B/uncertain)
            final_confidence: The final confidence score
            total_weight: Total weight used

        Returns:
            Explanation string
        """
        if not responses:
            return "All LLM providers failed to respond. Unable to make a decision."

        # Count votes by decision.
        decision_counts = {DisputeDecisionType.A: 0, DisputeDecisionType.B: 0, DisputeDecisionType.UNCERTAIN: 0}
        for response in responses:
            if response.winning_party is not None:
                decision_counts[response.winning_party] += 1
            else:
                # Fallback: map decision to DisputeDecisionType
                if response.decision == DecisionType.YES:
                    decision_counts[DisputeDecisionType.A] += 1
                elif response.decision == DecisionType.NO:
                    decision_counts[DisputeDecisionType.B] += 1
                else:
                    decision_counts[DisputeDecisionType.UNCERTAIN] += 1

        # Build explanation.
        explanation_parts = [
            f"**Final Decision: {final_decision.value.upper()}** (confidence: {final_confidence:.2f})",
            "",
            "**Voting Summary:**",
        ]

        for decision, count in decision_counts.items():
            if count > 0:
                explanation_parts.append(f"- {decision.value.upper()}: {count} provider(s)")

        explanation_parts.extend(
            [
                "",
                f"**Total Weight Used:** {total_weight:.2f}",
                "",
                "**Individual Provider Assessments:**",
            ]
        )

        for response in responses:
            provider_name = response.provider.upper()
            if response.winning_party is not None:
                decision_str = response.winning_party.value.upper()
            else:
                decision_str = response.decision.value.upper()
            confidence = response.confidence
            weight = self.weights.get(response.provider, 1.0)

            explanation_parts.append(
                f"- **{provider_name}** (weight: {weight:.1f}): {decision_str} "
                f"(confidence: {confidence:.2f})"
            )

        return "\n".join(explanation_parts)
    
    def _generate_explanation(
        self,
        responses: list[LLMResponse],
        final_decision: DecisionType,
        final_confidence: float,
        total_weight: float,
    ) -> str:
        """Generate human-readable explanation of the decision (legacy yes/no format).

        Args:
            responses: Valid LLM responses
            final_decision: The final aggregated decision
            final_confidence: The final confidence score
            total_weight: Total weight used

        Returns:
            Explanation string
        """
        if not responses:
            return "All LLM providers failed to respond. Unable to make a decision."

        # Count votes by decision.
        decision_counts = {DecisionType.YES: 0, DecisionType.NO: 0, DecisionType.UNCERTAIN: 0}
        for response in responses:
            decision_counts[response.decision] += 1

        # Build explanation.
        explanation_parts = [
            f"**Final Decision: {final_decision.value.upper()}** (confidence: {final_confidence:.2f})",
            "",
            "**Voting Summary:**",
        ]

        for decision, count in decision_counts.items():
            if count > 0:
                explanation_parts.append(f"- {decision.value.upper()}: {count} provider(s)")

        explanation_parts.extend(
            [
                "",
                f"**Total Weight Used:** {total_weight:.2f}",
                "",
                "**Individual Provider Assessments:**",
            ]
        )

        for response in responses:
            provider_name = response.provider.upper()
            decision = response.decision.value.upper()
            confidence = response.confidence
            weight = self.weights.get(response.provider, 1.0)

            explanation_parts.append(
                f"- **{provider_name}** (weight: {weight:.1f}): {decision} "
                f"(confidence: {confidence:.2f})"
            )

        return "\n".join(explanation_parts)

    def aggregate_tweet_responses(
        self, tweet_url: str, responses: list[TweetLLMResponse]
    ) -> TweetAnalysisResult:
        """Aggregate multiple LLM tweet analyses into a single credibility assessment.

        Args:
            tweet_url: The analyzed tweet URL
            responses: List of LLM tweet analysis responses

        Returns:
            TweetAnalysisResult with final verdict and summary
        """
        # Calculate weighted scores for each verdict type.
        verdict_scores = {
            TweetVerdictType.CREDIBLE: 0.0,
            TweetVerdictType.QUESTIONABLE: 0.0,
            TweetVerdictType.MISLEADING: 0.0,
            TweetVerdictType.OPINION: 0.0,
        }

        total_weight = 0.0
        valid_responses = [r for r in responses if r.error is None]

        # Accumulate weighted scores.
        for response in valid_responses:
            weight = self.weights.get(response.provider, 1.0)
            total_weight += weight

            # Add weighted confidence to the verdict score.
            verdict_scores[response.verdict] += weight * response.confidence

        # Determine final verdict.
        if total_weight == 0:
            # All providers failed.
            final_verdict = TweetVerdictType.QUESTIONABLE
            final_confidence = 0.0
        else:
            # Normalize scores by total weight.
            normalized_scores = {
                verdict: score / total_weight for verdict, score in verdict_scores.items()
            }

            # Find verdict with highest normalized score.
            winning_verdict, winning_score = max(
                normalized_scores.items(), key=lambda item: item[1]
            )

            if winning_score <= 0.0:
                # No positive scores.
                final_verdict = TweetVerdictType.QUESTIONABLE
                final_confidence = 0.0
            else:
                final_verdict = winning_verdict
                final_confidence = winning_score

        # Generate analysis summary.
        analysis_summary = self._generate_tweet_summary(
            valid_responses, final_verdict, final_confidence, total_weight
        )

        # Create a TweetData object with just the URL.
        # Grok will fetch and analyze the tweet directly.
        tweet_data = TweetData(url=tweet_url)

        return TweetAnalysisResult(
            tweet=tweet_data,
            final_verdict=final_verdict,
            final_confidence=final_confidence,
            analysis_summary=analysis_summary,
            llm_responses=responses,
            total_weight=total_weight,
            timestamp=datetime.now(UTC),
        )

    def _generate_tweet_summary(
        self,
        responses: list[TweetLLMResponse],
        final_verdict: TweetVerdictType,
        final_confidence: float,
        total_weight: float,
    ) -> str:
        """Generate human-readable summary of tweet credibility analysis.

        Args:
            responses: Valid LLM responses
            final_verdict: The final aggregated verdict
            final_confidence: The final confidence score
            total_weight: Total weight used

        Returns:
            Summary string
        """
        if not responses:
            return "All LLM providers failed to respond. Unable to analyze tweet credibility."

        # Count votes by verdict.
        verdict_counts = {
            TweetVerdictType.CREDIBLE: 0,
            TweetVerdictType.QUESTIONABLE: 0,
            TweetVerdictType.MISLEADING: 0,
            TweetVerdictType.OPINION: 0,
        }
        for response in responses:
            verdict_counts[response.verdict] += 1

        # Collect all identified claims and red flags across providers.
        all_claims = set()
        all_red_flags = set()
        for response in responses:
            all_claims.update(response.identified_claims)
            all_red_flags.update(response.red_flags)

        # Build summary.
        summary_parts = [
            f"**Final Verdict: {final_verdict.value.upper()}** (confidence: {final_confidence:.2f})",
            "",
            "**Analysis Summary:**",
        ]

        for verdict, count in verdict_counts.items():
            if count > 0:
                summary_parts.append(f"- {verdict.value.upper()}: {count} provider(s)")

        summary_parts.extend(
            [
                "",
                f"**Total Weight Used:** {total_weight:.2f}",
            ]
        )

        # Add claims if any were identified.
        if all_claims:
            summary_parts.extend(
                [
                    "",
                    "**Identified Claims:**",
                ]
            )
            for claim in list(all_claims)[:5]:  # Limit to top 5
                summary_parts.append(f"- {claim}")

        # Add red flags if any were found.
        if all_red_flags:
            summary_parts.extend(
                [
                    "",
                    "**Red Flags:**",
                ]
            )
            for flag in list(all_red_flags)[:5]:  # Limit to top 5
                summary_parts.append(f"- {flag}")

        summary_parts.extend(
            [
                "",
                "**Individual Provider Analyses:**",
            ]
        )

        for response in responses:
            provider_name = response.provider.upper()
            verdict = response.verdict.value.upper()
            confidence = response.confidence
            weight = self.weights.get(response.provider, 1.0)

            summary_parts.append(
                f"- **{provider_name}** (weight: {weight:.1f}): {verdict} "
                f"(confidence: {confidence:.2f})"
            )

        return "\n".join(summary_parts)
