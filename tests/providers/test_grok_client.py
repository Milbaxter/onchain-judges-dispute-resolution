#!/usr/bin/env python3
"""Integration test for Grok (xAI) client.

Run with:
    GROK_API_KEY=your_xai_api_key python tests/providers/test_grok_client.py
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llm_clients.grok import GrokClient


async def test_grok_query():
    """Test Grok with a factual query."""
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        print("❌ GROK_API_KEY environment variable not set")
        print("   Usage: GROK_API_KEY=your_xai_api_key python tests/providers/test_grok_client.py")
        return False

    print("Testing Grok client...")
    print("=" * 60)

    client = GrokClient(api_key=api_key)
    query = "Did the Lakers win the 2020 NBA Championship?"

    print(f"Query: {query}")
    print()

    response = await client.query(query)

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Decision: {response.decision}")
    print(f"Confidence: {response.confidence}")
    print(f"Error: {response.error}")
    print()
    print(f"Reasoning: {response.reasoning}")
    print()
    print(f"Raw Response Preview: {response.raw_response[:200]}...")

    if response.error:
        print(f"\n❌ Test failed with error: {response.error}")
        return False

    print("\n✅ Grok query test passed")
    return True


async def test_grok_tweet_analysis():
    """Test Grok with tweet analysis."""
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        print("❌ GROK_API_KEY environment variable not set")
        return False

    print("\nTesting Grok tweet analysis...")
    print("=" * 60)

    client = GrokClient(api_key=api_key)
    tweet_url = "https://x.com/crypto_news/status/123"

    print(f"Tweet URL: {tweet_url}")
    print()

    response = await client.analyze_tweet(tweet_url)

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Verdict: {response.verdict}")
    print(f"Confidence: {response.confidence}")
    print(f"Error: {response.error}")
    print()
    print(f"Analysis: {response.analysis}")
    print()
    print(f"Identified Claims: {response.identified_claims}")
    print(f"Red Flags: {response.red_flags}")

    if response.error:
        print(f"\n❌ Test failed with error: {response.error}")
        return False

    print("\n✅ Grok tweet analysis test passed")
    return True


async def main():
    """Run all tests."""
    print("Grok (xAI) Integration Tests")
    print("=" * 60)
    print()

    test1 = await test_grok_query()
    test2 = await test_grok_tweet_analysis()

    print()
    print("=" * 60)
    if test1 and test2:
        print("✅ All Grok tests passed")
        sys.exit(0)
    else:
        print("❌ Some Grok tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
