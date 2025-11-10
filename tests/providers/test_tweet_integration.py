#!/usr/bin/env python3
"""Integration test for tweet analysis across all LLM providers.

Run with:
    python tests/providers/test_tweet_integration.py

Or with API keys from .env.mainnet:
    python tests/providers/test_tweet_integration.py --env .env.mainnet
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm_clients.grok import GrokClient  # noqa: E402

# Test tweet URLs
TEST_TWEET_URLS = [
    "https://x.com/GwartyGwart/status/1987536920109519258",
    "https://x.com/Mangan150/status/1987507977411252624",
    "https://x.com/uttam_singhk/status/1987433706903249367",
    "https://x.com/hybridathlete8/status/1987505766383677795",
    "https://x.com/_philschmid/status/1987444232785740102",
    "https://x.com/0xMert_/status/1987565252888707410",
    "https://x.com/unusual_whales/status/1987528051681280478",
    "https://x.com/ripeth/status/1987532836770029770",
    "https://x.com/petusr/status/1986716608606375947",
    "https://x.com/petusr/status/1986326278794854765",
    "https://x.com/petusr/status/1986545354351755548",
]


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_header(text):
    """Print a formatted header."""
    print_separator()
    print(f"  {text}")
    print_separator()


async def test_provider_tweet(provider_name: str, client, tweet_url):
    """Test a single provider with tweet URL."""
    print(f"\n{provider_name.upper()}:")
    print("-" * 60)

    try:
        response = await client.analyze_tweet(tweet_url)

        print(f"Verdict: {response.verdict}")
        print(f"Confidence: {response.confidence * 100:.1f}%")
        print(f"Model: {response.model}")
        print(f"\nAnalysis: {response.analysis[:200]}...")
        print(f"\nIdentified Claims: {response.identified_claims}")
        print(f"Red Flags: {response.red_flags}")

        if response.error:
            print(f"⚠️  Error: {response.error}")
            return False

        print("✅ Success")
        return True

    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def test_tweet(providers, tweet_url):
    """Test all providers with a single tweet URL."""
    print_header(f"Testing Tweet: {tweet_url}")

    # Test each provider - they will fetch the tweet directly
    results = {}
    for provider_name, client in providers.items():
        if client:
            success = await test_provider_tweet(provider_name, client, tweet_url)
            results[provider_name] = success
        else:
            print(f"\n{provider_name.upper()}: ⚠️  Skipped (no API key)")
            results[provider_name] = None

    return {"url": tweet_url, "providers": results}


async def main():
    """Run integration tests for all providers."""
    # Check for --env flag
    env_file = ".env"
    if "--env" in sys.argv:
        env_idx = sys.argv.index("--env")
        if env_idx + 1 < len(sys.argv):
            env_file = sys.argv[env_idx + 1]

    # Load environment variables
    load_dotenv(env_file)
    print(f"📄 Loaded environment from: {env_file}\n")

    # Initialize all providers (skip if no API key)
    providers = {}

    # NOTE: Only Grok can access and analyze tweets directly from URLs
    # Other models (Claude, Gemini, OpenAI, Perplexity) cannot fetch tweet content

    # claude_key = os.getenv("CLAUDE_API_KEY")
    # if claude_key:
    #     providers["claude"] = ClaudeClient(api_key=claude_key)
    #     print("✅ Claude client initialized")
    # else:
    #     providers["claude"] = None
    #     print("⚠️  Claude API key not found")

    # gemini_key = os.getenv("GEMINI_API_KEY")
    # if gemini_key:
    #     providers["gemini"] = GeminiClient(api_key=gemini_key)
    #     print("✅ Gemini client initialized")
    # else:
    #     providers["gemini"] = None
    #     print("⚠️  Gemini API key not found")

    # openai_key = os.getenv("OPENAI_API_KEY")
    # if openai_key:
    #     providers["openai"] = OpenAIClient(api_key=openai_key)
    #     print("✅ OpenAI client initialized")
    # else:
    #     providers["openai"] = None
    #     print("⚠️  OpenAI API key not found")

    # perplexity_key = os.getenv("PERPLEXITY_API_KEY")
    # if perplexity_key:
    #     providers["perplexity"] = PerplexityClient(api_key=perplexity_key)
    #     print("✅ Perplexity client initialized")
    # else:
    #     providers["perplexity"] = None
    #     print("⚠️  Perplexity API key not found")

    grok_key = os.getenv("GROK_API_KEY")
    if grok_key:
        providers["grok"] = GrokClient(api_key=grok_key)
        print("✅ Grok client initialized")
    else:
        providers["grok"] = None
        print("⚠️  Grok API key not found")

    print()

    # Run tests for each tweet URL
    all_results = []
    for i, tweet_url in enumerate(TEST_TWEET_URLS, 1):
        print(f"\n{'=' * 80}")
        print(f"  Tweet {i}/{len(TEST_TWEET_URLS)}")
        print(f"{'=' * 80}\n")

        result = await test_tweet(providers, tweet_url)
        all_results.append(result)

        # Small delay between tweets to avoid rate limiting
        if i < len(TEST_TWEET_URLS):
            await asyncio.sleep(2)

    # Print summary
    print_header("SUMMARY")
    print()

    for i, result in enumerate(all_results, 1):
        print(f"{i}. {result['url']}")
        for provider, success in result["providers"].items():
            if success is True:
                print(f"   ✅ {provider}")
            elif success is False:
                print(f"   ❌ {provider}")
            else:
                print(f"   ⚠️  {provider} (skipped)")
        print()

    # Calculate success rates
    print_separator()
    print("\nProvider Success Rates:")
    print("-" * 60)

    for provider_name in providers.keys():
        total = sum(1 for r in all_results if r["providers"].get(provider_name) is not None)
        successes = sum(1 for r in all_results if r["providers"].get(provider_name) is True)

        if total > 0:
            rate = (successes / total) * 100
            print(f"{provider_name.upper()}: {successes}/{total} ({rate:.1f}%)")
        else:
            print(f"{provider_name.upper()}: Skipped (no API key)")

    print()
    print_separator()
    print("\n✅ Integration test completed!")


if __name__ == "__main__":
    asyncio.run(main())
