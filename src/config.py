"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    claude_api_key: str
    gemini_api_key: str
    perplexity_api_key: str

    # LLM Weights (equal by default)
    claude_weight: float = 1.0
    gemini_weight: float = 1.0
    perplexity_weight: float = 1.0

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
