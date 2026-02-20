"""
Application configuration using Pydantic Settings.

Supports environment variables, .env files, and sensible defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Ship Engine Anomaly Detection API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = Field(default="development", pattern="^(development|staging|production)$")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # API
    api_prefix: str = "/api/v1"
    docs_enabled: bool = True

    # Security
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    cors_origins: List[str] = ["*"]
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Model paths
    model_dir: Path = Field(default=Path("models"))
    ocsvm_model_path: Optional[Path] = None
    iforest_model_path: Optional[Path] = None

    # Inference
    default_model: str = Field(default="ocsvm", pattern="^(ocsvm|iforest|ensemble)$")
    max_batch_size: int = 10000
    strict_validation: bool = False

    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = "json"  # "json" or "text"
    log_requests: bool = True

    # Monitoring
    metrics_enabled: bool = True
    metrics_prefix: str = "anomaly_detection"

    @field_validator("model_dir", mode="before")
    @classmethod
    def resolve_model_dir(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# For direct imports
settings = get_settings()
