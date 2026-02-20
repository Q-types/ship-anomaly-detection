"""
FastAPI dependencies for dependency injection.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Generator

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from config.settings import Settings, get_settings
from inference.detect import AnomalyDetector, create_detector
from inference.load_model import ModelRegistry


# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify API key if configured."""
    # Skip verification if no API key is configured
    if not settings.api_key:
        return ""

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key


def get_model_registry(request: Request) -> ModelRegistry:
    """Get model registry from app state."""
    if not hasattr(request.app.state, "model_registry"):
        raise HTTPException(
            status_code=503,
            detail="Models not loaded",
        )
    return request.app.state.model_registry


def get_detector(
    request: Request,
    registry: ModelRegistry = Depends(get_model_registry),
) -> AnomalyDetector:
    """Get anomaly detector instance."""
    if not hasattr(request.app.state, "detector"):
        request.app.state.detector = create_detector(registry)
    return request.app.state.detector


# Re-export get_settings for routes
__all__ = [
    "verify_api_key",
    "get_model_registry",
    "get_detector",
    "get_settings",
]
