"""
Ship Engine Anomaly Detection API

A production-ready REST API for detecting anomalies in ship engine sensor data
using trained machine learning models (OCSVM and Isolation Forest).

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from api.routes import router as api_router
from api.middleware import setup_middleware
from inference.load_model import ModelRegistry


# Configure logging
def setup_logging(settings):
    """Configure application logging."""
    log_format = (
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": "%(message)s"}'
        if settings.log_format == "json"
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
        handlers=[logging.StreamHandler()],
    )

    # Reduce noise from third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    settings = get_settings()
    setup_logging(settings)

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Load models on startup
    model_dir = PROJECT_ROOT / settings.model_dir
    logger.info(f"Loading models from {model_dir}")

    try:
        registry = ModelRegistry(model_dir)
        registry.load_models()
        app.state.model_registry = registry
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Continue running - health endpoint will report degraded status
        app.state.model_registry = ModelRegistry(model_dir)

    yield

    # Shutdown
    logger.info("Shutting down application")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Ship Engine Anomaly Detection API

This API provides real-time anomaly detection for ship engine sensor data using
machine learning models trained on symbolic regression features.

### Models Available

- **OCSVM (One-Class SVM)**: Uses RBF kernel with scaled features. Good for
  detecting point anomalies with clear decision boundaries.

- **Isolation Forest**: Tree-based isolation approach. Effective for detecting
  anomalies through their isolation properties.

- **Ensemble**: Combines both models for higher confidence predictions.

### Input Features

The API accepts 6 sensor readings:
- `engine_rpm`: Engine RPM (0-3000)
- `lub_oil_pressure`: Lubricating oil pressure in bar (0-15)
- `fuel_pressure`: Fuel pressure in bar (0-50)
- `coolant_pressure`: Coolant pressure in bar (0-10)
- `oil_temp`: Oil temperature in °C (0-150)
- `coolant_temp`: Coolant temperature in °C (0-120)

### Output

Each prediction includes:
- `is_anomaly`: Boolean flag
- `label`: "normal", "anomaly", or "uncertain"
- `confidence`: Prediction confidence (0-1)
- `anomaly_score`: Raw model score
        """,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
    )

    # Setup middleware
    setup_middleware(app, settings)

    # Include API routes
    app.include_router(api_router, prefix=settings.api_prefix)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to API docs."""
        return RedirectResponse(url="/docs")

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=settings.workers if not settings.is_development else 1,
    )
