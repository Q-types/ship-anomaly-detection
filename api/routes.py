"""
API routes for anomaly detection endpoints.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.schemas import (
    SensorReadingRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    EnsemblePredictionResponse,
    PredictionResult,
    BatchPredictionSummary,
    HealthResponse,
    HealthStatus,
    ModelsInfoResponse,
    ModelInfo,
    APIInfoResponse,
    AnomalyLabelEnum,
)
from api.dependencies import get_detector, get_settings
from inference.detect import AnomalyDetector, AnomalyLabel
from config.settings import Settings

router = APIRouter()


def _convert_label(label: AnomalyLabel) -> AnomalyLabelEnum:
    """Convert internal label to API enum."""
    return AnomalyLabelEnum(label.value)


# ============================================================================
# Prediction Endpoints
# ============================================================================


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict anomaly for single reading",
    description="Analyze a single sensor reading and predict if it represents an anomaly.",
    tags=["Predictions"],
)
async def predict_single(
    reading: SensorReadingRequest,
    model: str = Query(default="ocsvm", pattern="^(ocsvm|iforest)$"),
    detector: AnomalyDetector = Depends(get_detector),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """Predict anomaly for a single sensor reading."""
    start_time = time.perf_counter()

    # Convert to DataFrame
    df = pd.DataFrame([reading.model_dump()])

    # Run detection
    result = detector.detect_single(df, model_name=model)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return PredictionResponse(
        prediction=PredictionResult(
            is_anomaly=result.is_anomaly,
            label=_convert_label(result.label),
            confidence=result.confidence,
            anomaly_score=result.anomaly_score,
            model_used=result.model_used,
        ),
        processing_time_ms=round(elapsed_ms, 2),
        timestamp=datetime.now(timezone.utc),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict anomalies for batch of readings",
    description="Analyze multiple sensor readings in a single request.",
    tags=["Predictions"],
)
async def predict_batch(
    request: BatchPredictionRequest,
    detector: AnomalyDetector = Depends(get_detector),
    settings: Settings = Depends(get_settings),
) -> BatchPredictionResponse:
    """Predict anomalies for a batch of sensor readings."""
    start_time = time.perf_counter()

    # Check batch size
    if len(request.readings) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.readings)} exceeds maximum {settings.max_batch_size}",
        )

    # Convert to DataFrame
    df = pd.DataFrame([r.model_dump() for r in request.readings])

    # Run detection
    batch_result = detector.detect_batch(df, model_name=request.model)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return BatchPredictionResponse(
        results=[
            PredictionResult(
                is_anomaly=r.is_anomaly,
                label=_convert_label(r.label),
                confidence=r.confidence,
                anomaly_score=r.anomaly_score,
                model_used=r.model_used,
            )
            for r in batch_result.results
        ],
        summary=BatchPredictionSummary(**batch_result.summary),
        processing_time_ms=round(elapsed_ms, 2),
        timestamp=datetime.now(timezone.utc),
    )


@router.post(
    "/predict/ensemble",
    response_model=EnsemblePredictionResponse,
    summary="Predict using both models",
    description="Run prediction using both OCSVM and Isolation Forest, with consensus.",
    tags=["Predictions"],
)
async def predict_ensemble(
    reading: SensorReadingRequest,
    require_agreement: bool = Query(default=False),
    detector: AnomalyDetector = Depends(get_detector),
) -> EnsemblePredictionResponse:
    """Predict anomaly using ensemble of both models."""
    start_time = time.perf_counter()

    # Convert to DataFrame
    df = pd.DataFrame([reading.model_dump()])

    # Run ensemble detection
    result = detector.detect_ensemble(df, require_agreement=require_agreement)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Build response
    ocsvm_result = None
    if result.ocsvm_result:
        ocsvm_result = PredictionResult(
            is_anomaly=result.ocsvm_result.is_anomaly,
            label=_convert_label(result.ocsvm_result.label),
            confidence=result.ocsvm_result.confidence,
            anomaly_score=result.ocsvm_result.anomaly_score,
            model_used=result.ocsvm_result.model_used,
        )

    iforest_result = None
    if result.iforest_result:
        iforest_result = PredictionResult(
            is_anomaly=result.iforest_result.is_anomaly,
            label=_convert_label(result.iforest_result.label),
            confidence=result.iforest_result.confidence,
            anomaly_score=result.iforest_result.anomaly_score,
            model_used=result.iforest_result.model_used,
        )

    return EnsemblePredictionResponse(
        ocsvm=ocsvm_result,
        iforest=iforest_result,
        consensus={
            "label": result.consensus_label.value,
            "confidence": round(result.consensus_confidence, 4),
            "models_agree": result.agreement,
        },
        processing_time_ms=round(elapsed_ms, 2),
        timestamp=datetime.now(timezone.utc),
    )


# ============================================================================
# Health & Info Endpoints
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and model availability.",
    tags=["Health"],
)
async def health_check(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Check API health status."""
    checks = {
        "api": True,
        "models": False,
    }

    # Check if models are loaded
    try:
        registry = request.app.state.model_registry
        checks["models"] = registry.is_loaded()
    except Exception:
        pass

    # Determine overall status
    if all(checks.values()):
        status = HealthStatus.HEALTHY
    elif checks["api"]:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.UNHEALTHY

    return HealthResponse(
        status=status,
        version=settings.app_version,
        environment=settings.environment,
        models_loaded=checks["models"],
        checks=checks,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/models",
    response_model=ModelsInfoResponse,
    summary="Model information",
    description="Get information about loaded models.",
    tags=["Info"],
)
async def get_models_info(request: Request) -> ModelsInfoResponse:
    """Get information about loaded models."""
    try:
        registry = request.app.state.model_registry
        info = registry.get_info()

        models = {}
        for name, data in info.get("models", {}).items():
            models[name] = ModelInfo(**data)

        return ModelsInfoResponse(loaded=info["loaded"], models=models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")


@router.get(
    "/info",
    response_model=APIInfoResponse,
    summary="API information",
    description="Get API metadata and available endpoints.",
    tags=["Info"],
)
async def get_api_info(settings: Settings = Depends(get_settings)) -> APIInfoResponse:
    """Get API information."""
    return APIInfoResponse(
        name=settings.app_name,
        version=settings.app_version,
        description="Ship engine anomaly detection API using ML models",
        docs_url="/docs" if settings.docs_enabled else "",
        endpoints=[
            "POST /api/v1/predict - Single prediction",
            "POST /api/v1/predict/batch - Batch predictions",
            "POST /api/v1/predict/ensemble - Ensemble prediction",
            "GET /api/v1/health - Health check",
            "GET /api/v1/models - Model information",
            "GET /api/v1/info - API information",
        ],
    )
