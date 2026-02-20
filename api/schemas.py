"""
Pydantic schemas for API request/response models.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Request Schemas
# ============================================================================


class SensorReadingRequest(BaseModel):
    """Single sensor reading for prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "engine_rpm": 750.0,
                "lub_oil_pressure": 3.5,
                "fuel_pressure": 6.0,
                "coolant_pressure": 2.5,
                "oil_temp": 78.0,
                "coolant_temp": 72.0,
            }
        }
    )

    engine_rpm: float = Field(..., ge=0, le=3000, description="Engine RPM (0-3000)")
    lub_oil_pressure: float = Field(..., ge=0, le=15, description="Lubricating oil pressure (bar)")
    fuel_pressure: float = Field(..., ge=0, le=50, description="Fuel pressure (bar)")
    coolant_pressure: float = Field(..., ge=0, le=10, description="Coolant pressure (bar)")
    oil_temp: float = Field(..., ge=0, le=150, description="Oil temperature (°C)")
    coolant_temp: float = Field(..., ge=0, le=120, description="Coolant temperature (°C)")


class BatchPredictionRequest(BaseModel):
    """Batch of sensor readings for bulk prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "readings": [
                    {
                        "engine_rpm": 750.0,
                        "lub_oil_pressure": 3.5,
                        "fuel_pressure": 6.0,
                        "coolant_pressure": 2.5,
                        "oil_temp": 78.0,
                        "coolant_temp": 72.0,
                    },
                    {
                        "engine_rpm": 900.0,
                        "lub_oil_pressure": 1.0,
                        "fuel_pressure": 20.0,
                        "coolant_pressure": 1.5,
                        "oil_temp": 95.0,
                        "coolant_temp": 88.0,
                    },
                ],
                "model": "ocsvm",
            }
        }
    )

    readings: List[SensorReadingRequest] = Field(
        ..., min_length=1, max_length=10000, description="List of sensor readings"
    )
    model: str = Field(default="ocsvm", pattern="^(ocsvm|iforest)$", description="Model to use")


class PredictionOptions(BaseModel):
    """Options for prediction request."""

    model: str = Field(default="ocsvm", pattern="^(ocsvm|iforest|ensemble)$")
    include_scores: bool = Field(default=True, description="Include raw anomaly scores")
    include_features: bool = Field(default=False, description="Include computed features")


# ============================================================================
# Response Schemas
# ============================================================================


class AnomalyLabelEnum(str, Enum):
    NORMAL = "normal"
    ANOMALY = "anomaly"
    UNCERTAIN = "uncertain"


class PredictionResult(BaseModel):
    """Result for a single prediction."""

    is_anomaly: bool = Field(..., description="Whether the sample is classified as anomaly")
    label: AnomalyLabelEnum = Field(..., description="Classification label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    anomaly_score: float = Field(..., description="Raw anomaly score from model")
    model_used: str = Field(..., description="Model that produced this prediction")


class PredictionResponse(BaseModel):
    """Response for single prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": {
                    "is_anomaly": False,
                    "label": "normal",
                    "confidence": 0.87,
                    "anomaly_score": 0.234,
                    "model_used": "OCSVM",
                },
                "processing_time_ms": 12.5,
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }
    )

    prediction: PredictionResult
    processing_time_ms: float
    timestamp: datetime


class EnsemblePredictionResponse(BaseModel):
    """Response for ensemble prediction using both models."""

    ocsvm: Optional[PredictionResult] = None
    iforest: Optional[PredictionResult] = None
    consensus: Dict[str, Any] = Field(
        ...,
        description="Consensus prediction combining both models"
    )
    processing_time_ms: float
    timestamp: datetime


class BatchPredictionSummary(BaseModel):
    """Summary statistics for batch prediction."""

    total_samples: int
    anomalies_detected: int
    anomaly_rate: float
    uncertain_count: int
    mean_confidence: float
    model_used: str


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""

    results: List[PredictionResult]
    summary: BatchPredictionSummary
    processing_time_ms: float
    timestamp: datetime


# ============================================================================
# Health & Info Schemas
# ============================================================================


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response."""

    status: HealthStatus
    version: str
    environment: str
    models_loaded: bool
    checks: Dict[str, bool]
    timestamp: datetime


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    version: str
    created_utc: str
    train_anomaly_rate: float
    params: Dict[str, Any]
    feature_names: List[str]


class ModelsInfoResponse(BaseModel):
    """Information about all loaded models."""

    loaded: bool
    models: Dict[str, ModelInfo]


class APIInfoResponse(BaseModel):
    """API information response."""

    name: str
    version: str
    description: str
    docs_url: str
    endpoints: List[str]


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorDetail(BaseModel):
    """Error detail for validation errors."""

    field: str
    message: str
    value: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime
