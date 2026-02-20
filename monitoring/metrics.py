"""
Prometheus metrics for monitoring the anomaly detection API.
"""
from __future__ import annotations

from functools import wraps
import time
from typing import Callable, Optional

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

# ============================================================================
# Metric Definitions
# ============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    "anomaly_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "anomaly_api_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "anomaly_predictions_total",
    "Total number of predictions made",
    ["model", "result"]
)

PREDICTION_LATENCY = Histogram(
    "anomaly_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
)

PREDICTION_CONFIDENCE = Histogram(
    "anomaly_prediction_confidence",
    "Prediction confidence distribution",
    ["model", "result"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

BATCH_SIZE = Histogram(
    "anomaly_batch_size",
    "Batch prediction size distribution",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000)
)

# Model metrics
MODEL_LOADED = Gauge(
    "anomaly_model_loaded",
    "Whether the model is loaded (1) or not (0)",
    ["model"]
)

MODEL_INFO = Info(
    "anomaly_model",
    "Model information"
)

# System metrics
ACTIVE_REQUESTS = Gauge(
    "anomaly_active_requests",
    "Number of currently active requests"
)


# ============================================================================
# Metrics Router
# ============================================================================

metrics_router = APIRouter()


@metrics_router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Endpoint for Prometheus metric scraping",
    tags=["Monitoring"],
)
async def get_metrics() -> Response:
    """Return Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Metric Helpers
# ============================================================================

def track_prediction(
    model: str,
    is_anomaly: bool,
    confidence: float,
    latency: float
) -> None:
    """Track metrics for a single prediction."""
    result = "anomaly" if is_anomaly else "normal"

    PREDICTION_COUNT.labels(model=model, result=result).inc()
    PREDICTION_LATENCY.labels(model=model).observe(latency)
    PREDICTION_CONFIDENCE.labels(model=model, result=result).observe(confidence)


def track_batch_prediction(
    model: str,
    batch_size: int,
    anomaly_count: int,
    latency: float
) -> None:
    """Track metrics for a batch prediction."""
    BATCH_SIZE.observe(batch_size)
    PREDICTION_COUNT.labels(model=model, result="anomaly").inc(anomaly_count)
    PREDICTION_COUNT.labels(model=model, result="normal").inc(batch_size - anomaly_count)
    PREDICTION_LATENCY.labels(model=model).observe(latency)


def track_request(method: str, endpoint: str, status_code: int, latency: float) -> None:
    """Track metrics for an API request."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)


def set_model_loaded(model: str, loaded: bool) -> None:
    """Set model loaded status."""
    MODEL_LOADED.labels(model=model).set(1 if loaded else 0)


def set_model_info(ocsvm_version: str, iforest_version: str) -> None:
    """Set model info."""
    MODEL_INFO.info({
        "ocsvm_version": ocsvm_version,
        "iforest_version": iforest_version,
    })


# ============================================================================
# Decorators
# ============================================================================

def track_time(metric: Histogram, labels: Optional[dict] = None):
    """Decorator to track function execution time."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if labels:
                    metric.labels(**labels).observe(elapsed)
                else:
                    metric.observe(elapsed)
        return wrapper
    return decorator
