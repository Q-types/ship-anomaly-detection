"""
Structured logging configuration for the anomaly detection API.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import structlog


def add_timestamp(logger, method_name, event_dict):
    """Add ISO timestamp to log events."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_service_info(logger, method_name, event_dict):
    """Add service information to log events."""
    event_dict["service"] = "anomaly-detection-api"
    return event_dict


def setup_structured_logging(
    log_level: str = "INFO",
    json_format: bool = True,
    log_file: str = None
) -> None:
    """
    Configure structured logging with structlog.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output JSON logs. If False, output human-readable.
        log_file: Optional file path for logging output
    """
    # Determine processors based on format
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_timestamp,
        add_service_info,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Human-readable format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
    )

    # Reduce noise from third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestLogger:
    """Context manager for request logging with timing."""

    def __init__(self, request_id: str, method: str, path: str):
        self.request_id = request_id
        self.method = method
        self.path = path
        self.logger = get_logger("request")
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        self.logger.info(
            "request_started",
            request_id=self.request_id,
            method=self.method,
            path=self.path,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            self.logger.error(
                "request_failed",
                request_id=self.request_id,
                method=self.method,
                path=self.path,
                elapsed_ms=round(elapsed_ms, 2),
                error=str(exc_val),
                error_type=exc_type.__name__,
            )
        else:
            self.logger.info(
                "request_completed",
                request_id=self.request_id,
                method=self.method,
                path=self.path,
                elapsed_ms=round(elapsed_ms, 2),
            )

        return False


def log_prediction(
    request_id: str,
    model: str,
    is_anomaly: bool,
    confidence: float,
    anomaly_score: float,
    elapsed_ms: float
) -> None:
    """Log a prediction event."""
    logger = get_logger("prediction")
    logger.info(
        "prediction_made",
        request_id=request_id,
        model=model,
        is_anomaly=is_anomaly,
        confidence=round(confidence, 4),
        anomaly_score=round(anomaly_score, 6),
        elapsed_ms=round(elapsed_ms, 2),
    )


def log_batch_prediction(
    request_id: str,
    model: str,
    batch_size: int,
    anomaly_count: int,
    anomaly_rate: float,
    elapsed_ms: float
) -> None:
    """Log a batch prediction event."""
    logger = get_logger("prediction")
    logger.info(
        "batch_prediction_made",
        request_id=request_id,
        model=model,
        batch_size=batch_size,
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_rate, 4),
        elapsed_ms=round(elapsed_ms, 2),
    )


def log_model_loaded(model: str, version: str, created_utc: str) -> None:
    """Log model loading event."""
    logger = get_logger("model")
    logger.info(
        "model_loaded",
        model=model,
        version=version,
        created_utc=created_utc,
    )


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log an error with context."""
    logger = get_logger("error")
    logger.error(
        "error_occurred",
        error=str(error),
        error_type=type(error).__name__,
        **(context or {}),
    )
