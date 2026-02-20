"""Inference package for anomaly detection."""
from inference.preprocess import (
    SensorReading,
    BatchSensorReadings,
    preprocess_dataframe,
    preprocess_single_reading,
    normalize_columns,
)
from inference.load_model import (
    ModelArtifact,
    ModelRegistry,
    ModelLoadError,
    get_model_registry,
    reset_registry,
)
from inference.detect import (
    AnomalyLabel,
    DetectionResult,
    BatchDetectionResult,
    EnsembleResult,
    AnomalyDetector,
    create_detector,
)

__all__ = [
    # Preprocessing
    "SensorReading",
    "BatchSensorReadings",
    "preprocess_dataframe",
    "preprocess_single_reading",
    "normalize_columns",
    # Model loading
    "ModelArtifact",
    "ModelRegistry",
    "ModelLoadError",
    "get_model_registry",
    "reset_registry",
    # Detection
    "AnomalyLabel",
    "DetectionResult",
    "BatchDetectionResult",
    "EnsembleResult",
    "AnomalyDetector",
    "create_detector",
]
