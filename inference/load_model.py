"""
Model loading utilities for ship engine anomaly detection.

Handles:
- Loading trained model artifacts from joblib files
- Model validation and metadata extraction
- Lazy loading with caching for performance
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging
from functools import lru_cache

from joblib import load
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    """Container for loaded model and its metadata."""

    model: Any
    scaler: Optional[Any]
    feature_names: list[str]
    symbolic_meta: Dict[str, Any]
    model_type: str
    version: str
    train_anom_rate: float
    params: Dict[str, Any]
    created_utc: str

    @property
    def requires_scaling(self) -> bool:
        """Whether this model requires input scaling."""
        return self.scaler is not None


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


def validate_artifact(artifact: Dict[str, Any], model_type: str) -> None:
    """
    Validate that a loaded artifact has all required fields.

    Args:
        artifact: Loaded dict from joblib
        model_type: Expected model type (OCSVM or IsolationForest)

    Raises:
        ModelLoadError: If validation fails
    """
    required_fields = [
        "model",
        "feature_names",
        "symbolic_meta",
        "model_type",
        "version",
        "train_anom_rate",
        "params",
        "created_utc",
    ]

    missing = [f for f in required_fields if f not in artifact]
    if missing:
        raise ModelLoadError(f"Artifact missing required fields: {missing}")

    if artifact["model_type"] != model_type:
        raise ModelLoadError(
            f"Expected model type '{model_type}', got '{artifact['model_type']}'"
        )

    # Validate symbolic metadata
    sym_meta = artifact["symbolic_meta"]
    if "spec" not in sym_meta or "output_order" not in sym_meta:
        raise ModelLoadError("Invalid symbolic_meta structure")


def load_model_artifact(path: Path, expected_type: str) -> ModelArtifact:
    """
    Load a model artifact from disk.

    Args:
        path: Path to the joblib file
        expected_type: Expected model type for validation

    Returns:
        ModelArtifact instance

    Raises:
        ModelLoadError: If loading or validation fails
    """
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    try:
        artifact = load(path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {path}: {e}")

    validate_artifact(artifact, expected_type)

    return ModelArtifact(
        model=artifact["model"],
        scaler=artifact.get("scaler"),
        feature_names=artifact["feature_names"],
        symbolic_meta=artifact["symbolic_meta"],
        model_type=artifact["model_type"],
        version=artifact["version"],
        train_anom_rate=artifact["train_anom_rate"],
        params=artifact["params"],
        created_utc=artifact["created_utc"],
    )


class ModelRegistry:
    """
    Registry for managing loaded models.

    Provides lazy loading and caching of models.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._ocsvm: Optional[ModelArtifact] = None
        self._iforest: Optional[ModelArtifact] = None
        self._loaded = False

    def _find_latest_model(self, prefix: str) -> Optional[Path]:
        """Find the most recent model file with given prefix."""
        pattern = f"{prefix}*.joblib"
        matches = list(self.model_dir.glob(pattern))

        if not matches:
            return None

        # Sort by modification time, newest first
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def load_models(
        self,
        ocsvm_path: Optional[Path] = None,
        iforest_path: Optional[Path] = None,
    ) -> None:
        """
        Load both models into the registry.

        Args:
            ocsvm_path: Explicit path to OCSVM model (auto-detect if None)
            iforest_path: Explicit path to IsolationForest model (auto-detect if None)
        """
        # Find OCSVM model
        if ocsvm_path is None:
            ocsvm_path = self._find_latest_model("ocsvm_symbolic")

        if ocsvm_path:
            logger.info(f"Loading OCSVM model from {ocsvm_path}")
            self._ocsvm = load_model_artifact(ocsvm_path, "OCSVM")
        else:
            logger.warning("No OCSVM model found")

        # Find IsolationForest model
        if iforest_path is None:
            iforest_path = self._find_latest_model("if_symbolic")

        if iforest_path:
            logger.info(f"Loading IsolationForest model from {iforest_path}")
            self._iforest = load_model_artifact(iforest_path, "IsolationForest")
        else:
            logger.warning("No IsolationForest model found")

        self._loaded = True

    @property
    def ocsvm(self) -> ModelArtifact:
        """Get the OCSVM model artifact."""
        if not self._loaded:
            self.load_models()
        if self._ocsvm is None:
            raise ModelLoadError("OCSVM model not available")
        return self._ocsvm

    @property
    def iforest(self) -> ModelArtifact:
        """Get the IsolationForest model artifact."""
        if not self._loaded:
            self.load_models()
        if self._iforest is None:
            raise ModelLoadError("IsolationForest model not available")
        return self._iforest

    def get_model(self, model_name: str) -> ModelArtifact:
        """
        Get a model by name.

        Args:
            model_name: One of 'ocsvm', 'iforest', or 'both'

        Returns:
            ModelArtifact instance
        """
        if model_name.lower() in ("ocsvm", "ocsvm_symbolic"):
            return self.ocsvm
        elif model_name.lower() in ("iforest", "if", "isolation_forest", "if_symbolic"):
            return self.iforest
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded

    def get_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {"loaded": self._loaded, "models": {}}

        if self._ocsvm:
            info["models"]["ocsvm"] = {
                "version": self._ocsvm.version,
                "created_utc": self._ocsvm.created_utc,
                "train_anomaly_rate": self._ocsvm.train_anom_rate,
                "params": self._ocsvm.params,
                "feature_names": self._ocsvm.feature_names,
            }

        if self._iforest:
            info["models"]["iforest"] = {
                "version": self._iforest.version,
                "created_utc": self._iforest.created_utc,
                "train_anomaly_rate": self._iforest.train_anom_rate,
                "params": self._iforest.params,
                "feature_names": self._iforest.feature_names,
            }

        return info


# Global registry instance (lazy-loaded)
_registry: Optional[ModelRegistry] = None


def get_model_registry(model_dir: Optional[Path] = None) -> ModelRegistry:
    """
    Get or create the global model registry.

    Args:
        model_dir: Path to models directory. Required on first call.

    Returns:
        ModelRegistry instance
    """
    global _registry

    if _registry is None:
        if model_dir is None:
            raise ValueError("model_dir required for first registry access")
        _registry = ModelRegistry(model_dir)

    return _registry


def reset_registry() -> None:
    """Reset the global model registry. Useful for testing."""
    global _registry
    _registry = None
