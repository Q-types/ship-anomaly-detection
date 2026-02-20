"""
Anomaly detection module for ship engine data.

Provides the core detection logic using trained OCSVM and IsolationForest models.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from features.symbolic import compute_symbolic_features
from inference.load_model import ModelArtifact, ModelRegistry


class AnomalyLabel(str, Enum):
    """Anomaly classification labels."""
    NORMAL = "normal"
    ANOMALY = "anomaly"
    UNCERTAIN = "uncertain"


@dataclass
class DetectionResult:
    """Result from anomaly detection for a single sample."""

    is_anomaly: bool
    label: AnomalyLabel
    confidence: float
    anomaly_score: float
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "anomaly_score": round(self.anomaly_score, 6),
            "model_used": self.model_used,
        }


@dataclass
class BatchDetectionResult:
    """Results from batch anomaly detection."""

    results: List[DetectionResult]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
        }


@dataclass
class EnsembleResult:
    """Result from ensemble detection using multiple models."""

    ocsvm_result: Optional[DetectionResult]
    iforest_result: Optional[DetectionResult]
    consensus_label: AnomalyLabel
    consensus_confidence: float
    agreement: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocsvm": self.ocsvm_result.to_dict() if self.ocsvm_result else None,
            "iforest": self.iforest_result.to_dict() if self.iforest_result else None,
            "consensus": {
                "label": self.consensus_label.value,
                "confidence": round(self.consensus_confidence, 4),
                "models_agree": self.agreement,
            }
        }


class AnomalyDetector:
    """
    Core anomaly detection engine.

    Supports single-model and ensemble predictions with confidence scoring.
    """

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute symbolic features from preprocessed data."""
        X_sym, _ = compute_symbolic_features(df)
        return X_sym

    def _predict_ocsvm(
        self,
        X: np.ndarray,
        artifact: ModelArtifact
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and scores from OCSVM model.

        Returns:
            Tuple of (predictions, decision_scores)
            predictions: -1 for anomaly, +1 for normal
            decision_scores: distance from decision boundary (negative = anomaly)
        """
        # Scale features
        if artifact.scaler is not None:
            X_scaled = artifact.scaler.transform(X)
        else:
            X_scaled = X

        predictions = artifact.model.predict(X_scaled)
        scores = artifact.model.decision_function(X_scaled)

        return predictions, scores

    def _predict_iforest(
        self,
        X: np.ndarray,
        artifact: ModelArtifact
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions and scores from IsolationForest model.

        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: -1 for anomaly, +1 for normal
            anomaly_scores: negative score (more negative = more anomalous)
        """
        predictions = artifact.model.predict(X)
        scores = artifact.model.score_samples(X)

        return predictions, scores

    def _score_to_confidence(
        self,
        score: float,
        model_type: str,
        threshold: float = 0.0
    ) -> float:
        """
        Convert raw anomaly score to confidence value [0, 1].

        Higher confidence means more certain about the prediction.
        """
        if model_type == "OCSVM":
            # OCSVM: score > 0 is normal, score < 0 is anomaly
            # Use sigmoid-like transformation
            confidence = 1.0 / (1.0 + np.exp(-2 * score))
        else:  # IsolationForest
            # IF: more negative scores = more anomalous
            # Typical scores range from -0.5 to 0.5
            confidence = 1.0 / (1.0 + np.exp(-5 * score))

        return float(np.clip(confidence, 0.0, 1.0))

    def _make_result(
        self,
        prediction: int,
        score: float,
        model_type: str,
        confidence_threshold: float = 0.6
    ) -> DetectionResult:
        """Create a DetectionResult from model output."""
        is_anomaly = prediction == -1
        confidence = self._score_to_confidence(score, model_type)

        # Determine label based on confidence
        if confidence < confidence_threshold:
            label = AnomalyLabel.UNCERTAIN
        elif is_anomaly:
            label = AnomalyLabel.ANOMALY
        else:
            label = AnomalyLabel.NORMAL

        return DetectionResult(
            is_anomaly=is_anomaly,
            label=label,
            confidence=confidence,
            anomaly_score=float(score),
            model_used=model_type,
        )

    def detect_single(
        self,
        df: pd.DataFrame,
        model_name: str = "ocsvm"
    ) -> DetectionResult:
        """
        Detect anomalies using a single model.

        Args:
            df: Preprocessed DataFrame (single row)
            model_name: Which model to use ('ocsvm' or 'iforest')

        Returns:
            DetectionResult for the input
        """
        artifact = self.registry.get_model(model_name)
        X = self._compute_features(df)

        if artifact.model_type == "OCSVM":
            preds, scores = self._predict_ocsvm(X, artifact)
        else:
            preds, scores = self._predict_iforest(X, artifact)

        return self._make_result(preds[0], scores[0], artifact.model_type)

    def detect_batch(
        self,
        df: pd.DataFrame,
        model_name: str = "ocsvm"
    ) -> BatchDetectionResult:
        """
        Detect anomalies for a batch of samples.

        Args:
            df: Preprocessed DataFrame (multiple rows)
            model_name: Which model to use

        Returns:
            BatchDetectionResult with per-sample results and summary
        """
        artifact = self.registry.get_model(model_name)
        X = self._compute_features(df)

        if artifact.model_type == "OCSVM":
            preds, scores = self._predict_ocsvm(X, artifact)
        else:
            preds, scores = self._predict_iforest(X, artifact)

        results = [
            self._make_result(p, s, artifact.model_type)
            for p, s in zip(preds, scores)
        ]

        # Compute summary statistics
        n_total = len(results)
        n_anomalies = sum(1 for r in results if r.is_anomaly)
        n_uncertain = sum(1 for r in results if r.label == AnomalyLabel.UNCERTAIN)

        summary = {
            "total_samples": n_total,
            "anomalies_detected": n_anomalies,
            "anomaly_rate": round(n_anomalies / n_total, 4) if n_total > 0 else 0.0,
            "uncertain_count": n_uncertain,
            "mean_confidence": round(np.mean([r.confidence for r in results]), 4),
            "model_used": artifact.model_type,
        }

        return BatchDetectionResult(results=results, summary=summary)

    def detect_ensemble(
        self,
        df: pd.DataFrame,
        require_agreement: bool = False
    ) -> EnsembleResult:
        """
        Detect anomalies using both models and combine results.

        Args:
            df: Preprocessed DataFrame (single row)
            require_agreement: If True, only flag anomaly if both models agree

        Returns:
            EnsembleResult with results from both models and consensus
        """
        X = self._compute_features(df)

        # Get OCSVM result
        ocsvm_result = None
        try:
            ocsvm_artifact = self.registry.ocsvm
            ocsvm_preds, ocsvm_scores = self._predict_ocsvm(X, ocsvm_artifact)
            ocsvm_result = self._make_result(
                ocsvm_preds[0], ocsvm_scores[0], "OCSVM"
            )
        except Exception:
            pass

        # Get IsolationForest result
        iforest_result = None
        try:
            iforest_artifact = self.registry.iforest
            if_preds, if_scores = self._predict_iforest(X, iforest_artifact)
            iforest_result = self._make_result(
                if_preds[0], if_scores[0], "IsolationForest"
            )
        except Exception:
            pass

        # Compute consensus
        results = [r for r in [ocsvm_result, iforest_result] if r is not None]

        if not results:
            return EnsembleResult(
                ocsvm_result=None,
                iforest_result=None,
                consensus_label=AnomalyLabel.UNCERTAIN,
                consensus_confidence=0.0,
                agreement=False,
            )

        # Check agreement
        anomaly_votes = sum(1 for r in results if r.is_anomaly)
        n_models = len(results)
        agreement = (anomaly_votes == 0 or anomaly_votes == n_models)

        # Average confidence
        avg_confidence = np.mean([r.confidence for r in results])

        # Determine consensus label
        if require_agreement:
            if agreement and anomaly_votes == n_models:
                consensus_label = AnomalyLabel.ANOMALY
            elif agreement and anomaly_votes == 0:
                consensus_label = AnomalyLabel.NORMAL
            else:
                consensus_label = AnomalyLabel.UNCERTAIN
        else:
            # Majority vote (if models disagree, use higher confidence)
            if anomaly_votes > n_models / 2:
                consensus_label = AnomalyLabel.ANOMALY
            elif anomaly_votes < n_models / 2:
                consensus_label = AnomalyLabel.NORMAL
            else:
                # Tie - use model with higher confidence
                if ocsvm_result and iforest_result:
                    if ocsvm_result.confidence >= iforest_result.confidence:
                        consensus_label = ocsvm_result.label
                    else:
                        consensus_label = iforest_result.label
                else:
                    consensus_label = results[0].label

        return EnsembleResult(
            ocsvm_result=ocsvm_result,
            iforest_result=iforest_result,
            consensus_label=consensus_label,
            consensus_confidence=float(avg_confidence),
            agreement=agreement,
        )


def create_detector(registry: ModelRegistry) -> AnomalyDetector:
    """Factory function to create an AnomalyDetector."""
    return AnomalyDetector(registry)
