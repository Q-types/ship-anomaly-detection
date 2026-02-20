"""
Tests for the inference pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.preprocess import (
    SensorReading,
    BatchSensorReadings,
    preprocess_dataframe,
    preprocess_single_reading,
    normalize_columns,
    COLUMN_MAPPING,
    SENSOR_RANGES,
)
from inference.load_model import ModelRegistry, ModelLoadError
from inference.detect import AnomalyDetector, AnomalyLabel, create_detector


class TestSensorReading:
    """Tests for the SensorReading model."""

    def test_valid_reading(self, sample_reading):
        """Test creating a valid sensor reading."""
        reading = SensorReading(**sample_reading)

        assert reading.engine_rpm == sample_reading["engine_rpm"]
        assert reading.oil_temp == sample_reading["oil_temp"]

    def test_reading_to_dict(self, sample_reading):
        """Test converting reading to dict."""
        reading = SensorReading(**sample_reading)
        result = reading.to_dataframe_row()

        assert isinstance(result, dict)
        assert all(k in result for k in sample_reading.keys())

    def test_invalid_rpm_too_high(self, sample_reading):
        """Test validation rejects RPM too high."""
        sample_reading["engine_rpm"] = 5000.0

        with pytest.raises(ValueError):
            SensorReading(**sample_reading)

    def test_invalid_negative_pressure(self, sample_reading):
        """Test validation rejects negative pressure."""
        sample_reading["lub_oil_pressure"] = -1.0

        with pytest.raises(ValueError):
            SensorReading(**sample_reading)

    def test_string_coercion(self, sample_reading):
        """Test that string values are coerced to float."""
        sample_reading["engine_rpm"] = "750.0"
        reading = SensorReading(**sample_reading)

        assert reading.engine_rpm == 750.0


class TestBatchSensorReadings:
    """Tests for batch sensor readings."""

    def test_valid_batch(self, batch_readings):
        """Test creating a valid batch."""
        batch = BatchSensorReadings(readings=[
            SensorReading(**r) for r in batch_readings
        ])

        assert len(batch.readings) == len(batch_readings)

    def test_batch_to_dataframe(self, batch_readings):
        """Test converting batch to DataFrame."""
        batch = BatchSensorReadings(readings=[
            SensorReading(**r) for r in batch_readings
        ])
        df = batch.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(batch_readings)


class TestNormalizeColumns:
    """Tests for column normalization."""

    def test_normalize_csv_columns(self):
        """Test normalizing CSV-format column names."""
        df = pd.DataFrame({
            "Engine rpm": [750.0],
            "Lub oil pressure": [3.5],
            "Fuel pressure": [6.0],
            "Coolant pressure": [2.5],
            "lub oil temp": [78.0],
            "Coolant temp": [72.0],
        })

        normalized = normalize_columns(df)

        assert "engine_rpm" in normalized.columns
        assert "oil_temp" in normalized.columns
        assert "coolant_temp" in normalized.columns

    def test_normalize_already_normalized(self, sample_reading):
        """Test normalizing already normalized columns."""
        df = pd.DataFrame([sample_reading])

        normalized = normalize_columns(df)

        # Should remain unchanged
        assert list(normalized.columns) == list(df.columns)


class TestPreprocessDataframe:
    """Tests for full preprocessing pipeline."""

    def test_preprocess_valid_data(self, sample_reading):
        """Test preprocessing valid data."""
        df = pd.DataFrame([sample_reading])

        processed, metadata = preprocess_dataframe(df)

        assert len(processed) == 1
        assert "warnings" in metadata
        assert "processed_shape" in metadata

    def test_preprocess_with_warnings(self):
        """Test preprocessing data with out-of-range values."""
        df = pd.DataFrame({
            "engine_rpm": [5000.0],  # Out of range
            "coolant_temp": [72.0],
            "coolant_pressure": [2.5],
            "oil_temp": [78.0],
            "lub_oil_pressure": [3.5],
            "fuel_pressure": [6.0],
        })

        processed, metadata = preprocess_dataframe(df, strict_validation=False)

        # Value should be clipped
        assert processed["engine_rpm"].iloc[0] <= SENSOR_RANGES["engine_rpm"][1]
        assert len(metadata["warnings"]) > 0


class TestModelRegistry:
    """Tests for the model registry."""

    def test_load_models(self, model_registry):
        """Test loading models into registry."""
        assert model_registry.is_loaded()

    def test_get_ocsvm(self, model_registry):
        """Test getting OCSVM model."""
        artifact = model_registry.ocsvm

        assert artifact is not None
        assert artifact.model_type == "OCSVM"
        assert artifact.scaler is not None

    def test_get_iforest(self, model_registry):
        """Test getting IsolationForest model."""
        artifact = model_registry.iforest

        assert artifact is not None
        assert artifact.model_type == "IsolationForest"

    def test_get_model_by_name(self, model_registry):
        """Test getting model by name."""
        ocsvm = model_registry.get_model("ocsvm")
        iforest = model_registry.get_model("iforest")

        assert ocsvm.model_type == "OCSVM"
        assert iforest.model_type == "IsolationForest"

    def test_get_info(self, model_registry):
        """Test getting registry info."""
        info = model_registry.get_info()

        assert info["loaded"] is True
        assert "ocsvm" in info["models"]
        assert "iforest" in info["models"]


class TestAnomalyDetector:
    """Tests for the anomaly detector."""

    @pytest.fixture
    def detector(self, model_registry):
        """Create an anomaly detector."""
        return create_detector(model_registry)

    @pytest.fixture
    def sample_df(self, sample_reading):
        """Create sample DataFrame."""
        return pd.DataFrame([sample_reading])

    def test_detect_single_ocsvm(self, detector, sample_df):
        """Test single detection with OCSVM."""
        result = detector.detect_single(sample_df, model_name="ocsvm")

        assert result is not None
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.label, AnomalyLabel)
        assert 0 <= result.confidence <= 1
        assert result.model_used == "OCSVM"

    def test_detect_single_iforest(self, detector, sample_df):
        """Test single detection with IsolationForest."""
        result = detector.detect_single(sample_df, model_name="iforest")

        assert result is not None
        assert result.model_used == "IsolationForest"

    def test_detect_batch(self, detector, batch_readings):
        """Test batch detection."""
        df = pd.DataFrame(batch_readings)
        result = detector.detect_batch(df, model_name="ocsvm")

        assert len(result.results) == len(batch_readings)
        assert result.summary["total_samples"] == len(batch_readings)
        assert 0 <= result.summary["anomaly_rate"] <= 1

    def test_detect_ensemble(self, detector, sample_df):
        """Test ensemble detection."""
        result = detector.detect_ensemble(sample_df)

        assert result.ocsvm_result is not None
        assert result.iforest_result is not None
        assert isinstance(result.consensus_label, AnomalyLabel)
        assert isinstance(result.agreement, bool)

    def test_result_to_dict(self, detector, sample_df):
        """Test converting result to dict."""
        result = detector.detect_single(sample_df, model_name="ocsvm")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "is_anomaly" in result_dict
        assert "label" in result_dict
        assert "confidence" in result_dict
