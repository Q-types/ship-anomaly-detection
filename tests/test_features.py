"""
Tests for the feature engineering module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.symbolic import (
    compute_symbolic_features,
    evaluate_symbolic_equation,
    get_symbolic_spec_metadata,
    SYMBOLIC_SPEC,
    OUTPUT_ORDER,
)


class TestSymbolicFeatures:
    """Tests for symbolic feature computation."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with required columns."""
        return pd.DataFrame({
            "engine_rpm": [750.0, 800.0, 850.0],
            "coolant_temp": [72.0, 75.0, 78.0],
            "coolant_pressure": [2.5, 2.8, 3.0],
            "oil_temp": [78.0, 80.0, 82.0],
        })

    def test_compute_symbolic_features_shape(self, sample_df):
        """Test that computed features have correct shape."""
        X_sym, names = compute_symbolic_features(sample_df)

        assert X_sym.shape == (len(sample_df), len(OUTPUT_ORDER))
        assert len(names) == len(OUTPUT_ORDER)

    def test_compute_symbolic_features_names(self, sample_df):
        """Test that feature names are correct."""
        _, names = compute_symbolic_features(sample_df)

        expected_names = [f"sym_{name}" for name in OUTPUT_ORDER]
        assert names == expected_names

    def test_compute_symbolic_features_no_nan(self, sample_df):
        """Test that computed features have no NaN values."""
        X_sym, _ = compute_symbolic_features(sample_df)

        assert not np.isnan(X_sym).any()

    def test_compute_symbolic_features_finite(self, sample_df):
        """Test that computed features are finite."""
        X_sym, _ = compute_symbolic_features(sample_df)

        assert np.isfinite(X_sym).all()

    def test_missing_column_raises(self):
        """Test that missing required column raises error."""
        df = pd.DataFrame({
            "engine_rpm": [750.0],
            # Missing coolant_temp, coolant_pressure, oil_temp
        })

        with pytest.raises(KeyError):
            compute_symbolic_features(df)


class TestSymbolicEquationEvaluation:
    """Tests for individual equation evaluation."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for equation evaluation."""
        return pd.DataFrame({
            "engine_rpm": [750.0, 800.0],
            "coolant_temp": [72.0, 75.0],
            "coolant_pressure": [2.5, 2.8],
            "oil_temp": [78.0, 80.0],
        })

    def test_evaluate_oil_temp_equation(self, sample_df):
        """Test oil_temp symbolic equation."""
        spec = SYMBOLIC_SPEC["oil_temp"]
        result = evaluate_symbolic_equation(
            sample_df,
            spec["equation"],
            spec["variables"]
        )

        assert len(result) == len(sample_df)
        assert np.isfinite(result).all()

    def test_evaluate_coolant_temp_equation(self, sample_df):
        """Test coolant_temp symbolic equation."""
        spec = SYMBOLIC_SPEC["coolant_temp"]
        result = evaluate_symbolic_equation(
            sample_df,
            spec["equation"],
            spec["variables"]
        )

        assert len(result) == len(sample_df)
        assert np.isfinite(result).all()

    def test_evaluate_oil_pressure_equation(self, sample_df):
        """Test oil_pressure symbolic equation."""
        spec = SYMBOLIC_SPEC["oil_pressure"]
        result = evaluate_symbolic_equation(
            sample_df,
            spec["equation"],
            spec["variables"]
        )

        assert len(result) == len(sample_df)
        assert np.isfinite(result).all()

    def test_evaluate_fuel_pressure_equation(self, sample_df):
        """Test fuel_pressure symbolic equation."""
        spec = SYMBOLIC_SPEC["fuel_pressure"]
        result = evaluate_symbolic_equation(
            sample_df,
            spec["equation"],
            spec["variables"]
        )

        assert len(result) == len(sample_df)
        assert np.isfinite(result).all()


class TestSymbolicMetadata:
    """Tests for symbolic specification metadata."""

    def test_metadata_structure(self):
        """Test metadata has required fields."""
        meta = get_symbolic_spec_metadata()

        assert "output_order" in meta
        assert "spec" in meta
        assert "required_input_columns" in meta
        assert "feature_names" in meta
        assert "version" in meta

    def test_metadata_version(self):
        """Test metadata version is set."""
        meta = get_symbolic_spec_metadata()
        assert meta["version"] == "v1.0"

    def test_metadata_feature_names(self):
        """Test metadata feature names match output order."""
        meta = get_symbolic_spec_metadata()

        expected = [f"sym_{name}" for name in OUTPUT_ORDER]
        assert meta["feature_names"] == expected

    def test_required_columns(self):
        """Test required input columns are correct."""
        meta = get_symbolic_spec_metadata()

        required = meta["required_input_columns"]
        assert "engine_rpm" in required
        assert "coolant_temp" in required
        assert "coolant_pressure" in required
        assert "oil_temp" in required
