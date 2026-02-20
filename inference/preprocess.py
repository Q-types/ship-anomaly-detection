"""
Preprocessing module for ship engine anomaly detection.

Handles:
- Column name normalization from various input formats
- Input validation with physical range checks
- Data type coercion
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


# Column mapping from common input formats to internal names
COLUMN_MAPPING: Dict[str, str] = {
    # CSV format
    "Engine rpm": "engine_rpm",
    "Lub oil pressure": "lub_oil_pressure",
    "Fuel pressure": "fuel_pressure",
    "Coolant pressure": "coolant_pressure",
    "lub oil temp": "oil_temp",
    "Coolant temp": "coolant_temp",
    # API format (already normalized)
    "engine_rpm": "engine_rpm",
    "lub_oil_pressure": "lub_oil_pressure",
    "fuel_pressure": "fuel_pressure",
    "coolant_pressure": "coolant_pressure",
    "oil_temp": "oil_temp",
    "coolant_temp": "coolant_temp",
}

# Physical validity ranges for sensor values
SENSOR_RANGES: Dict[str, Tuple[float, float]] = {
    "engine_rpm": (0.0, 3000.0),           # RPM: 0-3000 typical for marine engines
    "lub_oil_pressure": (0.0, 15.0),       # Bar: lubricating oil pressure
    "fuel_pressure": (0.0, 50.0),          # Bar: fuel injection pressure
    "coolant_pressure": (0.0, 10.0),       # Bar: cooling system pressure
    "oil_temp": (0.0, 150.0),              # Celsius: oil temperature
    "coolant_temp": (0.0, 120.0),          # Celsius: coolant temperature
}

# Required columns for symbolic feature computation
REQUIRED_COLUMNS: List[str] = [
    "engine_rpm",
    "coolant_temp",
    "coolant_pressure",
    "oil_temp",
]


class SensorReading(BaseModel):
    """Single sensor reading with validation."""

    engine_rpm: float = Field(..., ge=0, le=3000, description="Engine RPM (0-3000)")
    lub_oil_pressure: float = Field(..., ge=0, le=15, description="Lubricating oil pressure in bar")
    fuel_pressure: float = Field(..., ge=0, le=50, description="Fuel pressure in bar")
    coolant_pressure: float = Field(..., ge=0, le=10, description="Coolant pressure in bar")
    oil_temp: float = Field(..., ge=0, le=150, description="Oil temperature in Celsius")
    coolant_temp: float = Field(..., ge=0, le=120, description="Coolant temperature in Celsius")

    @field_validator("*", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        """Coerce string inputs to float."""
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Cannot convert '{v}' to numeric")
        return v

    def to_dataframe_row(self) -> Dict[str, float]:
        """Convert to dict suitable for DataFrame."""
        return self.model_dump()


class BatchSensorReadings(BaseModel):
    """Batch of sensor readings for bulk prediction."""

    readings: List[SensorReading] = Field(..., min_length=1, max_length=10000)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert batch to DataFrame."""
        return pd.DataFrame([r.model_dump() for r in self.readings])


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from various input formats.

    Args:
        df: Input DataFrame with potentially non-standard column names

    Returns:
        DataFrame with normalized column names

    Raises:
        KeyError: If required columns cannot be found
    """
    df = df.copy()

    # Build rename mapping
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            rename_map[col] = COLUMN_MAPPING[col]

    df = df.rename(columns=rename_map)

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after normalization: {missing}")

    return df


def validate_ranges(
    df: pd.DataFrame,
    strict: bool = False
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Validate sensor values are within physical ranges.

    Args:
        df: DataFrame with normalized column names
        strict: If True, raise on any out-of-range values. If False, clip and warn.

    Returns:
        Tuple of (processed DataFrame, list of warnings)
    """
    df = df.copy()
    warnings = []

    for col, (min_val, max_val) in SENSOR_RANGES.items():
        if col not in df.columns:
            continue

        out_of_range = (df[col] < min_val) | (df[col] > max_val)
        n_invalid = out_of_range.sum()

        if n_invalid > 0:
            warning = {
                "column": col,
                "count": int(n_invalid),
                "min_allowed": min_val,
                "max_allowed": max_val,
                "actual_min": float(df[col].min()),
                "actual_max": float(df[col].max()),
            }
            warnings.append(warning)

            if strict:
                raise ValueError(
                    f"Column '{col}' has {n_invalid} values outside range "
                    f"[{min_val}, {max_val}]"
                )
            else:
                # Clip to valid range
                df[col] = df[col].clip(min_val, max_val)

    return df, warnings


def check_missing_values(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check for missing values in required columns.

    Returns:
        Tuple of (has_missing, list of columns with missing values)
    """
    missing_cols = []
    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isna().any():
            missing_cols.append(col)

    return len(missing_cols) > 0, missing_cols


def preprocess_dataframe(
    df: pd.DataFrame,
    strict_validation: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full preprocessing pipeline for input data.

    Steps:
    1. Normalize column names
    2. Check for missing values
    3. Validate physical ranges
    4. Coerce data types

    Args:
        df: Raw input DataFrame
        strict_validation: If True, raise on any validation issues

    Returns:
        Tuple of (processed DataFrame, metadata dict with warnings)
    """
    metadata = {
        "original_shape": df.shape,
        "warnings": [],
        "columns_renamed": [],
    }

    # Step 1: Normalize columns
    original_cols = set(df.columns)
    df = normalize_columns(df)
    renamed = original_cols - set(df.columns)
    metadata["columns_renamed"] = list(renamed)

    # Step 2: Check missing values
    has_missing, missing_cols = check_missing_values(df)
    if has_missing:
        if strict_validation:
            raise ValueError(f"Missing values in columns: {missing_cols}")
        metadata["warnings"].append({
            "type": "missing_values",
            "columns": missing_cols,
        })

    # Step 3: Validate ranges
    df, range_warnings = validate_ranges(df, strict=strict_validation)
    metadata["warnings"].extend(range_warnings)

    # Step 4: Ensure numeric types
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    metadata["processed_shape"] = df.shape

    return df, metadata


def preprocess_single_reading(reading: Dict[str, float]) -> pd.DataFrame:
    """
    Preprocess a single sensor reading dict to DataFrame.

    Args:
        reading: Dict with sensor values

    Returns:
        Single-row DataFrame ready for inference
    """
    validated = SensorReading(**reading)
    return pd.DataFrame([validated.model_dump()])
