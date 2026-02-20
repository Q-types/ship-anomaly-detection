"""
Pytest fixtures and configuration for the test suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import app
from inference.load_model import ModelRegistry, reset_registry


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model_dir(project_root) -> Path:
    """Get the models directory."""
    return project_root / "models"


@pytest.fixture(scope="session")
def data_dir(project_root) -> Path:
    """Get the data directory."""
    return project_root / "data"


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def model_registry(model_dir) -> ModelRegistry:
    """Create a model registry for testing."""
    reset_registry()
    registry = ModelRegistry(model_dir)
    registry.load_models()
    return registry


@pytest.fixture
def sample_reading() -> dict:
    """Sample sensor reading for testing."""
    return {
        "engine_rpm": 750.0,
        "lub_oil_pressure": 3.5,
        "fuel_pressure": 6.0,
        "coolant_pressure": 2.5,
        "oil_temp": 78.0,
        "coolant_temp": 72.0,
    }


@pytest.fixture
def anomalous_reading() -> dict:
    """Potentially anomalous sensor reading for testing."""
    return {
        "engine_rpm": 900.0,
        "lub_oil_pressure": 0.5,  # Very low oil pressure
        "fuel_pressure": 45.0,   # Very high fuel pressure
        "coolant_pressure": 0.2,
        "oil_temp": 140.0,       # High oil temperature
        "coolant_temp": 110.0,   # High coolant temperature
    }


@pytest.fixture
def batch_readings(sample_reading) -> list:
    """Batch of sensor readings for testing."""
    import random

    readings = []
    for i in range(10):
        reading = sample_reading.copy()
        # Add some variation
        reading["engine_rpm"] = 500 + random.random() * 500
        reading["oil_temp"] = 70 + random.random() * 20
        reading["coolant_temp"] = 65 + random.random() * 20
        readings.append(reading)

    return readings


@pytest.fixture
def invalid_reading() -> dict:
    """Invalid sensor reading for validation testing."""
    return {
        "engine_rpm": 5000.0,  # Exceeds max
        "lub_oil_pressure": -1.0,  # Negative
        "fuel_pressure": 6.0,
        "coolant_pressure": 2.5,
        "oil_temp": 78.0,
        "coolant_temp": 72.0,
    }
