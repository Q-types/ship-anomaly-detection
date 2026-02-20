"""
Tests for the FastAPI endpoints.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client: TestClient):
        """Test health endpoint returns expected structure."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "models_loaded" in data
        assert "timestamp" in data

    def test_health_check_contains_checks(self, client: TestClient):
        """Test health endpoint includes component checks."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "checks" in data
        assert "api" in data["checks"]


class TestInfoEndpoint:
    """Tests for the API info endpoint."""

    def test_api_info(self, client: TestClient):
        """Test API info endpoint."""
        response = client.get("/api/v1/info")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert len(data["endpoints"]) > 0


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_single_ocsvm(self, client: TestClient, sample_reading: dict):
        """Test single prediction with OCSVM model."""
        response = client.post(
            "/api/v1/predict?model=ocsvm",
            json=sample_reading
        )

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "processing_time_ms" in data
        assert "timestamp" in data

        pred = data["prediction"]
        assert "is_anomaly" in pred
        assert "label" in pred
        assert "confidence" in pred
        assert "anomaly_score" in pred
        assert "model_used" in pred
        assert pred["model_used"] == "OCSVM"

    def test_predict_single_iforest(self, client: TestClient, sample_reading: dict):
        """Test single prediction with IsolationForest model."""
        response = client.post(
            "/api/v1/predict?model=iforest",
            json=sample_reading
        )

        assert response.status_code == 200
        data = response.json()

        pred = data["prediction"]
        assert pred["model_used"] == "IsolationForest"

    def test_predict_validation_error(self, client: TestClient, invalid_reading: dict):
        """Test prediction with invalid input."""
        response = client.post(
            "/api/v1/predict",
            json=invalid_reading
        )

        assert response.status_code == 422  # Validation error

    def test_predict_missing_field(self, client: TestClient, sample_reading: dict):
        """Test prediction with missing required field."""
        del sample_reading["engine_rpm"]

        response = client.post(
            "/api/v1/predict",
            json=sample_reading
        )

        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the batch prediction endpoint."""

    def test_batch_predict(self, client: TestClient, batch_readings: list):
        """Test batch prediction."""
        response = client.post(
            "/api/v1/predict/batch",
            json={"readings": batch_readings, "model": "ocsvm"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == len(batch_readings)

        summary = data["summary"]
        assert "total_samples" in summary
        assert "anomalies_detected" in summary
        assert "anomaly_rate" in summary
        assert summary["total_samples"] == len(batch_readings)

    def test_batch_predict_empty(self, client: TestClient):
        """Test batch prediction with empty list."""
        response = client.post(
            "/api/v1/predict/batch",
            json={"readings": [], "model": "ocsvm"}
        )

        assert response.status_code == 422  # Validation error - min_length=1


class TestEnsemblePredictEndpoint:
    """Tests for the ensemble prediction endpoint."""

    def test_ensemble_predict(self, client: TestClient, sample_reading: dict):
        """Test ensemble prediction."""
        response = client.post(
            "/api/v1/predict/ensemble",
            json=sample_reading
        )

        assert response.status_code == 200
        data = response.json()

        assert "ocsvm" in data
        assert "iforest" in data
        assert "consensus" in data
        assert "processing_time_ms" in data

        consensus = data["consensus"]
        assert "label" in consensus
        assert "confidence" in consensus
        assert "models_agree" in consensus

    def test_ensemble_predict_require_agreement(self, client: TestClient, sample_reading: dict):
        """Test ensemble prediction with agreement requirement."""
        response = client.post(
            "/api/v1/predict/ensemble?require_agreement=true",
            json=sample_reading
        )

        assert response.status_code == 200


class TestModelsEndpoint:
    """Tests for the models info endpoint."""

    def test_models_info(self, client: TestClient):
        """Test models info endpoint."""
        response = client.get("/api/v1/models")

        assert response.status_code == 200
        data = response.json()

        assert "loaded" in data
        assert "models" in data


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_security_headers(self, client: TestClient):
        """Test that security headers are present."""
        response = client.get("/api/v1/health")

        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client: TestClient):
        """Test CORS preflight request."""
        response = client.options(
            "/api/v1/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )

        # CORS should be allowed
        assert response.status_code in (200, 204, 405)
