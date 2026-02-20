# Ship Engine Anomaly Detection API

A production-ready REST API for real-time anomaly detection in ship engine sensor data using machine learning.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This API analyzes ship engine sensor data to detect anomalies that may indicate equipment malfunction, maintenance needs, or operational issues. It uses two complementary machine learning models trained on **symbolic regression features** discovered through automated equation discovery.

### Key Features

- **Real-time Prediction**: Sub-100ms inference for single samples
- **Dual Model Architecture**: OCSVM and Isolation Forest with ensemble support
- **Symbolic Feature Engineering**: Interpretable equations derived from data
- **Production Ready**: Docker, monitoring, logging, security headers
- **Comprehensive API**: Single, batch, and ensemble prediction endpoints

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FastAPI Application                         │
├─────────────────────────────────────────────────────────────────────┤
│  Endpoints                                                          │
│  ├── POST /api/v1/predict          (single prediction)             │
│  ├── POST /api/v1/predict/batch    (batch predictions)             │
│  ├── POST /api/v1/predict/ensemble (multi-model consensus)         │
│  ├── GET  /api/v1/health           (health check)                  │
│  └── GET  /api/v1/models           (model information)             │
├─────────────────────────────────────────────────────────────────────┤
│  Inference Pipeline                                                  │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    │
│  │ Preprocess  │ -> │ Symbolic Features │ -> │ Model Inference │    │
│  │ (Validate)  │    │ (4 equations)     │    │ (OCSVM/IF)      │    │
│  └─────────────┘    └──────────────────┘    └─────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│  Models                                                              │
│  ├── OCSVM (One-Class SVM)      - RBF kernel, scaled features       │
│  └── IsolationForest            - Tree-based isolation              │
└─────────────────────────────────────────────────────────────────────┘
```

## Input Features

The API accepts 6 sensor readings from ship engine monitoring systems:

| Feature | Description | Unit | Valid Range |
|---------|-------------|------|-------------|
| `engine_rpm` | Engine revolutions per minute | RPM | 0-3000 |
| `lub_oil_pressure` | Lubricating oil pressure | bar | 0-15 |
| `fuel_pressure` | Fuel injection pressure | bar | 0-50 |
| `coolant_pressure` | Cooling system pressure | bar | 0-10 |
| `oil_temp` | Oil temperature | °C | 0-150 |
| `coolant_temp` | Coolant temperature | °C | 0-120 |

## Symbolic Feature Engineering

Unlike black-box feature engineering, this system uses **symbolic regression** to discover interpretable equations that capture physical relationships:

```
oil_temp_predicted = log(coolant_temp × √engine_rpm) + 69.69828
coolant_temp_predicted = √oil_temp + 69.59798
oil_pressure_predicted = exp(exp(√(√(log(engine_rpm) × 0.827) / oil_temp)))
fuel_pressure_predicted = 40.246 / (1619.76 - engine_rpm) + 6.426
```

These equations were discovered using PySR (Symbolic Regression) from 19,535 training samples.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and navigate to the project
cd ship_anomaly_detection

# Build and run
docker compose --profile prod up --build

# API available at http://localhost:8000
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Train models (if needed)
PYTHONPATH=. python scripts/train_models.py --train_csv data/train.csv

# Run the API
uvicorn main:app --reload
```

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict?model=ocsvm" \
  -H "Content-Type: application/json" \
  -d '{
    "engine_rpm": 750,
    "lub_oil_pressure": 3.5,
    "fuel_pressure": 6.0,
    "coolant_pressure": 2.5,
    "oil_temp": 78,
    "coolant_temp": 72
  }'
```

Response:
```json
{
  "prediction": {
    "is_anomaly": false,
    "label": "normal",
    "confidence": 0.87,
    "anomaly_score": 0.234,
    "model_used": "OCSVM"
  },
  "processing_time_ms": 12.5,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "readings": [
      {"engine_rpm": 750, "lub_oil_pressure": 3.5, "fuel_pressure": 6, "coolant_pressure": 2.5, "oil_temp": 78, "coolant_temp": 72},
      {"engine_rpm": 900, "lub_oil_pressure": 1.0, "fuel_pressure": 25, "coolant_pressure": 0.5, "oil_temp": 120, "coolant_temp": 100}
    ],
    "model": "ocsvm"
  }'
```

### Ensemble Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/ensemble" \
  -H "Content-Type: application/json" \
  -d '{
    "engine_rpm": 750,
    "lub_oil_pressure": 3.5,
    "fuel_pressure": 6.0,
    "coolant_pressure": 2.5,
    "oil_temp": 78,
    "coolant_temp": 72
  }'
```

### Python Client

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")

# Single prediction
response = client.post("/api/v1/predict", params={"model": "ocsvm"}, json={
    "engine_rpm": 750,
    "lub_oil_pressure": 3.5,
    "fuel_pressure": 6.0,
    "coolant_pressure": 2.5,
    "oil_temp": 78,
    "coolant_temp": 72
})
result = response.json()
print(f"Anomaly: {result['prediction']['is_anomaly']}")
print(f"Confidence: {result['prediction']['confidence']}")
```

## Configuration

Configuration via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | development/staging/production | development |
| `PORT` | API port | 8000 |
| `LOG_LEVEL` | DEBUG/INFO/WARNING/ERROR | INFO |
| `LOG_FORMAT` | json/text | json |
| `API_KEY` | Optional API key for authentication | None |
| `CORS_ORIGINS` | Allowed CORS origins | * |
| `DEFAULT_MODEL` | ocsvm/iforest/ensemble | ocsvm |
| `MAX_BATCH_SIZE` | Maximum batch size | 10000 |

## Model Performance

Training results on 19,535 samples:

| Model | Anomaly Rate | Parameters |
|-------|--------------|------------|
| OCSVM | 2.00% | nu=0.02, gamma=0.2, kernel=RBF |
| Isolation Forest | 2.00% | contamination=0.02, n_estimators=300 |

## Project Structure

```
ship_anomaly_detection/
├── main.py                 # FastAPI application entry point
├── api/                    # API layer
│   ├── routes.py          # Endpoint definitions
│   ├── schemas.py         # Pydantic models
│   ├── dependencies.py    # Dependency injection
│   └── middleware.py      # Security, logging middleware
├── config/                 # Configuration
│   ├── settings.py        # Pydantic settings
│   └── model_config.yaml  # Model configuration
├── features/               # Feature engineering
│   └── symbolic.py        # Symbolic feature computation
├── inference/              # Inference pipeline
│   ├── preprocess.py      # Input validation & normalization
│   ├── load_model.py      # Model loading & registry
│   └── detect.py          # Anomaly detection logic
├── models/                 # Trained model artifacts
│   ├── ocsvm_symbolic.joblib
│   └── if_symbolic.joblib
├── monitoring/             # Observability
│   ├── metrics.py         # Prometheus metrics
│   └── logging_config.py  # Structured logging
├── scripts/                # Training & utilities
│   └── train_models.py    # Model training script
├── tests/                  # Test suite
├── Dockerfile             # Production container
├── docker-compose.yml     # Container orchestration
└── requirements.txt       # Python dependencies
```

## Testing

```bash
# Run all tests
PYTHONPATH=. pytest

# Run with coverage
PYTHONPATH=. pytest --cov=. --cov-report=html

# Run specific test file
PYTHONPATH=. pytest tests/test_api.py -v
```

## Monitoring

### Prometheus Metrics

Available at `/api/v1/metrics`:

- `anomaly_api_requests_total` - Request count by endpoint/status
- `anomaly_api_request_latency_seconds` - Request latency histogram
- `anomaly_predictions_total` - Predictions by model/result
- `anomaly_prediction_latency_seconds` - Prediction latency
- `anomaly_prediction_confidence` - Confidence distribution
- `anomaly_model_loaded` - Model status gauge

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

## Security

- Input validation with Pydantic
- Physical range checks for sensor values
- Security headers (X-Content-Type-Options, X-Frame-Options, etc.)
- Optional API key authentication
- CORS configuration
- Non-root Docker user
- No secrets in images

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read the contributing guidelines first.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML models from [scikit-learn](https://scikit-learn.org/)
- Symbolic regression via [PySR](https://github.com/MilesCranmer/PySR)
