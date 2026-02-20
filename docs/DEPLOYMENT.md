# Deployment Guide

This guide covers deploying the Ship Engine Anomaly Detection API to various environments.

## Prerequisites

- Docker 24+ and Docker Compose v2
- Python 3.11+ (for local development)
- Trained model artifacts in `models/` directory

## Local Development

### Quick Start

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Train models (if needed)
PYTHONPATH=. python scripts/train_models.py --train_csv data/train.csv

# Run with auto-reload
uvicorn main:app --reload --port 8000
```

### Using Docker Compose (Development)

```bash
docker compose --profile dev up --build
```

This mounts the code directory for hot reload.

## Production Deployment

### Docker Production Build

```bash
# Build production image
docker build -t ship-anomaly-api:latest .

# Run container
docker run -d \
  --name anomaly-api \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e LOG_FORMAT=json \
  -e WORKERS=4 \
  ship-anomaly-api:latest
```

### Docker Compose (Production)

```bash
docker compose --profile prod up -d
```

### Configuration

Set environment variables for production:

```bash
# Required
ENVIRONMENT=production

# Recommended
LOG_LEVEL=INFO
LOG_FORMAT=json
WORKERS=4
DOCS_ENABLED=false  # Disable Swagger in production

# Security (optional)
API_KEY=your-secret-key
CORS_ORIGINS=https://your-domain.com
```

## Cloud Deployment

### AWS ECS/Fargate

1. Build and push to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker build -t ship-anomaly-api .
docker tag ship-anomaly-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ship-anomaly-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ship-anomaly-api:latest
```

2. Create ECS task definition with:
   - CPU: 512 (0.5 vCPU)
   - Memory: 1024 MB
   - Port: 8000
   - Health check: `/api/v1/health`

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/ship-anomaly-api
gcloud run deploy ship-anomaly-api \
  --image gcr.io/PROJECT_ID/ship-anomaly-api \
  --platform managed \
  --memory 1Gi \
  --port 8000 \
  --set-env-vars ENVIRONMENT=production
```

### Kubernetes

Example deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ship-anomaly-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ship-anomaly-api
  template:
    metadata:
      labels:
        app: ship-anomaly-api
    spec:
      containers:
      - name: api
        image: ship-anomaly-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: production
        - name: WORKERS
          value: "2"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ship-anomaly-api
spec:
  selector:
    app: ship-anomaly-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring Setup

### Prometheus & Grafana

With Docker Compose:

```bash
docker compose --profile prod --profile monitoring up -d
```

Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Key Metrics to Monitor

1. **Request Rate**: `rate(anomaly_api_requests_total[5m])`
2. **Error Rate**: `rate(anomaly_api_requests_total{status_code=~"5.."}[5m])`
3. **Latency P95**: `histogram_quantile(0.95, anomaly_api_request_latency_seconds_bucket)`
4. **Anomaly Rate**: `rate(anomaly_predictions_total{result="anomaly"}[5m])`

## Health Checks

The `/api/v1/health` endpoint returns:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "models_loaded": true,
  "checks": {
    "api": true,
    "models": true
  }
}
```

Status codes:
- `healthy`: All systems operational
- `degraded`: API running, models may have issues
- `unhealthy`: API not functioning

## Scaling Considerations

### Horizontal Scaling

- Stateless design allows multiple instances
- Use load balancer with health check routing
- Each instance loads models independently (~50MB)

### Vertical Scaling

- CPU-bound inference
- 512MB minimum memory
- 2+ workers per CPU core recommended

### Batch Processing

For high-throughput scenarios:
- Use `/api/v1/predict/batch` endpoint
- Maximum batch size: 10,000 samples
- Consider async processing for very large batches

## Troubleshooting

### Models Not Loading

```bash
# Check model files exist
ls -la models/

# Check permissions
chmod 644 models/*.joblib

# View startup logs
docker logs anomaly-api
```

### High Latency

1. Check batch sizes (larger = more efficient)
2. Monitor memory usage
3. Consider adding workers
4. Enable connection pooling in load balancer

### Memory Issues

```bash
# Check container memory
docker stats anomaly-api

# Increase memory limit
docker run --memory=2g ...
```

## Security Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Disable docs in production (`DOCS_ENABLED=false`)
- [ ] Configure CORS origins
- [ ] Enable API key authentication
- [ ] Use HTTPS (via load balancer/proxy)
- [ ] Review security headers
- [ ] Set up rate limiting
- [ ] Enable structured logging
- [ ] Configure log aggregation
