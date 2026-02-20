# Free Cloud Deployment Guide

This guide covers deploying the Ship Engine Anomaly Detection system to **free tier** cloud platforms.

## Quick Comparison

| Platform | Free Tier | Best For | Limits |
|----------|-----------|----------|--------|
| **Google Cloud Run** | ✅ Generous | API + Dashboard | 2M requests/month, 360k vCPU-seconds |
| **Streamlit Cloud** | ✅ Free forever | Dashboard only | Public repos, 1GB memory |
| **Railway** | ✅ $5 credit/month | Full stack | 500 hours/month |
| **Render** | ✅ Free tier | API | Spins down after 15min inactivity |
| **Fly.io** | ✅ Free tier | API | 3 shared VMs, 160GB bandwidth |

**Recommendation**: Use **Streamlit Cloud** for the dashboard (easiest) + **Google Cloud Run** for the API (most generous free tier).

---

## Option 1: Streamlit Cloud (Dashboard Only) - EASIEST

Streamlit Cloud provides **free hosting** for Streamlit apps from public GitHub repos.

### Step 1: Push to GitHub

```bash
# Initialize git if not already
cd ship_anomaly_detection
git init

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
.venv/
venv/
*.egg-info/
.pytest_cache/
.mypy_cache/
EOF

# Add and commit
git add .
git commit -m "Ship Engine Anomaly Detection System"

# Create GitHub repo and push
# Go to github.com/new and create a new public repo
git remote add origin https://github.com/YOUR_USERNAME/ship-anomaly-detection.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set:
   - **Main file path**: `dashboard/app.py`
   - **Python version**: 3.11
6. Click **"Deploy"**

**That's it!** Your dashboard will be live at `https://your-app.streamlit.app`

### Streamlit Cloud Limits (Free Tier)
- Public repos only
- 1GB memory
- Apps sleep after 7 days of inactivity (wake on visit)
- Unlimited apps

---

## Option 2: Google Cloud Run (API + Dashboard) - RECOMMENDED

Google Cloud Run offers the **most generous free tier** for containerized apps.

### Free Tier Limits
- 2 million requests/month
- 360,000 vCPU-seconds/month
- 180,000 GiB-seconds memory/month
- 1 GB outbound data/month

### Prerequisites

```bash
# Install Google Cloud SDK
# macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install

# Initialize and login
gcloud init
gcloud auth login
```

### Step 1: Create Google Cloud Project

```bash
# Create new project (or use existing)
gcloud projects create ship-anomaly-detection --name="Ship Anomaly Detection"

# Set as active project
gcloud config set project ship-anomaly-detection

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Step 2: Deploy API to Cloud Run

```bash
cd ship_anomaly_detection

# Build and deploy in one command
gcloud run deploy ship-anomaly-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --cpu 1 \
  --port 8000 \
  --allow-unauthenticated \
  --set-env-vars "ENVIRONMENT=production,LOG_FORMAT=json"
```

**Output**: You'll get a URL like `https://ship-anomaly-api-xxxxx-uc.a.run.app`

### Step 3: Deploy Dashboard to Cloud Run

Create a Dockerfile for the dashboard:

```bash
cat > dashboard/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY dashboard/requirements.txt ./requirements.txt
COPY requirements.txt ./api-requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir scikit-learn pandas numpy joblib

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF
```

Deploy:

```bash
gcloud run deploy ship-anomaly-dashboard \
  --source . \
  --dockerfile dashboard/Dockerfile \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --port 8501 \
  --allow-unauthenticated
```

### Step 4: Set Up Budget Alert (Important!)

```bash
# Create budget alert to avoid unexpected charges
gcloud billing budgets create \
  --billing-account=$(gcloud billing accounts list --format='value(name)' | head -1) \
  --display-name="Free Tier Alert" \
  --budget-amount=1USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

---

## Option 3: Railway (Full Stack)

Railway offers $5 free credit per month - enough for light usage.

### Step 1: Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub

### Step 2: Deploy from GitHub

1. Click **"New Project"** → **"Deploy from GitHub repo"**
2. Select your repository
3. Railway auto-detects the Dockerfile
4. Click **"Deploy"**

### Environment Variables

Add in Railway dashboard:
```
ENVIRONMENT=production
PORT=8000
LOG_FORMAT=json
```

### Custom Domain (Optional)

Railway provides free `*.up.railway.app` domains.

---

## Option 4: Render (API)

Render offers a free tier but services spin down after 15 minutes of inactivity.

### Step 1: Create render.yaml

```yaml
# render.yaml
services:
  - type: web
    name: ship-anomaly-api
    env: docker
    plan: free
    healthCheckPath: /api/v1/health
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: PORT
        value: 8000
```

### Step 2: Deploy

1. Go to [render.com](https://render.com)
2. Connect GitHub repository
3. Render auto-deploys from `render.yaml`

---

## Option 5: Fly.io (API)

Fly.io offers 3 shared VMs for free.

### Step 1: Install Fly CLI

```bash
# macOS
brew install flyctl

# Or
curl -L https://fly.io/install.sh | sh
```

### Step 2: Deploy

```bash
cd ship_anomaly_detection

# Login
fly auth login

# Launch (creates fly.toml)
fly launch --name ship-anomaly-api

# Deploy
fly deploy
```

### fly.toml Configuration

```toml
app = "ship-anomaly-api"
primary_region = "ord"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8000
  force_https = true
  auto_start_machines = true
  auto_stop_machines = true
  min_machines_running = 0

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 512
```

---

## Verifying Deployment

### Test API

```bash
# Replace with your actual URL
API_URL="https://your-api-url.run.app"

# Health check
curl $API_URL/api/v1/health

# Test prediction
curl -X POST "$API_URL/api/v1/predict" \
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

### Test Dashboard

Simply visit your dashboard URL in a browser.

---

## Cost Optimization Tips

1. **Use Cloud Run min-instances=0**: Scales to zero when not in use
2. **Set memory limits**: 512MB-1GB is usually sufficient
3. **Enable Cloud Run CPU throttling**: Saves costs during idle
4. **Use Streamlit Cloud for dashboard**: Completely free
5. **Set up billing alerts**: Catch issues before they cost money

---

## Monitoring Free Tier Usage

### Google Cloud

```bash
# Check Cloud Run metrics
gcloud run services describe ship-anomaly-api --region us-central1

# View logs
gcloud run logs read ship-anomaly-api --region us-central1 --limit 50
```

### View in Console

- Google Cloud Console: https://console.cloud.google.com/run
- Streamlit Cloud: https://share.streamlit.io

---

## Recommended Setup Summary

For a **portfolio demonstration**:

| Component | Platform | Cost |
|-----------|----------|------|
| API | Google Cloud Run | Free (within limits) |
| Dashboard | Streamlit Cloud | Free |
| Code | GitHub | Free |

**Total Monthly Cost: $0** (within free tier limits)

### Quick Deploy Commands

```bash
# 1. Push to GitHub
git add . && git commit -m "Deploy" && git push

# 2. Deploy API to Google Cloud Run
gcloud run deploy ship-anomaly-api --source . --region us-central1 --allow-unauthenticated

# 3. Deploy Dashboard to Streamlit Cloud
# Go to share.streamlit.io and connect your repo
```

Your portfolio project will be live at:
- **API**: `https://ship-anomaly-api-xxxxx.run.app`
- **Dashboard**: `https://your-app.streamlit.app`
- **Docs**: `https://ship-anomaly-api-xxxxx.run.app/docs`
