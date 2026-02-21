"""
Ship Engine Anomaly Detection - Interactive Dashboard

A professional Streamlit dashboard for demonstrating the anomaly detection system.
Includes real-time predictions, batch analysis, model comparison, and visualizations.

Run with: streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

from features.symbolic import compute_symbolic_features, SYMBOLIC_SPEC
from joblib import load


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Ship Engine Anomaly Detection",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .status-normal {
        color: #28a745;
        font-weight: bold;
    }
    .status-anomaly {
        color: #dc3545;
        font-weight: bold;
    }
    .status-uncertain {
        color: #ffc107;
        font-weight: bold;
    }
    .component-highlight {
        border: 2px solid #1f77b4;
        border-radius: 8px;
        padding: 5px;
        background-color: rgba(31, 119, 180, 0.1);
    }
    .preset-btn {
        margin: 2px;
    }
    .regime-indicator {
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 8px;
    }
    .regime-normal { background-color: #28a745; color: white; }
    .regime-warning { background-color: #ffc107; color: black; }
    .regime-danger { background-color: #dc3545; color: white; }
    .tip-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Preset Scenarios & Operating Regimes
# ============================================================================

PRESET_SCENARIOS = {
    # === NORMAL STATES ===
    "🟢 Normal Cruising": {
        "engine_rpm": 1200,
        "lub_oil_pressure": 4.5,
        "fuel_pressure": 6.5,
        "coolant_pressure": 2.8,
        "oil_temp": 82,
        "coolant_temp": 78,
        "description": "Typical steady-state operation at moderate load",
        "is_anomaly": False
    },
    "🔵 Cold Start": {
        "engine_rpm": 650,
        "lub_oil_pressure": 3.8,
        "fuel_pressure": 5.5,
        "coolant_pressure": 2.0,
        "oil_temp": 45,
        "coolant_temp": 35,
        "description": "Engine warming up - low but acceptable temps",
        "is_anomaly": False
    },
    # === ANOMALY STATES ===
    "🔴 Overheating Crisis": {
        "engine_rpm": 1800,
        "lub_oil_pressure": 2.0,
        "fuel_pressure": 7.0,
        "coolant_pressure": 0.8,
        "oil_temp": 135,
        "coolant_temp": 115,
        "description": "⚠️ CRITICAL: Cooling system failure - oil degrading, bearings at risk",
        "cause": "Coolant pump failure or blockage",
        "fix": "Reduce RPM, check coolant level, inspect pump",
        "is_anomaly": True
    },
    "⚫ Oil System Failure": {
        "engine_rpm": 1400,
        "lub_oil_pressure": 0.8,
        "fuel_pressure": 6.5,
        "coolant_pressure": 2.5,
        "oil_temp": 125,
        "coolant_temp": 85,
        "description": "⚠️ CRITICAL: Oil pressure collapsed - imminent bearing damage",
        "cause": "Oil pump failure, severe leak, or clogged filter",
        "fix": "STOP ENGINE IMMEDIATELY, check oil level and pump",
        "is_anomaly": True
    },
    "⛽ Fuel System Leak": {
        "engine_rpm": 1600,
        "lub_oil_pressure": 4.0,
        "fuel_pressure": 2.5,
        "coolant_pressure": 2.8,
        "oil_temp": 88,
        "coolant_temp": 82,
        "description": "⚠️ DANGER: Low fuel pressure - incomplete combustion, power loss",
        "cause": "Fuel line leak, failing fuel pump, or clogged injectors",
        "fix": "Inspect fuel lines, check pump pressure, replace filters",
        "is_anomaly": True
    },
    "🌊 Extreme Cold Weather": {
        "engine_rpm": 800,
        "lub_oil_pressure": 6.5,
        "fuel_pressure": 4.0,
        "coolant_pressure": 3.5,
        "oil_temp": 15,
        "coolant_temp": 8,
        "description": "⚠️ Arctic conditions - oil too viscous, fuel gelling risk",
        "cause": "Operating in sub-zero temperatures without proper warm-up",
        "fix": "Extended idle warm-up, use winter-grade fluids",
        "is_anomaly": True
    },
    "🔥 Bearing Failure Imminent": {
        "engine_rpm": 2200,
        "lub_oil_pressure": 1.5,
        "fuel_pressure": 8.5,
        "coolant_pressure": 3.0,
        "oil_temp": 140,
        "coolant_temp": 95,
        "description": "⚠️ CRITICAL: High load + low oil + extreme heat = catastrophic failure",
        "cause": "Oil starvation under high load - metal-on-metal contact",
        "fix": "REDUCE LOAD IMMEDIATELY, check oil level and quality",
        "is_anomaly": True
    },
    "💨 Coolant Pressure Surge": {
        "engine_rpm": 1500,
        "lub_oil_pressure": 4.2,
        "fuel_pressure": 6.8,
        "coolant_pressure": 8.5,
        "oil_temp": 92,
        "coolant_temp": 98,
        "description": "⚠️ WARNING: Excessive coolant pressure - head gasket stress",
        "cause": "Thermostat stuck closed or radiator blockage",
        "fix": "Check thermostat, flush cooling system, inspect head gasket",
        "is_anomaly": True
    },
    "📉 Sensor Malfunction": {
        "engine_rpm": 1100,
        "lub_oil_pressure": 12.0,
        "fuel_pressure": 45.0,
        "coolant_pressure": 0.3,
        "oil_temp": 5,
        "coolant_temp": 150,
        "description": "⚠️ SENSORS: Impossible readings indicate sensor failure",
        "cause": "Electrical fault, damaged sensors, or wiring issues",
        "fix": "Diagnose sensor circuits, replace faulty sensors",
        "is_anomaly": True
    },
    "🎲 Random Scenario": {
        "engine_rpm": None,
        "description": "Generate a random engine state for exploration",
        "is_anomaly": None
    },
}

# Operating regime thresholds for each parameter
OPERATING_REGIMES = {
    "engine_rpm": {
        "normal": (500, 2000),
        "warning": (2000, 2500),
        "danger_low": (0, 300),
        "danger_high": (2500, 3000),
        "unit": "RPM",
        "tip": "RPM affects oil pressure and temperatures. Higher RPM = more heat generation."
    },
    "lub_oil_pressure": {
        "normal": (3.0, 6.0),
        "warning": (2.0, 3.0),
        "danger_low": (0, 2.0),
        "danger_high": (8.0, 15.0),
        "unit": "bar",
        "tip": "Oil pressure lubricates bearings. Too low = metal-on-metal contact damage."
    },
    "fuel_pressure": {
        "normal": (5.0, 8.0),
        "warning": (4.0, 5.0),
        "danger_low": (0, 4.0),
        "danger_high": (10.0, 50.0),
        "unit": "bar",
        "tip": "Fuel pressure ensures proper injection. Affects combustion efficiency."
    },
    "coolant_pressure": {
        "normal": (2.0, 3.5),
        "warning": (1.5, 2.0),
        "danger_low": (0, 1.5),
        "danger_high": (5.0, 10.0),
        "unit": "bar",
        "tip": "Coolant circulates heat away from engine. High pressure may indicate blockage."
    },
    "oil_temp": {
        "normal": (70, 95),
        "warning": (95, 110),
        "danger_low": (0, 40),
        "danger_high": (110, 150),
        "unit": "°C",
        "tip": "Oil temperature affects viscosity. Too hot = breakdown, too cold = poor flow."
    },
    "coolant_temp": {
        "normal": (70, 90),
        "warning": (90, 100),
        "danger_low": (0, 50),
        "danger_high": (100, 120),
        "unit": "°C",
        "tip": "Coolant temp indicates heat removal efficiency. Watch for rising trends."
    },
}

def get_regime_status(param: str, value: float) -> tuple[str, str]:
    """Return regime status (normal/warning/danger) and color for a parameter value."""
    regime = OPERATING_REGIMES[param]

    if regime["normal"][0] <= value <= regime["normal"][1]:
        return "normal", "#28a745"
    elif regime.get("warning") and regime["warning"][0] <= value <= regime["warning"][1]:
        return "warning", "#ffc107"
    elif regime.get("danger_low") and regime["danger_low"][0] <= value <= regime["danger_low"][1]:
        return "danger", "#dc3545"
    elif regime.get("danger_high") and regime["danger_high"][0] <= value <= regime["danger_high"][1]:
        return "danger", "#dc3545"
    else:
        return "warning", "#ffc107"


def create_engine_diagram(values: dict, selected_component: str = None) -> go.Figure:
    """Create an interactive engine schematic showing component relationships."""
    fig = go.Figure()

    # === ENGINE BLOCK (central) ===
    fig.add_shape(type="rect", x0=2, y0=2, x1=5, y1=5,
                  fillcolor="#34495e", line=dict(color="#2c3e50", width=3),
                  layer="below")
    fig.add_annotation(x=3.5, y=3.5, text="⚙️<br>ENGINE<br>BLOCK",
                       showarrow=False, font=dict(size=11, color="white"))

    # === OIL SYSTEM (left side) ===
    fig.add_shape(type="rect", x0=0.3, y0=2.5, x1=1.5, y1=4.5,
                  fillcolor="#8B4513", line=dict(color="#654321", width=2),
                  layer="below")
    fig.add_annotation(x=0.9, y=4.2, text="OIL", showarrow=False,
                       font=dict(size=9, color="white"))

    # === COOLING SYSTEM (right side) ===
    fig.add_shape(type="rect", x0=5.5, y0=2.5, x1=6.7, y1=4.5,
                  fillcolor="#3498db", line=dict(color="#2980b9", width=2),
                  layer="below")
    fig.add_annotation(x=6.1, y=4.2, text="COOLANT", showarrow=False,
                       font=dict(size=9, color="white"))

    # === FUEL SYSTEM (bottom) ===
    fig.add_shape(type="rect", x0=2.5, y0=0.3, x1=4.5, y1=1.5,
                  fillcolor="#e67e22", line=dict(color="#d35400", width=2),
                  layer="below")
    fig.add_annotation(x=3.5, y=0.6, text="FUEL", showarrow=False,
                       font=dict(size=9, color="white"))

    # === RPM INDICATOR (top) ===
    fig.add_shape(type="circle", x0=2.8, y0=5.5, x1=4.2, y1=6.7,
                  fillcolor="#2c3e50", line=dict(color="#1a252f", width=2),
                  layer="below")

    # Component positions with better layout
    components = {
        "engine_rpm": {"x": 3.5, "y": 6.1, "label": "RPM", "icon": "🔄", "size": 45},
        "lub_oil_pressure": {"x": 0.9, "y": 3.5, "label": "Oil P", "icon": "🛢️", "size": 38},
        "oil_temp": {"x": 0.9, "y": 2.8, "label": "Oil T", "icon": "🌡️", "size": 32},
        "fuel_pressure": {"x": 3.5, "y": 1.1, "label": "Fuel P", "icon": "⛽", "size": 38},
        "coolant_pressure": {"x": 6.1, "y": 3.5, "label": "Cool P", "icon": "💧", "size": 38},
        "coolant_temp": {"x": 6.1, "y": 2.8, "label": "Cool T", "icon": "🌡️", "size": 32},
    }

    # Draw flow arrows showing relationships
    arrows = [
        # Oil flows through engine
        {"x": 1.5, "y": 3.5, "ax": 2, "ay": 3.5, "color": "#8B4513"},
        # Coolant flows through engine
        {"x": 5.5, "y": 3.5, "ax": 5, "ay": 3.5, "color": "#3498db"},
        # Fuel enters engine
        {"x": 3.5, "y": 1.5, "ax": 3.5, "ay": 2, "color": "#e67e22"},
        # RPM drives everything
        {"x": 3.5, "y": 5.5, "ax": 3.5, "ay": 5, "color": "#95a5a6"},
    ]

    for arrow in arrows:
        fig.add_annotation(
            x=arrow["ax"], y=arrow["ay"],
            ax=arrow["x"], ay=arrow["y"],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor=arrow["color"], opacity=0.6
        )

    # Draw component indicators
    for param, comp in components.items():
        value = values.get(param, 0)
        status, color = get_regime_status(param, value)

        # Highlight selected component
        is_selected = param == selected_component
        marker_line = dict(color="#FFD700", width=4) if is_selected else dict(color="white", width=2)
        marker_size = comp["size"] + 8 if is_selected else comp["size"]

        # Format value display
        if isinstance(value, float):
            val_str = f"{value:.1f}"
        else:
            val_str = str(value)

        fig.add_trace(go.Scatter(
            x=[comp["x"]], y=[comp["y"]],
            mode="markers+text",
            marker=dict(
                size=marker_size,
                color=color,
                line=marker_line,
                symbol="circle"
            ),
            text=[f"{comp['icon']}<br>{val_str}"],
            textposition="middle center",
            textfont=dict(size=10, color="white", family="Arial Black"),
            name=param,
            hovertemplate=(
                f"<b>{comp['label']}</b><br>"
                f"Value: {val_str}<br>"
                f"Status: {status.upper()}<br>"
                f"<extra></extra>"
            ),
        ))

    # Legend
    fig.add_annotation(x=3.5, y=7.2, text="🟢 Normal  🟡 Warning  🔴 Danger",
                       showarrow=False, font=dict(size=11))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.2, 7.2]),
        yaxis=dict(visible=False, range=[-0.2, 7.5]),
        height=380,
        margin=dict(l=5, r=5, t=10, b=5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ============================================================================
# Model Loading
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models with caching."""
    model_dir = PROJECT_ROOT / "models"

    ocsvm_artifact = load(model_dir / "ocsvm_symbolic.joblib")
    if_artifact = load(model_dir / "if_symbolic.joblib")

    return {
        "ocsvm": ocsvm_artifact,
        "iforest": if_artifact,
    }


@st.cache_data
def load_training_data():
    """Load training data for visualization."""
    data_path = PROJECT_ROOT / "data" / "train.csv"
    df = pd.read_csv(data_path)

    # Normalize column names
    column_mapping = {
        "Engine rpm": "engine_rpm",
        "Lub oil pressure": "lub_oil_pressure",
        "Fuel pressure": "fuel_pressure",
        "Coolant pressure": "coolant_pressure",
        "lub oil temp": "oil_temp",
        "Coolant temp": "coolant_temp",
    }
    df = df.rename(columns=column_mapping)
    return df


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_single(reading: dict, models: dict, model_name: str = "ocsvm"):
    """Make a prediction for a single reading."""
    df = pd.DataFrame([reading])
    X_sym, _ = compute_symbolic_features(df)

    artifact = models[model_name]

    if model_name == "ocsvm":
        X_scaled = artifact["scaler"].transform(X_sym)
        pred = artifact["model"].predict(X_scaled)[0]
        score = artifact["model"].decision_function(X_scaled)[0]
    else:
        pred = artifact["model"].predict(X_sym)[0]
        score = artifact["model"].score_samples(X_sym)[0]

    is_anomaly = pred == -1

    # Calculate anomaly confidence (high = definitely anomaly, low = definitely normal)
    # Negative scores = anomaly, positive scores = normal
    if model_name == "ocsvm":
        # Sigmoid flipped: very negative score → ~100%, positive score → ~0%
        anomaly_confidence = 1.0 / (1.0 + np.exp(2 * score))
    else:
        anomaly_confidence = 1.0 / (1.0 + np.exp(5 * score))

    return {
        "is_anomaly": is_anomaly,
        "confidence": float(anomaly_confidence),
        "score": float(score),
        "prediction": pred,
    }


def predict_batch(df: pd.DataFrame, models: dict, model_name: str = "ocsvm"):
    """Make predictions for a batch of readings."""
    X_sym, _ = compute_symbolic_features(df)
    artifact = models[model_name]

    if model_name == "ocsvm":
        X_scaled = artifact["scaler"].transform(X_sym)
        preds = artifact["model"].predict(X_scaled)
        scores = artifact["model"].decision_function(X_scaled)
    else:
        preds = artifact["model"].predict(X_sym)
        scores = artifact["model"].score_samples(X_sym)

    return preds, scores


# ============================================================================
# Visualization Functions
# ============================================================================

def create_gauge_chart(value: float, title: str, color: str = "blue"):
    """Create a gauge chart for confidence/score display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#ffebee"},
                {'range': [40, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#e8f5e9"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_feature_distribution(df: pd.DataFrame, feature: str):
    """Create distribution plot for a feature."""
    fig = px.histogram(
        df, x=feature,
        nbins=50,
        title=f"Distribution of {feature}",
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_scatter_matrix(df: pd.DataFrame, preds: np.ndarray):
    """Create scatter matrix with anomaly highlighting."""
    df_plot = df.copy()
    df_plot["Status"] = ["Anomaly" if p == -1 else "Normal" for p in preds]

    fig = px.scatter_matrix(
        df_plot,
        dimensions=["engine_rpm", "oil_temp", "coolant_temp", "lub_oil_pressure"],
        color="Status",
        color_discrete_map={"Normal": "#28a745", "Anomaly": "#dc3545"},
        title="Feature Relationships with Anomaly Highlighting"
    )
    fig.update_layout(height=600)
    return fig


def create_anomaly_timeline(scores: np.ndarray, threshold: float = 0):
    """Create timeline visualization of anomaly scores."""
    fig = go.Figure()

    colors = ['#dc3545' if s < threshold else '#28a745' for s in scores]

    fig.add_trace(go.Scatter(
        y=scores,
        mode='markers+lines',
        marker=dict(color=colors, size=6),
        line=dict(color='#1f77b4', width=1),
        name='Anomaly Score'
    ))

    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text="Threshold")

    fig.update_layout(
        title="Anomaly Scores Over Time",
        xaxis_title="Sample Index",
        yaxis_title="Anomaly Score",
        height=400,
    )
    return fig


@st.cache_data
def compute_pca(df: pd.DataFrame, features: list):
    """Compute PCA projection."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca.explained_variance_ratio_


@st.cache_data
def compute_umap(df: pd.DataFrame, features: list, n_neighbors: int = 15, min_dist: float = 0.1):
    """Compute UMAP projection."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    return X_umap


def create_pca_plot(X_pca: np.ndarray, labels: np.ndarray, variance_ratio: np.ndarray):
    """Create PCA scatter plot with anomaly coloring."""
    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Status': ['Anomaly' if l == -1 else 'Normal' for l in labels]
    })

    fig = px.scatter(
        df_plot, x='PC1', y='PC2', color='Status',
        color_discrete_map={'Normal': '#28a745', 'Anomaly': '#dc3545'},
        title=f"PCA Projection (Explained Variance: PC1={variance_ratio[0]:.1%}, PC2={variance_ratio[1]:.1%})",
        opacity=0.7
    )

    fig.update_layout(
        height=500,
        xaxis_title=f"PC1 ({variance_ratio[0]:.1%} variance)",
        yaxis_title=f"PC2 ({variance_ratio[1]:.1%} variance)",
    )

    return fig


def create_umap_plot(X_umap: np.ndarray, labels: np.ndarray):
    """Create UMAP scatter plot with anomaly coloring."""
    df_plot = pd.DataFrame({
        'UMAP1': X_umap[:, 0],
        'UMAP2': X_umap[:, 1],
        'Status': ['Anomaly' if l == -1 else 'Normal' for l in labels]
    })

    fig = px.scatter(
        df_plot, x='UMAP1', y='UMAP2', color='Status',
        color_discrete_map={'Normal': '#28a745', 'Anomaly': '#dc3545'},
        title="UMAP Projection (Non-linear Dimensionality Reduction)",
        opacity=0.7
    )

    fig.update_layout(height=500)

    return fig


def create_3d_pca_plot(df: pd.DataFrame, features: list, labels: np.ndarray):
    """Create 3D PCA scatter plot."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Status': ['Anomaly' if l == -1 else 'Normal' for l in labels]
    })

    fig = px.scatter_3d(
        df_plot, x='PC1', y='PC2', z='PC3', color='Status',
        color_discrete_map={'Normal': '#28a745', 'Anomaly': '#dc3545'},
        title=f"3D PCA Projection (Total Variance: {sum(pca.explained_variance_ratio_):.1%})",
        opacity=0.7
    )

    fig.update_layout(height=600)

    return fig


# ============================================================================
# Main Application
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🚢 Ship Engine Anomaly Detection</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666;">
    Real-time anomaly detection using ML models with symbolic regression features
    </p>
    """, unsafe_allow_html=True)

    # Load models
    try:
        models = load_models()
        training_data = load_training_data()
        st.sidebar.success("✅ Models loaded successfully")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["ocsvm", "iforest"],
        format_func=lambda x: "One-Class SVM" if x == "ocsvm" else "Isolation Forest"
    )

    # Display model info
    artifact = models[model_choice]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Info**")
    st.sidebar.text(f"Version: {artifact['version']}")
    st.sidebar.text(f"Train Anomaly Rate: {artifact['train_anom_rate']:.2%}")

    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🔍 Real-time Detection", "📊 Batch Analysis", "🔬 Model Comparison", "📈 Data Explorer"]
    )

    # ========================================================================
    # Page: Real-time Detection (Enhanced with 3 UX Features)
    # ========================================================================
    if page == "🔍 Real-time Detection":
        st.header("Real-time Anomaly Detection")
        st.markdown("Explore engine states and detect anomalies in real-time.")

        # Initialize session state for values if not exists
        if "sensor_values" not in st.session_state:
            st.session_state.sensor_values = {
                "engine_rpm": 1200,
                "lub_oil_pressure": 4.5,
                "fuel_pressure": 6.5,
                "coolant_pressure": 2.8,
                "oil_temp": 82,
                "coolant_temp": 78,
            }
        if "selected_component" not in st.session_state:
            st.session_state.selected_component = None
        if "guided_mode" not in st.session_state:
            st.session_state.guided_mode = True

        # Guided Exploration Toggle
        st.session_state.guided_mode = st.toggle(
            "🎓 Guided Exploration Mode",
            value=st.session_state.guided_mode,
            help="Shows operating regime indicators and educational tips"
        )

        if st.session_state.guided_mode:
            st.markdown("""
            <div class="tip-box">
            💡 <b>Guided Mode Active</b>: Slider colors indicate operating regimes.
            <span style="color:#28a745">●</span> Normal
            <span style="color:#ffc107">●</span> Warning
            <span style="color:#dc3545">●</span> Danger
            — Click components in the diagram to highlight corresponding sliders.
            </div>
            """, unsafe_allow_html=True)

        # Main layout: 3 columns
        col_diagram, col_sliders, col_result = st.columns([1.2, 1.3, 1])

        # === Column 1: Engine Diagram ===
        with col_diagram:
            st.subheader("🔧 Engine Diagram")

            # Create and display engine diagram
            engine_fig = create_engine_diagram(
                st.session_state.sensor_values,
                st.session_state.selected_component
            )
            st.plotly_chart(engine_fig, use_container_width=True, key="engine_diagram")

            # Component selection buttons
            st.markdown("**Click to highlight:**")
            comp_cols = st.columns(3)
            component_labels = {
                "engine_rpm": "🔄 RPM",
                "lub_oil_pressure": "🛢️ Oil P",
                "fuel_pressure": "⛽ Fuel P",
                "coolant_pressure": "💧 Cool P",
                "oil_temp": "🌡️ Oil T",
                "coolant_temp": "🌡️ Cool T",
            }
            for i, (comp, label) in enumerate(component_labels.items()):
                with comp_cols[i % 3]:
                    if st.button(label, key=f"btn_{comp}", use_container_width=True):
                        st.session_state.selected_component = comp
                        st.rerun()

        # === Column 2: Sensor Sliders ===
        with col_sliders:
            st.subheader("🎛️ Sensor Inputs")

            def create_slider_with_regime(param: str, label: str, min_val, max_val, step, format_str=None):
                """Create a slider with regime indicator."""
                current_val = st.session_state.sensor_values[param]
                regime = OPERATING_REGIMES[param]
                status, color = get_regime_status(param, current_val)

                # Highlight if selected
                is_selected = st.session_state.selected_component == param
                if is_selected and st.session_state.guided_mode:
                    st.markdown(f'<div class="component-highlight">', unsafe_allow_html=True)

                # Status indicator
                if st.session_state.guided_mode:
                    status_icon = {"normal": "🟢", "warning": "🟡", "danger": "🔴"}.get(status, "⚪")
                    st.markdown(f"**{label}** {status_icon}")
                else:
                    st.markdown(f"**{label}**")

                # Slider
                new_val = st.slider(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=current_val if isinstance(current_val, int) else float(current_val),
                    step=step,
                    format=format_str,
                    key=f"slider_{param}",
                    label_visibility="collapsed"
                )
                st.session_state.sensor_values[param] = new_val

                # Educational tip in guided mode
                if st.session_state.guided_mode and is_selected:
                    st.caption(f"💡 {regime['tip']}")

                if is_selected and st.session_state.guided_mode:
                    st.markdown('</div>', unsafe_allow_html=True)

                return new_val

            engine_rpm = create_slider_with_regime("engine_rpm", "Engine RPM", 0, 3000, 10)
            lub_oil_pressure = create_slider_with_regime("lub_oil_pressure", "Lub Oil Pressure (bar)", 0.0, 15.0, 0.1)
            fuel_pressure = create_slider_with_regime("fuel_pressure", "Fuel Pressure (bar)", 0.0, 50.0, 0.5)
            coolant_pressure = create_slider_with_regime("coolant_pressure", "Coolant Pressure (bar)", 0.0, 10.0, 0.1)
            oil_temp = create_slider_with_regime("oil_temp", "Oil Temperature (°C)", 0, 150, 1)
            coolant_temp = create_slider_with_regime("coolant_temp", "Coolant Temperature (°C)", 0, 120, 1)

            # === Preset Conditions Section ===
            st.markdown("---")
            st.subheader("⚡ Preset Conditions")
            st.caption("Click to load a predefined engine state:")

            def apply_preset(preset_vals: dict, is_random: bool = False):
                """Apply preset values to both sensor_values and slider keys."""
                if is_random:
                    new_values = {
                        "engine_rpm": int(np.random.uniform(300, 2800)),
                        "lub_oil_pressure": round(np.random.uniform(0.5, 12.0), 1),
                        "fuel_pressure": round(np.random.uniform(2.0, 15.0), 1),
                        "coolant_pressure": round(np.random.uniform(0.3, 8.0), 1),
                        "oil_temp": int(np.random.uniform(5, 145)),
                        "coolant_temp": int(np.random.uniform(5, 118)),
                    }
                else:
                    new_values = {k: v for k, v in preset_vals.items()
                                  if k not in ["description", "is_anomaly", "cause", "fix"]}

                # Update session state values
                st.session_state.sensor_values = new_values

                # CRITICAL: Also update the slider widget keys directly
                for param, val in new_values.items():
                    st.session_state[f"slider_{param}"] = val

            preset_cols = st.columns(2)
            for i, (preset_name, preset_vals) in enumerate(PRESET_SCENARIOS.items()):
                with preset_cols[i % 2]:
                    # Color code the button based on anomaly status
                    is_anomaly = preset_vals.get("is_anomaly", None)
                    if st.button(preset_name, key=f"preset_{i}", use_container_width=True):
                        apply_preset(preset_vals, is_random=(preset_name == "🎲 Random Scenario"))
                        st.rerun()

            # Show preset description if applicable
            if st.session_state.guided_mode:
                st.caption("ℹ️ Presets simulate common operating conditions for exploration.")

        # === Column 3: Prediction Result ===
        with col_result:
            st.subheader("📊 Prediction Result")

            reading = {
                "engine_rpm": float(st.session_state.sensor_values["engine_rpm"]),
                "lub_oil_pressure": float(st.session_state.sensor_values["lub_oil_pressure"]),
                "fuel_pressure": float(st.session_state.sensor_values["fuel_pressure"]),
                "coolant_pressure": float(st.session_state.sensor_values["coolant_pressure"]),
                "oil_temp": float(st.session_state.sensor_values["oil_temp"]),
                "coolant_temp": float(st.session_state.sensor_values["coolant_temp"]),
            }

            result = predict_single(reading, models, model_choice)

            # Status display
            if result["is_anomaly"]:
                st.error("🚨 **ANOMALY DETECTED**")
            else:
                st.success("✅ **Normal Operation**")

            # Metrics
            st.metric("Anomaly Confidence", f"{result['confidence']:.1%}")
            st.metric("Anomaly Score", f"{result['score']:.4f}")

            # Gauge chart
            color = "#dc3545" if result["confidence"] > 0.5 else "#28a745"
            gauge = create_gauge_chart(result["confidence"], "Anomaly Confidence", color)
            st.plotly_chart(gauge, use_container_width=True)

            # Educational explanation in guided mode
            if st.session_state.guided_mode:
                st.markdown("---")

                # Diagnostic analysis - show which parameters are problematic
                st.markdown("**🔍 Diagnostic Analysis:**")
                problem_params = []
                for param, val in reading.items():
                    status, _ = get_regime_status(param, val)
                    if status == "danger":
                        problem_params.append((param, "🔴 DANGER", val))
                    elif status == "warning":
                        problem_params.append((param, "🟡 WARNING", val))

                if problem_params:
                    for param, status_str, val in problem_params:
                        regime = OPERATING_REGIMES[param]
                        normal_range = regime["normal"]
                        st.markdown(f"- **{param}**: {val} {status_str} (normal: {normal_range[0]}-{normal_range[1]} {regime['unit']})")
                else:
                    st.markdown("✅ All parameters within normal ranges")

                st.markdown("---")
                st.markdown("**📖 Understanding the Result:**")
                if result["confidence"] > 0.8:
                    st.error("🚨 **HIGH CONFIDENCE ANOMALY**")
                    st.markdown("""
                    This combination of sensor values is **significantly unusual**
                    compared to normal operation patterns. The ML model has detected
                    a pattern that deviates strongly from the training data.
                    """)

                    # Show potential causes and fixes
                    st.markdown("**🔧 Potential Causes & Actions:**")
                    if reading["lub_oil_pressure"] < 2.0:
                        st.warning("• **Low Oil Pressure**: Check oil level, inspect pump, look for leaks")
                    if reading["oil_temp"] > 110:
                        st.warning("• **High Oil Temp**: Reduce load, check cooling, verify oil quality")
                    if reading["coolant_temp"] > 100:
                        st.warning("• **Overheating**: Check coolant level, inspect thermostat, clean radiator")
                    if reading["fuel_pressure"] < 4.0:
                        st.warning("• **Low Fuel Pressure**: Inspect fuel lines, check pump, replace filters")
                    if reading["coolant_pressure"] > 5.0:
                        st.warning("• **High Coolant Pressure**: Check for blockages, inspect head gasket")

                elif result["confidence"] > 0.5:
                    st.warning("⚠️ **MODERATE ANOMALY CONFIDENCE**")
                    st.markdown("Some parameters may be outside typical ranges. Monitor closely.")
                elif result["confidence"] > 0.2:
                    st.info("ℹ️ Low anomaly confidence - readings mostly normal, minor deviations detected.")
                else:
                    st.success("✅ Very low anomaly confidence - this looks like normal engine operation.")

    # ========================================================================
    # Page: Batch Analysis
    # ========================================================================
    elif page == "📊 Batch Analysis":
        st.header("Batch Analysis")
        st.markdown("Upload a CSV file or analyze the training dataset.")

        tab1, tab2 = st.tabs(["📁 Upload CSV", "📊 Training Data"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload sensor data CSV",
                type=["csv"],
                help="CSV must contain: engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, oil_temp, coolant_temp"
            )

            if uploaded_file is not None:
                df_upload = pd.read_csv(uploaded_file)

                # Try to normalize columns
                column_mapping = {
                    "Engine rpm": "engine_rpm",
                    "Lub oil pressure": "lub_oil_pressure",
                    "Fuel pressure": "fuel_pressure",
                    "Coolant pressure": "coolant_pressure",
                    "lub oil temp": "oil_temp",
                    "Coolant temp": "coolant_temp",
                }
                df_upload = df_upload.rename(columns=column_mapping)

                st.success(f"Loaded {len(df_upload)} samples")

                with st.spinner("Running predictions..."):
                    preds, scores = predict_batch(df_upload, models, model_choice)

                # Summary
                n_anomalies = np.sum(preds == -1)
                anomaly_rate = n_anomalies / len(preds)

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Samples", len(preds))
                col2.metric("Anomalies Detected", n_anomalies)
                col3.metric("Anomaly Rate", f"{anomaly_rate:.2%}")

                # Visualizations
                st.plotly_chart(create_anomaly_timeline(scores), use_container_width=True)
                st.plotly_chart(create_scatter_matrix(df_upload, preds), use_container_width=True)

        with tab2:
            st.subheader("Training Data Analysis")

            n_samples = st.slider("Number of samples to analyze", 100, len(training_data), 1000)
            df_sample = training_data.sample(n_samples, random_state=42)

            with st.spinner("Running predictions..."):
                preds, scores = predict_batch(df_sample, models, model_choice)

            # Summary
            n_anomalies = np.sum(preds == -1)
            anomaly_rate = n_anomalies / len(preds)

            col1, col2, col3 = st.columns(3)
            col1.metric("Samples Analyzed", len(preds))
            col2.metric("Anomalies Detected", n_anomalies)
            col3.metric("Anomaly Rate", f"{anomaly_rate:.2%}")

            # Visualizations
            st.plotly_chart(create_anomaly_timeline(scores), use_container_width=True)
            st.plotly_chart(create_scatter_matrix(df_sample, preds), use_container_width=True)

    # ========================================================================
    # Page: Model Comparison
    # ========================================================================
    elif page == "🔬 Model Comparison":
        st.header("Model Comparison")
        st.markdown("Compare predictions between OCSVM and Isolation Forest models.")

        n_samples = st.slider("Number of samples", 100, 5000, 500)
        df_sample = training_data.sample(n_samples, random_state=42)

        with st.spinner("Running both models..."):
            preds_ocsvm, scores_ocsvm = predict_batch(df_sample, models, "ocsvm")
            preds_if, scores_if = predict_batch(df_sample, models, "iforest")

        # Agreement analysis
        agreement = np.mean(preds_ocsvm == preds_if)
        both_anomaly = np.sum((preds_ocsvm == -1) & (preds_if == -1))
        only_ocsvm = np.sum((preds_ocsvm == -1) & (preds_if == 1))
        only_if = np.sum((preds_ocsvm == 1) & (preds_if == -1))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model Agreement", f"{agreement:.1%}")
        col2.metric("Both Flag Anomaly", both_anomaly)
        col3.metric("Only OCSVM", only_ocsvm)
        col4.metric("Only IsolationForest", only_if)

        # Comparison chart
        fig = make_subplots(rows=1, cols=2, subplot_titles=("OCSVM Scores", "Isolation Forest Scores"))

        fig.add_trace(
            go.Histogram(x=scores_ocsvm, name="OCSVM", marker_color='#1f77b4'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=scores_if, name="IsolationForest", marker_color='#ff7f0e'),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Scatter comparison
        fig_scatter = px.scatter(
            x=scores_ocsvm, y=scores_if,
            labels={"x": "OCSVM Score", "y": "IsolationForest Score"},
            title="Model Score Correlation",
            color_discrete_sequence=['#1f77b4']
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ========================================================================
    # Page: Data Explorer
    # ========================================================================
    elif page == "📈 Data Explorer":
        st.header("Data Explorer")
        st.markdown("Explore the training data with dimensionality reduction visualizations.")

        features = ["engine_rpm", "lub_oil_pressure", "fuel_pressure",
                   "coolant_pressure", "oil_temp", "coolant_temp"]

        # Sample size for visualization
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Visualization Settings**")
        n_samples_viz = st.sidebar.slider("Samples for PCA/UMAP", 500, 5000, 2000, 100)

        df_sample = training_data.sample(min(n_samples_viz, len(training_data)), random_state=42)

        # Run predictions for coloring
        with st.spinner("Computing predictions for visualization..."):
            preds, scores = predict_batch(df_sample, models, model_choice)

        # ====== PCA & UMAP Section ======
        st.subheader("🔮 Dimensionality Reduction Visualizations")
        st.markdown("Visualize high-dimensional sensor data in 2D/3D space with anomaly highlighting.")

        tab_pca, tab_umap, tab_3d = st.tabs(["📊 PCA (2D)", "🌀 UMAP (2D)", "🎲 PCA (3D)"])

        with tab_pca:
            st.markdown("**Principal Component Analysis** - Linear projection preserving maximum variance")
            with st.spinner("Computing PCA..."):
                X_pca, variance_ratio = compute_pca(df_sample, features)
                fig_pca = create_pca_plot(X_pca, preds, variance_ratio)
            st.plotly_chart(fig_pca, use_container_width=True)

            # PCA insights
            col1, col2, col3 = st.columns(3)
            col1.metric("PC1 Variance", f"{variance_ratio[0]:.1%}")
            col2.metric("PC2 Variance", f"{variance_ratio[1]:.1%}")
            col3.metric("Total Explained", f"{sum(variance_ratio):.1%}")

        with tab_umap:
            st.markdown("**UMAP** - Non-linear projection preserving local structure")

            col1, col2 = st.columns(2)
            with col1:
                n_neighbors = st.slider("n_neighbors", 5, 50, 15, help="Larger = more global structure")
            with col2:
                min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, 0.05, help="Smaller = tighter clusters")

            with st.spinner("Computing UMAP (this may take a moment)..."):
                X_umap = compute_umap(df_sample, features, n_neighbors, min_dist)
                fig_umap = create_umap_plot(X_umap, preds)
            st.plotly_chart(fig_umap, use_container_width=True)

        with tab_3d:
            st.markdown("**3D PCA** - Interactive 3D projection (drag to rotate)")
            with st.spinner("Computing 3D PCA..."):
                fig_3d = create_3d_pca_plot(df_sample, features, preds)
            st.plotly_chart(fig_3d, use_container_width=True)

        # Summary stats for anomalies in projections
        n_anomalies = np.sum(preds == -1)
        st.info(f"📊 Showing {len(df_sample)} samples | 🚨 {n_anomalies} anomalies ({n_anomalies/len(preds):.1%}) | Model: {model_choice.upper()}")

        st.markdown("---")

        # ====== Feature Distributions ======
        st.subheader("📈 Feature Distributions")

        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                fig = create_feature_distribution(training_data, feature)
                st.plotly_chart(fig, use_container_width=True)

        # ====== Correlation Heatmap ======
        st.subheader("🔥 Feature Correlations")
        corr = training_data[features].corr()

        fig_corr = px.imshow(
            corr,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix",
            text_auto=".2f"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        # ====== Symbolic Equations ======
        st.subheader("🧮 Symbolic Regression Equations")
        st.markdown("These equations were **automatically discovered** through symbolic regression (PySR):")

        for name, spec in SYMBOLIC_SPEC.items():
            with st.expander(f"📐 {name}", expanded=True):
                st.code(f"{name} = {spec['equation']}", language="python")
                st.caption(f"**Input Variables:** {', '.join(spec['variables'])}")

        # ====== Data Statistics ======
        st.subheader("📋 Data Statistics")
        st.dataframe(training_data.describe().T.style.format("{:.2f}").background_gradient(cmap='Blues'))

    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #888; font-size: 0.8rem;">
    Ship Engine Anomaly Detection System | Built with FastAPI + Streamlit |
    Models: OCSVM + Isolation Forest with Symbolic Regression Features
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
