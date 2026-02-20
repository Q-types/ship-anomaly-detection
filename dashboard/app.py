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
</style>
""", unsafe_allow_html=True)


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

    # Calculate confidence
    if model_name == "ocsvm":
        confidence = 1.0 / (1.0 + np.exp(-2 * score))
    else:
        confidence = 1.0 / (1.0 + np.exp(-5 * score))

    return {
        "is_anomaly": is_anomaly,
        "confidence": float(confidence),
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
    # Page: Real-time Detection
    # ========================================================================
    if page == "🔍 Real-time Detection":
        st.header("Real-time Anomaly Detection")
        st.markdown("Enter sensor values to check for anomalies in real-time.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sensor Inputs")

            engine_rpm = st.slider("Engine RPM", 0, 3000, 750, 10)
            lub_oil_pressure = st.slider("Lub Oil Pressure (bar)", 0.0, 15.0, 3.5, 0.1)
            fuel_pressure = st.slider("Fuel Pressure (bar)", 0.0, 50.0, 6.0, 0.5)
            coolant_pressure = st.slider("Coolant Pressure (bar)", 0.0, 10.0, 2.5, 0.1)
            oil_temp = st.slider("Oil Temperature (°C)", 0, 150, 78, 1)
            coolant_temp = st.slider("Coolant Temperature (°C)", 0, 120, 72, 1)

        with col2:
            st.subheader("Prediction Result")

            reading = {
                "engine_rpm": float(engine_rpm),
                "lub_oil_pressure": float(lub_oil_pressure),
                "fuel_pressure": float(fuel_pressure),
                "coolant_pressure": float(coolant_pressure),
                "oil_temp": float(oil_temp),
                "coolant_temp": float(coolant_temp),
            }

            result = predict_single(reading, models, model_choice)

            # Status display
            if result["is_anomaly"]:
                st.error("🚨 **ANOMALY DETECTED**")
                status_class = "status-anomaly"
            else:
                st.success("✅ **Normal Operation**")
                status_class = "status-normal"

            # Metrics
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            with col2b:
                st.metric("Anomaly Score", f"{result['score']:.4f}")

            # Gauge chart
            color = "#dc3545" if result["is_anomaly"] else "#28a745"
            gauge = create_gauge_chart(result["confidence"], "Confidence Level", color)
            st.plotly_chart(gauge, use_container_width=True)

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
