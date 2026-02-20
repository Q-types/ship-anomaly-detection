from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from features.symbolic import compute_symbolic_features, get_symbolic_spec_metadata


# Column mapping from CSV format to internal names
COLUMN_MAPPING = {
    "Engine rpm": "engine_rpm",
    "Lub oil pressure": "lub_oil_pressure",
    "Fuel pressure": "fuel_pressure",
    "Coolant pressure": "coolant_pressure",
    "lub oil temp": "oil_temp",
    "Coolant temp": "coolant_temp",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names from CSV format."""
    rename_map = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    return df.rename(columns=rename_map)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train anomaly detection models on symbolic features.")
    p.add_argument("--train_csv", type=str, required=True, help="Path to training CSV.")
    p.add_argument("--model_dir", type=str, default="models", help="Directory to save model artefacts.")
    p.add_argument("--timestamp", action="store_true", help="Add timestamp to saved artefact filenames.")

    # OCSVM params
    p.add_argument("--ocsvm_nu", type=float, default=0.02)
    p.add_argument("--ocsvm_gamma", type=float, default=0.2)

    # IF params
    p.add_argument("--if_contamination", type=float, default=0.02)
    p.add_argument("--if_estimators", type=int, default=300)
    p.add_argument("--random_state", type=int, default=42)

    return p.parse_args()


def compute_anom_rate_from_predict(pred: np.ndarray) -> float:
    # sklearn anomaly convention: -1 = anomaly, +1 = normal
    return float(np.mean(pred == -1))


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data
    df = pd.read_csv(args.train_csv)

    # Normalize column names
    df = normalize_columns(df)
    print(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

    # ---- Build symbolic feature matrix
    X_sym, sym_feature_names = compute_symbolic_features(df)
    meta = get_symbolic_spec_metadata()

    # ---- Train OCSVM (scaled)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sym)

    ocsvm = OneClassSVM(
        kernel="rbf",
        nu=args.ocsvm_nu,
        gamma=args.ocsvm_gamma,
    )
    ocsvm.fit(X_scaled)
    ocsvm_pred = ocsvm.predict(X_scaled)
    ocsvm_rate = compute_anom_rate_from_predict(ocsvm_pred)

    # ---- Train Isolation Forest (unscaled)
    iforest = IsolationForest(
        n_estimators=args.if_estimators,
        contamination=args.if_contamination,
        random_state=args.random_state,
    )
    iforest.fit(X_sym)
    if_pred = iforest.predict(X_sym)
    if_rate = compute_anom_rate_from_predict(if_pred)

    # ---- Build artefacts
    created_utc = datetime.now(timezone.utc).isoformat()

    ocsvm_artifact = {
        "model": ocsvm,
        "scaler": scaler,
        "symbolic_meta": meta,
        "feature_names": sym_feature_names,
        "train_anom_rate": ocsvm_rate,
        "params": {"nu": args.ocsvm_nu, "gamma": args.ocsvm_gamma},
        "created_utc": created_utc,
        "version": meta.get("version", "v1.0"),
        "model_type": "OCSVM",
    }

    if_artifact = {
        "model": iforest,
        "symbolic_meta": meta,
        "feature_names": sym_feature_names,
        "train_anom_rate": if_rate,
        "params": {"contamination": args.if_contamination, "n_estimators": args.if_estimators},
        "created_utc": created_utc,
        "version": meta.get("version", "v1.0"),
        "model_type": "IsolationForest",
    }

    # ---- Save
    suffix = ""
    if args.timestamp:
        suffix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    ocsvm_path = model_dir / f"ocsvm_symbolic{suffix}.joblib"
    if_path = model_dir / f"if_symbolic{suffix}.joblib"

    dump(ocsvm_artifact, ocsvm_path)
    dump(if_artifact, if_path)

    print("Saved artefacts:")
    print(f"  OCSVM -> {ocsvm_path} (train anomaly rate={ocsvm_rate:.4f})")
    print(f"  IF    -> {if_path} (train anomaly rate={if_rate:.4f})")
    print("Feature names:", sym_feature_names)


if __name__ == "__main__":
    main()
    
    
# python -m training.train_models --train_csv data/train.csv --timestamp