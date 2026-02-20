# features/symbolic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# ---------------------------
# 1) Canonical specification
# ---------------------------

# IMPORTANT:
# x0, x1, x2 correspond to the variables list order below.
SYMBOLIC_SPEC: Dict[str, Dict[str, object]] = {
    "oil_temp": {
        "variables": ["engine_rpm", "coolant_temp", "coolant_pressure"],
        "equation": "log(x1 * sqrt(x0)) + 69.69828",
    },
    "coolant_temp": {
        "variables": ["engine_rpm", "coolant_pressure", "oil_temp"],
        "equation": "sqrt(x2) + 69.59798",
    },
    "oil_pressure": {
        "variables": ["engine_rpm", "oil_temp"],
        "equation": "exp(exp(sqrt(sqrt(log(x0) * 0.82669675) / x1)))",
    },
    "fuel_pressure": {
        "variables": ["engine_rpm"],
        "equation": "(40.246246 / (1619.7604 - x0)) + 6.4258323",
    },
}

# The order of outputs in the final feature matrix.
# Keep this fixed for v1.0.
OUTPUT_ORDER: List[str] = ["oil_temp", "coolant_temp", "oil_pressure", "fuel_pressure"]


# -----------------------------------
# 2) Safe evaluation helper functions
# -----------------------------------

_ALLOWED_FUNCS = {
    "log": np.log,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "abs": np.abs,
    "clip": np.clip,
    "maximum": np.maximum,
    "minimum": np.minimum,
}

_SAFE_GLOBALS = {"__builtins__": {}}  # disables builtins in eval


def _required_input_columns(spec: Dict[str, Dict[str, object]]) -> List[str]:
    cols = set()
    for target, entry in spec.items():
        vars_list = entry["variables"]
        for v in vars_list:
            cols.add(v)
    return sorted(cols)


def _validate_inputs(df: pd.DataFrame, spec: Dict[str, Dict[str, object]]) -> None:
    required = _required_input_columns(spec)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required input columns for symbolic features: {missing}")


def evaluate_symbolic_equation(
    df: pd.DataFrame,
    equation: str,
    variables: List[str],
    *,
    clip: tuple[float, float] | None = (-1e6, 1e6),
) -> np.ndarray:
    """
    Evaluates a symbolic regression equation with a locked mapping:
      x0 -> variables[0], x1 -> variables[1], ...

    - Sanitises NaN / inf
    - Optionally clips extreme values
    - Returns a 1D numpy array of length len(df)
    """
    # Map x0, x1, ... to numpy arrays
    local = {f"x{i}": df[var].to_numpy(dtype=float) for i, var in enumerate(variables)}
    local.update(_ALLOWED_FUNCS)

    try:
        out = eval(equation, _SAFE_GLOBALS, local)
    except Exception as e:
        raise ValueError(
            f"Failed evaluating equation='{equation}' with variables={variables}. "
            f"Original error: {e}"
        )

    out = np.asarray(out, dtype=float)

    if out.ndim != 1 or out.shape[0] != len(df):
        raise ValueError(
            f"Equation output has wrong shape. Expected (n_samples,), got {out.shape}"
        )

    # Replace inf with NaN, then check
    out = np.where(np.isfinite(out), out, np.nan)

    if np.isnan(out).any():
        raise ValueError(
            f"Non-finite values produced by equation='{equation}'. "
            f"Check domain of variables={variables}."
        )

    if clip is not None:
        out = np.clip(out, clip[0], clip[1])

    return out


# ---------------------------
# 3) Public API for pipeline
# ---------------------------

def compute_symbolic_features(
    df: pd.DataFrame,
    output_order: List[str] = OUTPUT_ORDER,
    spec: Dict[str, Dict[str, object]] = SYMBOLIC_SPEC,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute symbolic features in a fixed, versionable order.

    Returns:
      X_sym: (n_samples, n_features) numpy array
      feature_names: list of feature names in the same column order as X_sym
    """
    _validate_inputs(df, spec)

    feats = []
    names = []

    for name in output_order:
        if name not in spec:
            raise KeyError(f"'{name}' is in output_order but missing from SYMBOLIC_SPEC")

        entry = spec[name]
        variables = entry["variables"]  # type: ignore[assignment]
        equation = entry["equation"]    # type: ignore[assignment]

        y = evaluate_symbolic_equation(df, equation, variables)
        feats.append(y)
        names.append(f"sym_{name}")

    X_sym = np.column_stack(feats)
    return X_sym, names


def get_symbolic_spec_metadata() -> Dict[str, object]:
    """
    Metadata that should be saved alongside models for reproducibility.
    """
    return {
        "output_order": OUTPUT_ORDER,
        "spec": SYMBOLIC_SPEC,
        "required_input_columns": _required_input_columns(SYMBOLIC_SPEC),
        "feature_names": [f"sym_{n}" for n in OUTPUT_ORDER],
        "version": "v1.0",
    }