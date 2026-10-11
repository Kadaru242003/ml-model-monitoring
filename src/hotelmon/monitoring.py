"""Turn the prediction log into per-window drift and performance summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from hotelmon import config, drift
from hotelmon.evaluation import bootstrap_auc_ci
from hotelmon.features import to_model_frame

# arrival_month is a calendar feature. The reference covers Jul to Sep only, so
# every later month "drifts" by construction. It is left out of drift scoring.
DRIFT_NUMERIC = list(config.NUMERIC_FEATURES)
DRIFT_CATEGORICAL = [c for c in config.CATEGORICAL_FEATURES if c != "arrival_month"]
DRIFT_FEATURES = DRIFT_NUMERIC + DRIFT_CATEGORICAL


def model_view(df: pd.DataFrame) -> pd.DataFrame:
    """Model features (as the model sees them) plus score and label columns."""
    out = to_model_frame(df)
    out["score"] = df["score"].to_numpy()
    out["label"] = df["label"].to_numpy() if "label" in df else np.nan
    return out


def reference_view(reference: pd.DataFrame) -> pd.DataFrame:
    ref = reference.copy()
    ref["label"] = ref[config.TARGET]
    return model_view(ref)


def log_view(log_df: pd.DataFrame) -> pd.DataFrame:
    df = log_df.rename(columns={"probability": "score"}).copy()
    out = model_view(df)
    out["arrival_date"] = pd.to_datetime(df["arrival_date"]).to_numpy()
    out["prediction_id"] = df["prediction_id"].to_numpy()
    return out


def performance(y: np.ndarray, p: np.ndarray, with_ci: bool = True) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = ~np.isnan(y)
    y, p = y[m].astype(int), p[m]
    out = {"n_labeled": int(len(y))}
    if len(y) == 0 or y.min() == y.max():
        return out
    out |= {
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    if with_ci:
        out["roc_auc_ci95"] = bootstrap_auc_ci(y, p, n_boot=300)
    return out


def window_summary(ref: pd.DataFrame, cur: pd.DataFrame, with_ci: bool = True) -> dict:
    """Drift and performance for one window of production data against the reference."""
    table = drift.feature_drift(ref, cur, DRIFT_NUMERIC, DRIFT_CATEGORICAL)
    sd = drift.score_drift(ref["score"].to_numpy(), cur["score"].to_numpy())
    perf = performance(cur["label"], cur["score"], with_ci=with_ci)
    return {
        "n": int(len(cur)),
        "feature_drift": table,
        "score_drift": sd,
        "performance": perf,
        "mean_feature_psi": float(table["psi"].mean()),
        "max_feature_psi": float(table["psi"].max()),
        "n_features_psi_moderate_or_more": int((table["psi"] >= drift.PSI_MODERATE).sum()),
        "n_features_test_flagged": int(table["test_flag"].sum()),
    }
