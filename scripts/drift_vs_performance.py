"""Does feature drift predict performance drop on this dataset?

For each production window (calendar month, and ISO week) we compute drift
summaries against the reference and the ROC AUC drop against validation. We then
measure rank correlation (Spearman) between each drift summary and the AUC drop.

Weekly windows are autocorrelated, so p-values use a circular-shift permutation
null (shift the AUC series against the drift series), which keeps that structure.

Usage: python scripts/drift_vs_performance.py [--db data/replay.db]
Writes: reports/drift_vs_performance.json, reports/figures/drift_vs_auc_drop.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from hotelmon import artifacts, config, monitoring  # noqa: E402
from hotelmon.logdb import PredictionLog  # noqa: E402

SIGNALS = [
    "score_psi",
    "score_ks",
    "mean_feature_psi",
    "max_feature_psi",
    "n_features_psi_ge_0.1",
    "importance_weighted_psi",
]
DEGRADED_DROP = 0.05  # a window is "degraded" if AUC is this far below validation


def importance_weights(model) -> pd.Series:
    """LightGBM gain importance, restricted to drift-monitored features, summing to 1."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "booster_"):
        coefs = pd.Series(1.0, index=monitoring.DRIFT_FEATURES)
        return coefs / coefs.sum()
    gain = pd.Series(
        clf.booster_.feature_importance(importance_type="gain"),
        index=clf.booster_.feature_name(),
    )
    w = gain.reindex(monitoring.DRIFT_FEATURES).fillna(0.0)
    return w / w.sum()


def windows(cur_all: pd.DataFrame, ref: pd.DataFrame, key: str, weights: pd.Series, ref_auc):
    rows = []
    for w, cur in cur_all.groupby(key):
        if len(cur) < 300 or cur["label"].nunique() < 2:
            continue
        s = monitoring.window_summary(ref, cur, with_ci=False)
        psi = s["feature_drift"].set_index("feature")["psi"]
        nodep = cur[cur["deposit_type"] == "No Deposit"]
        row = {
            "window": str(w),
            "n": s["n"],
            "roc_auc": s["performance"]["roc_auc"],
            "auc_drop": ref_auc - s["performance"]["roc_auc"],
            "auc_no_deposit_only": float(roc_auc_score(nodep["label"], nodep["score"])),
            "positive_rate": s["performance"]["positive_rate"],
            "share_non_refund": float((cur["deposit_type"] == "Non Refund").mean()),
            "score_psi": s["score_drift"]["psi"],
            "score_ks": s["score_drift"]["ks_statistic"],
            "mean_feature_psi": s["mean_feature_psi"],
            "max_feature_psi": s["max_feature_psi"],
            "n_features_psi_ge_0.1": s["n_features_psi_moderate_or_more"],
            "importance_weighted_psi": float((psi * weights.reindex(psi.index)).sum()),
        }
        row |= {f"psi_{f}": float(v) for f, v in psi.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_with_null(x, y, n_perm: int = 2000, circular: bool = True, seed: int = 0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    n = len(x)
    if circular:
        null = [stats.spearmanr(x, np.roll(y, k)).statistic for k in range(1, n)]
    else:
        null = [stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)]
    p = (1 + np.sum(np.abs(null) >= abs(rho))) / (1 + len(null))
    return float(rho), float(p)


def alarm_table(df: pd.DataFrame) -> list[dict]:
    degraded = df["auc_drop"] > DEGRADED_DROP
    rules = {
        "score PSI >= 0.1": df["score_psi"] >= 0.1,
        "score PSI >= 0.2": df["score_psi"] >= 0.2,
        "max feature PSI >= 0.2": df["max_feature_psi"] >= 0.2,
        ">= 5 features with PSI >= 0.1": df["n_features_psi_ge_0.1"] >= 5,
        "importance-weighted PSI >= 0.1": df["importance_weighted_psi"] >= 0.1,
    }
    out = []
    for name, alarm in rules.items():
        tp = int((alarm & degraded).sum())
        out.append(
            {
                "rule": name,
                "windows": int(len(df)),
                "degraded_windows": int(degraded.sum()),
                "alarms": int(alarm.sum()),
                "true_alarms": tp,
                "recall": tp / degraded.sum() if degraded.sum() else None,
                "precision": tp / alarm.sum() if alarm.sum() else None,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=config.DATA_DIR / "replay.db")
    args = ap.parse_args()

    art = artifacts.load(artifacts.resolve(), with_reference=True)
    ref = monitoring.reference_view(art.reference)
    ref_auc = float(roc_auc_score(ref["label"], ref["score"]))
    weights = importance_weights(art.model)

    log = PredictionLog(args.db)
    raw = log.read()
    log.close()
    cur_all = monitoring.log_view(raw)
    cur_all["deposit_type"] = raw["deposit_type"].to_numpy()
    cur_all["month"] = cur_all["arrival_date"].dt.strftime("%Y-%m")
    iso = cur_all["arrival_date"].dt.isocalendar()
    cur_all["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    result = {
        "reference_roc_auc": ref_auc,
        "degraded_definition": f"window ROC AUC more than {DEGRADED_DROP} below validation",
        "importance_weights": weights.round(4).to_dict(),
    }
    frames = {}
    for key, circular in (("month", False), ("week", True)):
        df = windows(cur_all, ref, key, weights, ref_auc)
        frames[key] = df
        corr = {}
        for sig in SIGNALS + ["share_non_refund"]:
            rho, p = spearman_with_null(df[sig], df["auc_drop"], circular=circular)
            corr[sig] = {"spearman_rho": rho, "p_value": p}
        per_feature = {}
        for f in monitoring.DRIFT_FEATURES:
            rho, p = spearman_with_null(df[f"psi_{f}"], df["auc_drop"], circular=circular)
            per_feature[f] = {"spearman_rho": rho, "p_value": p}
        result[key] = {
            "n_windows": int(len(df)),
            "null": "circular shift" if circular else "random permutation",
            "correlation_with_auc_drop": corr,
            "per_feature_psi_correlation": per_feature,
            "alarm_rules": alarm_table(df),
            "windows": df.round(5).to_dict(orient="records"),
        }
        print(f"\n{key}: {len(df)} windows")
        print(pd.DataFrame(corr).T.round(3).to_string())

    # Is the drop explained by losing the easy Non Refund cases?
    m = frames["month"]
    result["month"]["auc_no_deposit_only"] = {
        "valid": float(
            roc_auc_score(
                ref.loc[art.reference["deposit_type"] == "No Deposit", "label"],
                ref.loc[art.reference["deposit_type"] == "No Deposit", "score"],
            )
        ),
        "by_month": dict(zip(m["window"], m["auc_no_deposit_only"].round(4), strict=True)),
    }

    w = frames["week"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, sig in zip(
        axes, ["score_psi", "importance_weighted_psi", "share_non_refund"], strict=True
    ):
        ax.scatter(w[sig], w["auc_drop"], s=14, alpha=0.7)
        c = result["week"]["correlation_with_auc_drop"][sig]
        ax.set_title(f"{sig}: rho {c['spearman_rho']:.2f}, p {c['p_value']:.3f}")
        ax.set_xlabel(sig)
        ax.axhline(0, color="grey", lw=0.8)
    axes[0].set_ylabel("ROC AUC drop vs validation")
    fig.suptitle("Weekly production windows")
    fig.tight_layout()
    (config.REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(config.REPORTS_DIR / "figures" / "drift_vs_auc_drop.png", dpi=110)

    (config.REPORTS_DIR / "drift_vs_performance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(pd.DataFrame(result["week"]["alarm_rules"]).to_string(index=False))


if __name__ == "__main__":
    main()
