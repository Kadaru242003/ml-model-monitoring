"""Train the baseline and LightGBM models on the earliest period and save an artifact.

Usage: python scripts/train.py
Writes: models/<version>/, models/LATEST, reports/training_metrics.json,
        reports/figures/reliability_valid.png
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sklearn  # noqa: E402

from hotelmon import artifacts, config, data, evaluation, features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train")


def main() -> None:
    raw_path = data.download()
    df = data.load_clean(raw_path)
    parts = data.split(df)
    train, valid = parts["train"], parts["valid"]
    y_tr, y_va = train[config.TARGET], valid[config.TARGET]
    log.info("train rows=%d valid rows=%d prod rows=%d", len(train), len(valid), len(parts["prod"]))

    candidates = {
        "logreg": features.make_logreg(),
        "lightgbm": features.make_lightgbm(),
    }
    results, scores = {}, {}
    for name, pipe in candidates.items():
        features.fit(pipe, train, y_tr)
        p = pipe.predict_proba(valid)[:, 1]
        m = evaluation.classification_metrics(y_va, p)
        m["roc_auc_ci95"] = evaluation.bootstrap_auc_ci(y_va, p)
        m["reliability"] = evaluation.reliability_curve(y_va, p)
        results[name], scores[name] = m, p
        log.info(
            "%s valid ROC AUC=%.4f PR AUC=%.4f Brier=%.4f",
            name,
            m["roc_auc"],
            m["pr_auc"],
            m["brier"],
        )

    best = max(results, key=lambda k: results[k]["roc_auc"])
    data_sha = data.sha256_of(raw_path)
    commit = artifacts.git_commit()
    version = f"{best}-{data_sha[:8]}-{commit[:7]}"
    metadata = {
        "version": version,
        "model_type": best,
        "trained_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "data": {
            "source_url": config.DATA_URL,
            "sha256": data_sha,
            "license": "CC BY 4.0 (Antonio, Almeida and Nunes, 2019, Data in Brief 22:41-49)",
        },
        "train_period": {
            "start": config.TRAIN_START,
            "end_exclusive": config.TRAIN_END,
            "rows": len(train),
            "first_arrival": str(train[config.TIME_COL].min().date()),
            "last_arrival": str(train[config.TIME_COL].max().date()),
        },
        "valid_period": {
            "start": config.VALID_START,
            "end_exclusive": config.VALID_END,
            "rows": len(valid),
        },
        "input_fields": [c for c in train.columns if c != config.TARGET],
        "model_features": {
            "numeric": config.NUMERIC_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
        },
        "excluded_leaky_columns": config.LEAKY_COLUMNS,
        "decision_threshold": 0.5,
        "metrics_valid": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
    }
    reference = valid.copy()
    reference["score"] = scores[best]
    reference[config.TIME_COL] = reference[config.TIME_COL].dt.date.astype(str)
    out = artifacts.save(candidates[best], metadata, reference)
    log.info("saved %s (selected %s by validation ROC AUC)", out, best)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    summary = {
        "selected_model": best,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "prod_rows": len(parts["prod"]),
        "rows_after_cleaning": len(df),
        "valid_metrics": {
            k: {kk: v[kk] for kk in ("roc_auc", "roc_auc_ci95", "pr_auc", "brier", "ece")}
            | {"positive_rate": v["positive_rate"], "n": v["n"]}
            for k, v in results.items()
        },
    }
    (config.REPORTS_DIR / "training_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig_dir = config.REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect")
    for name, m in results.items():
        r = m["reliability"]
        ax.plot(
            r["mean_predicted"],
            r["fraction_positive"],
            marker="o",
            label=f"{name} (Brier {m['brier']:.3f})",
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed cancellation rate")
    ax.set_title("Reliability on validation (Jul to Sep 2016)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "reliability_valid.png", dpi=120)
    log.info("wrote reports/training_metrics.json and reliability figure")


if __name__ == "__main__":
    main()
