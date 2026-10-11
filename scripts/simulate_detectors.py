"""Validate the drift detectors with known, injected shifts.

Held-out data: the validation period (Jul to Sep 2016), split at random into a
reference half and a pool half. Each trial draws n rows from the pool, either
plainly (no shift) or with importance weights that create a shift of known size.
Shifts are made by resampling real rows, so every value stays realistic.

- numeric feature f: weight exp(beta * z), z = clipped standard score of f.
  Effect size: the standardized mean difference (SMD) measured in the drawn samples.
- categorical feature f, category c: reweight so the share of c rises by delta.
  Effect size: the share change measured in the drawn samples.

Detection rate = share of trials where the detector fires. At size 0 this is
the false alarm rate.

Usage: python scripts/simulate_detectors.py [--trials 200]
Writes: reports/detector_simulation.json, reports/figures/detector_power.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotelmon import artifacts, config, drift, monitoring  # noqa: E402
from hotelmon.features import to_model_frame  # noqa: E402
from hotelmon.simulation import categorical_weights, numeric_weights  # noqa: E402

SAMPLE_SIZES = [250, 1000, 2500]
NUMERIC_CASES = {
    "lead_time": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "adr": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "total_of_special_requests": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "previous_cancellations": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
}
CATEGORICAL_CASES = {
    ("deposit_type", "Non Refund"): [0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
    ("market_segment", "Groups"): [0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
    ("country", "GBR"): [0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
}


def detect_numeric(ref: np.ndarray, cur: np.ndarray) -> dict[str, bool]:
    psi = drift.psi_numeric(ref, cur)
    _, p = drift.ks_test(ref, cur)
    return {"psi>=0.1": psi >= 0.1, "psi>=0.2": psi >= 0.2, "test p<0.05": p < 0.05}


def detect_categorical(ref_counts: np.ndarray, cur_counts: np.ndarray) -> dict[str, bool]:
    # Same statistics as drift.psi_categorical and drift.chi_square_test, on counts.
    psi = drift.psi_from_proportions(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())
    _, p, _ = drift.chi_square_from_counts(ref_counts, cur_counts)
    return {"psi>=0.1": psi >= 0.1, "psi>=0.2": psi >= 0.2, "test p<0.05": p < 0.05}


def run_case(
    ref_col, pool_col, weights, n, trials, rng, numeric: bool, category: str | None = None
) -> dict[str, float]:
    """Detection rates over trials, plus the mean shift actually achieved in the samples.

    Rows are drawn WITHOUT replacement, so with no shift the sample and the reference
    are exchangeable random subsets of the same period and each test's null holds.
    (Drawing with replacement from a finite pool inflates false alarms at large n.)
    """
    hits = {"psi>=0.1": 0, "psi>=0.2": 0, "test p<0.05": 0}
    effects = []
    if numeric:
        ref_vals, pool_vals = ref_col.to_numpy(dtype=float), pool_col.to_numpy(dtype=float)
        ref_mean, ref_sd = ref_vals.mean(), ref_vals.std()
    else:
        cats = pd.Index(sorted(set(ref_col) | set(pool_col)))
        ref_counts = np.bincount(cats.get_indexer(ref_col), minlength=len(cats))
        pool_codes = cats.get_indexer(pool_col)
        c_idx = cats.get_loc(category)
        ref_share = ref_counts[c_idx] / ref_counts.sum()
    for _ in range(trials):
        idx = rng.choice(len(pool_col), size=n, replace=False, p=weights)
        if numeric:
            cur = pool_vals[idx]
            res = detect_numeric(ref_vals, cur)
            effects.append((cur.mean() - ref_mean) / ref_sd)
        else:
            cur_counts = np.bincount(pool_codes[idx], minlength=len(cats))
            res = detect_categorical(ref_counts, cur_counts)
            effects.append(cur_counts[c_idx] / n - ref_share)
        for k, v in res.items():
            hits[k] += bool(v)
    return {k: v / trials for k, v in hits.items()} | {"effect": round(float(np.mean(effects)), 4)}


def any_feature_false_alarm(ref, pool, n, trials, rng) -> dict[str, float]:
    """No shift: how often does at least one of the monitored features fire?"""
    k = len(monitoring.DRIFT_FEATURES)
    hits = {"psi>=0.1": 0, "psi>=0.2": 0, "test p<0.05": 0, "test p<0.05/k (Bonferroni)": 0}
    for _ in range(trials):
        cur = pool.iloc[rng.choice(len(pool), size=n, replace=False)]
        t = drift.feature_drift(ref, cur, monitoring.DRIFT_NUMERIC, monitoring.DRIFT_CATEGORICAL)
        hits["psi>=0.1"] += bool((t["psi"] >= 0.1).any())
        hits["psi>=0.2"] += bool((t["psi"] >= 0.2).any())
        hits["test p<0.05"] += bool((t["p_value"] < 0.05).any())
        hits["test p<0.05/k (Bonferroni)"] += bool((t["p_value"] < 0.05 / k).any())
    return {key: v / trials for key, v in hits.items()}


def natural_weekly_variation(ref_full: pd.DataFrame, dates: pd.Series) -> list[dict]:
    """Each week of the reference period vs the rest of it. No injected shift."""
    wk = dates.dt.to_period("W-SUN").astype(str)
    rows = []
    for w in sorted(wk.unique()):
        m = (wk == w).to_numpy()
        if m.sum() < 300:
            continue
        t = drift.feature_drift(
            ref_full[~m], ref_full[m], monitoring.DRIFT_NUMERIC, monitoring.DRIFT_CATEGORICAL
        )
        rows.append(
            {
                "week": w,
                "n": int(m.sum()),
                "n_features_psi_ge_0.1": int((t["psi"] >= 0.1).sum()),
                "n_features_test_flagged_bonferroni": int(t["test_flag"].sum()),
                "max_psi": float(t["psi"].max()),
                "max_psi_feature": t.loc[t["psi"].idxmax(), "feature"],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    art = artifacts.load(artifacts.resolve(), with_reference=True)
    full = to_model_frame(art.reference)
    perm = rng.permutation(len(full))
    half = len(full) // 2
    ref = full.iloc[perm[:half]].reset_index(drop=True)
    pool = full.iloc[perm[half:]].reset_index(drop=True)

    rows = []
    for f, sizes in NUMERIC_CASES.items():
        x = pool[f].to_numpy(dtype=float)
        for beta in sizes:
            w = numeric_weights(x, beta)
            for n in SAMPLE_SIZES:
                rates = run_case(ref[f], pool[f], w, n, args.trials, rng, numeric=True)
                rows.append(
                    {
                        "feature": f,
                        "kind": "numeric",
                        "shift": f"beta={beta}",
                        "size": beta,
                        "effect_unit": "SMD",
                        "n": n,
                    }
                    | rates
                )
                print(rows[-1])
    for (f, cat), sizes in CATEGORICAL_CASES.items():
        base = float((pool[f] == cat).mean())
        for delta in sizes:
            w = categorical_weights(pool[f], cat, delta)
            for n in SAMPLE_SIZES:
                rates = run_case(
                    ref[f], pool[f], w, n, args.trials, rng, numeric=False, category=cat
                )
                rows.append(
                    {
                        "feature": f,
                        "kind": "categorical",
                        "shift": f"{cat} +{delta:.2f}",
                        "size": delta,
                        "effect_unit": "share points",
                        "base_share": round(base, 4),
                        "n": n,
                    }
                    | rates
                )
                print(rows[-1])

    fa_any = {n: any_feature_false_alarm(ref, pool, n, args.trials, rng) for n in SAMPLE_SIZES}
    print("any-feature false alarm:", fa_any)
    dates = pd.to_datetime(art.reference[config.TIME_COL])
    natural = natural_weekly_variation(full, dates)

    out = {
        "trials_per_cell": args.trials,
        "seed": args.seed,
        "reference_rows": len(ref),
        "pool_rows": len(pool),
        "results": rows,
        "any_feature_false_alarm": {str(k): v for k, v in fa_any.items()},
        "n_monitored_features": len(monitoring.DRIFT_FEATURES),
        "natural_weekly_variation_in_reference": natural,
    }
    (config.REPORTS_DIR / "detector_simulation.json").write_text(json.dumps(out, indent=2) + "\n")

    df = pd.DataFrame(rows)
    cases = list(NUMERIC_CASES) + [f for f, _ in CATEGORICAL_CASES]
    fig, axes = plt.subplots(3, len(cases), figsize=(3.2 * len(cases), 8.5), sharey=True)
    for j, f in enumerate(cases):
        sub = df[df["feature"] == f]
        for i, det in enumerate(["test p<0.05", "psi>=0.1", "psi>=0.2"]):
            ax = axes[i, j]
            for n in SAMPLE_SIZES:
                s = sub[sub["n"] == n]
                ax.plot(s["effect"], s[det], marker="o", ms=3, label=f"n={n}")
            ax.axhline(0.05, color="grey", lw=0.7, ls=":")
            if i == 0:
                ax.set_title(f, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{det}\ndetection rate")
            if i == 2:
                ax.set_xlabel(sub["effect_unit"].iloc[0])
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Detector power vs injected shift size (rate at size 0 = false alarm rate)")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "figures" / "detector_power.png", dpi=100)


if __name__ == "__main__":
    main()
