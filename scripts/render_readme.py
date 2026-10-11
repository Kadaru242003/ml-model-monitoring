"""Fill the generated blocks in README.md from the JSON files in reports/.

Every result number in the README sits between <!-- BEGIN:x --> and <!-- END:x -->
markers and is written by this script. Hand-written text holds no results.

Usage: python scripts/render_readme.py          # rewrite README.md
       python scripts/render_readme.py --check  # exit 1 if README.md is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
README = ROOT / "README.md"
FORBIDDEN = {"—": "em dash", "–": "en dash"}


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def block_data() -> str:
    t = load("training_metrics.json")
    return table(
        ["Split", "Arrival dates", "Bookings"],
        [
            ["Train", "2015-07-01 to 2016-06-30", t["train_rows"]],
            ["Validation", "2016-07-01 to 2016-09-30", t["valid_rows"]],
            ["Production (replayed)", "2016-10-01 to 2017-08-31", t["prod_rows"]],
        ],
    ) + (
        f"\n\nRows after cleaning: {t['rows_after_cleaning']}. "
        "Source: `scripts/train.py`, file `reports/training_metrics.json`."
    )


def block_training() -> str:
    t = load("training_metrics.json")
    rows = []
    for name, m in t["valid_metrics"].items():
        lo, hi = m["roc_auc_ci95"]
        rows.append(
            [
                name + (" (served)" if name == t["selected_model"] else ""),
                f"{m['roc_auc']:.3f} ({lo:.3f} to {hi:.3f})",
                f"{m['pr_auc']:.3f}",
                f"{m['brier']:.3f}",
                f"{m['ece']:.3f}",
            ]
        )
    base = next(iter(t["valid_metrics"].values()))["positive_rate"]
    return (
        table(
            ["Model", "ROC AUC (95% bootstrap CI)", "PR AUC", "Brier", "ECE (10 bins)"],
            rows,
        )
        + f"\n\nValidation cancellation rate: {base:.3f} (the PR AUC of a random model). "
        "Source: `scripts/train.py`, file `reports/training_metrics.json`."
    )


def block_tuning() -> str:
    t = load("tuning_lightgbm.json")
    rows = [
        [
            r["max_cat_to_onehot"],
            r["n_estimators"],
            r["num_leaves"],
            f"{r['roc_auc']:.3f}",
            f"{r['brier']:.3f}",
        ]
        for r in t["results"]
    ]
    return (
        table(["max_cat_to_onehot", "n_estimators", "num_leaves", "ROC AUC", "Brier"], rows)
        + f"\n\nFit on {t['fit_rows']} bookings (Jul 2015 to Mar 2016), scored on "
        f"{t['eval_rows']} (Apr to Jun 2016). Source: `scripts/tune_lightgbm.py`."
    )


def block_monitoring() -> str:
    m = load("monitoring_summary.json")
    ref = m["reference"]
    rows = [
        [
            "validation (ref)",
            ref["n_labeled"],
            f"{ref['positive_rate']:.3f}",
            f"{ref['roc_auc']:.3f}",
            f"{ref['brier']:.3f}",
            "",
            "",
            "",
        ]
    ]
    for r in m["periods"]:
        lo, hi = r["roc_auc_ci95"]
        rows.append(
            [
                f"[{r['period']}](reports/drift/{r['period']}.md)",
                r["n"],
                f"{r['positive_rate']:.3f}",
                f"{r['roc_auc']:.3f} ({lo:.3f} to {hi:.3f})",
                f"{r['brier']:.3f}",
                f"{r['score_psi']:.3f}",
                f"{r['n_features_psi_ge_0.1']} / {r['n_features']}",
                f"{r['n_features_test_flagged']} / {r['n_features']}",
            ]
        )
    return (
        table(
            [
                "Arrival month",
                "Bookings",
                "Cancel rate",
                "ROC AUC (95% CI)",
                "Brier",
                "Score PSI",
                "Features PSI >= 0.1",
                "Features failing test",
            ],
            rows,
        )
        + "\n\nSource: `scripts/replay.py` then `scripts/monitor.py`, "
        "file `reports/monitoring_summary.json`."
    )


def block_drift_perf() -> str:
    d = load("drift_vs_performance.json")
    names = {
        "score_psi": "Score PSI",
        "score_ks": "Score KS statistic",
        "mean_feature_psi": "Mean feature PSI",
        "max_feature_psi": "Max feature PSI",
        "n_features_psi_ge_0.1": "Number of features with PSI >= 0.1",
        "importance_weighted_psi": "Importance-weighted feature PSI",
        "share_non_refund": "Share of Non Refund bookings (not a drift metric)",
    }
    rows = []
    for k, label in names.items():
        mo = d["month"]["correlation_with_auc_drop"][k]
        wk = d["week"]["correlation_with_auc_drop"][k]
        rows.append(
            [
                label,
                f"{mo['spearman_rho']:+.2f} (p {mo['p_value']:.2f})",
                f"{wk['spearman_rho']:+.2f} (p {wk['p_value']:.2f})",
            ]
        )
    corr = table(
        [
            "Signal",
            f"Monthly, n={d['month']['n_windows']}",
            f"Weekly, n={d['week']['n_windows']}",
        ],
        rows,
    )
    alarm_rows = [
        [
            a["rule"],
            a["alarms"],
            a["true_alarms"],
            "n/a" if a["precision"] is None else f"{a['precision']:.2f}",
            "n/a" if a["recall"] is None else f"{a['recall']:.2f}",
        ]
        for a in d["week"]["alarm_rules"]
    ]
    a0 = d["week"]["alarm_rules"][0]
    alarms = table(
        ["Alarm rule (weekly)", "Alarms", "On degraded weeks", "Precision", "Recall"], alarm_rows
    )
    nd = d["month"]["auc_no_deposit_only"]
    nd_rows = [["validation", f"{nd['valid']:.3f}"]] + [
        [k, f"{v:.3f}"] for k, v in nd["by_month"].items()
    ]
    return (
        "Spearman correlation between each signal and the ROC AUC drop "
        "(validation AUC minus window AUC). Positive means more drift goes with a bigger drop.\n\n"
        + corr
        + f"\n\nDegraded week: {d['degraded_definition']}. "
        f"{a0['degraded_windows']} of {a0['windows']} weeks are degraded, so a rule that fires "
        f"at random has precision about {a0['degraded_windows'] / a0['windows']:.2f}.\n\n"
        + alarms
        + "\n\nROC AUC on No Deposit bookings only:\n\n"
        + table(["Window", "ROC AUC, No Deposit only"], nd_rows)
        + "\n\nSource: `scripts/drift_vs_performance.py`, file "
        "`reports/drift_vs_performance.json`."
    )


def block_simulation() -> str:
    s = load("detector_simulation.json")
    res = s["results"]
    out = []
    # False alarms (no shift) per feature and n.
    fa = [r for r in res if r["size"] == 0]
    feats = list(dict.fromkeys(r["feature"] for r in fa))
    ns = sorted({r["n"] for r in fa})
    rows = []
    for f in feats:
        for n in ns:
            r = next(x for x in fa if x["feature"] == f and x["n"] == n)
            rows.append([f, n, pct(r["test p<0.05"]), pct(r["psi>=0.1"]), pct(r["psi>=0.2"])])
    out.append("**False alarm rate with no shift (single feature).**\n")
    out.append(table(["Feature", "n", "KS or chi2 p<0.05", "PSI>=0.1", "PSI>=0.2"], rows))

    out.append("\n\n**Detection rate by shift size, n = 1000.**\n")
    rows = []
    for f in feats:
        for r in [x for x in res if x["feature"] == f and x["n"] == 1000 and x["size"] > 0]:
            eff = (
                f"SMD {r['effect']:.2f}"
                if r["effect_unit"] == "SMD"
                else f"+{r['effect']:.2f} share ({r['shift'].split(' +')[0]})"
            )
            rows.append([f, eff, pct(r["test p<0.05"]), pct(r["psi>=0.1"]), pct(r["psi>=0.2"])])
    out.append(
        table(["Feature", "Injected shift", "KS or chi2 p<0.05", "PSI>=0.1", "PSI>=0.2"], rows)
    )

    out.append(
        f"\n\n**Any of the {s['n_monitored_features']} monitored features fires, no shift.**\n"
    )
    rows = [
        [
            n,
            pct(v["test p<0.05"]),
            pct(v["test p<0.05/k (Bonferroni)"]),
            pct(v["psi>=0.1"]),
            pct(v["psi>=0.2"]),
        ]
        for n, v in s["any_feature_false_alarm"].items()
    ]
    out.append(table(["n", "p<0.05", "p<0.05/k (Bonferroni)", "PSI>=0.1", "PSI>=0.2"], rows))

    nat = s["natural_weekly_variation_in_reference"]
    rows = [
        [
            w["week"],
            w["n"],
            w["n_features_psi_ge_0.1"],
            w["n_features_test_flagged_bonferroni"],
            f"{w['max_psi']:.2f} ({w['max_psi_feature']})",
        ]
        for w in nat
    ]
    out.append(
        "\n\n**Natural week-to-week variation inside the validation period** "
        "(each week against the rest of the period, no injected shift).\n"
    )
    out.append(
        table(
            [
                "Week",
                "Bookings",
                "Features PSI>=0.1",
                "Features failing test (Bonferroni)",
                "Max PSI",
            ],
            rows,
        )
    )
    out.append(
        f"\n\n{s['trials_per_cell']} trials per cell, seed {s['seed']}. Source: "
        "`scripts/simulate_detectors.py`, file `reports/detector_simulation.json`."
    )
    return "\n".join(out)


def block_latency() -> str:
    path = REPORTS / "latency.json"
    if not path.exists():
        return "Not measured yet. Run `make latency` with the container up."
    lat = json.loads(path.read_text())
    m = lat["machine"]
    return (
        table(
            ["Setup", "Requests", "p50 ms", "p95 ms", "p99 ms", "Sequential req/s"],
            [
                [
                    lat["setup"],
                    lat["requests"],
                    f"{lat['p50_ms']:.1f}",
                    f"{lat['p95_ms']:.1f}",
                    f"{lat['p99_ms']:.1f}",
                    f"{lat['throughput_rps_sequential']:.0f}",
                ]
            ],
        )
        + f"\n\nMachine: {m['cpu']}, {m['logical_cpus']} logical CPUs, {m['memory_gb']} GB RAM, "
        f"{m['os']}. One client, one request at a time, after {lat['warmup_requests']} warm-up "
        "requests. Time is measured at the client and includes HTTP, validation, inference, "
        "and the SQLite write. Source: `scripts/load_test.py`, file `reports/latency.json`."
    )


def block_headline() -> str:
    t = load("training_metrics.json")
    m = load("monitoring_summary.json")
    d = load("drift_vs_performance.json")
    s = load("detector_simulation.json")
    vm = t["valid_metrics"]
    sel = t["selected_model"]
    other = next(k for k in vm if k != sel)
    periods = m["periods"]
    worst = min(periods, key=lambda r: r["roc_auc"])
    best = max(periods, key=lambda r: r["roc_auc"])
    flagged = [r["n_features_test_flagged"] for r in periods]
    nf = periods[0]["n_features"]
    wk = d["week"]
    iw = wk["correlation_with_auc_drop"]["importance_weighted_psi"]
    nr = wk["correlation_with_auc_drop"]["share_non_refund"]
    rules = [a for a in wk["alarm_rules"] if a["precision"] is not None]
    best_rule = max(rules, key=lambda a: a["precision"])
    base = rules[0]["degraded_windows"] / rules[0]["windows"]
    res = s["results"]
    lt = {r["size"]: r for r in res if r["feature"] == "lead_time" and r["n"] == 1000}
    small = lt[0.1]
    fa = s["any_feature_false_alarm"]["1000"]
    sign_note = "the wrong sign for an alarm" if iw["spearman_rho"] < 0 else "a positive link"
    lines = [
        f"* Served model: {sel}, validation ROC AUC {vm[sel]['roc_auc']:.3f} "
        f"(baseline {other}: {vm[other]['roc_auc']:.3f}).",
        f"* Production ROC AUC by arrival month ranges from {worst['roc_auc']:.3f} "
        f"({worst['period']}) to {best['roc_auc']:.3f} ({best['period']}). "
        f"Validation: {m['reference']['roc_auc']:.3f}.",
        f"* Every production month has {min(flagged)} to {max(flagged)} of {nf} features "
        "failing a KS or chi-square test, even with a Bonferroni correction. "
        "Significance alone says little at this sample size.",
        f"* Feature drift does not predict the AUC drop. Weekly Spearman correlation between "
        f"importance-weighted PSI and AUC drop: {iw['spearman_rho']:+.2f} "
        f"(p {iw['p_value']:.2f}), {sign_note}. The best alarm rule "
        f'("{best_rule["rule"]}") has precision {best_rule["precision"]:.2f}, against '
        f"{base:.2f} for random alarms.",
        f"* The share of Non Refund bookings tracks the drop much better: weekly Spearman "
        f"{nr['spearman_rho']:+.2f} (p {nr['p_value']:.2f}). Fewer easy cases means lower AUC.",
        f"* Detector check, n = 1000 rows: a lead_time shift of SMD {small['effect']:.2f} is "
        f"caught by KS in {pct(small['test p<0.05'])} of trials and by PSI >= 0.1 in "
        f"{pct(small['psi>=0.1'])}. With no shift, at least one of "
        f"{s['n_monitored_features']} features fails p < 0.05 in {pct(fa['test p<0.05'])} "
        f"of trials ({pct(fa['test p<0.05/k (Bonferroni)'])} with Bonferroni).",
    ]
    lat_path = REPORTS / "latency.json"
    if lat_path.exists():
        lat = json.loads(lat_path.read_text())
        lines.append(
            f"* Single prediction latency in Docker: p50 {lat['p50_ms']:.1f} ms, "
            f"p95 {lat['p95_ms']:.1f} ms ({lat['machine']['cpu']}, "
            f"{lat['machine']['logical_cpus']} vCPUs)."
        )
    return "\n".join(lines)


BLOCKS = {
    "headline": block_headline,
    "data": block_data,
    "training": block_training,
    "tuning": block_tuning,
    "monitoring": block_monitoring,
    "drift_perf": block_drift_perf,
    "simulation": block_simulation,
    "latency": block_latency,
}


def render(text: str) -> str:
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README is missing the {name} block markers")
        body = fn()
        text = pat.sub(lambda m, body=body: m.group(1) + body + m.group(2), text)
    return text


def dash_problems() -> list[str]:
    files = [README, *sorted((ROOT / "reports").rglob("*.md")), *sorted(ROOT.glob("docs/**/*.md"))]
    bad = []
    for f in files:
        txt = f.read_text()
        for ch, name in FORBIDDEN.items():
            if ch in txt:
                bad.append(f"{f.relative_to(ROOT)} contains an {name}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    current = README.read_text()
    new = render(current)
    if args.check:
        problems = dash_problems()
        if new != current:
            problems.append("README.md is out of date: run python scripts/render_readme.py")
        if problems:
            print("\n".join(problems))
            sys.exit(1)
        print("README is in sync with reports/ and has no em or en dashes")
        return
    README.write_text(new)
    print("README.md updated")


if __name__ == "__main__":
    main()
