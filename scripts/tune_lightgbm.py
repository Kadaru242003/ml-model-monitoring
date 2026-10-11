"""Small LightGBM search on an inner time split of the training period only.

Fit on arrivals Jul 2015 to Mar 2016, score arrivals Apr to Jun 2016.
The validation and production periods are never touched here.

Usage: python scripts/tune_lightgbm.py
Writes: reports/tuning_lightgbm.json
"""

from __future__ import annotations

import itertools
import json

from sklearn.metrics import brier_score_loss, roc_auc_score

from hotelmon import config, data, features

INNER_CUTOFF = "2016-04-01"
GRID = {
    "max_cat_to_onehot": [4, 64],  # 4 is the LightGBM default (native category splits)
    "n_estimators": [200, 400],
    "num_leaves": [15, 31],
}


def main() -> None:
    train = data.split(data.load_clean(data.download()))["train"]
    fit_part = train[train[config.TIME_COL] < INNER_CUTOFF]
    eval_part = train[train[config.TIME_COL] >= INNER_CUTOFF]
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values, strict=True))
        pipe = features.make_lightgbm(**params)
        features.fit(pipe, fit_part, fit_part[config.TARGET])
        p = pipe.predict_proba(eval_part)[:, 1]
        y = eval_part[config.TARGET]
        rows.append(params | {"roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p)})
        print(rows[-1])
    best = max(rows, key=lambda r: r["roc_auc"])
    out = {
        "fit_rows": len(fit_part),
        "eval_rows": len(eval_part),
        "inner_cutoff": INNER_CUTOFF,
        "results": rows,
        "best": best,
        "selected_in_code": features.LIGHTGBM_PARAMS,
    }
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "tuning_lightgbm.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
