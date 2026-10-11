"""Download, verify, clean, and split the hotel bookings data."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from hotelmon import config
from hotelmon.schema import INPUT_FIELDS, NUMERIC_RANGES

log = logging.getLogger(__name__)

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path = config.RAW_FILE, url: str = config.DATA_URL) -> Path:
    """Download the raw CSV if missing and verify its SHA-256 checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        log.info("downloading %s", url)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    digest = sha256_of(dest)
    if digest != config.DATA_SHA256:
        raise ValueError(f"checksum mismatch for {dest}: {digest} != {config.DATA_SHA256}")
    return dest


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    # "NULL" is a real category in agent/company/country; only "NA" means missing.
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=["NA"],
        dtype={"agent": str, "company": str, "country": str},
    )


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the modelling table: one row per booking, sorted by arrival date.

    Drops leaky columns and rows that the API schema would reject, so that every
    row can later be replayed through the service.
    """
    df = raw.copy()
    month_num = df["arrival_date_month"].map({m: i for i, m in enumerate(MONTHS, 1)})
    df[config.TIME_COL] = pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": month_num,
            "day": df["arrival_date_day_of_month"],
        }
    )
    df["children"] = df["children"].fillna(0).astype(int)
    for col in ("agent", "company", "country"):
        df[col] = df[col].astype(str).str.strip().replace({"": "NULL"})

    keep = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in NUMERIC_RANGES.items():
        keep &= df[col].between(lo, hi).to_numpy()
    keep &= (df["adults"] + df["children"] + df["babies"] >= 1).to_numpy()
    n_drop = int((~keep).sum())
    if n_drop:
        log.info("dropping %d rows outside the input schema", n_drop)
    df = df.loc[keep]

    df = df[INPUT_FIELDS + [config.TARGET]]
    df = df.sort_values(config.TIME_COL, kind="stable").reset_index(drop=True)
    return df


def _between(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    t = df[config.TIME_COL]
    return df.loc[(t >= start) & (t < end)].reset_index(drop=True)


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Time-based split into train, validation, and production."""
    return {
        "train": _between(df, config.TRAIN_START, config.TRAIN_END),
        "valid": _between(df, config.VALID_START, config.VALID_END),
        "prod": _between(df, config.PROD_START, config.PROD_END),
    }


def load_clean(path: Path = config.RAW_FILE) -> pd.DataFrame:
    return clean(load_raw(path))
