import hashlib

import pandas as pd
import pytest

from hotelmon import config, data
from hotelmon.schema import INPUT_FIELDS


def test_clean_drops_leaky_columns_and_keeps_schema(clean_df):
    assert list(clean_df.columns) == INPUT_FIELDS + [config.TARGET]
    for col in config.LEAKY_COLUMNS:
        assert col not in clean_df.columns


def test_clean_builds_arrival_date_and_sorts(raw_df, clean_df):
    assert clean_df[config.TIME_COL].is_monotonic_increasing
    row = raw_df.iloc[0]
    month = data.MONTHS.index(row["arrival_date_month"]) + 1
    expected = pd.Timestamp(
        int(row["arrival_date_year"]), month, int(row["arrival_date_day_of_month"])
    )
    assert expected in set(clean_df[config.TIME_COL])


def test_clean_fills_children_and_keeps_null_strings(clean_df):
    assert clean_df["children"].isna().sum() == 0
    assert "NULL" in set(clean_df["agent"])
    assert clean_df["agent"].map(type).eq(str).all()


def test_clean_drops_rows_outside_schema(raw_df):
    bad = raw_df.copy()
    bad.loc[0, ["adults", "children", "babies"]] = [0, 0, 0]
    bad.loc[1, "adr"] = -6.38
    bad.loc[2, "adr"] = 5400.0
    out = data.clean(bad)
    assert len(out) == len(raw_df) - 3


def test_split_is_time_ordered_and_disjoint(clean_df):
    parts = data.split(clean_df)
    t = config.TIME_COL
    assert parts["train"][t].max() < pd.Timestamp(config.TRAIN_END)
    assert parts["valid"][t].min() >= pd.Timestamp(config.VALID_START)
    assert parts["valid"][t].max() < pd.Timestamp(config.VALID_END)
    assert parts["prod"][t].min() >= pd.Timestamp(config.PROD_START)
    assert parts["train"][t].max() < parts["valid"][t].min() < parts["prod"][t].min()
    assert sum(len(p) for p in parts.values()) <= len(clean_df)


def test_sha256_of(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    assert data.sha256_of(f) == hashlib.sha256(b"hello").hexdigest()


def test_download_rejects_bad_checksum(tmp_path):
    src = tmp_path / "src.csv"
    src.write_text("not the real data\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.download(dest=tmp_path / "out.csv", url=src.as_uri())


def test_load_raw_roundtrip_keeps_null_category(tmp_path, raw_df):
    p = tmp_path / "raw.csv"
    raw_df.to_csv(p, index=False, na_rep="NA")  # the real file writes missing as NA
    back = data.load_raw(p)
    assert (back["agent"] == "NULL").sum() == (raw_df["agent"] == "NULL").sum()
    assert back["children"].isna().sum() == raw_df["children"].isna().sum()
