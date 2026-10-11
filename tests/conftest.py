from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hotelmon import artifacts, config, features
from hotelmon.data import MONTHS


def make_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows in the raw CSV layout, with a learnable cancellation signal."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2015-07-01") + pd.to_timedelta(rng.integers(0, 790, n), unit="D")
    deposit = rng.choice(["No Deposit", "Non Refund", "Refundable"], n, p=[0.85, 0.13, 0.02])
    lead = rng.integers(0, 400, n)
    logit = -1.5 + 0.006 * lead + 3.0 * (deposit == "Non Refund")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
            "is_canceled": y,
            "lead_time": lead,
            "arrival_date_year": dates.year,
            "arrival_date_month": [MONTHS[m - 1] for m in dates.month],
            "arrival_date_week_number": dates.isocalendar().week.to_numpy(),
            "arrival_date_day_of_month": dates.day,
            "stays_in_weekend_nights": rng.integers(0, 3, n),
            "stays_in_week_nights": rng.integers(0, 5, n),
            "adults": rng.integers(1, 3, n),
            "children": rng.choice([0.0, 1.0, np.nan], n, p=[0.9, 0.09, 0.01]),
            "babies": 0,
            "meal": rng.choice(["BB", "HB", "SC"], n),
            "country": rng.choice(["PRT", "GBR", "FRA", "NULL"], n),
            "market_segment": rng.choice(["Online TA", "Groups", "Direct"], n),
            "distribution_channel": rng.choice(["TA/TO", "Direct"], n),
            "is_repeated_guest": rng.integers(0, 2, n),
            "previous_cancellations": rng.integers(0, 2, n),
            "previous_bookings_not_canceled": rng.integers(0, 2, n),
            "reserved_room_type": rng.choice(["A", "D", "E"], n),
            "assigned_room_type": "A",
            "booking_changes": 0,
            "deposit_type": deposit,
            "agent": rng.choice(["9", "240", "NULL"], n),
            "company": rng.choice(["NULL", "40"], n, p=[0.95, 0.05]),
            "days_in_waiting_list": 0,
            "customer_type": rng.choice(["Transient", "Transient-Party"], n),
            "adr": rng.uniform(40, 200, n).round(2),
            "required_car_parking_spaces": 0,
            "total_of_special_requests": rng.integers(0, 3, n),
            "reservation_status": np.where(y == 1, "Canceled", "Check-Out"),
            "reservation_status_date": "2016-01-01",
        }
    )
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    from hotelmon.data import clean

    return clean(raw_df)


@pytest.fixture
def model_dir(tmp_path, clean_df):
    """A small trained artifact saved to a temp directory."""
    pipe = features.make_lightgbm(n_estimators=20)
    features.fit(pipe, clean_df, clean_df[config.TARGET])
    ref = clean_df.copy()
    ref["score"] = pipe.predict_proba(clean_df)[:, 1]
    ref[config.TIME_COL] = ref[config.TIME_COL].dt.strftime("%Y-%m-%d")
    meta = {"version": "test-0001", "model_type": "lightgbm", "decision_threshold": 0.5}
    return artifacts.save(pipe, meta, ref, models_dir=tmp_path / "models")


@pytest.fixture
def booking() -> dict:
    return {
        "hotel": "City Hotel",
        "arrival_date": "2016-10-03",
        "lead_time": 50,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "reserved_room_type": "A",
        "deposit_type": "No Deposit",
        "agent": "9",
        "customer_type": "Transient",
        "adr": 95.0,
    }
