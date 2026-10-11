"""Project-wide constants: data source, time splits, feature lists, paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Hotel Booking Demand (Antonio, Almeida and Nunes, 2019, Data in Brief, CC BY 4.0),
# as mirrored by the TidyTuesday project on GitHub.
DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-02-11/hotels.csv"
)
DATA_SHA256 = "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
RAW_FILE = DATA_DIR / "hotels.csv"

TARGET = "is_canceled"
TIME_COL = "arrival_date"

# Time-based split on arrival date (inclusive start, exclusive end).
TRAIN_START, TRAIN_END = "2015-07-01", "2016-07-01"  # 12 months, one full season cycle
VALID_START, VALID_END = "2016-07-01", "2016-10-01"  # 3 months
PROD_START, PROD_END = "2016-10-01", "2017-09-01"  # 11 monthly "production" periods

NUMERIC_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_month",  # derived from arrival_date
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
    "company",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns in the raw file that are recorded after the booking is made, so they
# leak the outcome. They are never used as features.
LEAKY_COLUMNS = [
    "reservation_status",  # is the label in another form
    "reservation_status_date",  # date of cancellation or check-out
    "assigned_room_type",  # set at check-in
    "booking_changes",  # counted until check-in or cancellation
    "required_car_parking_spaces",  # only recorded for guests who arrive
]

RANDOM_SEED = 42
