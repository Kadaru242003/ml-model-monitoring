"""Input schema shared by the API and the data cleaning step.

Ranges are deliberately a bit wider than what the raw data shows, so that
plausible new bookings pass while typos and unit errors are rejected.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "lead_time": (0, 800),
    "stays_in_weekend_nights": (0, 30),
    "stays_in_week_nights": (0, 60),
    "adults": (0, 60),
    "children": (0, 10),
    "babies": (0, 10),
    "is_repeated_guest": (0, 1),
    "previous_cancellations": (0, 30),
    "previous_bookings_not_canceled": (0, 100),
    "days_in_waiting_list": (0, 400),
    "adr": (0, 1000),
    "total_of_special_requests": (0, 5),
}

Hotel = Literal["City Hotel", "Resort Hotel"]
Meal = Literal["BB", "HB", "FB", "SC", "Undefined"]
MarketSegment = Literal[
    "Online TA",
    "Offline TA/TO",
    "Groups",
    "Direct",
    "Corporate",
    "Complementary",
    "Aviation",
    "Undefined",
]
DistributionChannel = Literal["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
DepositType = Literal["No Deposit", "Non Refund", "Refundable"]
CustomerType = Literal["Transient", "Transient-Party", "Contract", "Group"]


def _rng(name: str) -> dict[str, float]:
    lo, hi = NUMERIC_RANGES[name]
    return {"ge": lo, "le": hi}


class Booking(BaseModel):
    """One hotel booking, as known at booking time."""

    model_config = ConfigDict(extra="forbid")

    hotel: Hotel
    arrival_date: dt.date
    lead_time: int = Field(..., **_rng("lead_time"), description="Days from booking to arrival")
    stays_in_weekend_nights: int = Field(..., **_rng("stays_in_weekend_nights"))
    stays_in_week_nights: int = Field(..., **_rng("stays_in_week_nights"))
    adults: int = Field(..., **_rng("adults"))
    children: int = Field(0, **_rng("children"))
    babies: int = Field(0, **_rng("babies"))
    meal: Meal
    country: str = Field(
        "NULL", pattern=r"^([A-Z]{2,3}|NULL)$", description="ISO 3166 alpha-3 code, or NULL"
    )
    market_segment: MarketSegment
    distribution_channel: DistributionChannel
    is_repeated_guest: int = Field(0, **_rng("is_repeated_guest"))
    previous_cancellations: int = Field(0, **_rng("previous_cancellations"))
    previous_bookings_not_canceled: int = Field(0, **_rng("previous_bookings_not_canceled"))
    reserved_room_type: str = Field(..., pattern=r"^[A-Z]$")
    deposit_type: DepositType
    agent: str = Field("NULL", pattern=r"^(\d{1,4}|NULL)$", description="Agent ID or NULL")
    company: str = Field("NULL", pattern=r"^(\d{1,4}|NULL)$", description="Company ID or NULL")
    days_in_waiting_list: int = Field(0, **_rng("days_in_waiting_list"))
    customer_type: CustomerType
    adr: float = Field(..., **_rng("adr"), description="Average daily rate in EUR")
    total_of_special_requests: int = Field(0, **_rng("total_of_special_requests"))

    @model_validator(mode="after")
    def _at_least_one_guest(self) -> Booking:
        if self.adults + self.children + self.babies < 1:
            raise ValueError("a booking needs at least one guest (adults + children + babies >= 1)")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[Booking] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    prediction_id: int
    probability: float
    will_cancel: bool


class SingleResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: int = Field(..., ge=1)
    label: int = Field(..., ge=0, le=1)


class LabelBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[LabelIn] = Field(..., min_length=1, max_length=10000)


INPUT_FIELDS = list(Booking.model_fields)
