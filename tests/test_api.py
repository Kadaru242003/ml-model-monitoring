import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hotelmon import artifacts, config
from hotelmon.api import create_app
from hotelmon.logdb import PredictionLog


@pytest.fixture
def client(model_dir, tmp_path):
    db = tmp_path / "pred.db"
    with TestClient(create_app(model_dir=model_dir, db_path=db)) as c:
        c.db_path = db
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test-0001"}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "test-0001"
    assert body["decision_threshold"] == 0.5


def test_predict_single(client, booking):
    r = client.post("/predict", json=booking)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction_id", "probability", "will_cancel", "model_version"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["will_cancel"] == (body["probability"] >= 0.5)


def test_predict_batch_matches_offline_model(client, booking, model_dir):
    b2 = dict(booking, lead_time=300, deposit_type="Non Refund")
    r = client.post("/predict", json={"instances": [booking, b2]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["prediction_id"] < preds[1]["prediction_id"]

    model = artifacts.load(model_dir).model
    frame = pd.DataFrame([booking, b2])
    frame[config.TIME_COL] = pd.to_datetime(frame[config.TIME_COL])
    for col, default in [
        ("children", 0),
        ("babies", 0),
        ("is_repeated_guest", 0),
        ("previous_cancellations", 0),
        ("previous_bookings_not_canceled", 0),
        ("days_in_waiting_list", 0),
        ("total_of_special_requests", 0),
        ("company", "NULL"),
    ]:
        frame[col] = default
    offline = model.predict_proba(frame)[:, 1]
    np.testing.assert_allclose([p["probability"] for p in preds], offline, rtol=1e-9)


def _locs(r):
    return {tuple(e["loc"][-1:]) + (e["type"],) for e in r.json()["detail"]}


def test_missing_field_is_422_with_location(client, booking):
    bad = dict(booking)
    bad.pop("adults")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert ("adults", "missing") in _locs(r)


@pytest.mark.parametrize(
    "field,value,err",
    [
        ("lead_time", -1, "greater_than_equal"),
        ("lead_time", 5000, "less_than_equal"),
        ("adr", 20000.0, "less_than_equal"),
        ("hotel", "Motel", "literal_error"),
        ("country", "portugal", "string_pattern_mismatch"),
        ("arrival_date", "2016-13-01", "date_from_datetime_parsing"),
    ],
)
def test_out_of_range_is_422(client, booking, field, value, err):
    r = client.post("/predict", json=dict(booking, **{field: value}))
    assert r.status_code == 422
    assert (field, err) in _locs(r)


def test_zero_guests_rejected(client, booking):
    r = client.post("/predict", json=dict(booking, adults=0, children=0, babies=0))
    assert r.status_code == 422
    assert "at least one guest" in r.text


def test_extra_field_rejected(client, booking):
    r = client.post("/predict", json=dict(booking, is_canceled=1))
    assert r.status_code == 422
    assert ("is_canceled", "extra_forbidden") in _locs(r)


def test_batch_errors_point_to_the_bad_row(client, booking):
    r = client.post("/predict", json={"instances": [booking, dict(booking, lead_time=-5)]})
    assert r.status_code == 422
    locs = [e["loc"] for e in r.json()["detail"]]
    assert ["body", "batch", "instances", 1, "lead_time"] in locs


def test_batch_size_limits(client, booking):
    assert client.post("/predict", json={"instances": []}).status_code == 422
    assert client.post("/predict", json={"instances": [booking] * 1001}).status_code == 422


def test_every_prediction_is_logged(client, booking):
    client.post("/predict", json=booking)
    client.post("/predict", json={"instances": [booking, booking, booking]})
    log = PredictionLog(client.db_path)
    df = log.read()
    log.close()
    assert len(df) == 4
    assert (df["model_version"] == "test-0001").all()
    assert df["request_id"].nunique() == 2
    assert df["ts_utc"].str.endswith("+00:00").all()
    assert (df["lead_time"] == booking["lead_time"]).all()
    assert (df["arrival_date"] == booking["arrival_date"]).all()
    assert df["probability"].between(0, 1).all()


def test_labels_endpoint(client, booking):
    ids = [
        p["prediction_id"]
        for p in client.post("/predict", json={"instances": [booking, booking]}).json()[
            "predictions"
        ]
    ]
    r = client.post(
        "/labels",
        json={
            "labels": [{"prediction_id": ids[0], "label": 1}, {"prediction_id": 999999, "label": 0}]
        },
    )
    assert r.status_code == 200
    assert r.json() == {"stored": 1, "unknown_prediction_ids": [999999]}
    assert (
        client.post("/labels", json={"labels": [{"prediction_id": 999999, "label": 0}]}).status_code
        == 404
    )
    assert (
        client.post("/labels", json={"labels": [{"prediction_id": ids[1], "label": 2}]}).status_code
        == 422
    )
