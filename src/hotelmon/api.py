"""FastAPI service: /predict, /labels, /health, /model-info."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import Discriminator, Tag

from hotelmon import artifacts, config
from hotelmon.logdb import PredictionLog
from hotelmon.schema import (
    BatchRequest,
    BatchResponse,
    Booking,
    LabelBatch,
    Prediction,
    SingleResponse,
)


def _request_kind(v: Any) -> str:
    return "batch" if isinstance(v, dict) and "instances" in v else "single"


PredictBody = Annotated[
    Annotated[Booking, Tag("single")] | Annotated[BatchRequest, Tag("batch")],
    Discriminator(_request_kind),
]


def create_app(model_dir: str | Path | None = None, db_path: str | Path | None = None) -> FastAPI:
    if model_dir is None and os.environ.get("MODEL_DIR"):
        model_dir = os.environ["MODEL_DIR"]
    if model_dir is None:
        models_root = Path(os.environ.get("MODELS_DIR", config.MODELS_DIR))
        model_dir = artifacts.resolve(models_root)
    model_dir = Path(model_dir)
    db_path = db_path or os.environ.get("PREDICTION_DB") or (config.DATA_DIR / "predictions.db")
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["artifact"] = artifacts.load(model_dir)
        state["log"] = PredictionLog(db_path)
        yield
        state["log"].close()

    app = FastAPI(
        title="Hotel cancellation model",
        description="Predicts the probability that a hotel booking will be canceled.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def _score(bookings: list[Booking]) -> tuple[str, list[Prediction]]:
        art = state["artifact"]
        records = [b.model_dump() for b in bookings]
        frame = pd.DataFrame(records)
        frame[config.TIME_COL] = pd.to_datetime(frame[config.TIME_COL])
        probs = art.model.predict_proba(frame)[:, 1].tolist()
        ids = state["log"].log_predictions(uuid.uuid4().hex, art.version, records, probs)
        threshold = art.metadata.get("decision_threshold", 0.5)
        preds = [
            Prediction(prediction_id=i, probability=p, will_cancel=p >= threshold)
            for i, p in zip(ids, probs, strict=True)
        ]
        return art.version, preds

    @app.post("/predict", response_model=SingleResponse | BatchResponse)
    def predict(body: Annotated[PredictBody, Body()]) -> SingleResponse | BatchResponse:
        """Score one booking (send the booking object) or many ({"instances": [...]})."""
        if isinstance(body, BatchRequest):
            version, preds = _score(body.instances)
            return BatchResponse(model_version=version, predictions=preds)
        version, preds = _score([body])
        return SingleResponse(model_version=version, **preds[0].model_dump())

    @app.post("/labels")
    def labels(body: LabelBatch) -> dict:
        """Attach observed outcomes (1 = canceled) to earlier predictions."""
        stored, unknown = state["log"].log_labels([(x.prediction_id, x.label) for x in body.labels])
        if unknown and not stored:
            raise HTTPException(status_code=404, detail={"unknown_prediction_ids": unknown[:50]})
        return {"stored": stored, "unknown_prediction_ids": unknown[:50]}

    @app.get("/health")
    def health() -> dict:
        art = state.get("artifact")
        return {"status": "ok" if art is not None else "loading", "model_version": art.version}

    @app.get("/model-info")
    def model_info() -> dict:
        md = state["artifact"].metadata
        keys = [
            "version",
            "model_type",
            "trained_at_utc",
            "git_commit",
            "data",
            "train_period",
            "valid_period",
            "input_fields",
            "model_features",
            "excluded_leaky_columns",
            "decision_threshold",
        ]
        info = {k: md.get(k) for k in keys}
        info["metrics_valid"] = {
            name: {k: v for k, v in m.items() if k != "reliability"}
            for name, m in md.get("metrics_valid", {}).items()
        }
        return info

    return app


def app_from_env() -> FastAPI:
    """Entry point for uvicorn --factory."""
    return create_app()
