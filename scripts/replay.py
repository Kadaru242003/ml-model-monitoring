"""Replay the production periods (Oct 2016 to Aug 2017) through the prediction service.

Bookings are sent in arrival order, in batches, to POST /predict. Labels are then
sent to POST /labels, as if the hotel reported outcomes after each arrival date.
By default the app runs in-process (FastAPI TestClient, same code path as the
container). Pass --url to replay against a running server instead.

Usage: python scripts/replay.py [--url http://localhost:8000] [--db data/replay.db]
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from hotelmon import config, data
from hotelmon.schema import INPUT_FIELDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
log = logging.getLogger("replay")


def to_payload(df: pd.DataFrame) -> list[dict]:
    out = df[INPUT_FIELDS].copy()
    out[config.TIME_COL] = out[config.TIME_COL].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@contextmanager
def client_for(url: str | None, db: Path):
    if url:
        import httpx2

        with httpx2.Client(base_url=url, timeout=60) as c:
            yield c
    else:
        from fastapi.testclient import TestClient

        from hotelmon.api import create_app

        with TestClient(create_app(db_path=db)) as c:
            yield c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--db", type=Path, default=config.DATA_DIR / "replay.db")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="replay only the first N rows")
    args = ap.parse_args()

    if not args.url and args.db.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(args.db) + suffix).unlink(missing_ok=True)

    prod = data.split(data.load_clean(data.download()))["prod"]
    if args.limit:
        prod = prod.iloc[: args.limit]
    t0 = time.perf_counter()
    labels = []
    with client_for(args.url, args.db) as c:
        for start in range(0, len(prod), args.batch_size):
            chunk = prod.iloc[start : start + args.batch_size]
            r = c.post("/predict", json={"instances": to_payload(chunk)})
            r.raise_for_status()
            ids = [p["prediction_id"] for p in r.json()["predictions"]]
            labels += [
                {"prediction_id": i, "label": int(y)}
                for i, y in zip(ids, chunk[config.TARGET], strict=True)
            ]
        # Outcomes arrive later. Send them in chunks, as a label feed would.
        for start in range(0, len(labels), 5000):
            r = c.post("/labels", json={"labels": labels[start : start + 5000]})
            r.raise_for_status()
    log.info(
        "replayed %d bookings in %.1fs; labels sent for %d",
        len(prod),
        time.perf_counter() - t0,
        len(labels),
    )


if __name__ == "__main__":
    main()
