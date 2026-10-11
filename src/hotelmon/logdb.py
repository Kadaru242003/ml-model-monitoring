"""SQLite log of every prediction, plus labels that arrive later."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    ts_utc TEXT NOT NULL,
    model_version TEXT NOT NULL,
    arrival_date TEXT NOT NULL,
    inputs TEXT NOT NULL,
    probability REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_arrival ON predictions(arrival_date);
CREATE TABLE IF NOT EXISTS labels (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id),
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    received_ts_utc TEXT NOT NULL
);
"""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds")


class PredictionLog:
    """Thread-safe append-only log. One row per scored booking."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def log_predictions(
        self,
        request_id: str,
        model_version: str,
        inputs: list[dict],
        probabilities: list[float],
    ) -> list[int]:
        ts = _now()
        ids = []
        with self._lock:
            cur = self._conn.cursor()
            for rec, p in zip(inputs, probabilities, strict=True):
                cur.execute(
                    "INSERT INTO predictions "
                    "(request_id, ts_utc, model_version, arrival_date, inputs, probability) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        request_id,
                        ts,
                        model_version,
                        str(rec["arrival_date"]),
                        json.dumps(rec, default=str, sort_keys=True),
                        float(p),
                    ),
                )
                ids.append(int(cur.lastrowid))
            self._conn.commit()
        return ids

    def log_labels(self, labels: list[tuple[int, int]]) -> tuple[int, list[int]]:
        """Store (prediction_id, label) pairs. Returns (stored count, unknown ids)."""
        ts = _now()
        with self._lock:
            ids = [pid for pid, _ in labels]
            known = set()
            for i in range(0, len(ids), 900):
                chunk = ids[i : i + 900]
                q = f"SELECT id FROM predictions WHERE id IN ({','.join('?' * len(chunk))})"
                known.update(r[0] for r in self._conn.execute(q, chunk))
            rows = [(pid, lab, ts) for pid, lab in labels if pid in known]
            self._conn.executemany(
                "INSERT OR REPLACE INTO labels (prediction_id, label, received_ts_utc) "
                "VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows), [pid for pid in ids if pid not in known]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def read(self) -> pd.DataFrame:
        """All predictions with inputs expanded to columns and labels joined (may be NaN)."""
        with self._lock:
            df = pd.read_sql_query(
                "SELECT p.id AS prediction_id, p.request_id, p.ts_utc, p.model_version, "
                "p.arrival_date, p.inputs, p.probability, l.label "
                "FROM predictions p LEFT JOIN labels l ON l.prediction_id = p.id ORDER BY p.id",
                self._conn,
            )
        if df.empty:
            return df
        inputs = pd.DataFrame([json.loads(s) for s in df.pop("inputs")])
        inputs = inputs.drop(columns=["arrival_date"])
        return pd.concat([df, inputs], axis=1)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
