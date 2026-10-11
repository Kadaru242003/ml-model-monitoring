"""Measure single-prediction latency against a running server.

Sends real validation-period bookings one at a time (sequential, one client),
after a warm-up. Latency is client-side wall time per request, including HTTP,
validation, model inference, and the SQLite write.

Usage: python scripts/load_test.py --url http://localhost:8000 [--n 2000] [--label docker]
Writes: reports/latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import httpx2
import numpy as np

from hotelmon import config, data
from hotelmon.schema import INPUT_FIELDS


def machine_info() -> dict:
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            mem_gb = round(int(f.readline().split()[1]) / 1024**2, 1)
    except (OSError, ValueError):
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "memory_gb": mem_gb,
        "os": f"{platform.system()} {platform.release()}",
        "python_client": platform.python_version(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--label", default="docker")
    ap.add_argument("--out", default=str(config.REPORTS_DIR / "latency.json"))
    args = ap.parse_args()

    valid = data.split(data.load_clean(data.download()))["valid"]
    rows = valid[INPUT_FIELDS].copy()
    rows[config.TIME_COL] = rows[config.TIME_COL].dt.strftime("%Y-%m-%d")
    payloads = rows.sample(n=args.n + args.warmup, replace=True, random_state=0).to_dict("records")

    lat_ms = []
    with httpx2.Client(base_url=args.url, timeout=30) as c:
        c.get("/health").raise_for_status()
        for i, body in enumerate(payloads):
            t0 = time.perf_counter()
            r = c.post("/predict", json=body)
            dt_ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            if i >= args.warmup:
                lat_ms.append(dt_ms)
        version = c.get("/health").json()["model_version"]

    a = np.array(lat_ms)
    result = {
        "setup": args.label,
        "url": args.url,
        "model_version": version,
        "requests": len(a),
        "warmup_requests": args.warmup,
        "concurrency": 1,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "throughput_rps_sequential": float(1000 / a.mean()),
        "machine": machine_info(),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
