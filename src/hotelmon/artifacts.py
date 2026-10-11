"""Save and load versioned model artifacts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hotelmon import config

MODEL_FILE = "model.joblib"
META_FILE = "metadata.json"
REFERENCE_FILE = "reference.csv.gz"
LATEST_FILE = "LATEST"


def git_commit(root: Path = config.ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class ModelArtifact:
    model: Any
    metadata: dict
    reference: pd.DataFrame | None
    path: Path

    @property
    def version(self) -> str:
        return self.metadata["version"]


def save(
    model: Any,
    metadata: dict,
    reference: pd.DataFrame | None = None,
    models_dir: Path = config.MODELS_DIR,
) -> Path:
    version = metadata["version"]
    out = models_dir / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILE)
    (out / META_FILE).write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    if reference is not None:
        reference.to_csv(out / REFERENCE_FILE, index=False)
    (models_dir / LATEST_FILE).write_text(version + "\n")
    return out


def resolve(models_dir: Path = config.MODELS_DIR, version: str | None = None) -> Path:
    if version is None:
        version = (models_dir / LATEST_FILE).read_text().strip()
    return models_dir / version


def load(path: Path, with_reference: bool = False) -> ModelArtifact:
    model = joblib.load(path / MODEL_FILE)
    metadata = json.loads((path / META_FILE).read_text())
    ref = None
    if with_reference and (path / REFERENCE_FILE).exists():
        ref = pd.read_csv(
            path / REFERENCE_FILE,
            keep_default_na=False,
            dtype={"agent": str, "company": str, "country": str},
        )
    return ModelArtifact(model=model, metadata=metadata, reference=ref, path=path)
