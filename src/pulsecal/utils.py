"""Configuration, deterministic seeding and result serialisation.

Every script reads the same YAML file and derives its randomness from one
master seed via ``SeedSequence``, so any figure can be regenerated in
isolation and still be bit-identical to the one produced by a full run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "transmon.yaml"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_config(path: Path | str = CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_device(cfg: dict):
    from .device import Transmon
    return Transmon(**cfg["device"])


def default_pulse(cfg: dict):
    """Nominal pi pulse: duration and sigma from config, amplitude uncalibrated."""
    from .pulses import PulseSpec
    p = cfg["pulse"]
    return PulseSpec(duration=p["duration"], sigma=p["sigma_ratio"] * p["duration"],
                     amp=p["amp_guess"])


def rng(cfg: dict, stream: str) -> np.random.Generator:
    """Independent generator per named experiment, reproducible from one seed."""
    entropy = [cfg["measurement"]["seed"], int.from_bytes(stream.encode(), "little")]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _plain(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    raise TypeError(f"unserialisable: {type(obj)}")


def save(name: str, payload: dict) -> Path:
    """Write a result record to ``results/<name>.json`` and return the path."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_plain))
    return path


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())
