from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel.config import load_settings  # noqa: E402
from sentinel.data import load_csv  # noqa: E402

SAMPLES = ROOT / "samples"


def sample(kind: str, i: int) -> Path:
    name = {"cold_spot": "cold_spot", "combustion_dynamics": "comb_dyn", "hets_trip": "hets_trip"}[kind]
    return SAMPLES / kind / f"{name}_{i:02d}.csv"


@lru_cache(maxsize=None)
def _cached(path: str):
    return load_csv(path)


def case_for(kind: str, i: int):
    return _cached(str(sample(kind, i)))


# Ground truth, reproduced from scripts/generate_scenario_csvs.py
def cold_truth(i: int) -> dict:
    return {
        "tc": 1 + ((i - 1) * 3) % 27,
        "depth": 18 + (i % 5) * 4,
        "width": 3 if i % 3 else 5,
        "onset_i": int(8640 * (0.35 + 0.03 * (i % 4))),
    }


def dyn_truth(i: int) -> dict:
    return {"severity": 0.8 + 0.25 * i, "period": 3 + (i % 5), "onset_i": int(14400 * (0.25 + 0.04 * (i % 5)))}


def hets_truth(i: int) -> dict:
    return {"trip_i": int(8640 * (0.55 + 0.02 * i)), "peak": 48 + 4 * i}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads TS Pulse weights or runs a fine-tune")


@pytest.fixture
def settings(tmp_path):
    """Isolated store, LLM and TS Pulse off (fast, deterministic)."""
    return load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
