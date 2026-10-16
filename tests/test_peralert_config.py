"""Per-alert mode and the constants the acceptance run is not allowed to retune.

Offline. No model weights.
"""

from __future__ import annotations

from sentinel.config import (
    COVERAGE_WARN_FRACTION,
    KNN_VOTE_MIN,
    MISSING_TAG_CONFIDENCE_CAP,
    NOVELTY_QUANTILE,
    PROFILE_SCHEMA,
    load_settings,
)
from sentinel.e2e.model import Calibration


def test_default_mode_stays_legacy(tmp_path, monkeypatch):
    """The learned path stays on the 14-channel bundle until acceptance passes."""
    monkeypatch.delenv("SENTINEL_E2E_MODE", raising=False)
    bare = load_settings({"data_dir": tmp_path / "data"})
    assert bare.e2e_mode == "legacy"
    assert PROFILE_SCHEMA == 2


def test_peralert_mode_is_honoured(tmp_path, monkeypatch):
    """SENTINEL_E2E_MODE=peralert is a real choice. The old name loads as legacy."""
    monkeypatch.setenv("SENTINEL_E2E_MODE", "peralert")
    chosen = load_settings({"data_dir": tmp_path / "data"})
    assert chosen.e2e_mode == "peralert"

    monkeypatch.setenv("SENTINEL_E2E_MODE", "profiles")
    still = load_settings({"data_dir": tmp_path / "data"})
    assert still.e2e_mode == "legacy"

    # An unknown mode is ignored. The dataclass default remains.
    monkeypatch.delenv("SENTINEL_E2E_MODE", raising=False)
    kept = load_settings({"e2e_mode": "both"})
    assert kept.e2e_mode == "legacy"


def test_novelty_quantile_matches_the_legacy_calibration_default():
    """The per-alert radius uses the same quantile the legacy head was calibrated with."""
    fresh = Calibration(novelty_threshold=0.5, match_threshold=0.25)
    assert NOVELTY_QUANTILE == fresh.novelty_quantile
    assert NOVELTY_QUANTILE == 0.98
    assert KNN_VOTE_MIN == 0.6
    assert MISSING_TAG_CONFIDENCE_CAP == 0.6
    assert COVERAGE_WARN_FRACTION == 0.5
