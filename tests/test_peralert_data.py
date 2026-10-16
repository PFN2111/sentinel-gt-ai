"""Per-alert training rows from both generators.

Short CSVs stand in for the two datasets. Nothing here scores the 320 held-out
rows, and nothing loads TS Pulse weights.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sentinel.e2e.dataset import decompressed_csv
from sentinel.e2e.encoder import StatsEmbedder
from sentinel.e2e.train import acceptance_blocklist, alert_training_packs, alert_training_rows

# A real side-car id. The row below is a train split, so only the blocklist can drop it.
SIDE_CAR_ID = "test-normal-base-000"

_COLUMNS = (
    "load_MW",
    "fuel_flow_kg_s",
    "IGV_deg",
    "EGT_spread_C",
    "EGT_avg_C",
    "CDP_bar",
    "comb_dyn_psi",
    "vib_DE_mm_s",
    "vib_NDE_mm_s",
    "TC1",
    "TC2",
    "TC3",
    "TC4",
    "trip_active",
    "lube_oil_C",
    "speed_rpm",
    "inlet_dp_mbar",
    "ambient_C",
)


def _write_csv(path, salt: float) -> str:
    """A short gzip CSV. The salt keeps two files from sharing a content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 80
    clock = np.linspace(0.0, 6.0, n)
    frame = {"timestamp": pd.date_range("2024-06-01", periods=n, freq="min").strftime("%Y-%m-%d %H:%M:%S")}
    for index, name in enumerate(_COLUMNS):
        frame[name] = np.round(40 + salt + (index + 1) * np.sin(clock + salt), 4)
    pd.DataFrame(frame).to_csv(path, index=False, compression="gzip")
    return hashlib.sha256(decompressed_csv(path)).hexdigest()


def _row(case_id, split, signature, analytic, rel, profile=None, action="monitor"):
    row = {
        "case_id": case_id,
        "split": split,
        "signature": signature,
        "analytic": analytic,
        "file": rel,
        "action": action,
        "layout": "ring",
        "alt_tags": False,
    }
    if profile is not None:
        row["profile"] = profile
    return row


def _build(tmp_path):
    """One tiny original tree and one tiny alert-type tree."""
    original = tmp_path / "original"
    profile = tmp_path / "profile"
    files = {}

    def add(root, rel, salt):
        files[rel] = _write_csv(root / rel, salt)

    original_rows = []
    # Rule A names. Each normal lands on its own alert type, not on every combustion type.
    plan = [
        ("p2fix-cs-train", "train", "normal", "Exhaust Cold Spot", "train/p2fix-cs-train.csv.gz", 1.0),
        ("p2fix-cs-val", "val", "normal", "Exhaust Cold Spot", "val/p2fix-cs-val.csv.gz", 2.0),
        ("p2fix-cs-var", "variant", "cold_spot_hardware", "Exhaust Cold Spot", "variant/p2fix-cs-var.csv.gz", 3.0),
        ("p2fix-cs-sf", "sensorfault", "tc_sensor_fault", "Exhaust Cold Spot", "sensorfault/p2fix-cs-sf.csv.gz", 4.0),
        ("p2fix-cs-test", "test", "cold_spot_hardware", "Exhaust Cold Spot", "test/p2fix-cs-test.csv.gz", 5.0),
        ("p2fix-cs-unseen", "unseen", "normal", "Exhaust Cold Spot", "unseen/p2fix-cs-unseen.csv.gz", 6.0),
        ("p2fix-cd-train", "train", "normal", "Combustion Dynamics High", "train/p2fix-cd-train.csv.gz", 7.0),
        (SIDE_CAR_ID, "train", "normal", "Combustion Dynamics High", "train/p2fix-cd-sidecar.csv.gz", 8.0),
        ("p2fix-cd-hash", "train", "combustion_instability", "Combustion Dynamics High", "train/p2fix-cd-hash.csv.gz", 9.0),
        ("p2fix-brg-orig", "train", "normal", "Bearing vibration high", "train/p2fix-brg-orig.csv.gz", 10.0),
    ]
    for case_id, split, signature, analytic, rel, salt in plan:
        add(original, rel, salt)
        original_rows.append(_row(case_id, split, signature, analytic, rel))
    # A manifest row with no file is skipped, not raised.
    original_rows.append(_row("p2fix-missing", "train", "normal", "Exhaust Cold Spot", "train/missing.csv.gz"))
    (original / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in original_rows),
        encoding="utf-8",
    )

    profile_rows = []
    profile_plan = [
        ("p2fix-cs-p-train", "train", "normal", "Exhaust Cold Spot", "exhaust_cold_spot", 11.0),
        ("p2fix-cs-p-val", "val", "normal", "Exhaust Cold Spot", "exhaust_cold_spot", 12.0),
        ("p2fix-cs-p-test", "test", "cold_spot_hardware", "Exhaust Cold Spot", "exhaust_cold_spot", 13.0),
        ("p2fix-cs-p-unseen", "unseen", "tc_harness_fault", "Exhaust Cold Spot", "exhaust_cold_spot", 14.0),
        ("p2fix-cs-p-var", "variant", "cold_spot_hardware", "Exhaust Cold Spot", "exhaust_cold_spot", 15.0),
        ("p2fix-cd-p-train", "train", "normal", "Combustion Dynamics High", "combustion_dynamics", 16.0),
        ("p2fix-brg-p-train", "train", "normal", "Bearing vibration high", "bearing_vibration", 17.0),
        ("p2fix-brg-p-test", "test", "bearing_wear_trend", "Bearing vibration high", "bearing_vibration", 18.0),
        ("p2fix-heat-train", "train", "normal", "Fuel gas heater outlet low", "fuel_gas_heater", 19.0),
    ]
    for case_id, split, signature, analytic, pid, salt in profile_plan:
        rel = f"{split}/{pid}/{case_id}.csv.gz"
        add(profile, rel, salt)
        profile_rows.append(_row(case_id, split, signature, analytic, rel, profile=pid))
    (profile / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in profile_rows),
        encoding="utf-8",
    )
    blocked_hash = files["train/p2fix-cd-hash.csv.gz"]
    return original, profile, blocked_hash


def _by_id(rows):
    return {row["id"]: row for row in rows}


def _all(grouped):
    return grouped["train"] + grouped["val"] + grouped["heldout"]


def test_acceptance_blocklist_reads_the_sidecar(tmp_path):
    """The committed side-car is 320 ids and 320 content hashes. A missing file is an empty block."""
    block = acceptance_blocklist()
    assert len(block["ids"]) == 320
    assert len(block["hashes"]) == 320
    assert SIDE_CAR_ID in block["ids"]
    assert acceptance_blocklist(tmp_path / "missing.jsonl") == {"ids": set(), "hashes": set()}


def test_splits_roles_blocklist_and_spread(tmp_path, settings, monkeypatch):
    """Train, val and held-out follow section 4.2. The 320 rows and both unseen splits stay out."""
    original, profile, blocked_hash = _build(tmp_path)
    reader = acceptance_blocklist

    def widened(path=None):
        found = reader(path)
        if path is not None:
            return found
        return {"ids": set(found["ids"]), "hashes": set(found["hashes"]) | {blocked_hash}}

    monkeypatch.setattr("sentinel.e2e.train.acceptance_blocklist", widened)
    embedder = StatsEmbedder()
    cold = alert_training_rows(settings, original, profile, "exhaust_cold_spot", embedder)
    dyn = alert_training_rows(settings, original, profile, "combustion_dynamics", embedder)
    bearing = alert_training_rows(settings, original, profile, "bearing_vibration", embedder)

    cold_train = _by_id(cold["train"])
    assert set(cold_train) == {"p2fix-cs-train", "p2fix-cs-p-train"}
    assert cold_train["p2fix-cs-train"]["source"] == "synthetic_original"
    assert cold_train["p2fix-cs-train"]["signature"] == "normal"
    assert cold_train["p2fix-cs-train"]["alert_name"] == "Exhaust Cold Spot"
    assert cold_train["p2fix-cs-p-train"]["source"] == "synthetic_profile"
    assert "p2fix-cd-train" not in cold_train

    cold_val = _by_id(cold["val"])
    assert set(cold_val) == {"p2fix-cs-val", "p2fix-cs-p-val"}
    assert cold_val["p2fix-cs-val"]["source"] == "synthetic_original"
    assert cold_val["p2fix-cs-p-val"]["source"] == "synthetic_profile"

    cold_hold = _by_id(cold["heldout"])
    assert set(cold_hold) == {"p2fix-cs-var", "p2fix-cs-sf", "p2fix-cs-p-test"}
    assert cold_hold["p2fix-cs-var"]["source"] == "synthetic_original"
    assert cold_hold["p2fix-cs-var"]["split"] == "variant"
    assert cold_hold["p2fix-cs-sf"]["source"] == "synthetic_original"
    assert cold_hold["p2fix-cs-sf"]["split"] == "sensorfault"
    assert cold_hold["p2fix-cs-p-test"]["source"] == "synthetic_profile"
    assert cold_hold["p2fix-cs-p-test"]["split"] == "test"

    cold_ids = set(_by_id(_all(cold)))
    for dropped in ("p2fix-cs-test", "p2fix-cs-unseen", "p2fix-cs-p-unseen", "p2fix-cs-p-var", "p2fix-missing", "p2fix-heat-train"):
        assert dropped not in cold_ids

    dyn_ids = _by_id(_all(dyn))
    assert "p2fix-cd-train" in _by_id(dyn["train"])
    assert _by_id(dyn["train"])["p2fix-cd-train"]["source"] == "synthetic_original"
    assert "p2fix-cd-p-train" in _by_id(dyn["train"])
    assert "p2fix-cs-train" not in dyn_ids
    # Train-split rows. Absence is the side-car, not the split rule.
    assert SIDE_CAR_ID not in dyn_ids
    assert "p2fix-cd-hash" not in dyn_ids
    assert "p2fix-brg-orig" not in dyn_ids

    bearing_ids = _by_id(_all(bearing))
    assert set(bearing_ids) == {"p2fix-brg-p-train", "p2fix-brg-p-test"}
    assert bearing_ids["p2fix-brg-p-train"]["role"] == "train"
    assert bearing_ids["p2fix-brg-p-train"]["source"] == "synthetic_profile"
    assert bearing_ids["p2fix-brg-p-test"]["role"] == "heldout"
    assert "p2fix-brg-orig" not in bearing_ids

    for pid, grouped in (
        ("exhaust_cold_spot", cold),
        ("combustion_dynamics", dyn),
        ("bearing_vibration", bearing),
    ):
        assert grouped["scaler"].mean.shape == (48,)
        assert grouped["scaler"].std.shape == (48,)
        assert np.isfinite(grouped["scaler"].mean).all()
        assert np.isfinite(grouped["scaler"].std).all()
        for row in _all(grouped):
            assert row["alert_type"] == pid
            assert row["u"].ndim == 2 and row["u"].shape[1] == 48
            assert np.isfinite(row["u"]).all()
            assert "path" not in row and "x" not in row
            assert row["split"] != "unseen"
            if row["source"] == "synthetic_original":
                assert row["split"] not in {"test", "unseen"}
            assert len(row["csv_sha256"]) == 64
    assert not np.allclose(cold["scaler"].mean, 0.0)

    with pytest.raises(ValueError, match="unknown alert type"):
        alert_training_rows(settings, original, profile, "not_a_type", embedder)


class _CountingEmbedder(StatsEmbedder):
    """Counts the files embedded, so a repeated pass shows up."""

    def __init__(self):
        super().__init__()
        self.cases = 0

    def embed_channels(self, x, mask):
        self.cases += int(np.asarray(x).shape[0])
        return super().embed_channels(x, mask)


def test_packs_embed_each_file_once_and_match_the_single_calls(tmp_path, settings):
    """Several alert types in one pass give the rows the one-at-a-time calls give."""
    original, profile, _blocked = _build(tmp_path)
    pids = ["exhaust_cold_spot", "combustion_dynamics", "bearing_vibration"]
    single_counter = _CountingEmbedder()
    single = {pid: alert_training_rows(settings, original, profile, pid, single_counter) for pid in pids}
    pack_counter = _CountingEmbedder()
    packs = alert_training_packs(settings, original, profile, pids, pack_counter)
    assert set(packs) == set(pids)
    # One pass embeds every file once. Three single calls embed them three times.
    assert single_counter.cases == 3 * pack_counter.cases
    assert pack_counter.cases > 0
    for pid in pids:
        assert np.allclose(packs[pid]["scaler"].mean, single[pid]["scaler"].mean)
        for role in ("train", "val", "heldout"):
            got = packs[pid][role]
            want = single[pid][role]
            assert [row["id"] for row in got] == [row["id"] for row in want]
            for a, b in zip(got, want):
                assert np.allclose(a["u"], b["u"])
                assert a["frame"] == b["frame"]
