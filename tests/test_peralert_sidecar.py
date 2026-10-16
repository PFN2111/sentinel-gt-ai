"""The frozen alert type of each of the 320 held-out cases.

The side-car is committed before any per-alert model is trained. These tests pin
its bytes and the design section 7.1 counts. They do not score a model.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from conftest import ROOT
from sentinel.e2e.dataset import GENERIC_ALERT_NAMES
from sentinel.profiles import alert_type_for_name, load_profiles

# UTF-8, LF, json.dumps(..., sort_keys=True). The writer emits these bytes.
PINNED_SHA256 = "85611457ad9bf527bb237ecc82f67b5c43c4fab3591ee91e4bb55cf5bb63b3c1"

# The five generator names that are not alert names. Rule B is only these.
GENERIC = (
    "GT alarm review",
    "Tier 1 escalation",
    "Exhaust temperature deviation advisory",
    "Unit advisory",
    "Performance deviation",
)

# Design section 7.1. A later edit of the side-car must fail here, not be retuned.
EXPECTED = {
    "combustion_dynamics": {
        "n": 92,
        "normal": 12,
        "faults": {"combustion_instability": 40, "dynamics_probe_fault": 40},
        "rules": {"A": 82, "B-owned": 8, "B-hash": 2},
    },
    "igv_fuel_control": {
        "n": 89,
        "normal": 9,
        "faults": {"igv_lvdt_drift": 40, "fuel_control_oscillation": 40},
        "rules": {"A": 77, "B-owned": 12, "B-hash": 0},
    },
    "exhaust_cold_spot": {
        "n": 84,
        "normal": 14,
        "faults": {"cold_spot_hardware": 40, "tc_sensor_fault": 30},
        "rules": {"A": 76, "B-owned": 1, "B-hash": 7},
    },
    "exhaust_spread_trip": {
        "n": 55,
        "normal": 5,
        "faults": {"hets_trip_hardware": 40, "tc_sensor_fault": 10},
        "rules": {"A": 51, "B-owned": 2, "B-hash": 2},
    },
}

FIELDS = {
    "case_id",
    "file",
    "csv_sha256",
    "truth",
    "frame",
    "original_analytic",
    "alert_type",
    "alert_name",
    "rule",
}

SIDE_CAR = ROOT / "eval" / "acceptance_peralert.jsonl"
DATASET = ROOT / "data" / "e2e" / "dataset" / "manifest.jsonl"


def _bytes() -> bytes:
    """Side-car bytes with newlines folded to LF, which is what the writer emits."""
    return SIDE_CAR.read_bytes().replace(b"\r\n", b"\n")


def _rows() -> list:
    text = _bytes().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_sidecar_sha256_is_pinned():
    """The committed file is the frozen assignment. A rewrite changes this hash."""
    assert hashlib.sha256(_bytes()).hexdigest() == PINNED_SHA256
    assert _bytes().endswith(b"\n")
    assert b"\r" not in _bytes()


def test_sidecar_counts_match_the_design_table():
    """Per alert type, per truth signature, and the 286 / 23 / 11 rule split."""
    rows = _rows()
    assert len(rows) == 320
    assert {row["alert_type"] for row in rows} == set(EXPECTED)
    rules = {"A": 0, "B-owned": 0, "B-hash": 0}
    normal = 0
    for alert_type, expect in EXPECTED.items():
        group = [row for row in rows if row["alert_type"] == alert_type]
        truths: dict = {}
        by_rule: dict = {}
        for row in group:
            truths[row["truth"]] = truths.get(row["truth"], 0) + 1
            by_rule[row["rule"]] = by_rule.get(row["rule"], 0) + 1
            rules[row["rule"]] = rules[row["rule"]] + 1
        assert len(group) == expect["n"]
        assert truths.get("normal", 0) == expect["normal"]
        normal += truths.get("normal", 0)
        faults = {key: value for key, value in truths.items() if key != "normal"}
        assert faults == expect["faults"]
        assert {rule: by_rule.get(rule, 0) for rule in ("A", "B-owned", "B-hash")} == expect["rules"]
    assert rules == {"A": 286, "B-owned": 23, "B-hash": 11}
    assert normal == 40
    assert len({row["case_id"] for row in rows}) == 320


def test_every_alert_type_lists_its_truth_and_rule_b_is_only_the_generic_names():
    """Rule A keeps the original name. Rule B is one of the five names, renamed to the type."""
    assert tuple(GENERIC_ALERT_NAMES) == GENERIC
    book = load_profiles()
    for row in _rows():
        assert set(row) == FIELDS
        listed = {str(sig) for sig in book[row["alert_type"]].signatures}
        assert row["truth"] in listed
        assert len(row["csv_sha256"]) == 64
        if row["rule"] == "A":
            assert row["original_analytic"] not in GENERIC
            assert row["alert_name"] == row["original_analytic"]
            found = alert_type_for_name(row["alert_name"], book)
            assert found is not None and found.id == row["alert_type"]
        else:
            assert row["rule"] in {"B-owned", "B-hash"}
            assert row["original_analytic"] in GENERIC
            assert row["alert_name"] == book[row["alert_type"]].name
            assert alert_type_for_name(row["original_analytic"], book) is None


@pytest.mark.skipif(not DATASET.is_file(), reason="data/e2e/dataset is absent")
def test_recompute_matches_the_committed_sidecar_and_the_manifest_ids():
    """Built-ins only. A site profile must not move a held-out case. Same ids, same order."""
    from sentinel.e2e.dataset import acceptance_sidecar_rows, load_manifest

    root = DATASET.parent
    fresh = acceptance_sidecar_rows(root)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in fresh)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PINNED_SHA256
    manifest_ids = [row["case_id"] for row in load_manifest(root) if str(row.get("split") or "") == "test"]
    assert [row["case_id"] for row in fresh] == manifest_ids
    assert len(manifest_ids) == 320
