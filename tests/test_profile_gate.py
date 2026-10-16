"""Per-profile quick retrain: no gate under 10 labels, CV from 10, held-out from 50, force log."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from conftest import ROOT
from sentinel.config import load_settings
from sentinel.e2e.labels import LabelStore, Metrics, ProfileEmbeddings, Registry
from sentinel.e2e.train import (
    assign_heldout,
    cv_accuracies,
    held_out_case,
    profile_gate_reasons,
    profile_labels_since_retrain,
    retrain_profile,
)


def _settings(tmp_path):
    return load_settings(
        {
            "data_dir": tmp_path / "data",
            "e2e_model_dir": tmp_path / "models",
            "use_llm": False,
            "use_tspulse": False,
            "embedder": "stats",
            "e2e_retrain_every": 100,
            "profiles_dir": str(tmp_path / "profiles"),
        }
    )


def _vector(kind: int, channels: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Two classes sit on different axes, so a neighbour of the same class is the nearest."""
    vec = np.zeros((channels, 48), dtype=np.float32)
    vec[:, kind] = 1.0
    return vec, np.ones(channels, dtype=bool)


def _seed(settings, profile: str, n: int, *, frames: list[str] | None = None) -> list[str]:
    """Write n labeled embeddings, alternating two real signatures."""
    store = ProfileEmbeddings(settings)
    labels = LabelStore(settings)
    ids = []
    names = ("cold_spot_hardware", "tc_sensor_fault")
    actions = {"cold_spot_hardware": "borescope", "tc_sensor_fault": "sensor_fix_replace"}
    for i in range(n):
        case_id = f"{profile}-{i:04d}"
        signature = names[i % 2]
        vec, mask = _vector(i % 2)
        store.save(profile, case_id, vec, vec, mask, ["a", "b"], "hash", "e0000", {})
        frame = "" if not frames else frames[i % len(frames)]
        labels.add(
            {
                "case_id": case_id,
                "signature": signature,
                "action": actions[signature],
                "profile": profile,
                "frame_key": frame,
                "verdict": "good",
            }
        )
        ids.append(case_id)
    return ids


def test_under_ten_labels_updates_the_index_with_no_gate(tmp_path):
    settings = _settings(tmp_path)
    _seed(settings, "bearing_vibration", 4)
    _seed(settings, "inlet_filter_dp", 4)
    other = retrain_profile(settings, "inlet_filter_dp", trigger="other")
    out = retrain_profile(settings, "bearing_vibration", trigger="small")
    assert out["promoted"] is True and out["gate"]["status"] == "none"
    assert "no gate" in out["gate"]["reasons"][0]
    folder = Registry(settings, "bearing_vibration").folder(out["version"])
    assert (folder / "index.npz").is_file() and not (folder / "head.npz").is_file()
    assert Registry(settings, "bearing_vibration").current() == out["version"]
    assert Registry(settings, "inlet_filter_dp").current() == other["version"]
    pointer = json.loads((tmp_path / "data" / "e2e" / "registry" / "current.json").read_text(encoding="utf-8"))
    assert "version" not in pointer or pointer.get("version") in (None, "v0000")


def test_cv_gate_blocks_a_worse_candidate_and_force_logs_the_profile(tmp_path, monkeypatch):
    from sentinel.e2e import train

    settings = _settings(tmp_path)
    _seed(settings, "exhaust_cold_spot", 12)

    def worse(rows, tax, *, profile_id, active_head, weights=None, **kwargs):
        good = {"n": len(rows), "action_acc": 0.95, "signature_acc": 0.95, "false_novelty": 0.0}
        bad = {"n": len(rows), "action_acc": 0.40, "signature_acc": 0.40, "false_novelty": 0.0}
        return good, bad

    monkeypatch.setattr(train, "cv_accuracies", worse)
    blocked = retrain_profile(settings, "exhaust_cold_spot", trigger="cv")
    assert blocked["promoted"] is False and blocked["gate"]["status"] == "fail"
    assert Registry(settings, "exhaust_cold_spot").current() == "v0000"
    assert (Registry(settings, "exhaust_cold_spot").folder(blocked["version"]) / "meta.json").is_file()
    forced = retrain_profile(settings, "exhaust_cold_spot", force=True, trigger="cv force")
    assert forced["promoted"] is True and forced["gate"]["status"] == "forced"
    assert Registry(settings, "exhaust_cold_spot").current() == forced["version"]
    log = (tmp_path / "data" / "e2e" / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"type": "force_promote"' in log and "exhaust_cold_spot" in log and forced["version"] in log and forced["force_reason"]


def test_real_cv_on_separated_cases_is_not_worse(tmp_path):
    from sentinel.e2e.taxonomy import load_taxonomy

    settings = _settings(tmp_path)
    _seed(settings, "exhaust_cold_spot", 12)
    from sentinel.e2e.train import collect_profile_rows

    rows = collect_profile_rows(settings, "exhaust_cold_spot")
    before, after = cv_accuracies(rows, load_taxonomy(settings.data_dir), profile_id="exhaust_cold_spot", active_head=False)
    assert after["signature_acc"] + 1e-9 >= before["signature_acc"]
    assert after["action_acc"] + 1e-9 >= before["action_acc"]


def test_head_tier_promotes_when_the_folds_hold(tmp_path):
    settings = _settings(tmp_path)
    _seed(settings, "exhaust_spread_trip", 24)
    out = retrain_profile(settings, "exhaust_spread_trip", trigger="head")
    assert out["promoted"] is True and out["gate"]["status"] == "pass"
    folder = Registry(settings, "exhaust_spread_trip").folder(out["version"])
    assert (folder / "head.npz").is_file() and (folder / "calibration.json").is_file()


def test_heldout_gate_blocks_a_slice_and_false_novelty(tmp_path, monkeypatch):
    from sentinel.e2e import train

    settings = _settings(tmp_path)
    ids = _seed(settings, "lube_oil_system", 50, frames=["site_a", "site_b"])
    assert assign_heldout(settings, "lube_oil_system", ids)
    seen = {}

    def worse(train_rows, held_rows, tax, *, profile_id, active_head, weights=None, **kwargs):
        seen["held"] = len(held_rows)
        before = {
            "n": len(held_rows), "action_acc": 0.9, "signature_acc": 0.9, "false_novelty": 0.0,
            "by_frame": {"site_a": {"n": 10, "action_acc": 0.9, "signature_acc": 0.9}},
        }
        after = {
            "n": len(held_rows), "action_acc": 0.9, "signature_acc": 0.9, "false_novelty": 0.05,
            "by_frame": {"site_a": {"n": 10, "action_acc": 0.5, "signature_acc": 0.9}},
        }
        return before, after

    monkeypatch.setattr(train, "heldout_scores", worse)
    out = retrain_profile(settings, "lube_oil_system", trigger="heldout")
    assert seen["held"] > 0
    assert out["promoted"] is False and out["gate"]["status"] == "fail"
    assert any("slice" in reason or "false novelty" in reason for reason in out["gate"]["reasons"])
    text = (Registry(settings, "lube_oil_system").folder(out["version"]) / "index.jsonl").read_text(encoding="utf-8")
    saved_ids = {json.loads(line)["id"] for line in text.splitlines() if line.strip()}
    assert saved_ids.isdisjoint(assign_heldout(settings, "lube_oil_system", ids))


def test_gate_numbers_for_slices_and_false_novelty():
    before = {
        "action_acc": 0.9, "signature_acc": 0.9, "false_novelty": 0.02,
        "by_frame": {"site_a": {"n": 8, "action_acc": 0.9, "signature_acc": 0.9}},
    }
    slice_drop = {
        **before,
        "by_frame": {"site_a": {"n": 8, "action_acc": 0.5, "signature_acc": 0.9}},
    }
    assert profile_gate_reasons(before, slice_drop)
    small = {
        **before,
        "by_frame": {"site_a": {"n": 7, "action_acc": 0.1, "signature_acc": 0.1}},
    }
    assert profile_gate_reasons(before, small) == []
    bumped = {**before, "false_novelty": 0.03}
    assert profile_gate_reasons(before, bumped) == []
    over = {**before, "false_novelty": 0.031}
    assert any("false novelty" in reason for reason in profile_gate_reasons(before, over))


def test_heldout_ids_are_frozen_and_absent_below_fifty(tmp_path):
    settings = _settings(tmp_path)
    ids = [f"case-{i:04d}" for i in range(60)]
    assert assign_heldout(settings, "igv_fuel_control", ids[:40]) == set()
    assert not (tmp_path / "data" / "e2e" / "heldout" / "igv_fuel_control.json").is_file()
    first = assign_heldout(settings, "igv_fuel_control", ids[:50])
    assert first == {item for item in ids[:50] if held_out_case(item)}
    path = tmp_path / "data" / "e2e" / "heldout" / "igv_fuel_control.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ids"].append("kept-by-hand")
    path.write_text(json.dumps(data), encoding="utf-8")
    second = assign_heldout(settings, "igv_fuel_control", ids[:50] + ["kept-by-hand"])
    assert "kept-by-hand" in second and first <= second


def test_profile_label_counter_ignores_the_legacy_retrain(tmp_path):
    settings = _settings(tmp_path)
    metrics = Metrics(settings)
    metrics.log("label", case_id="a", profile="exhaust_cold_spot")
    metrics.log("retrain", version="v0001")
    metrics.log("label", case_id="b", profile="exhaust_cold_spot")
    metrics.log("label", case_id="c")
    assert profile_labels_since_retrain(settings, "exhaust_cold_spot") == 2
    metrics.log("retrain", profile="exhaust_cold_spot", version="v0001")
    metrics.log("label", case_id="d", profile="exhaust_cold_spot")
    assert profile_labels_since_retrain(settings, "exhaust_cold_spot") == 1


def test_train_cli_accepts_profile_and_all():
    proc = subprocess.run([sys.executable, str(ROOT / "app.py"), "train", "--help"], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert "--profile" in proc.stdout and "--all" in proc.stdout
