"""Not-worse gate against the served alert-type model.

These cases use the statistical embedder. They do not load TS Pulse weights
and they do not score the 320 acceptance rows.
"""

from __future__ import annotations

import json

import numpy as np

from sentinel.config import load_settings
from sentinel.e2e.labels import LabelStore, ProfileEmbeddings, Registry
from sentinel.e2e.model import ChannelScaler
from sentinel.e2e.train import (
    _model_from_rows,
    assign_heldout,
    retrain_profile,
    shipped_heldout,
    shipped_rows,
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


def _vector(kind: int, channels: int = 2):
    vec = np.zeros((channels, 48), dtype=np.float32)
    vec[:, kind] = 1.0
    return vec, np.ones(channels, dtype=bool)


def _scaled(kind: int):
    vec, mask = _vector(kind)
    scaler = ChannelScaler(np.zeros(48, dtype=np.float64), np.ones(48, dtype=np.float64))
    return np.asarray(scaler.transform(vec, mask), dtype=np.float32), mask


def _seed(settings, profile: str, n: int, *, frames=None) -> list:
    store = ProfileEmbeddings(settings)
    labels = LabelStore(settings)
    ids = []
    names = ("cold_spot_hardware", "tc_sensor_fault")
    actions = {"cold_spot_hardware": "borescope", "tc_sensor_fault": "sensor_fix_replace"}
    for i in range(n):
        case_id = f"{profile}-{i:04d}"
        signature = names[i % 2]
        vec, mask = _vector(i % 2)
        frame = "" if not frames else frames[i % len(frames)]
        store.save(profile, case_id, vec, vec, mask, ["a", "b"], "hash", "e0000", {})
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


def _plant(settings, pid: str, *, n_train: int = 24, n_held: int = 16, frame: str = "7EA"):
    """A small shipped head bundle in the stats space, with a held-out file."""
    names = ("cold_spot_hardware", "tc_sensor_fault")
    actions = {"cold_spot_hardware": "borescope", "tc_sensor_fault": "sensor_fix_replace"}
    rows = []
    for i in range(n_train):
        vec, mask = _scaled(i % 2)
        rows.append(
            {
                "id": f"ship-{i:04d}",
                "signature": names[i % 2],
                "action": actions[names[i % 2]],
                "frame": "7EA",
                "u": vec,
                "mask": mask,
                "source": "synthetic",
            }
        )
    model = _model_from_rows(pid, rows, use_head=True)
    folder = settings.e2e_model_dir / "profiles" / pid
    folder.mkdir(parents=True, exist_ok=True)
    model.head.save(folder)
    model.index.save(folder)
    (folder / "calibration.json").write_text(json.dumps(model.calibration), encoding="utf-8")
    held_u, held_m, sig, act, ids, frames, source = [], [], [], [], [], [], []
    for i in range(n_held):
        vec, mask = _scaled(i % 2)
        held_u.append(vec)
        held_m.append(mask)
        sig.append(names[i % 2])
        act.append(actions[names[i % 2]])
        ids.append(f"held-{i:04d}")
        frames.append(frame if frame != "split" else ("site_a" if i < n_held // 2 else "site_b"))
        source.append("synthetic")
    np.savez_compressed(
        folder / "heldout.npz",
        U=np.stack(held_u).astype(np.float16),
        mask=np.stack(held_m).astype(np.uint8),
        signature=np.array(sig),
        case_id=np.array(ids),
        action=np.array(act),
        frame=np.array(frames),
        source=np.array(source),
    )
    (folder / "meta.json").write_text(
        json.dumps(
            {
                "profile": pid,
                "embedder": "stats",
                "tier": "head",
                "gate": {"status": "pass", "reasons": ["shipped synthetic bundle"]},
            }
        ),
        encoding="utf-8",
    )
    return folder


def test_shipped_rows_stay_in_their_embedder(tmp_path):
    """A tspulse bundle is invisible to a stats retrain. A stats bundle keeps frame and source."""
    settings = _settings(tmp_path)
    folder = _plant(settings, "exhaust_cold_spot")
    rows = shipped_rows(settings, "exhaust_cold_spot")
    held = shipped_heldout(settings, "exhaust_cold_spot")
    assert len(rows) == 24 and {row["source"] for row in rows} == {"synthetic"}
    assert len(held) == 16 and held[0]["frame"] == "7EA" and held[0]["source"] == "synthetic"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    meta["embedder"] = "tspulse_ci"
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert shipped_rows(settings, "exhaust_cold_spot") == []
    assert shipped_heldout(settings, "exhaust_cold_spot") == []
    assert shipped_rows(settings, "site_only_alarm") == []


def test_worse_candidate_loses_to_the_served_model(tmp_path):
    """The served head is scored as saved. A refit of both sides would tie and promote."""
    settings = _settings(tmp_path)
    pid = "site_alarm"
    ids = _seed(settings, pid, 60)
    first = retrain_profile(settings, pid, trigger="good")
    assert first["promoted"] is True and first["gate"]["status"] == "pass"
    assert (Registry(settings, pid).folder(first["version"]) / "head.npz").is_file()
    held = assign_heldout(settings, pid, ids)
    store = ProfileEmbeddings(settings)
    for case_id in ids:
        if case_id in held:
            continue
        stored = store.load(pid, case_id)
        hot = int(np.argmax(np.abs(stored["stats"]).sum(axis=0)))
        flipped = np.zeros_like(stored["stats"])
        flipped[:, 1 - (hot % 2)] = 1.0
        store.save(
            pid, case_id, flipped, flipped, stored["mask"],
            stored["channel_ids"], stored["channels_hash"], stored["encoder_id"], stored["resolved"],
        )
    second = retrain_profile(settings, pid, trigger="flipped")
    assert second["promoted"] is False and second["gate"]["status"] == "fail"
    assert Registry(settings, pid).current() == first["version"]


def test_five_labels_keep_the_shipped_head_unless_the_candidate_passes(tmp_path, monkeypatch):
    """Five labels do not replace the shipped head with a five-row index."""
    settings = _settings(tmp_path)
    pid = "exhaust_cold_spot"
    _plant(settings, pid)
    _seed(settings, pid, 5)
    passed = retrain_profile(settings, pid, trigger="five")
    assert passed["gate"]["status"] == "pass" and passed["promoted"] is True
    assert passed["n_train"] > 5
    assert (Registry(settings, pid).folder(passed["version"]) / "head.npz").is_file()
    saved = json.loads((Registry(settings, pid).folder(passed["version"]) / "meta.json").read_text(encoding="utf-8"))
    assert any(str(item).startswith("ship-") for item in saved["index_ids"])
    assert saved["gate_rows"]["n"] >= 8

    blocked_dir = tmp_path / "blocked"
    blocked = _settings(blocked_dir)
    _plant(blocked, pid)
    _seed(blocked, pid, 5)
    from sentinel.e2e import train as train_mod

    state = {"n": 0}

    def fake(model, rows, tax):
        state["n"] += 1
        base = {"n": len(rows), "false_novelty": 0.0, "by_frame": {}, "by_alert": {}}
        if state["n"] == 1:
            return {**base, "action_acc": 0.95, "signature_acc": 0.95}
        return {**base, "action_acc": 0.40, "signature_acc": 0.40}

    monkeypatch.setattr(train_mod, "_score_profile_rows", fake)
    out = retrain_profile(blocked, pid, trigger="five worse")
    assert out["promoted"] is False and out["gate"]["status"] == "fail"
    assert Registry(blocked, pid).current() == "v0000"
    assert (Registry(blocked, pid).folder("v0000") / "head.npz").is_file()


def test_stats_subbundle_is_the_served_model(tmp_path, monkeypatch):
    """v0000's folder is the channel-independent bundle. Stats still scores its own."""
    settings = _settings(tmp_path)
    pid = "exhaust_cold_spot"
    planted = _plant(settings, pid)
    stats_dir = planted / "stats"
    stats_dir.mkdir()
    for item in list(planted.iterdir()):
        if item.is_file():
            item.replace(stats_dir / item.name)
    (planted / "meta.json").write_text(
        json.dumps(
            {
                "profile": pid,
                "embedder": "tspulse_ci",
                "tier": "head",
                "gate": {"status": "pass", "reasons": ["channel-independent bundle"]},
            }
        ),
        encoding="utf-8",
    )
    from sentinel.e2e.train import load_profile_model

    served = load_profile_model(settings, pid)
    assert served is not None and served.embedder == "stats"
    _seed(settings, pid, 5)
    from sentinel.e2e import train as train_mod

    state = {"n": 0}

    def fake(model, rows, tax):
        state["n"] += 1
        base = {"n": len(rows), "false_novelty": 0.0, "by_frame": {}, "by_alert": {}}
        if state["n"] == 1:
            return {**base, "action_acc": 0.95, "signature_acc": 0.95}
        return {**base, "action_acc": 0.40, "signature_acc": 0.40}

    monkeypatch.setattr(train_mod, "_score_profile_rows", fake)
    out = retrain_profile(settings, pid, trigger="stats worse")
    assert state["n"] >= 2
    assert out["promoted"] is False and out["gate"]["status"] == "fail"
    assert Registry(settings, pid).current() == "v0000"


def test_no_new_labels_tie_on_the_cv_gate(tmp_path):
    """The served ids and the current ids are the same, so the fold scores tie."""
    settings = _settings(tmp_path)
    pid = "site_cv"
    _seed(settings, pid, 16)
    first = retrain_profile(settings, pid, trigger="first")
    assert first["promoted"] is True
    second = retrain_profile(settings, pid, trigger="again")
    assert second["promoted"] is True and second["gate"]["status"] == "pass"
    assert second["before"]["action_acc"] == second["after"]["action_acc"]
    assert second["before"]["signature_acc"] == second["after"]["signature_acc"]


def test_frame_slice_drop_blocks_and_force_leaves_the_other_pointer(tmp_path, monkeypatch):
    """A frame with at least 8 rows on both sides blocks. Force logs this alert type only."""
    settings = _settings(tmp_path)
    pid = "exhaust_spread_trip"
    other = "inlet_filter_dp"
    _plant(settings, pid, frame="split")
    _seed(settings, other, 4)
    kept = retrain_profile(settings, other, trigger="other")
    assert kept["promoted"] is True and kept["gate"]["status"] == "none"
    from sentinel.e2e import train as train_mod

    state = {"n": 0}

    def fake(model, rows, tax):
        state["n"] += 1
        before = {
            "n": len(rows), "action_acc": 0.75, "signature_acc": 0.75, "false_novelty": 0.0,
            "by_frame": {
                "site_a": {"n": 10, "action_acc": 0.9, "signature_acc": 0.9},
                "site_b": {"n": 10, "action_acc": 0.6, "signature_acc": 0.6},
            },
            "by_alert": {},
        }
        after = {
            "n": len(rows), "action_acc": 0.75, "signature_acc": 0.75, "false_novelty": 0.0,
            "by_frame": {
                "site_a": {"n": 10, "action_acc": 0.4, "signature_acc": 0.9},
                "site_b": {"n": 10, "action_acc": 0.9, "signature_acc": 0.9},
            },
            "by_alert": {},
        }
        return before if state["n"] % 2 == 1 else after

    monkeypatch.setattr(train_mod, "_score_profile_rows", fake)
    blocked = retrain_profile(settings, pid, trigger="slice")
    assert blocked["promoted"] is False
    assert any("slice" in reason for reason in blocked["gate"]["reasons"])
    assert Registry(settings, other).current() == kept["version"]
    forced = retrain_profile(settings, pid, force=True, trigger="slice force")
    assert forced["promoted"] is True and forced["gate"]["status"] == "forced"
    assert Registry(settings, other).current() == kept["version"]
    log = (tmp_path / "data" / "e2e" / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"type": "force_promote"' in log and pid in log and forced["version"] in log
    pointer = json.loads((tmp_path / "data" / "e2e" / "registry" / "current.json").read_text(encoding="utf-8"))
    assert pointer["profiles"][other] == kept["version"]
    assert pointer["profiles"][pid] == forced["version"]


def test_retrain_without_an_engine_keeps_the_slot_weights(tmp_path):
    """A CLI retrain has no cached model. It uses the declared slot weights, not uniform ones."""
    from sentinel.e2e.model import channel_weights
    from sentinel.e2e.train import _weights_for
    from sentinel.profiles import compile_channels, load_profiles

    settings = _settings(tmp_path)
    profile = load_profiles(settings)["exhaust_cold_spot"]
    expected = channel_weights(profile, compile_channels(profile))
    assert expected.min() < 1.0
    rows = [{"u": np.zeros((len(expected), 48), dtype=np.float32)}]
    assert np.allclose(_weights_for(settings, "exhaust_cold_spot", rows, None), expected)
    # Rows on another channel list, or an unknown alert type, stay uniform.
    short = [{"u": np.zeros((len(expected) - 1, 48), dtype=np.float32)}]
    assert np.allclose(_weights_for(settings, "exhaust_cold_spot", short, None), 1.0)
    assert np.allclose(_weights_for(settings, "no_such_alert_type", rows, None), 1.0)
