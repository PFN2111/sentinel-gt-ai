"""Shipped per-alert bundles on the statistical embedder.

Short CSVs stand in for both generators. A scale-0.1 generator train split is
18 rows, under the 20-label head tier, so each alert type here has 22 training
rows of two signatures. Nothing loads TS Pulse weights and nothing scores the
320 held-out rows.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.e2e.model import channel_weights
from sentinel.e2e.train import _choose_encoder, _install_encoder, load_profile_model, train_alert_bundles
from sentinel.profiles import channels_hash, compile_channels, load_profiles

# A real side-car id. It is planted as a train split so only the blocklist can drop it.
SIDE_CAR_ID = "test-normal-base-000"
ORIG_TEST_ID = "bundle-orig-split-row"
ROOT = Path(__file__).resolve().parents[1]

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


def _write_csv(path: Path, salt: float) -> None:
    """A short CSV. The salt keeps two files from sharing a content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 48
    clock = np.linspace(0.0, 6.0, n)
    frame = {"timestamp": pd.date_range("2024-06-01", periods=n, freq="min").strftime("%Y-%m-%d %H:%M:%S")}
    for index, name in enumerate(_COLUMNS):
        frame[name] = np.round(40 + salt + (index + 1) * np.sin(clock + salt), 4)
    pd.DataFrame(frame).to_csv(path, index=False)


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


def _add(root, rows, case_id, split, signature, analytic, salt, profile=None, action="monitor"):
    rel = f"{split}/{case_id}.csv"
    _write_csv(root / rel, salt)
    rows.append(_row(case_id, split, signature, analytic, rel, profile=profile, action=action))


def _build(tmp_path: Path):
    """Two alert types with a head-sized train split, plus rows the trainer must drop."""
    original = tmp_path / "original"
    profile = tmp_path / "profile"
    original_rows = []
    profile_rows = []
    salt = 1.0

    def take():
        nonlocal salt
        salt += 0.17
        return salt

    for index in range(11):
        _add(profile, profile_rows, f"bundle-cs-tr-n-{index}", "train", "normal", "Exhaust Cold Spot", take(), "exhaust_cold_spot", "monitor_only")
        _add(profile, profile_rows, f"bundle-cs-tr-f-{index}", "train", "cold_spot_hardware", "Exhaust Cold Spot", take(), "exhaust_cold_spot", "borescope")
        _add(profile, profile_rows, f"bundle-br-tr-n-{index}", "train", "normal", "Bearing vibration high", take(), "bearing_vibration", "monitor_only")
        _add(profile, profile_rows, f"bundle-br-tr-f-{index}", "train", "bearing_wear_trend", "Bearing vibration high", take(), "bearing_vibration", "monitor_only")
    for index in range(2):
        _add(profile, profile_rows, f"bundle-cs-va-n-{index}", "val", "normal", "Exhaust Cold Spot", take(), "exhaust_cold_spot")
        _add(profile, profile_rows, f"bundle-cs-va-f-{index}", "val", "cold_spot_hardware", "Exhaust Cold Spot", take(), "exhaust_cold_spot", "borescope")
        _add(profile, profile_rows, f"bundle-br-va-n-{index}", "val", "normal", "Bearing vibration high", take(), "bearing_vibration")
        _add(profile, profile_rows, f"bundle-br-va-f-{index}", "val", "bearing_wear_trend", "Bearing vibration high", take(), "bearing_vibration")
    _add(profile, profile_rows, "bundle-cs-hold", "test", "cold_spot_hardware", "Exhaust Cold Spot", take(), "exhaust_cold_spot", "borescope")
    _add(profile, profile_rows, "bundle-br-hold", "test", "bearing_wear_trend", "Bearing vibration high", take(), "bearing_vibration")
    _add(profile, profile_rows, "bundle-cs-unseen", "unseen", "normal", "Exhaust Cold Spot", take(), "exhaust_cold_spot")
    _add(profile, profile_rows, "bundle-cs-variant", "variant", "cold_spot_hardware", "Exhaust Cold Spot", take(), "exhaust_cold_spot", "borescope")

    _add(original, original_rows, "bundle-cs-orig-tr", "train", "normal", "Exhaust Cold Spot", take(), action="monitor_only")
    _add(original, original_rows, "bundle-cs-orig-va", "val", "cold_spot_hardware", "Exhaust Cold Spot", take(), action="borescope")
    _add(original, original_rows, ORIG_TEST_ID, "test", "cold_spot_hardware", "Exhaust Cold Spot", take(), action="borescope")
    # Train split of this alert type. Only the side-car blocklist keeps it out of the index.
    _add(original, original_rows, SIDE_CAR_ID, "train", "normal", "Exhaust Cold Spot", take())
    _add(original, original_rows, "bundle-br-orig-tr", "train", "normal", "Bearing vibration high", take())

    (original / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in original_rows), encoding="utf-8")
    (profile / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in profile_rows), encoding="utf-8")
    return original, profile


def _case_ids(folder: Path) -> set:
    found = set()
    index_text = folder / "index.jsonl"
    if index_text.is_file():
        for line in index_text.read_text(encoding="utf-8").splitlines():
            if line.strip():
                found.add(str(json.loads(line).get("id") or ""))
    held = folder / "heldout.npz"
    if held.is_file():
        blob = np.load(held)
        if "case_id" in blob.files:
            for item in blob["case_id"]:
                found.add(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return found


def test_choose_encoder_tie_or_unavailable_stays_frozen():
    """Validation picks the higher signature accuracy. A tie or a missing score stays frozen."""
    assert _choose_encoder({"frozen": 0.5, "finetuned": 0.5}) == "frozen"
    assert _choose_encoder({"frozen": 0.4, "finetuned": 0.4 + 1e-12}) == "frozen"
    assert _choose_encoder({"frozen": 0.40, "finetuned": 0.41}) == "finetuned"
    assert _choose_encoder({"frozen": 0.5, "finetuned": None}) == "frozen"
    assert _choose_encoder({"frozen": None, "finetuned": 0.5}) == "finetuned"


def test_chosen_encoder_replaces_the_weights_already_installed(tmp_path, settings):
    """A frozen choice overwrites a previous run's fine-tuned weights; the bundles were embedded with frozen."""
    clf = tmp_path / "clf"
    clf.mkdir()
    (clf / "config.json").write_text("frozen config", encoding="utf-8")
    (clf / "model.safetensors").write_bytes(b"frozen weights")
    out = tmp_path / "models"
    fine = out / "_candidates" / "finetuned"
    fine.mkdir(parents=True)
    (fine / "config.json").write_text("fine config", encoding="utf-8")
    (fine / "model.safetensors").write_bytes(b"fine weights")
    settings.tspulse_clf_dir = clf

    enc = _install_encoder("finetuned", settings, out)
    assert (enc / "model.safetensors").read_bytes() == b"fine weights"
    enc = _install_encoder("frozen", settings, out)
    assert (enc / "model.safetensors").read_bytes() == b"frozen weights"
    assert (enc / "config.json").read_text(encoding="utf-8") == "frozen config"
    # stats has no weights and leaves the folder alone.
    assert (_install_encoder("stats", settings, out) / "model.safetensors").read_bytes() == b"frozen weights"


def test_refuses_without_the_sidecar(tmp_path, settings):
    """No side-car, or a side-car with no case ids, does not write a bundle."""
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="acceptance side-car"):
        train_alert_bundles(
            settings, tmp_path, tmp_path, out, encoders=["stats"], sidecar=tmp_path / "missing.jsonl",
        )
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no case ids"):
        train_alert_bundles(settings, tmp_path, tmp_path, out, encoders=["stats"], sidecar=empty)
    assert not (out / "profiles").exists()


def test_train_script_offers_peralert():
    """The training script grows a --peralert flag and keeps the 14-channel path."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_e2e.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--peralert" in proc.stdout
    assert "--profiles" in proc.stdout


def test_bundles_load_at_head_and_exclude_test_rows(tmp_path, settings):
    """Head-tier bundles load. Sources have no test role. Acceptance and original test rows stay out."""
    original, profile = _build(tmp_path)
    out = tmp_path / "models"
    settings.embedder = "stats"
    settings.e2e_model_dir = out
    result = train_alert_bundles(
        settings,
        original,
        profile,
        out,
        device="cpu",
        encoders=["stats"],
        profiles=["exhaust_cold_spot", "bearing_vibration"],
    )
    assert result["chosen"] == "stats"
    assert set(result["profiles"]) == {"exhaust_cold_spot", "bearing_vibration"}

    note = json.loads((out / "encoder" / "training.json").read_text(encoding="utf-8"))
    assert note["chosen"] == "stats"
    assert note["embedder"] == "stats"
    assert note["candidates"]["stats"]["signature_acc"] is not None
    prior = json.loads((out / "encoder" / "novelty_prior.json").read_text(encoding="utf-8"))
    assert prior["embedder"] == "stats"
    assert (out / "encoder" / "channel_scaler.npz").is_file()

    book = load_profiles()
    for pid, hold_id, original_source in (
        ("exhaust_cold_spot", "bundle-cs-hold", True),
        ("bearing_vibration", "bundle-br-hold", False),
    ):
        model = load_profile_model(settings, pid)
        assert model is not None
        assert model.tier() == "head"
        assert model.embedder == "stats"
        folder = out / "profiles" / pid
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        assert meta["embedder"] == "stats"
        assert meta["tier"] == "head"
        assert meta["channels_hash"] == channels_hash(book[pid])
        assert meta["gate"]["status"] == "pass"
        roles = meta["sources"]["roles"]
        assert "test" not in roles
        assert set(roles) <= {"train", "val", "heldout"}
        assert roles["train"] >= 22
        assert meta["sources"]["original"] == "original"
        assert meta["sources"]["profile"] == "profile"
        assert ":" not in meta["sources"]["original"]
        by_source = meta["sources"]["by_source"]
        if original_source:
            assert by_source.get("synthetic_original", 0) > 0
        else:
            assert by_source.get("synthetic_original", 0) == 0
        assert by_source.get("synthetic_profile", 0) > 0
        ids = _case_ids(folder) | _case_ids(folder / "stats")
        assert SIDE_CAR_ID not in ids
        assert ORIG_TEST_ID not in ids
        assert "bundle-cs-unseen" not in ids
        assert "bundle-cs-variant" not in ids
        assert "bundle-br-orig-tr" not in ids
        assert hold_id in ids
        stats_meta = json.loads((folder / "stats" / "meta.json").read_text(encoding="utf-8"))
        assert stats_meta["embedder"] == "stats"
        assert "test" not in stats_meta["sources"]["roles"]
        # The bundle's prior is the one this run pooled, not a file left by an earlier encoder.
        # The index carries the slot weights (load is 0.5), as the step-20 bundles did.
        expected = channel_weights(book[pid], compile_channels(book[pid]))
        assert expected.min() < 1.0
        for sub in (folder, folder / "stats"):
            cal = json.loads((sub / "calibration.json").read_text(encoding="utf-8"))
            assert cal["prior"] == {key: prior[key] for key in ("n", "radius", "match")}
            assert np.allclose(np.load(sub / "index.npz")["weights"], expected)


def test_a_stats_bundle_is_served_with_the_scaler_it_was_built_with(tmp_path, settings, monkeypatch):
    """A training row decided through the app lands on its own index row at distance 0.

    The trainer standardises statistical blocks before it indexes them. The
    decider used the identity scaler for that space, so the same file sat far
    from itself and every stats query looked unfamiliar.
    """
    from types import SimpleNamespace

    from sentinel.data import load_csv
    from sentinel.e2e.labels import Registry
    from sentinel.e2e.model import Calibration, ChannelScaler, ProfileCache
    from sentinel.e2e.pipeline import decide_alert_case
    from sentinel.e2e.train import stats_scaler
    from sentinel.profiles import resolve_tags

    original, profile = _build(tmp_path)
    out = tmp_path / "models"
    settings.embedder = "stats"
    settings.e2e_model_dir = out
    train_alert_bundles(settings, original, profile, out, device="cpu", encoders=["stats"], profiles=["exhaust_cold_spot"])
    saved = out / "encoder" / "stats_scaler.npz"
    assert saved.is_file()
    fitted = stats_scaler(settings)
    assert not np.allclose(fitted.mean, 0.0)
    assert np.allclose(ChannelScaler.load(saved).std, fitted.std)

    engine = SimpleNamespace(
        settings=settings,
        registry=Registry(settings),
        profile_cache=ProfileCache(),
        cal=Calibration(novelty_threshold=0.5, match_threshold=0.4, low_confidence=0.6),
    )
    book = load_profiles()
    prof = book["exhaust_cold_spot"]
    case = load_csv(profile / "train" / "bundle-cs-tr-n-3.csv")
    resolved = resolve_tags(prof, case, frame=getattr(case, "frame_type", None))
    out_case = decide_alert_case(case, prof, resolved, settings, engine)
    nearest = out_case["decision"]["similar"][0]
    assert nearest["id"] == "bundle-cs-tr-n-3"
    assert nearest["distance"] == pytest.approx(0.0, abs=1e-3)

    # The old serving: identity for the statistical space. The same file is no longer itself.
    identity = ChannelScaler(np.zeros(48), np.ones(48))
    monkeypatch.setattr("sentinel.e2e.train.stats_scaler", lambda _settings: identity)
    engine.profile_cache = ProfileCache()
    old = decide_alert_case(case, prof, resolved, settings, engine)
    mine = [hit for hit in old["decision"]["similar"] if hit["id"] == "bundle-cs-tr-n-3"]
    assert not mine or mine[0]["distance"] > 0.01
