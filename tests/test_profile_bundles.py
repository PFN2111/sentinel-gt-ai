"""Shipped per-profile bundles, and the 14-channel bundle moved without a byte change."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from conftest import ROOT

# SHA-256 of the 14-channel files before they moved, and of the classification snapshot
# that the channel-independent encoder is a copy of. Compared after the move.
_LEGACY = {
    "encoder/model.safetensors": "6bcf469afccb8eb96953447c1069548767f755e480befab8fec6002aa553f443",
    "encoder/config.json": "cfa1671af8f64875196893c3906358d6dd66ce63e716196036d6f615bfa2d2be",
    "encoder/classes.json": "18f0cd863009874140a80032937665e2db5503713007e73ac9337fb2a18c92b7",
    "base/head.npz": "d57ae2f34c0dff0d692266cc8461cab04c66a5353803b3578dc67562681d1b0c",
    "base/scaler.npz": "b21bfd6df4c3b9780cc693b329ea0d54442be1de251dfcfcf8eec8a7f84e52a5",
    "base/index.npz": "834814edd1a2846713eb551e5b8a9252acb305df85588a353ad1ebf56c00f7dd",
    "base/calibration.json": "a83da9442459861f134df6b682110c243bdac8363b254d0ba43fe61eb02f4172",
}
# The shared encoder is the fine-tuned candidate chosen on validation, not the
# frozen classification snapshot (71fddf71...).
_CI_SAFETENSORS = "6f84b9c490d84302ded03a4a466fdd035043bebb425baf1f4f93e41fafd04659"
SHIPPED = (
    "exhaust_cold_spot",
    "combustion_dynamics",
    "exhaust_spread_trip",
    "igv_fuel_control",
    "bearing_vibration",
    "inlet_filter_dp",
    "lube_oil_system",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_legacy_model_dir_prefers_the_moved_bundle(tmp_path):
    from sentinel.e2e.labels import legacy_model_dir

    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "scaler.npz").write_bytes(b"old")
    assert legacy_model_dir(tmp_path, "base") == tmp_path / "base"
    legacy = tmp_path / "legacy" / "base"
    legacy.mkdir(parents=True)
    (legacy / "scaler.npz").write_bytes(b"kept")
    assert legacy_model_dir(tmp_path, "base") == legacy

    enc = tmp_path / "encoder"
    enc.mkdir()
    (enc / "config.json").write_text("{}", encoding="utf-8")
    (enc / "classes.json").write_text("[]", encoding="utf-8")
    assert legacy_model_dir(tmp_path, "encoder") == enc
    moved = tmp_path / "legacy" / "encoder"
    moved.mkdir()
    (moved / "classes.json").write_text("[]", encoding="utf-8")
    assert legacy_model_dir(tmp_path, "encoder") == moved


def test_legacy_bundle_bytes_were_kept():
    """The 14-channel files moved under legacy/ and were not rewritten."""
    root = ROOT / "models" / "e2e"
    assert not (root / "base").exists()
    for rel, digest in _LEGACY.items():
        path = root / "legacy" / rel
        assert path.is_file(), rel
        assert _sha256(path) == digest, rel
    ci = root / "encoder" / "model.safetensors"
    assert ci.is_file()
    assert _sha256(ci) == _CI_SAFETENSORS
    assert _sha256(ci) != _LEGACY["encoder/model.safetensors"]
    assert not (root / "encoder" / "classes.json").exists()
    note = json.loads((root / "encoder" / "training.json").read_text(encoding="utf-8"))
    assert note["embedder"] == "tspulse_ci" and note["chosen"] == "finetuned"
    assert float(note["candidates"]["finetuned"]["signature_acc"]) > float(note["candidates"]["frozen"]["signature_acc"])


def test_shipped_profile_bundles_load():
    """Each built-in alert type loads. The heater cold-start split is not a bundle."""
    from sentinel.config import load_settings
    from sentinel.e2e.train import load_profile_model

    root = ROOT / "models" / "e2e"
    assert (root / "encoder" / "channel_scaler.npz").is_file()
    prior = json.loads((root / "encoder" / "novelty_prior.json").read_text(encoding="utf-8"))
    assert prior["embedder"] == "tspulse_ci"
    assert 0 < float(prior["radius"]) < 2
    settings = load_settings()
    card = json.loads((root / "card.json").read_text(encoding="utf-8"))
    assert "channels" in card and "signatures" in card
    assert set(card["profiles"]) == set(SHIPPED)
    for pid in SHIPPED:
        model = load_profile_model(settings, pid)
        assert model is not None, pid
        assert len(model.index) > 0
        assert model.tier() in {"head", "knn"}
        assert (root / "profiles" / pid / "index.npz").is_file()
        assert (root / "profiles" / pid / "calibration.json").is_file()
        assert (root / "profiles" / pid / "meta.json").is_file()
        meta = json.loads((root / "profiles" / pid / "meta.json").read_text(encoding="utf-8"))
        assert meta["gate"]["status"] == "pass"
        assert meta["embedder"] == "tspulse_ci"
    assert not (root / "profiles" / "fuel_gas_heater").exists()
    # A stats query loads the stats sub-bundle. It does not load the tspulse index.
    stats = load_settings({"embedder": "stats"})
    stats_model = load_profile_model(stats, "exhaust_cold_spot")
    assert stats_model is not None and stats_model.embedder == "stats"
    ci_model = load_profile_model(settings, "exhaust_cold_spot")
    assert ci_model is not None and ci_model.embedder == "tspulse_ci"
    assert float(stats_model.index.U.reshape(-1)[0]) != float(ci_model.index.U.reshape(-1)[0])
