"""Per-profile windows and the statistical fallback embedder.

The channel-independent TS Pulse tests that need weights live with the slow marker later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.data import load_csv
from sentinel.e2e.encoder import StatsEmbedder, get_embedder
from sentinel.e2e.features import WINDOW, profile_cache_key, profile_windows
from sentinel.profiles import profile_from_dict, resolve_tags


def _case_from(frame: pd.DataFrame):
    raw = frame.to_csv(index=False).encode("utf-8")
    return load_csv(raw, name="window.csv")


def _profile(slots: list, include_other: bool = False):
    return profile_from_dict(
        {
            "schema": 1,
            "id": "window_demo",
            "name": "Window demo",
            "include_other": include_other,
            "slots": slots,
            "signatures": ["normal"],
        }
    )


def _process(slot_id: str, column: str, role: str = "process", required: bool = False) -> dict:
    return {
        "id": slot_id,
        "role": role,
        "required": required,
        "aliases": {"*": [column]},
        **({"legacy_role": "load"} if role == "context" else {}),
    }


def _frame(columns: dict, rows: int = 80) -> pd.DataFrame:
    data = {"timestamp": pd.date_range("2024-01-01", periods=rows, freq="10min").strftime("%Y-%m-%d %H:%M:%S")}
    data.update(columns)
    return pd.DataFrame(data)


def test_profile_windows_shapes_for_several_widths():
    rows = 80
    clock = np.linspace(0, 1, rows)
    # 1 channel.
    one = _profile([_process("fuel", "fuel_flow")])
    frame = _frame({"fuel_flow": 3 + clock})
    feat = profile_windows(_case_from(frame), one, resolve_tags(one, frame))
    assert feat.x.shape == (WINDOW, 1)
    assert feat.mask.shape == (1,) and bool(feat.mask[0])
    # 4 channels: load plus three process tags.
    four_slots = [
        _process("load", "load_MW", "context"),
        _process("fuel", "fuel_flow"),
        _process("igv", "IGV_deg"),
        _process("cdp", "CDP_bar"),
    ]
    four = _profile(four_slots)
    frame = _frame({"load_MW": 40 + clock, "fuel_flow": 3 + clock, "IGV_deg": 50 + clock, "CDP_bar": 11 + clock})
    feat = profile_windows(_case_from(frame), four, resolve_tags(four, frame))
    assert feat.x.shape == (WINDOW, 4)
    assert feat.channel_ids == ["load", "fuel", "igv", "cdp"]
    # 12 channels: load, a ring (six) and five process tags. No _other.
    ring = {
        "id": "exh",
        "role": "ring",
        "aliases": {"*": [f"TC{i}" for i in range(1, 7)]},
        "legacy_role": "ring",
    }
    twelve_slots = [_process("load", "load_MW", "context"), ring] + [
        _process(f"p{i}", f"P{i}") for i in range(5)
    ]
    twelve = _profile(twelve_slots)
    cols = {"load_MW": np.full(rows, 40.0)}
    cols.update({f"TC{i}": np.full(rows, 540.0) for i in range(1, 7)})
    cols.update({f"P{i}": np.full(rows, float(i)) for i in range(5)})
    frame = _frame(cols)
    feat = profile_windows(_case_from(frame), twelve, resolve_tags(twelve, frame))
    assert feat.x.shape == (WINDOW, 12)
    assert feat.channel_ids[1:7] == ["exh:cold", "exh:cold_pair", "exh:hot", "exh:spread", "exh:avg", "exh:flat"]
    # 40 process channels.
    wide_slots = [_process(f"c{i}", f"C{i}") for i in range(40)]
    wide = _profile(wide_slots)
    frame = _frame({f"C{i}": np.full(rows, float(i)) for i in range(40)})
    feat = profile_windows(_case_from(frame), wide, resolve_tags(wide, frame))
    assert feat.x.shape == (WINDOW, 40)
    assert feat.mask.shape == (40,) and bool(feat.mask.all())


def test_a_missing_slot_is_masked_and_not_a_flat_signal():
    prof = _profile(
        [
            _process("load", "load_MW", "context", required=True),
            _process("fuel", "fuel_flow", required=True),
            _process("igv", "IGV_deg"),
        ]
    )
    rows = 80
    step = np.concatenate([np.full(40, 40.0), np.full(40, 55.0)])
    frame = _frame({"load_MW": step, "fuel_flow": 3 + np.linspace(0, 1, rows)})
    resolved = resolve_tags(prof, frame)
    assert resolved.by_id("igv").status == "missing"
    feat = profile_windows(_case_from(frame), prof, resolved)
    assert feat.channel_ids == ["load", "fuel", "igv"]
    assert list(feat.mask) == [True, True, False]
    # The masked channel is zeros in the window. It is not presented as a measured flat tag.
    assert np.all(feat.x[:, 2] == 0)
    assert np.any(feat.x[:, 0] != 0)


def test_ring_size_does_not_change_channel_ids_or_the_cold_sign():
    def build(n_sensors: int):
        rows = 120
        load = np.full(rows, 50.0)
        data = {"load_MW": load}
        for i in range(1, n_sensors + 1):
            series = np.full(rows, 540.0)
            if i == 1:
                series[60:] = 510.0
            data[f"TC{i}"] = series
        frame = _frame(data, rows=rows)
        prof = _profile(
            [
                _process("load", "load_MW", "context"),
                {
                    "id": "exh",
                    "role": "ring",
                    "aliases": {"*": {"prefix": "TC"}},
                    "legacy_role": "ring",
                },
            ]
        )
        resolved = resolve_tags(prof, frame)
        case = _case_from(frame)
        return prof, resolved, case, profile_windows(case, prof, resolved)

    small_prof, small_res, small_case, small = build(16)
    _large_prof, large_res, _large_case, large = build(36)
    assert [hit.tags and len(hit.tags) for hit in (small_res.by_id("exh"),)] == [16]
    assert len(large_res.by_id("exh").tags) == 36
    assert small.channel_ids == large.channel_ids
    cold = small.channel_ids.index("exh:cold")
    # A cold sensor that appears after the baseline pulls the cold channel down on both rings.
    assert small.x[:, cold].min() < -0.5
    assert large.x[:, cold].min() < -0.5
    assert profile_cache_key(small_case, small_prof, small_res).startswith("features3_")
    # A different resolution (the 36-sensor ring) is a different cache entry.
    assert profile_cache_key(small_case, small_prof, small_res) != profile_cache_key(_large_case, small_prof, large_res)


def test_stats_embedder_is_deterministic_and_48d():
    rng = np.random.default_rng(9)
    window = rng.normal(size=(WINDOW, 4)).astype(np.float32)
    window[:, 2] = 1.0  # a live flat channel is still embedded
    embedder = StatsEmbedder()
    first = embedder.embed_channels(window)
    second = embedder.embed_channels(window)
    assert embedder.name == "stats" and embedder.dim == 48 and embedder.fallback is False
    assert first.shape == (4, 48)
    assert first.dtype == np.float32
    assert np.allclose(first, second)
    assert np.isfinite(first).all()
    # A live constant is not the all-zero vector a masked channel gets.
    assert np.any(first[2] != 0)
    shifted = window.copy()
    shifted[:, 0] += 3.0
    assert not np.allclose(embedder.embed_channels(shifted), first)
    batch = np.stack([window, shifted])
    batched = embedder.embed_channels(batch)
    assert batched.shape == (2, 4, 48)
    assert np.allclose(batched[0], first)
    mask = np.array([True, False, True, True])
    masked = embedder.embed_channels(window, mask)
    assert np.all(masked[1] == 0)
    assert np.allclose(masked[0], first[0])
    assert np.allclose(masked[2], first[2])


def test_get_embedder_reads_the_name_and_keeps_legacy_separate():
    class _Settings:
        embedder = "stats"

    chosen = get_embedder(settings=_Settings())
    assert isinstance(chosen, StatsEmbedder) and chosen.fallback is False
    # No name and no settings: the fast fallback, so a unit test does not load weights.
    assert isinstance(get_embedder(), StatsEmbedder)
    legacy = get_embedder("legacy14")
    assert legacy.name == "legacy14" and legacy.fallback is False
    with pytest.raises(RuntimeError):
        legacy.embed_channels(np.zeros((WINDOW, 1), dtype=np.float32))
    unknown = get_embedder("not-an-embedder")
    assert isinstance(unknown, StatsEmbedder) and unknown.fallback is True
    assert "not-an-embedder" in unknown.fallback_reason


def test_tspulse_ci_falls_back_when_the_encoder_cannot_load(monkeypatch):
    import sentinel.e2e.encoder as encoder

    encoder._CI_CACHE.clear()

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("weights missing")

    monkeypatch.setattr(encoder, "CIEncoder", _Boom)
    made = encoder.get_embedder("tspulse_ci")
    assert isinstance(made, encoder.StatsEmbedder)
    assert made.fallback is True
    assert "OSError" in made.fallback_reason


def test_ci_weight_folder_uses_the_active_encoder_then_the_classification_snapshot(tmp_path):
    """No weights are loaded. The active folder wins, and an empty one falls back."""
    import json

    from sentinel.config import ROOT
    from sentinel.e2e.encoder import ci_weight_folder

    model_dir = tmp_path / "models"
    active = model_dir / "encoder"
    active.mkdir(parents=True)
    clf = tmp_path / "clf"
    clf.mkdir()
    (clf / "model.safetensors").write_bytes(b"classification snapshot")

    class _Settings:
        def __init__(self, data_dir):
            self.e2e_model_dir = model_dir
            self.tspulse_clf_dir = clf
            self.data_dir = data_dir

    # No registry and no weights in the shipped encoder folder: the classification snapshot.
    assert ci_weight_folder(settings=_Settings(tmp_path / "data")) == clf
    (active / "model.safetensors").write_bytes(b"shipped encoder")
    assert ci_weight_folder(settings=_Settings(tmp_path / "data")) == active

    # A registered eNNNN with weights replaces e0000. The top-level encoder stays on disk.
    site = tmp_path / "site"
    later = site / "e2e" / "registry" / "encoders" / "e0001"
    later.mkdir(parents=True)
    (later / "meta.json").write_text("{}\n", encoding="utf-8")
    (later / "model.safetensors").write_bytes(b"site encoder")
    current = site / "e2e" / "registry" / "current.json"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(json.dumps({"encoder": "e0001"}), encoding="utf-8")
    assert ci_weight_folder(settings=_Settings(site)) == later

    # An explicit path wins even when the active folder has weights.
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    assert ci_weight_folder(path=explicit, settings=_Settings(site)) == explicit
    # No settings: the shipped classification snapshot, not a guess at the active folder.
    assert ci_weight_folder() == ROOT / "models" / "tspulse" / "clf"


def test_embedder_cache_follows_the_resolved_weight_folder(monkeypatch, tmp_path):
    import sentinel.e2e.encoder as encoder

    encoder._CI_CACHE.clear()
    seen = []

    class _Fake:
        def __init__(self, path=None, device="auto", settings=None):
            self.path = encoder.ci_weight_folder(path, settings)
            self.fallback = False
            self.name = "tspulse_ci"
            seen.append(self.path)

    monkeypatch.setattr(encoder, "CIEncoder", _Fake)
    model_dir = tmp_path / "models"
    (model_dir / "encoder").mkdir(parents=True)
    clf = tmp_path / "clf"
    clf.mkdir()
    (clf / "model.safetensors").write_bytes(b"snapshot")

    class _Settings:
        def __init__(self):
            self.e2e_model_dir = model_dir
            self.tspulse_clf_dir = clf
            self.data_dir = tmp_path / "data"
            self.embedder = "tspulse_ci"

    first = encoder.get_embedder("tspulse_ci", _Settings())
    again = encoder.get_embedder("tspulse_ci", _Settings())
    assert again is first
    (model_dir / "encoder" / "model.safetensors").write_bytes(b"active")
    other = encoder.get_embedder("tspulse_ci", _Settings())
    assert other is not first
    assert seen == [clf, model_dir / "encoder"]


def test_used_key_check_ignores_the_head_and_flags_the_backbone():
    """The rule without weights: a used key refuses; the head and a channel mixer do not."""
    from sentinel.e2e.encoder import _dropped_channel_mixers, _head_notes, _load_problems

    info = {
        "missing_keys": {
            "decoder_with_head.head.projection.weight",
            "backbone.base_norm.bias",
        },
        "mismatched_keys": {
            ("decoder_with_head.head.projection.bias", (8, 994), (8, 71)),
            ("decoder_with_head.decoder.decoder_block.norm.weight", (4,), (8,)),
        },
        "unexpected_keys": {
            "decoder_with_head.decoder.decoder_block.mixers.0.channel_feature_mixer.mlp.fc1.weight",
            "backbone.scaler.affine_weight",
        },
    }
    problems = _load_problems(info)
    assert "missing backbone.base_norm.bias" in problems
    assert "size mismatch decoder_with_head.decoder.decoder_block.norm.weight" in problems
    assert all("projection" not in item for item in problems)
    assert all("channel_feature_mixer" not in item for item in problems)
    skipped, dropped = _dropped_channel_mixers(info)
    assert skipped == [
        "decoder_with_head.decoder.decoder_block.mixers.0.channel_feature_mixer.mlp.fc1.weight"
    ]
    assert dropped == ["unexpected backbone.scaler.affine_weight"]
    notes = _head_notes(info)
    assert "missing decoder_with_head.head.projection.weight" in notes
    assert "size mismatch decoder_with_head.head.projection.bias" in notes
    assert all(".head." in item for item in notes)
    assert all("base_norm" not in item for item in notes)


def test_gpu_oom_retries_the_batch_on_cpu():
    import sentinel.e2e.encoder as encoder

    class _Model:
        def __init__(self):
            self.placed = "cuda"

        def to(self, device):
            self.placed = device
            return self

    calls = {"n": 0}

    def run(device):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        assert device == "cpu"
        return "ok"

    model = _Model()
    value, device, reason = encoder._call_or_cpu("cuda", model, run)
    assert value == "ok" and device == "cpu" and model.placed == "cpu"
    assert "out of memory" in reason
    def _still_fails(_device):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        encoder._call_or_cpu("cpu", model, _still_fails)


@pytest.mark.slow
def test_ci_encoder_is_48d_per_channel_and_deterministic():
    from sentinel.e2e.encoder import CIEncoder

    encoder = CIEncoder(device="cpu")
    assert encoder.name == "tspulse_ci" and encoder.dim == 48 and encoder.fallback is False
    rng = np.random.default_rng(3)
    window = rng.normal(size=(WINDOW, 6)).astype(np.float32)
    first = encoder.embed_channels(window)
    second = encoder.embed_channels(window)
    assert first.shape == (6, 48)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.allclose(first, second)
    assert not np.allclose(first[0], first[1])
    mask = np.array([True, False, True, True, True, True])
    masked = encoder.embed_channels(window, mask)
    assert np.all(masked[1] == 0)
    assert np.allclose(masked[0], first[0])
    assert np.allclose(masked[2], first[2])
    # A masked channel is not the portrait of a live flat tag: zeros still move the encoder.
    flat = np.zeros((WINDOW, 2), dtype=np.float32)
    live = encoder.embed_channels(flat)
    assert np.any(live[0] != 0)
    skipped = encoder.embed_channels(flat, np.array([False, True]))
    assert np.all(skipped[0] == 0)
    assert np.allclose(skipped[1], live[1])
    batch = np.stack([window, window + 0.5])
    batched = encoder.embed_channels(batch)
    assert batched.shape == (2, 6, 48)
    assert np.allclose(batched[0], first)
    assert encoder.embed_rate > 0


@pytest.mark.slow
def test_finetuned_candidate_drops_only_channel_mixers(tmp_path):
    """The 14-channel fine-tune either loads with only channel mixers skipped, or is unavailable."""
    import json

    from sentinel.config import ROOT
    from sentinel.e2e.encoder import CIEncoder, ci_from_finetuned

    src = ROOT / "models" / "e2e" / "legacy" / "encoder"
    if not (src / "model.safetensors").is_file():
        pytest.skip("fine-tuned encoder is not in this checkout")
    dest = tmp_path / "candidate"
    report = ci_from_finetuned(src, dest)
    written = json.loads((dest / "training.json").read_text(encoding="utf-8"))
    assert written["available"] is report["available"]
    assert written["source"] == "models/e2e/legacy/encoder"
    if not report["available"]:
        assert report["reason"]
        assert not (dest / "model.safetensors").is_file()
        return
    assert report["skipped"]
    assert all("channel_feature_mixer" in name for name in report["skipped"])
    assert all(".head." in note for note in report["head_reinitialized"])
    encoder = CIEncoder(path=dest, device="cpu")
    rng = np.random.default_rng(5)
    window = rng.normal(size=(WINDOW, 3)).astype(np.float32)
    first = encoder.embed_channels(window)
    second = encoder.embed_channels(window)
    assert first.shape == (3, 48)
    assert np.allclose(first, second)
    assert np.isfinite(first).all()


@pytest.mark.slow
def test_missing_used_backbone_key_is_refused(tmp_path):
    """Deleting a backbone key the 1-channel model uses must not load as random weights."""
    import shutil

    from safetensors.numpy import load_file, save_file

    from sentinel.config import ROOT
    from sentinel.e2e.encoder import CIEncoder

    src = ROOT / "models" / "tspulse" / "clf"
    if not (src / "model.safetensors").is_file():
        pytest.skip("classification snapshot is not in this checkout")
    dest = tmp_path / "broken"
    dest.mkdir()
    shutil.copy(src / "config.json", dest / "config.json")
    weights = load_file(str(src / "model.safetensors"))
    victim = "backbone.base_norm.bias"
    assert victim in weights
    del weights[victim]
    save_file(weights, str(dest / "model.safetensors"))
    with pytest.raises(RuntimeError, match="missing backbone.base_norm.bias"):
        CIEncoder(path=dest, device="cpu")


def _unit(axis: int, dim: int = 48) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[axis] = 1.0
    return vec


def test_block_distance_properties():
    from sentinel.config import FRAME_COND_WEIGHT, FRAME_DIST_PENALTY, MASK_PENALTY
    from sentinel.e2e.model import (
        ChannelScaler,
        MaskedIndex,
        block_distances,
        head_input,
        set_pool,
        train_head,
    )

    # Two orthogonal unit blocks, both present: cosine term is 1, no mask or frame penalty.
    left = np.stack([_unit(0), _unit(1)])
    right = np.stack([_unit(1), _unit(0)])
    both = np.array([True, True])
    same = block_distances(left, both, "ring_tc", left[None, ...], both[None, ...], ["ring_tc"])
    assert same[0] == pytest.approx(0.0, abs=1e-6)
    ortho = block_distances(left, both, "ring_tc", right[None, ...], both[None, ...], ["ring_tc"])
    assert ortho[0] == pytest.approx(1.0, abs=1e-6)
    # Symmetric.
    back = block_distances(right, both, "ring_tc", left[None, ...], both[None, ...], ["ring_tc"])
    assert back[0] == pytest.approx(ortho[0], abs=1e-6)
    # One channel missing on the right: the shared channel matches, the mask penalty is the only cost.
    half = np.array([True, False])
    missing = block_distances(left, both, "", left[None, ...], half[None, ...], [""])
    assert missing[0] == pytest.approx(MASK_PENALTY * (1 - 1 / 2), abs=1e-6)
    # A known frame that differs adds the frame penalty. An unknown frame does not.
    shifted = block_distances(left, both, "ring_tc", left[None, ...], both[None, ...], ["zones"])
    assert shifted[0] == pytest.approx(FRAME_DIST_PENALTY, abs=1e-6)
    unknown = block_distances(left, both, "ring_tc", left[None, ...], both[None, ...], ["auto"])
    assert unknown[0] == pytest.approx(0.0, abs=1e-6)
    # ``*`` is the generic alias key, not a frame, on either side.
    wildcard = block_distances(left, both, "7HA", left[None, ...], both[None, ...], ["*"])
    assert wildcard[0] == pytest.approx(0.0, abs=1e-6)
    wildcard_query = block_distances(left, both, "*", left[None, ...], both[None, ...], ["7EA"])
    assert wildcard_query[0] == pytest.approx(0.0, abs=1e-6)
    # The matching channel outweighs the orthogonal one.
    heavy = block_distances(left, both, "", right[None, ...], both[None, ...], [""], weights=np.array([3.0, 1.0]))
    # dots are 0 and 0 with this pair? left is e0,e1 and right is e1,e0.
    # channel 0: <e0, e1> = 0, channel 1: <e1, e0> = 0. Weight does not change a total miss.
    assert heavy[0] == pytest.approx(1.0, abs=1e-6)
    partial = np.stack([_unit(0), _unit(0)])
    weighted = block_distances(left, both, "", partial[None, ...], both[None, ...], [""], weights=np.array([3.0, 1.0]))
    # dots: channel 0 = 1, channel 1 = 0. sim = 3/4. distance = 0.25.
    assert weighted[0] == pytest.approx(0.25, abs=1e-6)
    equal = block_distances(left, both, "", partial[None, ...], both[None, ...], [""])
    assert equal[0] == pytest.approx(0.5, abs=1e-6)

    # Scaler ignores masked rows, unit-normalises the rest, and keeps a masked channel at zero.
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(30, 3, 48))
    present = np.ones((30, 3), dtype=bool)
    present[:, 2] = False
    raw[:, 2] = 0.0
    scaler = ChannelScaler.fit(raw, present)
    only = ChannelScaler.fit(raw[:, :2].reshape(-1, 48))
    assert np.allclose(scaler.mean, only.mean)
    blocks = scaler.transform(raw[0], present[0])
    assert blocks.shape == (3, 48)
    assert np.allclose(np.linalg.norm(blocks[:2], axis=1), 1.0, atol=1e-5)
    assert np.all(blocks[2] == 0)

    pool = set_pool(left, both)
    assert pool.shape == (96,)
    assert np.allclose(np.linalg.norm(pool[:48]), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(pool[48:]), 1.0, atol=1e-5)
    # Order does not change the pool.
    assert np.allclose(pool, set_pool(left[::-1], both[::-1]))

    vector = head_input(left, both, pool, "ring_tc", ["ring_tc", "zones"])
    assert vector.shape == (48 * 2 + 96 + 2 + 2,)
    assert vector[48 * 2 + 96] == pytest.approx(0.5)
    assert vector[48 * 2 + 96 + 1] == pytest.approx(0.5)
    assert vector[-2] == pytest.approx(FRAME_COND_WEIGHT)
    assert vector[-1] == pytest.approx(0.0)
    head = train_head(np.stack([vector, -vector, vector, -vector]), ["a", "b", "a", "b"])
    assert head.W.shape[1] == vector.shape[0]

    index = MaskedIndex(weights=np.array([1.0, 1.0]))
    index.add(left, both, "ring_tc", {"id": "near", "signature": "normal"})
    index.add(right, both, "ring_tc", {"id": "far", "signature": "other"})
    hits = index.query(left, both, "ring_tc", k=2)
    assert [hit["id"] for hit in hits] == ["near", "far"]
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-4)
    assert hits[0]["common"] == 2 and hits[0]["channels"] == 2
    # A second label with the same id replaces the row.
    index.add(right, both, "zones", {"id": "near", "signature": "other"})
    replaced = index.query(left, both, "ring_tc", k=1, exclude=["far"])
    assert replaced[0]["id"] == "near"
    assert replaced[0]["distance"] == pytest.approx(1.0 + FRAME_DIST_PENALTY, abs=1e-4)


def test_unseen_query_frame_does_not_pay_the_frame_penalty():
    """A frame this index has never stored is passed as blank. The formula still penalises a seen mismatch."""
    from sentinel.config import FRAME_DIST_PENALTY
    from sentinel.e2e.model import MaskedIndex

    left = np.stack([_unit(0), _unit(1)])
    both = np.array([True, True])
    index = MaskedIndex(weights=np.array([1.0, 1.0]))
    index.add(left, both, "7EA", {"id": "a", "signature": "normal"})
    unseen = index.distances(left, both, "7HA")
    assert unseen[0] == pytest.approx(0.0, abs=1e-4)
    index.add(left, both, "7HA", {"id": "b", "signature": "normal"})
    mixed = index.distances(left, both, "7EA")
    assert mixed[0] == pytest.approx(0.0, abs=1e-4)
    assert mixed[1] == pytest.approx(FRAME_DIST_PENALTY, abs=1e-4)


def test_masked_index_roundtrip_keeps_float16_blocks(tmp_path):
    from sentinel.e2e.model import MaskedIndex

    left = np.stack([_unit(0), _unit(2), _unit(4)])
    mask = np.array([True, True, False])
    index = MaskedIndex(weights=np.array([0.5, 1.0, 1.0]))
    index.add(left, mask, "ring_tc", {"id": "c1", "signature": "normal"})
    index.save(tmp_path, "index")
    stored = np.load(tmp_path / "index.npz")
    assert stored["U"].dtype == np.float16
    loaded = MaskedIndex.load(tmp_path, "index")
    assert len(loaded) == 1
    assert np.allclose(loaded.weights, [0.5, 1.0, 1.0])
    distance = loaded.distances(left, mask, "ring_tc")
    assert distance[0] < 0.02
    hit = loaded.query(left, mask, "ring_tc", k=1)[0]
    assert hit["common"] == 2 and hit["channels"] == 3


def test_channel_weights_follow_the_slot():
    from sentinel.e2e.model import channel_weights
    from sentinel.profiles import compile_channels

    profile = _profile(
        [
            {**_process("load", "load_MW", "context"), "weight": 0.5},
            {**_process("fuel", "fuel_flow"), "health": False, "weight": 2.0},
        ]
    )
    specs = compile_channels(profile)
    weights = channel_weights(profile, specs)
    by_slot = {spec.slot_id: float(weight) for spec, weight in zip(specs, weights)}
    assert by_slot["load"] == pytest.approx(0.5)
    assert by_slot["fuel"] == pytest.approx(2.0)


def test_profile_embedding_store_roundtrip_and_selective_rebuild(tmp_path):
    from sentinel.config import load_settings
    from sentinel.e2e.labels import ProfileEmbeddings

    store = ProfileEmbeddings(load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False}))
    pulse = np.ones((2, 48), dtype=np.float32)
    pulse[0, 0] = 0.5
    stats = np.full((2, 48), 0.25, dtype=np.float32)
    mask = np.array([True, False])
    resolved = {"fuel": ["fuel_flow"], "load": ["load_MW"]}
    store.save("lube_oil_system", "case-a", pulse, stats, mask, ["fuel", "load"], "abc123", "e0000", resolved)
    store.save("lube_oil_system", "case-b", pulse, stats, mask, ["fuel", "load"], "abc123", "e0000", resolved)
    loaded = store.load("lube_oil_system", "case-a")
    assert loaded is not None
    assert np.load(tmp_path / "data" / "e2e" / "emb2" / "lube_oil_system" / "case-a.npz")["E_tspulse"].dtype == np.float16
    assert np.allclose(loaded["tspulse"][0, 0], 0.5, atol=1e-3)
    assert loaded["mask"].tolist() == [True, False]
    assert loaded["channel_ids"] == ["fuel", "load"]
    assert loaded["channels_hash"] == "abc123" and loaded["encoder_id"] == "e0000"
    # Same resolution, different key order, is not a new embedding.
    assert store.needs_embed("lube_oil_system", "case-a", "abc123", "e0000", {"load": ["load_MW"], "fuel": ["fuel_flow"]}) is False
    # An alias edit keeps the channels hash and still rebuilds only the case whose tags moved.
    edited = {"fuel": ["FQG"], "load": ["load_MW"]}
    assert store.needs_embed("lube_oil_system", "case-a", "abc123", "e0000", edited) is True
    assert store.needs_embed("lube_oil_system", "case-b", "abc123", "e0000", resolved) is False
    assert store.needs_embed("lube_oil_system", "case-b", "def456", "e0000", resolved) is True
    assert store.needs_embed("lube_oil_system", "case-c", "abc123", "e0000", resolved) is True
