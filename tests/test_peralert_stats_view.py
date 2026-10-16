"""The per-alert head reads the statistical view next to the TS Pulse blocks.

The kNN distance and the novelty radius stay on the TS Pulse blocks. No TS Pulse
weights are loaded and nothing scores the 320 acceptance rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.config import load_settings
from sentinel.e2e.labels import LabelStore, ProfileEmbeddings
from sentinel.e2e.model import MaskedIndex, ProfileModel, head_input, head_vector, set_pool
from sentinel.e2e.train import (
    _attach_stats,
    _fit_profile_head,
    _model_from_rows,
    _save_alert_bundle,
    collect_profile_rows,
    shipped_heldout,
    shipped_rows,
    stats_scaler,
)


def _unit(axis: int, channels: int = 2) -> np.ndarray:
    vec = np.zeros((channels, 48), dtype=np.float32)
    vec[:, axis] = 1.0
    return vec


def _rows(n: int, *, with_stats: bool = True, seed: int = 2) -> list:
    """Two signatures with the same TS Pulse blocks. Only the statistical view tells them apart."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        signature = "normal" if i % 2 else "dynamics_probe_fault"
        u = _unit(0) + rng.normal(0.0, 0.05, size=(2, 48)).astype(np.float32)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        row = {"id": f"r{seed}-{i}", "u": u, "mask": np.array([True, True]), "signature": signature, "frame": "7EA", "source": "synthetic_original"}
        if with_stats:
            stats = _unit(5 if signature == "normal" else 9) + rng.normal(0.0, 0.05, size=(2, 48)).astype(np.float32)
            row["s"] = stats / np.linalg.norm(stats, axis=1, keepdims=True)
        out.append(row)
    return out


def test_head_vector_appends_the_statistical_view_only_when_asked():
    u, s = _unit(0), _unit(3)
    mask = np.array([True, False])
    alone = head_vector(u, mask, "7EA", ["7EA"])
    assert np.array_equal(alone, head_input(u, mask, set_pool(u, mask), "7EA", ["7EA"]))
    both = head_vector(u, mask, "7EA", ["7EA"], s, stats_view=True)
    assert both.shape[0] == 2 * alone.shape[0]
    assert np.array_equal(both[: alone.shape[0]], alone)
    assert np.array_equal(both[alone.shape[0]:], head_input(s, mask, set_pool(s, mask), "7EA", ["7EA"]))
    with pytest.raises(ValueError, match="statistical view"):
        head_vector(u, mask, "7EA", ["7EA"], None, stats_view=True)
    batch = head_vector(np.stack([u, u]), np.stack([mask, mask]), ["7EA", "7EA"], ["7EA"], np.stack([s, s]), stats_view=True)
    assert batch.shape == (2, both.shape[0])


def test_a_head_on_both_views_separates_what_ts_pulse_alone_cannot():
    rows = _rows(40)
    both = _fit_profile_head(rows, ["7EA"])
    assert both.meta["stats_view"] is True
    alone = _fit_profile_head([{k: v for k, v in row.items() if k != "s"} for row in rows], ["7EA"])
    assert alone.meta["stats_view"] is False
    probe = _rows(20, seed=9)

    def accuracy(head):
        ok = 0
        for row in probe:
            vec = head_vector(row["u"], row["mask"], "7EA", ["7EA"], row["s"], stats_view=bool(head.meta["stats_view"]))
            ok += head.predict(vec[None, :])[0] == row["signature"]
        return ok / len(probe)

    assert accuracy(both) == 1.0
    assert accuracy(alone) < 0.8
    # One row without its statistical blocks keeps the whole head on TS Pulse alone.
    mixed = list(rows)
    mixed[3] = {k: v for k, v in mixed[3].items() if k != "s"}
    assert _fit_profile_head(mixed, ["7EA"]).meta["stats_view"] is False


def test_the_decider_reads_the_second_view_and_needs_it(tmp_path):
    rows = _rows(40)
    model = _model_from_rows("combustion_dynamics", rows, use_head=True, val_rows=_rows(30, seed=3), prior_path=tmp_path / "absent.json")
    assert model.tier() == "head" and model.head.meta["stats_view"] is True
    probe = _rows(4, seed=4)[1]
    decided = model.decide_profile(probe["u"], probe["mask"], "7EA", stats=probe["s"])
    assert decided["head_signature"] == probe["signature"]
    with pytest.raises(ValueError, match="statistical view"):
        model.decide_profile(probe["u"], probe["mask"], "7EA")
    # The distance never reads the second view.
    assert model.index.S is not None
    plain = _model_from_rows("combustion_dynamics", [{k: v for k, v in r.items() if k != "s"} for r in rows], use_head=False)
    assert np.allclose(
        model.index.distances(probe["u"], probe["mask"], "7EA"),
        plain.index.distances(probe["u"], probe["mask"], "7EA"),
    )


def test_index_keeps_statistical_blocks_only_while_every_row_has_them(tmp_path):
    index = MaskedIndex()
    index.add(_unit(0), np.array([True, True]), "7EA", {"id": "a", "signature": "normal"}, stats=_unit(4))
    index.add(_unit(1), np.array([True, True]), "7EA", {"id": "b", "signature": "normal"}, stats=_unit(5))
    assert index.S.shape == index.U.shape
    index.add(_unit(1), np.array([True, True]), "7EA", {"id": "b", "signature": "normal"}, stats=_unit(6))
    assert np.allclose(index.S[1], _unit(6))
    index.save(tmp_path / "with")
    back = MaskedIndex.load(tmp_path / "with")
    assert back.S is not None and np.allclose(back.S, index.S, atol=1e-3)
    index.add(_unit(2), np.array([True, True]), "7EA", {"id": "c", "signature": "normal"})
    assert index.S is None
    index.save(tmp_path / "without")
    assert "S" not in np.load(tmp_path / "without" / "index.npz").files
    assert MaskedIndex.load(tmp_path / "without").S is None


def test_attach_stats_matches_by_id_and_refuses_a_gap():
    pulse = {"cd": {"train": [{"id": "x", "u": _unit(0)}, {"id": "y", "u": _unit(1)}], "val": [], "heldout": []}}
    stats = {"cd": {"train": [{"id": "y", "u": _unit(7)}, {"id": "x", "u": _unit(6)}], "val": [], "heldout": []}}
    _attach_stats(pulse, stats)
    assert np.array_equal(pulse["cd"]["train"][0]["s"], _unit(6))
    assert np.array_equal(pulse["cd"]["train"][1]["s"], _unit(7))
    gap = {"cd": {"train": [{"id": "z", "u": _unit(0)}], "val": [], "heldout": []}}
    with pytest.raises(RuntimeError, match="no statistical blocks"):
        _attach_stats(gap, stats)


def test_bundle_rows_and_engineer_labels_carry_the_second_view(tmp_path):
    settings = load_settings(
        {
            "data_dir": tmp_path / "data",
            "e2e_model_dir": tmp_path / "models",
            "use_llm": False,
            "use_tspulse": False,
            "embedder": "tspulse_ci",
        }
    )
    rows = _rows(24)
    held = _rows(6, seed=5)
    model = _model_from_rows("exhaust_cold_spot", rows, use_head=True, val_rows=_rows(10, seed=6), prior_path=tmp_path / "absent.json")
    folder = tmp_path / "models" / "profiles" / "exhaust_cold_spot"
    _save_alert_bundle(folder, model, held, {"embedder": "tspulse_ci", "gate": {"status": "pass"}})
    ship = shipped_rows(settings, "exhaust_cold_spot")
    assert len(ship) == 24 and all(row.get("s") is not None for row in ship)
    assert np.allclose(ship[0]["s"], rows[0]["s"], atol=1e-3)
    hold = shipped_heldout(settings, "exhaust_cold_spot")
    assert len(hold) == 6 and all(row.get("s") is not None for row in hold)

    store = ProfileEmbeddings(settings)
    pulse, stats = _unit(0), _unit(8)
    store.save("exhaust_cold_spot", "eng-1", pulse, stats, np.array([True, True]), ["a", "b"], "hash", "e0000", {})
    LabelStore(settings).add({"case_id": "eng-1", "signature": "normal", "action": "monitor_only", "profile": "exhaust_cold_spot", "verdict": "good"})
    labeled = collect_profile_rows(settings, "exhaust_cold_spot")
    assert len(labeled) == 1
    expect = stats_scaler(settings).transform(stats, np.array([True, True]))
    assert np.allclose(labeled[0]["s"], expect)
    settings.embedder = "stats"
    assert "s" not in collect_profile_rows(settings, "exhaust_cold_spot")[0]
