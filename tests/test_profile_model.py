"""Per-profile tiers: cold start, kNN vote, head, and the engine's model cache."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sentinel.e2e.model import Head, MaskedIndex, ProfileCache, ProfileModel, capped_match
from sentinel.e2e.pipeline import Engine


def _unit(axis: int) -> np.ndarray:
    vec = np.zeros(48, dtype=np.float32)
    vec[axis] = 1.0
    return vec


def _pair() -> np.ndarray:
    return np.stack([_unit(0), _unit(1)])


def _mask() -> np.ndarray:
    return np.array([True, True])


def _index(*rows: tuple) -> MaskedIndex:
    index = MaskedIndex()
    block = _pair()
    for rec in rows:
        index.add(block, _mask(), "ring_tc", dict(rec))
    return index


def _prior(radius: float = 0.4, match: float = 0.3) -> dict:
    return {"prior": {"n": 0, "radius": radius, "match": match}}


def test_cold_start_asks_for_a_label_and_uses_the_rule_action():
    model = ProfileModel.empty("inlet_filter_dp")
    assert model.tier() == "cold"
    decided = model.decide_profile(_pair(), _mask(), "ring_tc")
    assert decided["signature"] == "novel_pattern"
    assert decided["basis"] == "cold_start"
    assert decided["needs_human"] is True
    assert decided["primary_action"] == "escalate"
    assert decided["rules_primary"] is True
    assert "label this case" in decided["note"]
    fired = model.decide_profile(_pair(), _mask(), rule_action="monitor_only")
    assert fired["primary_action"] == "monitor_only"
    # An ad-hoc header is cold even when somebody stored a label on it.
    adhoc = ProfileModel("_adhoc", _index({"id": "a", "signature": "normal", "source": "engineer"}))
    assert adhoc.tier() == "cold"


def test_knn_votes_and_flags_a_split():
    model = ProfileModel(
        "lube_oil_system",
        _index(
            {"id": "a", "signature": "normal", "source": "engineer", "action": "monitor_only"},
            {"id": "b", "signature": "lube_cooler_fouling", "source": "engineer", "action": "prompt"},
            {"id": "c", "signature": "bearing_wear_trend", "source": "engineer", "action": "prompt"},
        ),
        calibration=_prior(),
    )
    assert model.tier() == "knn"
    decided = model.decide_profile(_pair(), _mask(), "ring_tc")
    assert decided["basis"] == "knn"
    assert decided["vote_share"] < 0.6
    assert decided["needs_human"] is True
    assert decided["audit"].startswith("tier knn (3 labels")
    # One signature, well inside the radius: the vote is unanimous.
    sure = ProfileModel(
        "lube_oil_system",
        _index({"id": "a", "signature": "normal", "source": "synthetic", "action": "monitor_only"}),
        calibration=_prior(),
    )
    agreed = sure.decide_profile(_pair(), _mask(), "ring_tc")
    assert agreed["signature"] == "normal"
    assert agreed["needs_human"] is False
    assert agreed["one_shot"] is False


def test_radius_falls_from_frame_to_profile_to_prior():
    calibration = {
        "frame": {"ring_tc": {"n": 20, "radius": 0.2, "match": 0.1}},
        "profile": {"n": 25, "radius": 0.5, "match": 0.3},
        "prior": {"n": 0, "radius": 0.8, "match": 0.4},
    }
    model = ProfileModel("exhaust_cold_spot", _index({"id": "a", "signature": "normal", "source": "synthetic"}), calibration=calibration)
    assert model.radius_for("ring_tc")[:2] == (0.2, 0.1)
    assert model.radius_for("ring_tc")[2] == "frame"
    thin = dict(calibration)
    thin["frame"] = {"ring_tc": {"n": 5, "radius": 0.2, "match": 0.1}}
    model.calibration = thin
    assert model.radius_for("ring_tc")[2] == "profile"
    model.calibration = {"frame": {}, "profile": {"n": 4, "radius": 0.5, "match": 0.3}, "prior": calibration["prior"]}
    radius = model.radius_for("ring_tc")
    assert radius[2] == "prior" and radius[0] == 0.8
    # The statistical fallback on a prior radius still needs a person.
    decided = model.decide_profile(_pair(), _mask(), "ring_tc", fallback=True)
    assert decided["needs_human"] is True and decided["embedder_fallback"] is True
    # A missing required slot caps confidence.
    capped = model.decide_profile(_pair(), _mask(), "ring_tc", missing_required=True)
    assert capped["needs_human"] is True and capped["confidence"] <= 0.6


def test_head_tier_prefers_one_shot_then_the_classifier():
    rows = []
    for i in range(10):
        rows.append({"id": f"n{i}", "signature": "normal", "source": "synthetic", "action": "monitor_only"})
    for i in range(10):
        rows.append({"id": f"f{i}", "signature": "drain_tc_fault", "source": "synthetic", "action": "prompt"})
    block = _pair()
    mask = _mask()
    from sentinel.e2e.model import head_input, set_pool

    vector = head_input(block, mask, set_pool(block, mask), "ring_tc", ["ring_tc"])
    head = Head(["normal", "drain_tc_fault"], np.zeros((2, vector.shape[0]), np.float32), np.array([4.0, 0.0], np.float32))
    passed = ProfileModel("exhaust_cold_spot", _index(*rows), head=head, head_passed=True, frames=["ring_tc"], calibration=_prior(0.5, 0.4))
    assert passed.tier() == "head"
    classified = passed.decide_profile(block, mask, "ring_tc")
    assert classified["basis"] == "classifier" and classified["signature"] == "normal"
    assert classified["head_probs"]["normal"] >= classified["head_probs"]["drain_tc_fault"]
    assert set(classified["head_probs"]) == {"normal", "drain_tc_fault"}
    far = np.stack([_unit(2), _unit(3)])
    novel = passed.decide_profile(far, mask, "ring_tc")
    assert novel["basis"] == "novelty"
    assert novel["head_probs"]["normal"] > 0
    # Not enough agreement to leave kNN, even with a head sitting there.
    held = ProfileModel("exhaust_cold_spot", _index(*rows), head=head, head_passed=False, calibration=_prior())
    assert held.tier() == "knn"
    # An engineer label of the other signature, as close as the training rows, overrides the head.
    override_rows = list(rows)
    override_rows[0] = {"id": "eng", "signature": "drain_tc_fault", "source": "engineer", "action": "prompt"}
    override = ProfileModel(
        "exhaust_cold_spot",
        _index(*override_rows),
        head=head,
        head_passed=True,
        frames=["ring_tc"],
        calibration=_prior(0.5, 0.4),
    )
    decided = override.decide_profile(block, mask, "ring_tc")
    assert decided["basis"] == "one_shot" and decided["signature"] == "drain_tc_fault"


@pytest.mark.parametrize("source", ["synthetic_original", "synthetic_profile"])
def test_per_alert_bundle_rows_are_not_engineer_labels(source):
    """A shipped per-alert row of the other signature, as close as the rest, does not override the head."""
    from sentinel.e2e.model import head_input, is_synthetic_source, set_pool

    rows = [{"id": f"n{i}", "signature": "normal", "source": source, "action": "monitor_only"} for i in range(10)]
    rows += [{"id": f"f{i}", "signature": "drain_tc_fault", "source": source, "action": "prompt"} for i in range(10)]
    # The nearest row (index order breaks the tie) is the other signature.
    rows.insert(0, rows.pop())
    block = _pair()
    mask = _mask()
    vector = head_input(block, mask, set_pool(block, mask), "ring_tc", ["ring_tc"])
    head = Head(["normal", "drain_tc_fault"], np.zeros((2, vector.shape[0]), np.float32), np.array([4.0, 0.0], np.float32))
    model = ProfileModel("exhaust_cold_spot", _index(*rows), head=head, head_passed=True, frames=["ring_tc"], calibration=_prior(0.5, 0.4))
    assert model.decide_profile(block, mask, "ring_tc")["similar"][0]["signature"] == "drain_tc_fault"
    decided = model.decide_profile(block, mask, "ring_tc")
    assert decided["basis"] == "classifier" and decided["signature"] == "normal"
    knn = ProfileModel("exhaust_cold_spot", _index(*rows[:6]), calibration=_prior(0.5, 0.4))
    assert knn.tier() == "knn"
    assert knn.decide_profile(block, mask, "ring_tc")["one_shot"] is False
    assert is_synthetic_source(source) and is_synthetic_source("synthetic")
    assert not is_synthetic_source("engineer") and not is_synthetic_source("")


def test_a_far_neighbor_is_novel():
    model = ProfileModel(
        "bearing_vibration",
        _index({"id": "a", "signature": "normal", "source": "synthetic"}),
        calibration=_prior(radius=0.05, match=0.02),
    )
    other = np.stack([_unit(2), _unit(3)])
    decided = model.decide_profile(other, _mask(), "ring_tc")
    assert decided["signature"] == "novel_pattern" and decided["basis"] == "novelty"
    assert decided["needs_human"] is True


def _two_class_head(bias_normal: float):
    """A head whose answer does not depend on the input: ``normal`` with this bias over the fault."""
    from sentinel.e2e.model import head_input, set_pool

    block = _pair()
    vector = head_input(block, _mask(), set_pool(block, _mask()), "ring_tc", ["ring_tc"])
    return Head(["normal", "drain_tc_fault"], np.zeros((2, vector.shape[0]), np.float32), np.array([bias_normal, 0.0], np.float32))


def _agreement_model(bias_normal: float, floor):
    rows = [{"id": f"n{i}", "signature": "normal", "source": "synthetic_original", "action": "monitor_only"} for i in range(10)]
    rows += [{"id": f"f{i}", "signature": "drain_tc_fault", "source": "synthetic_original", "action": "prompt"} for i in range(10)]
    calibration = {"profile": {"n": 40, "radius": 0.5, "match": 0.4}, "prior": {"n": 0, "radius": 0.9, "match": 0.5}}
    if floor is not None:
        calibration["profile"]["confidence"] = floor
    return ProfileModel(
        "exhaust_cold_spot", _index(*rows), head=_two_class_head(bias_normal), head_passed=True,
        frames=["ring_tc"], calibration=calibration,
    )


def test_novel_needs_a_far_case_and_an_unsure_head():
    """Beyond the radius alone is not novel when the head is confident; both together are."""
    far = np.stack([_unit(2), _unit(3)])
    confident = _agreement_model(4.0, 0.6)  # p(normal) = 0.982
    assert confident.confidence_floor() == pytest.approx(0.6)
    kept = confident.decide_profile(far, _mask(), "ring_tc")
    assert kept["nearest_distance"] > kept["novelty_threshold"]
    assert kept["novel"] is False and kept["signature"] == "normal" and kept["basis"] == "classifier"
    assert kept["beyond_radius"] is True and kept["needs_human"] is True
    assert kept["confidence_floor"] == pytest.approx(0.6)

    unsure = _agreement_model(0.2, 0.6)  # p(normal) = 0.55
    novel = unsure.decide_profile(far, _mask(), "ring_tc")
    assert novel["novel"] is True and novel["signature"] == "novel_pattern" and novel["basis"] == "novelty"
    assert novel["head_confidence"] < 0.6

    # Near the library an unsure head is not novel either: it is the classifier, low confidence.
    near = unsure.decide_profile(_pair(), _mask(), "ring_tc")
    assert near["novel"] is False and near["signature"] == "normal" and "beyond_radius" not in near
    assert near["needs_human"] is False


def test_without_a_floor_the_distance_alone_decides():
    """A bundle calibrated before the floor existed keeps the old rule."""
    far = np.stack([_unit(2), _unit(3)])
    old = _agreement_model(4.0, None)
    assert old.confidence_floor() is None
    decided = old.decide_profile(far, _mask(), "ring_tc")
    assert decided["novel"] is True and decided["basis"] == "novelty"
    # A kNN-tier model has no head, so no floor, whatever the calibration says.
    knn = ProfileModel(
        "exhaust_cold_spot",
        _index({"id": "a", "signature": "normal", "source": "synthetic"}),
        calibration={"profile": {"n": 40, "radius": 0.05, "match": 0.02, "confidence": 0.9}, "prior": _prior()["prior"]},
    )
    assert knn.tier() == "knn" and knn.confidence_floor() is None
    assert knn.decide_profile(far, _mask(), "ring_tc")["novel"] is True


def test_empty_model_reads_the_prior_file(tmp_path):
    """The cold radius is the prior file. A missing file keeps 0.5."""
    path = tmp_path / "novelty_prior.json"
    path.write_text(json.dumps({"n": 12, "radius": 0.31, "match": 0.2}), encoding="utf-8")
    model = ProfileModel.empty("inlet_filter_dp", prior_path=path)
    assert model.radius_for("7HA") == (0.31, 0.2, "prior")
    missing = ProfileModel.empty("inlet_filter_dp", prior_path=tmp_path / "absent.json")
    assert missing.radius_for("7HA") == (0.5, 0.4, "prior")
    path.write_text("not json", encoding="utf-8")
    broken = ProfileModel.empty("inlet_filter_dp", prior_path=path)
    assert broken.radius_for("7HA")[0] == 0.5


def test_frame_match_is_the_median_capped_at_eighty_percent():
    """The median stands when it is inside the cap, and the cap wins when it is not."""
    assert capped_match([0.1, 0.2, 0.9], 0.5) == pytest.approx(0.2)
    assert capped_match([0.9, 0.9, 0.9], 0.5) == pytest.approx(0.4)
    assert capped_match([], 0.5) == pytest.approx(0.4)


def test_frame_bucket_match_is_not_eighty_percent_of_the_radius():
    """Nineteen close rows and one far row: the median is 0, so the match is not 0.8 times the radius."""
    from sentinel.e2e.train import _profile_calibration

    index = MaskedIndex()
    close = _unit(0).reshape(1, 48)
    far = _unit(1).reshape(1, 48)
    mask = np.array([True])
    for i in range(19):
        index.add(close, mask, "7EA", {"id": f"n{i}", "signature": "normal"})
    index.add(far, mask, "7EA", {"id": "far", "signature": "normal"})
    frame = _profile_calibration(index)["frame"]["7EA"]
    assert frame["n"] == 20
    assert frame["match"] == pytest.approx(0.0, abs=1e-4)
    assert frame["radius"] > 0.2
    assert frame["match"] < 0.8 * frame["radius"]


def test_engine_cache_drops_the_least_recent_profile():
    cache = ProfileCache(2)

    class _Host:
        profile_cache = cache

    built = []

    def build(profile_id):
        built.append(profile_id)
        return profile_id

    host = _Host()
    assert Engine.profile_model(host, "a", build=build) == "a"
    assert Engine.profile_model(host, "b", build=build) == "b"
    assert Engine.profile_model(host, "a", build=build) == "a"
    assert Engine.profile_model(host, "c", build=build) == "c"
    assert "b" not in cache
    assert cache.get("a") == "a" and cache.get("c") == "c"
    assert built == ["a", "b", "c"]
    fresh = Engine.profile_model(host, "fresh")
    assert isinstance(fresh, ProfileModel)
    assert cache.get("fresh").tier() == "cold"
