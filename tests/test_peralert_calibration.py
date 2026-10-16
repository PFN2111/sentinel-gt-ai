"""Novelty radius from cases the index was not fitted on, not from leave-one-out on the train rows.

Offline. No encoder weights. Decision D3: every validation signature counts.
The radius is the split-conformal order statistic, one per alert type.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sentinel.e2e.model import MaskedIndex, ProfileModel
from sentinel.e2e.train import (
    _fit_profile_head,
    _frame_names,
    _head_confidences,
    _model_from_rows,
    _profile_calibration,
    calibrate_alert_type,
    calibrate_site,
    conformal_lower,
    conformal_upper,
    fold_of_case,
    pooled_prior,
)


def _unit(axis: int) -> np.ndarray:
    vec = np.zeros(48, dtype=np.float32)
    vec[axis] = 1.0
    return vec.reshape(1, 48)


def _add(index: MaskedIndex, vec: np.ndarray, case_id: str, signature: str = "normal", frame: str = "") -> None:
    index.add(vec, np.array([True]), frame, {"id": case_id, "signature": signature})


def _index() -> MaskedIndex:
    return MaskedIndex()


def _val(vec: np.ndarray, case_id: str, signature: str = "normal", frame: str = "") -> dict:
    return {"id": case_id, "u": vec, "mask": np.array([True]), "signature": signature, "frame": frame}


def _ids_in_fold(fold: int, count: int, start: int = 0) -> list:
    found = []
    i = start
    while len(found) < count:
        case_id = f"site-{i}"
        if fold_of_case(case_id) == fold:
            found.append(case_id)
        i += 1
    return found


def test_builtin_radius_pools_validation_and_cross_fitted_rows_not_leave_one_out(tmp_path):
    """Twenty validation copies of one vector, measured against a different train vector."""
    missing = tmp_path / "absent.json"
    index = _index()
    for i in range(25):
        _add(index, _unit(2), f"train-{i}")
    val = [_val(_unit(0), f"val-{i}", signature="normal" if i < 15 else "fault") for i in range(20)]
    high = calibrate_alert_type(index, val, 0.98, prior_path=missing)
    mid = calibrate_alert_type(index, val[:10] + [_val(_unit(2), f"near-{i}") for i in range(10)], 0.50, prior_path=missing)
    # 20 validation rows plus the 25 train rows, each against the other four folds.
    assert high["profile"]["n"] == 45
    assert high["profile"]["radius"] == pytest.approx(1.0)
    # The fault rows are in the count. Calibrating on normal alone would be n=15.
    assert sum(1 for row in val if row["signature"] == "fault") == 5
    # A lower quantile on a mixed set is a smaller radius. The argument is used.
    assert mid["profile"]["radius"] < high["profile"]["radius"]
    loo = _profile_calibration(index, prior_path=missing)
    assert loo["profile"]["radius"] == pytest.approx(0.0)
    assert high["profile"]["radius"] != loo["profile"]["radius"]


def test_one_radius_per_alert_type_and_no_frame_bucket(tmp_path):
    """A frame with many rows still does not get its own radius. The decider uses the alert type's."""
    index = _index()
    _add(index, _unit(0), "train")
    val = [_val(_unit(0), f"a-{i}", frame="7EA") for i in range(19)]
    val += [_val(_unit(1), f"b-{i}", frame="7HA") for i in range(200)]
    cal = calibrate_alert_type(index, val, 0.98, prior_path=tmp_path / "absent.json")
    assert "frame" not in cal
    assert cal["profile"]["n"] == 219
    model = ProfileModel("combustion_dynamics", index, calibration=cal)
    assert model.radius_for("7HA") == (cal["profile"]["radius"], cal["profile"]["match"], "profile")
    assert model.radius_for("7EA")[2] == "profile"


def test_conformal_order_statistic():
    """k = ceil((n + 1) q). Never below np.quantile; the maximum when n is too small for q."""
    values = list(range(1, 100))  # n = 99
    assert conformal_upper(values, 0.98) == 98.0
    assert float(np.quantile(values, 0.98)) < 98.0
    assert conformal_upper(list(range(1, 49)), 0.98) == 48.0  # k = 48 = n
    assert conformal_upper(list(range(1, 21)), 0.98) == 20.0  # k = 21 > n: the largest
    assert conformal_upper([], 0.98) is None
    rng = np.random.default_rng(7)
    for n in (20, 33, 67, 88, 200, 606):
        draw = rng.gamma(4.0, 0.07, size=n)
        assert conformal_upper(draw, 0.98) >= float(np.quantile(draw, 0.98))
    # j = floor((n + 1) (1 - q)); the smallest when j < 1.
    assert conformal_lower(values, 0.98) == 2.0
    assert conformal_lower(list(range(1, 21)), 0.98) == 1.0
    assert conformal_lower([], 0.98) is None


def test_conformal_radius_holds_its_rate_on_new_draws():
    """Out of sample, the conformal radius on 99 rows flags 2 %; np.quantile on the same rows flags more.

    With 20 rows no order statistic reaches 0.98: the largest value flags 1 in 21.
    """
    rng = np.random.default_rng(11)
    flagged_conformal = []
    flagged_quantile = []
    flagged_small = []
    for _trial in range(400):
        calib = rng.gamma(4.0, 0.07, size=99)
        fresh = rng.gamma(4.0, 0.07, size=500)
        flagged_conformal.append(float(np.mean(fresh > conformal_upper(calib, 0.98))))
        flagged_quantile.append(float(np.mean(fresh > np.quantile(calib, 0.98))))
        flagged_small.append(float(np.mean(fresh > conformal_upper(calib[:20], 0.98))))
    assert np.mean(flagged_conformal) == pytest.approx(0.02, abs=0.003)
    assert np.mean(flagged_quantile) > np.mean(flagged_conformal) + 0.005
    assert np.mean(flagged_small) == pytest.approx(1.0 / 21.0, abs=0.006)


def test_site_with_25_labels_uses_cross_fitted_distances(tmp_path):
    """Same-fold neighbours do not set the radius. The other four folds do."""
    missing = tmp_path / "absent.json"
    same_fold = _ids_in_fold(0, 5)
    other = []
    i = 0
    while len(other) < 20:
        case_id = f"other-{i}"
        if fold_of_case(case_id) != 0:
            other.append(case_id)
        i += 1
    assert len({fold_of_case(case_id) for case_id in other}) >= 2
    index = _index()
    for case_id in same_fold:
        _add(index, _unit(0), case_id)
    for case_id in other:
        _add(index, _unit(1), case_id)
    cal = calibrate_site(index, 0.98, prior_path=missing)
    folds = [fold_of_case(rec["id"]) for rec in index.records]
    cross = []
    loo = []
    for i, _rec in enumerate(index.records):
        dist = np.asarray(index.distances(index.U[i], index.mask[i], index.frames[i]), dtype=np.float64).copy()
        held = dist.copy()
        held[i] = np.inf
        loo.append(float(np.min(held)))
        for j, fold in enumerate(folds):
            if fold == folds[i]:
                dist[j] = np.inf
        cross.append(float(np.min(dist)))
    expect = conformal_upper(cross, 0.98)
    assert cal["profile"]["n"] == 25
    assert cal["profile"]["radius"] == pytest.approx(round(expect, 4))
    assert float(np.quantile(np.asarray(loo), 0.98)) == pytest.approx(0.0)
    assert cal["profile"]["radius"] > 0


def test_site_under_20_uses_the_prior(tmp_path):
    """Nineteen labels do not estimate a radius. The prior file does."""
    path = tmp_path / "novelty_prior.json"
    path.write_text(json.dumps({"n": 4, "radius": 0.31, "match": 0.2}), encoding="utf-8")
    index = _index()
    for i in range(19):
        _add(index, _unit(0), f"few-{i}", frame="7EA")
    cal = calibrate_site(index, 0.98, prior_path=path)
    assert "profile" not in cal
    assert "frame" not in cal
    assert cal["prior"]["radius"] == pytest.approx(0.31)
    model = ProfileModel("site_alarm", index, calibration=cal)
    assert model.radius_for("7EA") == (0.31, 0.2, "prior")


def test_cross_fitted_train_rows_widen_a_tight_validation_radius(tmp_path):
    """Validation rows that all sit on the index would give radius 0. The train rows' far tail counts too."""
    missing = tmp_path / "absent.json"
    index = _index()
    for i in range(60):
        # Every sixth train row is alone on its own axis, so no other fold has a copy.
        _add(index, _unit(0) if i % 6 else _unit(3 + i // 6), f"train-{i}")
    val = [_val(_unit(0), f"val-{i}") for i in range(30)]
    alone = conformal_upper([0.0] * 30, 0.98)
    cal = calibrate_alert_type(index, val, 0.98, prior_path=missing)
    assert alone == 0.0
    assert cal["profile"]["n"] == 90
    assert cal["profile"]["radius"] > 0.0


def test_pooled_prior_is_not_the_median_of_the_radii():
    """One tight alert type and one wide one: the pool follows the wide tail."""
    tight = [0.1] * 50
    wide = [0.9] * 50
    per_alert = [float(np.quantile(np.asarray(group), 0.98)) for group in (tight, wide)]
    median = float(np.median(per_alert))
    prior = pooled_prior([tight, wide], 0.98)
    assert prior["n"] == 100
    assert prior["radius"] == pytest.approx(0.9)
    assert prior["radius"] != pytest.approx(median)
    assert prior["match"] == pytest.approx(round(0.9 * 0.8, 4))
    assert pooled_prior([], 0.98)["radius"] == pytest.approx(0.5)


def _rows(prefix: str, count: int, seed: int) -> list:
    """Two signatures on two axes of two channels, with noise, so a head is not certain of every row."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        signature = "normal" if i % 2 else "fault"
        block = np.zeros((2, 48), dtype=np.float32)
        block[0, 0 if signature == "normal" else 1] = 1.0
        block[1, 2] = 1.0
        block += rng.normal(0.0, 0.6, size=block.shape).astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        out.append({"id": f"{prefix}-{i}", "u": block, "mask": np.array([True, True]), "signature": signature, "frame": "7EA", "source": "synthetic_original"})
    return out


def test_confidence_floor_is_out_of_sample_and_at_the_same_level(tmp_path):
    """Served head on validation rows, fold heads on the train rows, then the conformal lower order statistic."""
    missing = tmp_path / "absent.json"
    train = _rows("train", 120, 3)
    val = _rows("val", 40, 4)
    model = _model_from_rows("combustion_dynamics", train, use_head=True, val_rows=val, quantile=0.98, prior_path=missing)
    frames = _frame_names(train)
    expected = _head_confidences(model.head, val, frames)
    folds = [fold_of_case(row["id"]) for row in train]
    for fold in sorted(set(folds)):
        fit = [row for row, other in zip(train, folds) if other != fold]
        held = [row for row, other in zip(train, folds) if other == fold]
        expected += _head_confidences(_fit_profile_head(fit, frames), held, frames)
    assert len(expected) == 160
    prof = model.calibration["profile"]
    assert prof["n"] == 160 and prof["confidence_n"] == 160
    assert prof["confidence"] == pytest.approx(round(conformal_lower(expected, 0.98), 4))
    # In-sample confidence (the served head on its own train rows) would sit higher.
    in_sample = _head_confidences(model.head, train, frames)
    assert conformal_lower(in_sample, 0.98) > prof["confidence"]
    assert model.confidence_floor() == pytest.approx(prof["confidence"])
    assert "frame" not in model.calibration


def test_site_floor_comes_from_the_folds_and_knn_has_none(tmp_path):
    """A site head gets its floor from cross-fitted heads. A model without a head stores no floor."""
    missing = tmp_path / "absent.json"
    train = _rows("site", 60, 5)
    with_head = _model_from_rows("site_alarm", train, use_head=True, prior_path=missing)
    assert with_head.head is not None
    assert with_head.calibration["profile"]["confidence_n"] == 60
    assert 0.0 < with_head.calibration["profile"]["confidence"] < 1.0
    no_head = _model_from_rows("site_alarm", train, use_head=False, prior_path=missing)
    assert no_head.head is None
    assert "confidence" not in no_head.calibration["profile"]
    assert no_head.confidence_floor() is None
