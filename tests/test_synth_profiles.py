"""Synthetic alert-type profiles: distinct tags, stable seeds, coldstart stays out of train."""

from __future__ import annotations

import gzip
import importlib.util
from collections import Counter

import pandas as pd
import pytest

from conftest import ROOT

FIELDS = ("profile", "dialect", "slots_present", "missing_tags", "extra_tags")
_IGV_REF = {"IGV_ref_deg", "CSRGV", "IGV_REF"}


def _gen():
    spec = importlib.util.spec_from_file_location("gen_profiles", ROOT / "scripts" / "generate_scenario_csvs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _gen()


@pytest.fixture(scope="module")
def smoke(gen, tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    rows = gen.generate_dataset(out, scale=0.1, workers=1, profiles=list(gen.ALL_SYNTH_PROFILES))
    return out, rows


def _core_tags(folder, row):
    path = folder / row["file"]
    cols = list(pd.read_csv(path, nrows=0).columns)
    extra = set(row["extra_tags"])
    return {c for c in cols if c != "timestamp" and c not in extra}


def test_smoke_has_seven_profiles_with_different_tag_sets(gen, smoke):
    folder, rows = smoke
    profiles = {r["profile"] for r in rows}
    assert profiles >= set(gen.SHIPPED_SYNTH_PROFILES)
    assert len(gen.SHIPPED_SYNTH_PROFILES) >= 7
    unions = {}
    for row in rows:
        if row["profile"] == "fuel_gas_heater":
            continue
        unions.setdefault(row["profile"], set()).update(_core_tags(folder, row))
    ids = list(unions)
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            assert unions[left] != unions[right], (left, right)
    bearing = unions["bearing_vibration"]
    assert not any(c.upper().startswith(("TC", "TTXD", "EGT")) for c in bearing)
    inlet = unions["inlet_filter_dp"]
    assert inlet.isdisjoint({c for c in unions["exhaust_cold_spot"] if c.upper().startswith(("TC", "TTXD", "EGT_"))})


def test_profile_dataset_is_deterministic(gen, tmp_path):
    a = gen.generate_dataset(tmp_path / "a", scale=0.1, workers=1, profiles=["inlet_filter_dp", "fuel_gas_heater"])
    b = gen.generate_dataset(tmp_path / "b", scale=0.1, workers=1, profiles=["inlet_filter_dp", "fuel_gas_heater"])
    assert a == b
    name = a[0]["file"]
    # gzip stores a timestamp in the header, so the check is the CSV bytes.
    assert gzip.open(tmp_path / "a" / name, "rb").read() == gzip.open(tmp_path / "b" / name, "rb").read()


def _unrelated(extra):
    return [name for name in extra if name not in _IGV_REF]


def _assert_slot_coverage(gen, rows, folder=None):
    """Every built-in slot is on the file, minus at most the one masked tag."""
    from sentinel.profiles import load_profiles

    book = load_profiles()
    for pid in gen.SHIPPED_SYNTH_PROFILES:
        wanted = {slot.id for slot in book[pid].slots}
        mine = [row for row in rows if row["profile"] == pid]
        assert mine, pid
        for row in mine:
            present = set(row["slots_present"])
            missing = list(row["missing_tags"])
            assert len(missing) <= 1
            assert "load" not in missing
            assert present.isdisjoint(missing)
            assert present | set(missing) == wanted
        train = [row for row in mine if row["split"] == "train"]
        assert train, pid
        # Scale 0.1 train is 18 cases. Two omissions of one slot are 88.9%, which this
        # seed does for one lube-oil tag. The 90% bar is the scale-1 train (180 cases).
        if len(train) >= 50:
            for slot in wanted:
                frac = sum(slot in row["slots_present"] for row in train) / len(train)
                assert frac >= 0.90, (pid, slot, round(frac, 3))
    for row in rows:
        if row["profile"] == "igv_fuel_control":
            refs = [name for name in row["extra_tags"] if name in _IGV_REF]
            assert refs == [refs[0]]
            assert "igv_ref" not in row["slots_present"]
            if folder is not None:
                cols = list(pd.read_csv(folder / row["file"], nrows=0).columns)
                assert refs[0] in cols
        else:
            assert all(name not in _IGV_REF for name in row["extra_tags"])
        assert row["analytic"] not in gen.GENERIC_ANALYTICS


def test_manifest_fields_and_rates(gen, smoke):
    folder, rows = smoke
    assert rows
    for row in rows:
        for field in FIELDS:
            assert field in row, field
        assert row["dialect"] in ("friendly", "historian", "site")
        assert isinstance(row["slots_present"], list) and isinstance(row["missing_tags"], list)
        assert len(row["missing_tags"]) <= 1
        assert len(_unrelated(row["extra_tags"])) <= 4
    # About one case in twenty omits a single non-context slot.
    rate = sum(1 for row in rows if row["missing_tags"]) / len(rows)
    assert 0.01 <= rate <= 0.12
    assert gen.MISSING_TAG_P == 0.05
    assert gen.GENERIC_NAME_P == 0
    extras = [len(_unrelated(row["extra_tags"])) for row in rows]
    assert min(extras) == 0 and max(extras) >= 1
    per_sig = Counter((row["profile"], row["signature"]) for row in rows if row["split"] == "train")
    assert per_sig
    assert set(per_sig.values()) == {6}
    _assert_slot_coverage(gen, rows, folder)


def test_scale1_slots_cover_each_builtin(gen):
    """Scale 1, in memory: the same slot rule on every train, val and test case."""
    rows = []
    for spec in gen.profile_specs(1.0):
        _frame, meta = gen.make_profile_case(spec)
        rows.append(meta)
    _assert_slot_coverage(gen, rows)


def test_coldstart_profile_is_absent_from_train_and_the_bundle(gen, smoke):
    _folder, rows = smoke
    assert any(r["profile"] == "fuel_gas_heater" and r["split"] == "coldstart" for r in rows)
    assert all(r["profile"] != "fuel_gas_heater" for r in rows if r["split"] != "coldstart")
    assert "fuel_gas_heater" not in gen.SHIPPED_SYNTH_PROFILES
    assert not (ROOT / "models" / "e2e" / "profiles" / "fuel_gas_heater").exists()


def test_legacy_dataset_counts_stay_put(gen):
    specs = gen.dataset_specs(1.0)
    by = {}
    for spec in specs:
        by[spec["split"]] = by.get(spec["split"], 0) + 1
    assert by == {"train": 1200, "val": 200, "test": 320, "variant": 128, "unseen": 80, "sensorfault": 40}
