"""Tag profiles: settings, schema, aliases, channels, built-ins and resolution.

Unit tests stay offline. The embedder is the statistical fallback unless a test
opts into the real encoder.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from sentinel.config import (
    CHANNEL_SOFT_MAX,
    CV_FOLDS,
    CV_MIN_LABELS,
    EMBED_BATCH_WINDOWS,
    FEATURE_VERSION,
    FRAME_DIST_PENALTY,
    HEAD_MIN_LABELS,
    HEAD_MIN_PER_CLASS,
    HELDOUT_FRACTION,
    HELDOUT_MIN_LABELS,
    LEGACY_FEATURE_VERSION,
    MASK_PENALTY,
    MISSING_MASK_FRACTION,
    NOVELTY_GATE_TOL,
    PROFILE_NOVELTY_MIN,
    PROFILE_SCHEMA,
    load_settings,
)


def test_profile_settings_round_trip(tmp_path, monkeypatch):
    """Defaults, an override dict, and the SENTINEL_* env vars all survive a JSON round trip."""
    monkeypatch.delenv("SENTINEL_E2E_MODE", raising=False)
    monkeypatch.delenv("SENTINEL_EMBEDDER", raising=False)
    monkeypatch.delenv("SENTINEL_PROFILES_DIR", raising=False)

    bare = load_settings({"data_dir": tmp_path / "data"})
    assert bare.e2e_mode == "legacy"
    assert bare.embedder == "tspulse_ci"
    assert bare.profiles_dir == tmp_path / "data" / "e2e" / "profiles"
    assert PROFILE_SCHEMA == 2
    assert FEATURE_VERSION == 3
    assert LEGACY_FEATURE_VERSION == 2
    assert HEAD_MIN_LABELS == 20 and HEAD_MIN_PER_CLASS == 3
    assert CV_MIN_LABELS == 10 and CV_FOLDS == 5
    assert HELDOUT_MIN_LABELS == 50 and HELDOUT_FRACTION == 0.2
    assert PROFILE_NOVELTY_MIN == 20
    assert MASK_PENALTY == 0.25 and FRAME_DIST_PENALTY == 0.05
    assert NOVELTY_GATE_TOL == 0.01
    assert CHANNEL_SOFT_MAX == 64 and MISSING_MASK_FRACTION == 0.5
    assert EMBED_BATCH_WINDOWS == 256

    described = bare.describe()
    assert described["e2e_mode"] == "legacy"
    assert described["embedder"] == "tspulse_ci"
    assert described["profiles_dir"]

    blob = {
        "e2e_mode": bare.e2e_mode,
        "embedder": "stats",
        "profiles_dir": "site/profiles",
        "data_dir": str(tmp_path / "data"),
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(blob), encoding="utf-8")
    again = load_settings(json.loads(path.read_text(encoding="utf-8")))
    assert again.e2e_mode == "legacy"
    assert again.embedder == "stats"
    assert again.profiles_dir.name == "profiles"
    assert again.describe()["embedder"] == "stats"

    monkeypatch.setenv("SENTINEL_E2E_MODE", "peralert")
    monkeypatch.setenv("SENTINEL_EMBEDDER", "legacy14")
    custom = tmp_path / "shared_profiles"
    monkeypatch.setenv("SENTINEL_PROFILES_DIR", str(custom))
    # Env loses to an explicit override, and wins over the dataclass default.
    from_env = load_settings({"data_dir": tmp_path / "data"})
    assert from_env.e2e_mode == "peralert"
    assert from_env.embedder == "legacy14"
    assert from_env.profiles_dir == custom
    forced = load_settings({"data_dir": tmp_path / "data", "e2e_mode": "legacy", "embedder": "stats"})
    assert forced.e2e_mode == "legacy"
    assert forced.embedder == "stats"
    # A value outside the allowed set is dropped. The override replaced the env
    # value in the bag, so the dataclass default is what remains.
    kept = load_settings({"e2e_mode": "both", "embedder": "no-such-embedder"})
    assert kept.e2e_mode == "legacy"
    assert kept.embedder == "tspulse_ci"


def _slot(slot_id="inlet_dp", role="process", **extra):
    slot = {
        "id": slot_id,
        "role": role,
        "label": extra.pop("label", slot_id),
        "unit": extra.pop("unit", ""),
        "required": extra.pop("required", True),
        "aliases": extra.pop("aliases", {"*": [slot_id]}),
    }
    slot.update(extra)
    return slot


def _valid(**over):
    """A small schema-1 profile that uses signatures the taxonomy already has."""
    data = {
        "schema": 1,
        "id": "inlet_filter_dp",
        "name": "Inlet filter high differential pressure",
        "rev": 1,
        "description": "Fouling versus a transmitter fault.",
        "match": {"names": ["Inlet filter high dp"], "keywords": ["inlet dp"]},
        "slots": [
            _slot("load", "context", required=True, weight=0.5, legacy_role="load"),
            _slot("inlet_dp", "process", required=True, load_fit=True, health=True),
        ],
        "include_other": True,
        "signatures": ["normal", "tc_sensor_fault"],
        "analyzer": "generic",
        "source": "file",
    }
    data.update(over)
    return data


def test_valid_profile_loads_and_round_trips(tmp_path):
    from sentinel.profiles import load_profiles, profile_from_dict, save_profile_file, validate_profile

    raw = _valid()
    assert validate_profile(raw) == []
    prof = profile_from_dict(raw)
    assert prof.id == "inlet_filter_dp"
    assert prof.signatures[0] == "normal"
    assert prof.slots[0].weight == 0.5
    # A process slot with a context sibling load-fits unless the file says otherwise.
    assert prof.slots[1].load_fit is True
    path = save_profile_file(prof, tmp_path / "profiles" / "inlet_filter_dp.json")
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    loaded_json = json.loads(text)
    assert list(loaded_json)[:8] == [
        "schema", "id", "name", "aliases", "rev", "description", "slots", "include_other",
    ]
    assert loaded_json["schema"] == 2
    assert loaded_json["aliases"] == ["Inlet filter high dp"]
    assert "match" not in loaded_json
    assert loaded_json["signatures"][0] == {"id": "normal", "action": "monitor_only"}
    assert all(slot.get("required") is None or slot.get("required") is True for slot in loaded_json["slots"])
    again = load_profiles(profiles_dir=tmp_path / "profiles", builtins=[])
    assert again["inlet_filter_dp"].name == prof.name
    assert again["inlet_filter_dp"].rev == 1
    assert again["inlet_filter_dp"].aliases == ["Inlet filter high dp"]
    assert all(slot.required for slot in again["inlet_filter_dp"].slots)
    assert (tmp_path / "profiles" / "_invalid.json").read_text(encoding="utf-8").strip() == "[]"


def test_validation_names_the_field():
    from sentinel.profiles import validate_profile

    def messages(raw):
        return " | ".join(validate_profile(raw))

    assert "id:" in messages(_valid(id="Bad-Id"))
    assert "reserved" in messages(_valid(id="_adhoc"))
    assert "id:" in messages(_valid(id="_legacy"))
    two = _valid()
    two["slots"] = [
        _slot("load", "context", legacy_role="load"),
        _slot("mw", "context", legacy_role="load"),
    ]
    assert "role" in messages(two) and "context" in messages(two)
    bad_role = _valid()
    bad_role["slots"] = [_slot("load", "temperature")]
    assert "role" in messages(bad_role)
    assert "signatures" in messages(_valid(signatures=["normal", "not_a_real_signature"]))
    assert "novel_pattern" in messages(_valid(signatures=["normal", "novel_pattern"]))
    cold = _valid(analyzer="cold_spot")
    cold_msg = messages(cold)
    assert "analyzer" in cold_msg and "REQUIRES" in cold_msg
    expect = _valid(residual_expect={"tc_sensor_fault": {"primary": ["no_such_slot"], "allowed": ["load"]}})
    assert "residual_expect" in messages(expect)
    assert "schema" in messages(_valid(schema=0))
    assert "load_fit" in messages(_valid(slots=[_slot("load", "context", load_fit=True, legacy_role="load")]))
    assert "legacy_role" in messages(_valid(slots=[_slot("load", "context", legacy_role="not_a_role")]))
    # Keywords are dropped on the way in, so a broken pattern is not an error.
    assert validate_profile(_valid(match={"names": ["Inlet filter high dp"], "keywords": ["("]})) == []
    assert "aliases" in messages(_valid(schema=2, match={"names": ["x"], "keywords": []}, aliases=["Inlet dp"]))
    assert "aliases" in messages(_valid(schema=2, aliases=["Cold spot", "cold-spot"]))
    assert "required" in messages(_valid(schema=2, aliases=["Inlet dp"], slots=[_slot("load", "context", required=False, legacy_role="load")]))
    assert "order" in messages(_valid(schema=2, aliases=["Inlet dp"], slots=[_slot("load", "context", legacy_role="load", order="listed")]))
    assert "members" in messages(_valid(schema=2, aliases=["Inlet dp"], slots=[_slot("inlet_dp", "process", members={"7FA": 1})]))
    assert "thresholds" in messages(_valid(schema=2, aliases=["Inlet dp"], thresholds={"novelty_quantile": 0.5}))
    assert "action" in messages(_valid(
        schema=2,
        aliases=["Inlet dp"],
        signatures=[{"id": "tc_sensor_fault", "action": "borescope"}],
    ))
    assert "reserved" in messages(_valid(id="_unknown"))


def test_normal_is_added_and_null_analyzer_is_generic():
    from sentinel.profiles import profile_from_dict

    prof = profile_from_dict(_valid(signatures=["tc_sensor_fault"], analyzer=None))
    assert prof.signatures[0] == "normal"
    assert "tc_sensor_fault" in prof.signatures
    assert prof.analyzer == "generic"


def test_site_file_overrides_builtin_and_invalid_file_is_listed(tmp_path):
    from sentinel.profiles import load_profiles

    builtin = _valid(id="inlet_filter_dp", name="Built in", rev=1, source="builtin")
    folder = tmp_path / "profiles"
    folder.mkdir()
    site = _valid(id="inlet_filter_dp", name="Site override", rev=4, source="file")
    (folder / "inlet_filter_dp.json").write_text(json.dumps(site), encoding="utf-8")
    (folder / "broken.json").write_text("{not json", encoding="utf-8")
    wrong = _valid(id="Bad", schema=9)
    (folder / "wrong.json").write_text(json.dumps(wrong), encoding="utf-8")
    # _invalid.json itself must not be loaded, even if it is valid JSON.
    (folder / "_invalid.json").write_text("[]", encoding="utf-8")
    loaded = load_profiles(profiles_dir=folder, builtins=[builtin])
    assert set(loaded) == {"inlet_filter_dp"}
    assert loaded["inlet_filter_dp"].name == "Site override"
    assert loaded["inlet_filter_dp"].rev == 4
    report = json.loads((folder / "_invalid.json").read_text(encoding="utf-8"))
    blob = json.dumps(report)
    assert "broken.json" in blob and "wrong.json" in blob
    assert "schema" in blob and "id" in blob
    # The override is schema 1 and still names a keyword, so it loads and is noted.
    assert "inlet_filter_dp.json" in blob
    assert "keywords no longer select an alert type" in blob


def test_needs_review_stub_may_have_no_slots():
    from sentinel.profiles import profile_from_dict, validate_profile

    stub = {
        "schema": 1,
        "id": "alert_custom",
        "name": "Custom alert",
        "slots": [],
        "signatures": ["normal"],
        "needs_review": True,
        "source": "migration",
    }
    assert validate_profile(stub) == []
    assert profile_from_dict(stub).needs_review is True
    refused = dict(stub)
    refused["needs_review"] = False
    assert any("slots" in msg for msg in validate_profile(refused))


def _alias_profile():
    """One profile that exercises every alias form. Signatures stay inside today's taxonomy."""
    from sentinel.profiles import profile_from_dict

    return profile_from_dict(
        {
            "schema": 1,
            "id": "alias_demo",
            "name": "Alias demo",
            "slots": [
                _slot("load", "context", required=True, legacy_role="load", aliases={"*": ["missing_mw", "load_MW"], "site_unit3": ["U3_GROSS_MW"]}),
                _slot("exh_tc", "ring", required=True, legacy_role="ring", health=True, aliases={"*": {"prefix": ["TTXD", "TC"]}}),
                _slot("brg_vib", "bank", required=False, legacy_role="vibration", health=True, aliases={"*": {"regex": r"^BRG\d+_VIB$"}}),
                _slot("inlet_dp", "process", required=False, aliases={"*": {"tags": ["AFPAP"]}, "site_unit3": ["U3_INL_DP"]}),
                _slot("fuel", "process", required=False, legacy_role="fuel", aliases={"*": {"pattern_role": "fuel"}}),
                _slot("ambient_t", "process", required=False, aliases={"site_unit3": ["U3_AMB"]}),
                _slot("cdp", "process", required=False, legacy_role="cdp", aliases={"site_unit3": ["U3_CDP"]}),
            ],
            "signatures": ["normal"],
        }
    )


_ALIAS_COLUMNS = [
    "Load_MW", "TC10", "TC2", "TC1", "BRG2_VIB", "brg1_vib", "AFPAP", "fuel_flow_kg_s",
    "U3_GROSS_MW", "U3_INL_DP", "U3_AMB", "U3_CDP", "DWATT",
]


def test_alias_forms_resolve_and_build_legacy():
    from sentinel.profiles import resolve_tags

    # No site columns are required for ``*``: drop the site-only tags so auto stays on ``*``.
    cols = [c for c in _ALIAS_COLUMNS if not c.startswith("U3_")]
    resolved = resolve_tags(_alias_profile(), cols, frame="*")
    assert resolved.frame_key == "*"
    by = {h.slot_id: h for h in resolved.slots}
    assert by["load"].tags == ["Load_MW"]
    assert by["load"].alias_key == "*"
    # Sensor number, not dictionary order: TC1, TC2, TC10. TTXD is listed first but has no hits.
    assert by["exh_tc"].tags == ["TC1", "TC2", "TC10"]
    assert by["brg_vib"].tags == ["brg1_vib", "BRG2_VIB"]
    assert by["inlet_dp"].tags == ["AFPAP"]
    assert by["fuel"].tags == ["fuel_flow_kg_s"]
    assert by["ambient_t"].status == "missing"
    assert resolved.legacy["roles"]["load"] == ["Load_MW"]
    assert resolved.legacy["roles"]["fuel"] == ["fuel_flow_kg_s"]
    assert resolved.legacy["ring"] == ["TC1", "TC2", "TC10"]
    assert resolved.legacy["roles"]["vibration"] == ["brg1_vib", "BRG2_VIB"]


def test_frame_specific_alias_beats_star():
    from sentinel.profiles import resolve_tags

    resolved = resolve_tags(_alias_profile(), _ALIAS_COLUMNS, frame="site_unit3")
    by = {h.slot_id: h for h in resolved.slots}
    assert resolved.frame_key == "site_unit3"
    assert by["load"].tags == ["U3_GROSS_MW"]
    assert by["load"].alias_key == "site_unit3"
    assert by["inlet_dp"].tags == ["U3_INL_DP"]
    # The ring has no site alias, so it falls through to ``*``.
    assert by["exh_tc"].tags == ["TC1", "TC2", "TC10"]
    assert by["exh_tc"].alias_key == "*"
    assert by["ambient_t"].tags == ["U3_AMB"]
    assert by["cdp"].tags == ["U3_CDP"]


def test_a_tag_is_not_taken_twice():
    from sentinel.profiles import profile_from_dict, resolve_tags

    prof = profile_from_dict(
        {
            "schema": 1,
            "id": "taken_twice",
            "name": "Taken twice",
            "slots": [
                _slot("first_tc", "process", aliases={"*": {"tags": ["TC1"]}}),
                _slot("ring", "ring", legacy_role="ring", aliases={"*": {"prefix": ["TC"]}}),
            ],
            "signatures": ["normal"],
        }
    )
    resolved = resolve_tags(prof, ["TC1", "TC2", "TC3", "TC4"])
    by = {h.slot_id: h for h in resolved.slots}
    assert by["first_tc"].tags == ["TC1"]
    assert "TC1" not in by["ring"].tags
    assert by["ring"].tags == ["TC2", "TC3", "TC4"]


def test_auto_frame_picks_the_key_with_most_hits_and_tie_goes_to_star():
    from sentinel.profiles import resolve_tags

    # site_unit3 resolves ambient, inlet and cdp (3). ``*`` resolves load, ring, vib, fuel (4)
    # on the full header, so this header drops the star-only tags that would outvote the site.
    site_cols = ["U3_GROSS_MW", "U3_INL_DP", "U3_AMB", "U3_CDP", "TC1", "TC2", "TC3"]
    picked = resolve_tags(_alias_profile(), site_cols)
    assert picked.frame_key == "site_unit3"
    assert picked.by_id("ambient_t").status == "present"
    assert picked.by_id("load").tags == ["U3_GROSS_MW"]

    # Both keys resolve the one load slot. The tie goes to ``*``, not to the site key.
    tie_prof_cols = ["DWATT", "load_MW"]
    from sentinel.profiles import profile_from_dict

    tie = profile_from_dict(
        {
            "schema": 1,
            "id": "tie_demo",
            "name": "Tie demo",
            "slots": [_slot("load", "context", legacy_role="load", aliases={"*": ["load_MW"], "site_unit3": ["DWATT"]})],
            "signatures": ["normal"],
        }
    )
    tied = resolve_tags(tie, tie_prof_cols)
    assert tied.frame_key == "*"
    assert tied.by_id("load").tags == ["load_MW"]
    assert "*" in (tied.frame_ties or ["*"]) or tied.frame_key == "*"


def test_mostly_empty_tag_is_masked():
    import pandas as pd

    from sentinel.profiles import profile_from_dict, resolve_tags

    prof = profile_from_dict(
        {
            "schema": 1,
            "id": "mask_demo",
            "name": "Mask demo",
            "slots": [
                _slot("load", "context", legacy_role="load", aliases={"*": ["load_MW"]}),
                _slot("inlet_dp", "process", required=True, aliases={"*": ["AFPAP"]}),
            ],
            "signatures": ["normal"],
        }
    )
    df = pd.DataFrame(
        {
            "load_MW": [100.0, 100.0, 100.0, 100.0],
            "AFPAP": [1.0, None, None, None],
        }
    )
    resolved = resolve_tags(prof, df)
    assert resolved.by_id("inlet_dp").status == "masked"
    assert resolved.by_id("inlet_dp").tags == ["AFPAP"]
    assert resolved.coverage()["required"] == (1, 2)
    assert any("masked" in w or "empty" in w for w in resolved.warnings)


def _layout_profile(**over):
    """One slot of each role, plus _other and one couple. Aliases are irrelevant to the hash."""
    from sentinel.profiles import profile_from_dict

    body = {
        "schema": 1,
        "id": "layout_demo",
        "name": "Layout demo",
        "include_other": True,
        "couple": [["fuel", "exh"]],
        "slots": [
            _slot("load", "context", label="unit load", aliases={"*": ["load_MW"]}),
            _slot("fuel", "process", label="fuel flow", load_fit=True, health=True, aliases={"*": ["FQG"]}),
            _slot("trip", "flag", label="trip", aliases={"*": ["trip_active"]}),
            _slot("exh", "ring", label="exhaust", health=True, aliases={"*": {"prefix": "TC"}}),
            _slot("vib", "bank", label="bearing vibration", health=True, aliases={"*": {"regex": "^BRG\\d+_VIB$"}}),
            _slot("note", "ignore", label="scratch", aliases={"*": ["NOTE"]}),
        ],
        "signatures": ["normal"],
    }
    body.update(over)
    return profile_from_dict(body)


def test_compile_channels_order_and_ids_per_role():
    from sentinel.profiles import compile_channels

    specs = compile_channels(_layout_profile())
    ids = [s.id for s in specs]
    assert ids == [
        "load",
        "fuel",
        "fuel:flat",
        "trip",
        "exh:cold",
        "exh:cold_pair",
        "exh:hot",
        "exh:spread",
        "exh:avg",
        "exh:flat",
        "vib:max",
        "vib:min",
        "vib:spread",
        "vib:flat",
        "_other",
        "fuel~exh",
    ]
    by = {s.id: s for s in specs}
    assert by["load"].transform == "robust_z" and by["load"].online_only is False
    assert by["fuel"].transform == "load_fit_z" and by["fuel"].online_only is True
    assert by["trip"].transform == "flag" and by["trip"].online_only is False
    assert by["exh:cold"].transform == "ring_cold" and by["exh:avg"].floor == 0.3
    assert by["vib:max"].transform == "bank_max"
    assert by["_other"].transform == "other"
    assert by["fuel~exh"].transform == "couple"
    assert all(not s.masked for s in specs)
    # An ignore slot records the engineer's choice and emits no channel.
    assert "note" not in ids


def test_short_ring_compiles_as_a_bank_with_the_same_ids():
    from sentinel.profiles import Resolved, SlotHit, compile_channels

    prof = _layout_profile()
    full = [s.id for s in compile_channels(prof) if s.slot_id == "exh"]
    two = Resolved("*" , [SlotHit("exh", "ring", False, ["TC1", "TC2"], "*", "present")])
    specs = compile_channels(prof, two)
    ring = [s for s in specs if s.slot_id == "exh"]
    assert [s.id for s in ring] == full
    by = {s.id: s for s in ring}
    assert by["exh:hot"].transform == "bank_max" and not by["exh:hot"].masked
    assert by["exh:cold"].transform == "bank_min" and not by["exh:cold"].masked
    assert by["exh:spread"].transform == "bank_spread"
    assert by["exh:flat"].transform == "bank_flat"
    assert by["exh:cold_pair"].masked and by["exh:avg"].masked
    # The declared transform stays on the spec so the hash does not follow the CSV.
    assert by["exh:hot"].declared == "ring_hot"
    assert by["exh:cold_pair"].declared == "ring_cold_pair"

    one = Resolved("*", [SlotHit("vib", "bank", False, ["BRG1_VIB"], "*", "present")])
    bank = {s.id: s for s in compile_channels(prof, one) if s.slot_id == "vib"}
    assert list(bank) == ["vib:max", "vib:min", "vib:spread", "vib:flat"]
    assert bank["vib:max"].transform == "robust_z" and not bank["vib:max"].masked
    assert bank["vib:flat"].transform == "flat" and not bank["vib:flat"].masked
    assert bank["vib:min"].masked and bank["vib:spread"].masked


def test_schema1_upgrades_without_changing_the_channels_hash(tmp_path):
    """A schema 1 file loads, writes back as schema 2, and keeps the same channel layout."""
    from sentinel.profiles import channels_hash, load_profiles, profile_from_dict, save_profile_file

    raw = _SCHEMA_EXAMPLE
    before = profile_from_dict(raw)
    assert before.schema == 2
    assert before.aliases == list(raw["match"]["names"])
    assert all(slot.required for slot in before.slots)
    assert before.signatures[0] == "normal"
    assert before.signatures[1].action == "monitor_only"
    path = save_profile_file(before, tmp_path / "profiles" / "inlet_filter_dp.json")
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["schema"] == 2
    assert "match" not in written
    assert written["signatures"][1] == {"id": "inlet_filter_fouling", "action": "monitor_only"}
    again = load_profiles(profiles_dir=tmp_path / "profiles", builtins=[])["inlet_filter_dp"]
    assert channels_hash(before) == channels_hash(again)
    notes = json.loads((tmp_path / "profiles" / "_invalid.json").read_text(encoding="utf-8"))
    assert notes == []


# The schema 2 example from the design. Members and the site alias are site data.
_SCHEMA2_EXAMPLE = {
    "schema": 2,
    "id": "exhaust_cold_spot",
    "name": "Exhaust cold spot",
    "aliases": [
        "Exhaust Cold Spot",
        "Exhaust temperature cold spot",
        "Cold spot TC ring",
        "TTXD cold spot",
        "exh cold spot alarm",
        "Exhaust TC deviation",
        "TC failure advisory",
    ],
    "rev": 2,
    "description": "Cold sector on the exhaust ring, or one thermocouple reading on its own.",
    "slots": [
        {"id": "load", "role": "context", "label": "unit load", "unit": "MW", "weight": 0.5,
         "aliases": {"*": [["load_MW", "DWATT", "power_MW"], {"pattern_role": "load"}]},
         "legacy_role": "load"},
        {"id": "exh_tc", "role": "ring", "label": "exhaust temperature", "unit": "C", "health": True,
         "order": "numeric", "members": {"7EA": 27, "7FA": 31},
         "aliases": {"*": [{"prefix": ["TC", "TTXD", "EGT_Z", "EGT_ZONE"]}, {"pattern_role": "ring"}, {"pattern_role": "zones"}],
                     "site_unit3": [{"tags": ["U3_EXH_01", "U3_EXH_02", "U3_EXH_03", "U3_EXH_04"]}]},
         "legacy_role": "ring"},
        {"id": "spread", "role": "process", "label": "exhaust spread", "unit": "C", "load_fit": True,
         "aliases": {"*": [["EGT_spread_C", "TTXSP1", "exh_spread_C"], {"pattern_role": "spread"}]}, "legacy_role": "spread"},
        {"id": "egt_avg", "role": "process", "label": "average exhaust temperature", "unit": "C", "load_fit": True,
         "aliases": {"*": [["EGT_avg_C", "TTXM", "exh_avg_C"], {"pattern_role": "egt_avg"}]}, "legacy_role": "egt_avg"},
        {"id": "fuel", "role": "process", "label": "fuel flow", "unit": "kg/s", "load_fit": True,
         "aliases": {"*": [["fuel_flow_kg_s", "FQG", "fuel_kg_s"], {"pattern_role": "fuel"}]}, "legacy_role": "fuel"},
        {"id": "igv", "role": "process", "label": "IGV position", "unit": "deg", "load_fit": True,
         "aliases": {"*": [["IGV_deg", "CSGV", "IGV"], {"pattern_role": "igv"}]}, "legacy_role": "igv"},
    ],
    "include_other": True,
    "couple": [],
    "signatures": [
        {"id": "normal", "action": "monitor_only"},
        {"id": "cold_spot_hardware", "action": "borescope"},
        {"id": "tc_sensor_fault", "action": "sensor_fix_replace"},
    ],
    "residual_expect": {
        "cold_spot_hardware": {"primary": ["exh_tc"], "allowed": ["exh_tc", "spread", "egt_avg"]},
        "tc_sensor_fault": {"primary": ["exh_tc"], "allowed": ["exh_tc"]},
    },
    "thresholds": {"novelty_quantile": 0.98, "knn_vote_min": 0.6, "residual": {}},
    "analyzer": "cold_spot",
    "docs": ["process_map_cold_spot.md", "process_map_exhaust_spread.md"],
    "source": "builtin",
}


def test_schema2_example_loads():
    """The design's schema 2 example is a valid alert type, including signature actions."""
    from sentinel.profiles import profile_from_dict, validate_profile

    assert validate_profile(_SCHEMA2_EXAMPLE) == []
    prof = profile_from_dict(_SCHEMA2_EXAMPLE)
    assert prof.schema == 2
    assert prof.aliases[0] == "Exhaust Cold Spot"
    assert prof.slots[1].order == "numeric"
    assert prof.slots[1].members == {"7EA": 27, "7FA": 31}
    assert all(slot.required for slot in prof.slots)
    assert [str(sig) for sig in prof.signatures] == ["normal", "cold_spot_hardware", "tc_sensor_fault"]
    assert prof.signatures[1].action == "borescope"
    assert prof.thresholds["novelty_quantile"] == 0.98
    assert prof.thresholds["knn_vote_min"] == 0.6
    written = prof.to_dict()
    assert list(written)[:12] == [
        "schema", "id", "name", "aliases", "rev", "description", "slots", "include_other",
        "couple", "signatures", "residual_expect", "thresholds",
    ]


def test_channels_hash_ignores_aliases_and_changes_on_a_role_edit():
    from sentinel.profiles import channels_hash

    base = _layout_profile()
    renamed = _layout_profile()
    renamed.slots[3].aliases = {"*": ["TC1", "TC2"]}
    assert channels_hash(base) == channels_hash(renamed)
    assert len(channels_hash(base)) == 12

    edited = _layout_profile()
    edited.slots[1].role = "flag"
    edited.slots[1].load_fit = False
    edited.slots[1].health = False
    assert channels_hash(edited) != channels_hash(base)


# The schema 1 example from the design. Signatures are the ones added with the built-ins.
_SCHEMA_EXAMPLE = {
    "schema": 1,
    "id": "inlet_filter_dp",
    "name": "Inlet filter high differential pressure",
    "rev": 3,
    "description": "Inlet filter dp alarm raised by Tier 1. Fouling versus transmitter fault.",
    "match": {
        "names": ["Inlet filter high dp", "Inlet dp high", "Filter house dp advisory"],
        "keywords": ["inlet (filter )?dp", "filter house", r"\bafpap\b"],
    },
    "slots": [
        {
            "id": "load", "role": "context", "label": "unit load", "unit": "MW",
            "required": True, "weight": 0.5,
            "aliases": {"*": ["load_MW", "DWATT"], "site_unit3": ["U3_GROSS_MW"]},
            "legacy_role": "load",
        },
        {
            "id": "inlet_dp", "role": "process", "label": "inlet filter differential pressure", "unit": "mbar",
            "required": True, "load_fit": True, "health": True,
            "aliases": {"*": ["inlet_dp_mbar", "AFPAP"], "site_unit3": ["U3_INL_DP"]},
        },
        {
            "id": "ambient_t", "role": "process", "label": "ambient temperature", "unit": "C",
            "required": False, "load_fit": False,
            "aliases": {"*": ["ambient_C", "AFTAMB", {"pattern_role": "ambient"}]},
        },
        {
            "id": "cdp", "role": "process", "label": "compressor discharge pressure", "unit": "bar",
            "required": False,
            "aliases": {"*": [{"pattern_role": "cdp"}]},
            "legacy_role": "cdp",
        },
    ],
    "include_other": True,
    "couple": [["inlet_dp", "ambient_t"]],
    "signatures": ["normal", "inlet_filter_fouling", "dp_transmitter_fault"],
    "residual_expect": {
        "inlet_filter_fouling": {"primary": ["inlet_dp"], "allowed": ["inlet_dp", "cdp", "load"]},
        "dp_transmitter_fault": {"primary": ["inlet_dp"], "allowed": ["inlet_dp"]},
    },
    "analyzer": "generic",
    "docs": ["knowledge/process_map_inlet_filter.md", "user/inlet-dp-walkthrough"],
    "source": "teach",
    "created": "2026-10-14T09:12:00",
    "updated": "2026-10-14T09:40:31",
    "note": "Sample CSV: unit 3, 10-min data, Sep 2026.",
}


def test_schema_example_loads_and_builtins_match_todays_samples(tmp_path):
    """The design's schema 1 example loads, and each built-in resolves the sample CSVs it owns."""
    import csv
    from pathlib import Path

    from sentinel.e2e.taxonomy import BASE_SIGNATURES
    from sentinel.profiles import BUILTIN_PROFILES, load_profiles, profile_from_dict, resolve_tags

    example = profile_from_dict(_SCHEMA_EXAMPLE)
    assert example.id == "inlet_filter_dp"
    assert example.rev == 3
    assert example.signatures == ["normal", "inlet_filter_fouling", "dp_transmitter_fault"]
    assert example.slots[0].weight == 0.5
    assert example.couple == [["inlet_dp", "ambient_t"]]
    assert [slot.id for slot in example.slots] == ["load", "inlet_dp", "ambient_t", "cdp"]

    actions = {
        "inlet_filter_fouling": ("monitor_only", "next_opportunity"),
        "dp_transmitter_fault": ("sensor_fix_replace", "prompt"),
        "drain_tc_fault": ("sensor_fix_replace", "prompt"),
        "vibration_probe_fault": ("sensor_fix_replace", "prompt"),
        "lube_cooler_fouling": ("escalate", "prompt"),
        "bearing_wear_trend": ("escalate", "prompt"),
    }
    for sid, (action, urgency) in actions.items():
        assert BASE_SIGNATURES[sid]["action"] == action
        assert BASE_SIGNATURES[sid]["urgency"] == urgency

    loaded = load_profiles(profiles_dir=tmp_path / "absent")
    assert list(loaded) == [
        "exhaust_cold_spot",
        "combustion_dynamics",
        "exhaust_spread_trip",
        "igv_fuel_control",
        "bearing_vibration",
        "inlet_filter_dp",
        "lube_oil_system",
    ]
    for prof in loaded.values():
        assert prof.match["keywords"] == []
        assert prof.thresholds["novelty_quantile"] == 0.98
        assert prof.thresholds["knn_vote_min"] == 0.6
        assert prof.thresholds["residual"] == {}
        assert all(slot.required for slot in prof.slots)

    root = Path(__file__).resolve().parents[1] / "samples"
    owned = {
        "cold_spot": "exhaust_cold_spot",
        "combustion_dynamics": "combustion_dynamics",
        "hets_trip": "exhaust_spread_trip",
    }
    for folder, pid in owned.items():
        paths = sorted((root / folder).glob("*.csv"))
        assert paths, folder
        for path in paths:
            with path.open(encoding="utf-8", newline="") as handle:
                columns = next(csv.reader(handle))
            resolved = resolve_tags(loaded[pid], columns)
            missing = [hit.slot_id for hit in resolved.slots if hit.required and hit.status != "present"]
            assert missing == [], (path.name, missing)
    with (root / "cold_spot" / "cold_spot_01.csv").open(encoding="utf-8", newline="") as handle:
        cold_cols = next(csv.reader(handle))
    cold = resolve_tags(loaded["exhaust_cold_spot"], cold_cols)
    assert cold.by_id("exh_tc").tags[0] == "TC1"
    assert len(cold.by_id("exh_tc").tags) >= 3
    with (root / "combustion_dynamics" / "comb_dyn_01.csv").open(encoding="utf-8", newline="") as handle:
        dyn_cols = next(csv.reader(handle))
    dyn = resolve_tags(loaded["combustion_dynamics"], dyn_cols)
    assert dyn.by_id("dyn").tags == ["comb_dyn_psi"]
    assert dyn.by_id("exh_tc").tags[0].startswith("EGT_Z")


def test_legacy_feature_cache_key_stays_on_version_2(tmp_path):
    """The 14-channel cache key must not move when FEATURE_VERSION becomes 3."""
    from sentinel.e2e.dataset import cached_features
    from sentinel.e2e.features import CHANNELS

    rows = [{"file": "train/a.csv"}]
    body = "|".join(r["file"] for r in rows) + "|" + ",".join(CHANNELS)
    key2 = hashlib.sha256(f"v{LEGACY_FEATURE_VERSION}|{body}".encode()).hexdigest()[:16]
    key3 = hashlib.sha256(f"v{FEATURE_VERSION}|{body}".encode()).hexdigest()[:16]
    assert key2 != key3
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez_compressed(cache / f"features_{key2}.npz", X=np.array([2], dtype=np.float32))
    np.savez_compressed(cache / f"features_{key3}.npz", X=np.array([3], dtype=np.float32))
    got = cached_features(tmp_path, rows, cache)
    assert int(got.reshape(-1)[0]) == 2


class _TinyHead:
    """Just enough of a head for the registry to store a version."""

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "head.npz", W=np.zeros((1, 1)), b=np.zeros(1))
        (folder / "head.json").write_text(json.dumps({"classes": ["normal"], "meta": {}}), encoding="utf-8")


def test_profile_registry_versions_stay_isolated(tmp_path):
    from sentinel.config import load_settings
    from sentinel.e2e.labels import BASE_VERSION, ENCODER_BASE, Registry

    settings = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    legacy = Registry(settings)
    legacy.register(_TinyHead(), {"trigger": "legacy"})
    cold = Registry(settings, profile="exhaust_cold_spot")
    lube = Registry(settings, profile="lube_oil_system")
    assert cold.current() == BASE_VERSION
    assert lube.current() == BASE_VERSION
    cold.register(_TinyHead(), {"trigger": "cold"})
    lube.register(None, {"trigger": "lube", "tier": "knn"})
    assert cold.current() == "v0001"
    assert lube.current() == "v0001"
    assert legacy.current() == "v0001"
    assert [row["version"] for row in cold.versions()] == [BASE_VERSION, "v0001"]
    assert [row["version"] for row in lube.versions()] == [BASE_VERSION, "v0001"]
    assert (cold.folder("v0001") / "head.npz").is_file()
    assert not (lube.folder("v0001") / "head.npz").is_file()
    assert (lube.folder("v0001") / "meta.json").is_file()
    # Rolling one profile back does not move the other, or the legacy head.
    assert cold.rollback() == BASE_VERSION
    assert cold.current() == BASE_VERSION
    assert lube.current() == "v0001"
    assert legacy.current() == "v0001"
    # A legacy activate rewrites the pointer file and must keep the profile map.
    legacy.activate(BASE_VERSION)
    assert legacy.current() == BASE_VERSION
    assert lube.current() == "v0001"
    assert cold.current() == BASE_VERSION
    legacy.activate("v0001")
    # The shared encoder line is a third pointer.
    assert legacy.current_encoder() == ENCODER_BASE
    assert legacy.register_encoder({"trigger": "tune"}) == "e0001"
    assert legacy.current_encoder() == "e0001"
    assert cold.current() == BASE_VERSION
    assert lube.current() == "v0001"
    assert legacy.current() == "v0001"
    assert legacy.rollback_encoder() == ENCODER_BASE
    assert lube.current() == "v0001"
    pointer = json.loads((tmp_path / "data" / "e2e" / "registry" / "current.json").read_text(encoding="utf-8"))
    assert pointer["version"] == "v0001"
    assert pointer["encoder"] == ENCODER_BASE
    assert pointer["profiles"]["lube_oil_system"] == "v0001"
    assert pointer["profiles"]["exhaust_cold_spot"] == BASE_VERSION
