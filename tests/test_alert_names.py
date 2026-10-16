"""The alert name selects the alert type. No tag scoring and no guessing.

Offline. Built-ins come from the code.
"""

from __future__ import annotations

import json

import pytest

from sentinel.profiles import (
    alert_type_for_name,
    load_profiles,
    resolve_alert_type,
)


def test_canonical_name_id_and_alias_map():
    book = load_profiles()
    assert alert_type_for_name("Exhaust cold spot", book).id == "exhaust_cold_spot"
    assert alert_type_for_name("exhaust_cold_spot", book).id == "exhaust_cold_spot"
    assert alert_type_for_name("TTXD cold spot", book).id == "exhaust_cold_spot"

    by_name = resolve_alert_type("Exhaust cold spot", book=book)
    assert by_name.how == "name" and by_name.profile.id == "exhaust_cold_spot"
    by_id = resolve_alert_type("exhaust_cold_spot", book=book)
    assert by_id.how == "name"
    by_alias = resolve_alert_type("TTXD cold spot", book=book)
    assert by_alias.how == "alias" and by_alias.matched == "TTXD cold spot"


def test_punctuation_and_case_are_ignored():
    book = load_profiles()
    assert alert_type_for_name("ttxd-cold.spot", book).id == "exhaust_cold_spot"
    assert alert_type_for_name("  EXHAUST   COLD SPOT ", book).id == "exhaust_cold_spot"


def test_unknown_and_empty_names_are_unknown():
    book = load_profiles()
    for name in ("", "   ", None, "not a listed alert"):
        choice = resolve_alert_type(name or "", book=book)
        assert choice.how == "unknown" and choice.profile is None
        assert alert_type_for_name(name or "", book) is None


def test_explicit_id_wins_over_the_name():
    book = load_profiles()
    choice = resolve_alert_type("HETS Trip", explicit="combustion_dynamics", book=book)
    assert choice.how == "explicit"
    assert choice.profile.id == "combustion_dynamics"


def test_unknown_explicit_id_raises():
    book = load_profiles()
    with pytest.raises(ValueError, match="no_such_alert_type"):
        resolve_alert_type("Exhaust cold spot", explicit="no_such_alert_type", book=book)


def _site(pid: str, name: str, aliases: list) -> dict:
    return {
        "schema": 2,
        "id": pid,
        "name": name,
        "aliases": aliases,
        "slots": [
            {
                "id": "load",
                "role": "context",
                "label": "unit load",
                "unit": "MW",
                "aliases": {"*": ["load_MW"]},
                "legacy_role": "load",
            }
        ],
        "signatures": [{"id": "normal", "action": "monitor_only"}],
        "analyzer": "generic",
        "source": "file",
    }


def test_site_alias_that_collides_with_a_builtin_is_refused(tmp_path):
    folder = tmp_path / "profiles"
    folder.mkdir()
    (folder / "site_cold.json").write_text(
        json.dumps(_site("site_cold", "Site cold", ["TTXD cold spot"])),
        encoding="utf-8",
    )
    book = load_profiles(profiles_dir=folder)
    assert "site_cold" not in book
    assert alert_type_for_name("TTXD cold spot", book).id == "exhaust_cold_spot"
    listed = (folder / "_invalid.json").read_text(encoding="utf-8")
    assert "site_cold.json" in listed
    assert "TTXD cold spot" in listed
    assert "already belongs to exhaust_cold_spot" in listed


# Design section 2.5, frozen before training. Order is part of the table.
_ALIAS_TABLE = {
    "exhaust_cold_spot": (
        "Exhaust cold spot",
        [
            "Exhaust Cold Spot",
            "Exhaust temperature cold spot",
            "Cold spot TC ring",
            "TTXD cold spot",
            "exh cold spot alarm",
            "Exhaust TC deviation",
            "TC failure advisory",
        ],
    ),
    "combustion_dynamics": (
        "Combustion dynamics",
        [
            "Combustion Dynamics High",
            "comb dyn high",
            "CDM pulsation alarm",
            "Combustion humming",
            "CDM channel alarm",
        ],
    ),
    "exhaust_spread_trip": (
        "High exhaust spread trip",
        ["HETS Trip", "Unit trip - exhaust spread", "High exhaust spread advisory"],
    ),
    "igv_fuel_control": (
        "IGV / fuel control",
        [
            "IGV position deviation",
            "IGV LVDT mismatch",
            "Fuel control oscillation",
            "Load hunting",
            "Fuel valve oscillation advisory",
            "Compressor IGV advisory",
        ],
    ),
    "bearing_vibration": (
        "Bearing vibration",
        ["Bearing vibration high", "Vibration advisory", "High vibration alarm", "Vibration advisory BB1"],
    ),
    "inlet_filter_dp": (
        "Inlet filter high differential pressure",
        ["Inlet filter high dp", "Inlet dp high", "Filter house dp advisory"],
    ),
    "lube_oil_system": (
        "Lube oil temperature",
        ["Lube oil temperature high", "Lube cooler advisory", "Drain temperature alarm"],
    ),
}


def test_builtin_alias_table_and_channels_hashes():
    """The frozen alias table, no name collisions, and the step-20 channels hashes."""
    import json
    from pathlib import Path

    from sentinel.profiles import channels_hash, name_index, profile_name_keys, validate_profile
    from sentinel.profiles import BUILTIN_PROFILES

    book = load_profiles()
    assert list(book) == list(_ALIAS_TABLE)
    seen = {}
    for pid, (canonical, aliases) in _ALIAS_TABLE.items():
        raw = next(row for row in BUILTIN_PROFILES if row["id"] == pid)
        assert validate_profile(raw) == [], pid
        prof = book[pid]
        assert prof.name == canonical
        assert prof.aliases == aliases
        assert prof.match["keywords"] == []
        for key in profile_name_keys(prof):
            assert key not in seen, (key, seen[key], pid)
            seen[key] = pid
        meta_path = Path(__file__).resolve().parents[1] / "models" / "e2e" / "profiles" / pid / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert channels_hash(prof) == meta["channels_hash"], pid
    assert len(name_index(book)) == len(seen)


@pytest.mark.parametrize(
    "phrase",
    [
        "cold_spot",
        "TTXD cold sector alarm",
        "comb dyn",
        "CDM pulsation alert",
        "High Exhaust Temperature Spread trip",
    ],
)
def test_keyword_phrases_are_not_alert_names(phrase):
    """Phrases the old keyword table matched are not alert names.

    The rule path keeps its own keyword table. An alert type is selected only
    by the canonical name, the id, or an alias in the frozen table.
    """
    assert alert_type_for_name(phrase, load_profiles()) is None


def test_generator_names_map_to_one_builtin_that_lists_the_signature():
    """Every real generator alert name maps to one built-in. Generic and heater names do not."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gen_names", root / "scripts" / "generate_scenario_csvs.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    ANALYTIC_NAMES = gen.ANALYTIC_NAMES
    GENERIC_ANALYTICS = gen.GENERIC_ANALYTICS
    PROFILE_NAMES = gen.PROFILE_NAMES
    PROFILE_SIGNATURES = gen.PROFILE_SIGNATURES
    SEEN = gen.SEEN

    book = load_profiles()
    skip = set(GENERIC_ANALYTICS) | {"Performance deviation", "Fuel gas heater outlet low", "Heater performance advisory"}
    raised = {}
    for sig, names in ANALYTIC_NAMES.items():
        if sig not in SEEN:
            continue
        for name in names:
            raised.setdefault(name, set()).add(sig)
    for pid, names in PROFILE_NAMES.items():
        if pid == "fuel_gas_heater":
            for name in names:
                assert name in skip
            continue
        for name in names:
            raised.setdefault(name, set())
            for sig in PROFILE_SIGNATURES[pid]:
                if sig in SEEN:
                    raised[name].add(sig)
    for name in skip:
        assert alert_type_for_name(name, book) is None, name
    for name, sigs in raised.items():
        if name in skip:
            continue
        prof = alert_type_for_name(name, book)
        assert prof is not None, name
        listed = {str(sig) for sig in prof.signatures}
        missing = sigs - listed
        assert not missing, (name, prof.id, missing)


def test_site_override_replaces_the_builtin_names(tmp_path):
    folder = tmp_path / "profiles"
    folder.mkdir()
    (folder / "exhaust_cold_spot.json").write_text(
        json.dumps(_site("exhaust_cold_spot", "Exhaust cold spot", ["Unit 3 cold sector"])),
        encoding="utf-8",
    )
    book = load_profiles(profiles_dir=folder)
    assert book["exhaust_cold_spot"].source == "file"
    assert alert_type_for_name("Unit 3 cold sector", book).id == "exhaust_cold_spot"
    assert alert_type_for_name("Exhaust cold spot", book).id == "exhaust_cold_spot"
    assert alert_type_for_name("TTXD cold spot", book) is None
    choice = resolve_alert_type("Unit 3 cold sector", book=book)
    assert choice.how == "alias"
