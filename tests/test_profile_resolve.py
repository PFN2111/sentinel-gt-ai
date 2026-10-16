"""The alert name selects the alert type. Tags do not, and a tie cannot happen.

Offline. Built-ins come from the code, not a folder.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sentinel.analytics import resolve
from sentinel.profiles import alert_type_for_name, load_profiles, resolve_alert_type, resolve_tags

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def _header(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


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


def test_name_selects_one_alert_type():
    book = load_profiles()
    by_name = resolve_alert_type("Exhaust cold spot", book=book)
    by_alias = resolve_alert_type("TTXD cold spot", book=book)
    by_id = resolve_alert_type("exhaust_cold_spot", book=book)
    assert by_name.profile.id == by_alias.profile.id == by_id.profile.id == "exhaust_cold_spot"
    assert by_name.how == "name" and by_alias.how == "alias" and by_id.how == "name"
    resolved = resolve_tags(by_name.profile, _header(SAMPLES / "cold_spot" / "cold_spot_01.csv"))
    assert all(hit.status == "present" for hit in resolved.slots if hit.required)


def test_a_shared_name_is_refused_so_no_tie_is_possible(tmp_path):
    """Two alert types cannot own one name, so nothing is left to rank or tie-break."""
    folder = tmp_path / "profiles"
    folder.mkdir()
    (folder / "left_alarm.json").write_text(json.dumps(_site("left_alarm", "Same alarm", [])), encoding="utf-8")
    (folder / "right_alarm.json").write_text(json.dumps(_site("right_alarm", "Same alarm", [])), encoding="utf-8")
    book = load_profiles(profiles_dir=folder)
    owner = alert_type_for_name("Same alarm", book)
    assert owner is not None and owner.id == "left_alarm"
    assert "right_alarm" not in book
    listed = (folder / "_invalid.json").read_text(encoding="utf-8")
    assert "already belongs to left_alarm" in listed


@pytest.mark.parametrize(
    "name",
    [
        "GT alarm review",
        "Tier 1 escalation",
        "Exhaust temperature deviation advisory",
        "Unit advisory",
        "Performance deviation",
    ],
)
def test_generic_names_are_unknown(name):
    assert resolve_alert_type(name).how == "unknown"
    assert resolve(name, columns=["load_MW", "inlet_dp_mbar", "ambient_C", "CDP_bar"]).analytic == "generic"


def test_a_missing_tag_does_not_change_the_alert_type():
    """A short file stays on the named alert type. Tags never pick a different one."""
    choice = resolve_alert_type("Exhaust cold spot")
    assert choice.profile is not None and choice.profile.id == "exhaust_cold_spot"
    resolved = resolve_tags(choice.profile, ["timestamp"])
    assert any(hit.status != "present" for hit in resolved.slots)
    assert resolve("Exhaust cold spot", columns=["timestamp"]).analytic == "cold_spot"


def test_generic_name_is_unknown_even_when_every_tag_is_present():
    """A complete inlet header used to win by tag coverage. The name is unknown now."""
    columns = ["load_MW", "inlet_dp_mbar", "ambient_C", "CDP_bar"]
    choice = resolve_alert_type("Unit advisory")
    assert choice.how == "unknown" and choice.profile is None
    assert alert_type_for_name("Unit advisory", load_profiles()) is None
    res = resolve("Unit advisory", columns=columns)
    assert res.analytic == "generic"
    assert "inlet_filter" not in res.reason


def test_explicit_id_wins_and_an_unknown_id_raises():
    header = _header(SAMPLES / "hets_trip" / "hets_trip_01.csv")
    choice = resolve_alert_type("HETS Trip", explicit="combustion_dynamics")
    assert choice.how == "explicit" and choice.profile.id == "combustion_dynamics"
    res = resolve("HETS Trip", columns=header, explicit="combustion_dynamics")
    assert res.analytic == "combustion_dynamics"
    with pytest.raises(ValueError, match="unknown alert type"):
        resolve("HETS Trip", columns=header, explicit="no_such_alert_type")


def test_known_alias_with_a_header_uses_that_alert_type_not_the_keyword_table():
    """Name-only stays on the keyword table. A header follows the alert type."""
    header = _header(SAMPLES / "cold_spot" / "cold_spot_01.csv")
    assert resolve("Exhaust TC deviation").analytic == "generic"
    res = resolve("Exhaust TC deviation", columns=header)
    assert res.analytic == "cold_spot"
    assert "exhaust_cold_spot" in res.reason


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Exhaust Cold Spot", "cold_spot"),
        ("Combustion Dynamics High", "combustion_dynamics"),
        ("HETS Trip", "hets_trip"),
        ("Bearing vibration high", "generic"),
        ("", "generic"),
    ],
)
def test_builtin_names_match_analytics_resolve(name, expected):
    """Exact alert names, with a header, land on the same analyzer as the rule path."""
    if expected == "cold_spot":
        columns = _header(SAMPLES / "cold_spot" / "cold_spot_01.csv")
    elif expected == "combustion_dynamics":
        columns = _header(SAMPLES / "combustion_dynamics" / "comb_dyn_01.csv")
    elif expected == "hets_trip":
        columns = _header(SAMPLES / "hets_trip" / "hets_trip_01.csv")
    elif name == "Bearing vibration high":
        columns = _header(SAMPLES / "gt_sensors_demo.csv")
    else:
        columns = ["timestamp"]
    choice = resolve_alert_type(name)
    if name == "":
        assert choice.how == "unknown"
    else:
        assert choice.profile is not None
        assert choice.profile.analyzer == expected
    assert resolve(name).analytic == expected
    assert resolve(name, columns=columns).analytic == expected


def test_every_sample_csv_follows_analytics_resolve():
    folders = {
        "cold_spot": "Exhaust Cold Spot",
        "combustion_dynamics": "Combustion Dynamics High",
        "hets_trip": "HETS Trip",
    }
    for folder, analytic in folders.items():
        expected = resolve(analytic).analytic
        choice = resolve_alert_type(analytic)
        assert choice.profile is not None and choice.profile.analyzer == expected
        for path in sorted((SAMPLES / folder).glob("*.csv")):
            header = _header(path)
            resolved = resolve_tags(choice.profile, header)
            assert all(hit.status == "present" for hit in resolved.slots if hit.required), path.name
            assert resolve(analytic, columns=header).analytic == expected


def test_keyword_phrase_with_a_header_stays_on_the_rule_table():
    """A phrase that is not an alert name does not select an alert type."""
    header = _header(SAMPLES / "cold_spot" / "cold_spot_01.csv")
    phrase = "cold sector on tc 12"
    assert alert_type_for_name(phrase, load_profiles()) is None
    assert resolve_alert_type(phrase).how == "unknown"
    assert resolve(phrase).analytic == "cold_spot"
    assert resolve(phrase, columns=header).analytic == "cold_spot"
