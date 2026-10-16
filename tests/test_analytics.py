"""Analyzers against the generator's ground truth on every sample CSV."""

from __future__ import annotations

import pytest

from conftest import case_for, cold_truth, dyn_truth, hets_truth
from sentinel.analytics import cold_spot, combustion_dynamics, hets_trip, resolve
from sentinel.notes import parse_notes

NO_NOTES = parse_notes("")


@pytest.mark.parametrize("i", range(1, 11))
def test_cold_spot_sector_found(i):
    f = cold_spot.analyze(case_for("cold_spot", i), NO_NOTES)
    t = cold_truth(i)
    s = f.signature
    assert f.detected
    assert s["primary_tc"] == f"TC{t['tc']}"
    assert s["sector_size"] == t["width"]
    assert s["adjacent"] is True
    assert abs(s["depth_C"] - t["depth"]) <= 3.0
    assert abs(f.events[0]["start_i"] - t["onset_i"]) <= 12  # within 2 h at 10 min
    assert s["step_onset"] is False
    assert f.top.cause == "combustion_hardware"
    assert f.top.action == "borescope"
    assert f.hypotheses[1].cause == "instrumentation"
    assert f.severity == ("high" if t["depth"] >= 25 else "medium")


@pytest.mark.parametrize("i", range(1, 11))
def test_combustion_dynamics_event_found(i):
    f = combustion_dynamics.analyze(case_for("combustion_dynamics", i), NO_NOTES)
    t = dyn_truth(i)
    s = f.signature
    assert f.detected
    assert abs(f.events[0]["start_i"] - t["onset_i"]) <= 10  # minutes
    assert s["modulation_period_min"] == pytest.approx(t["period"], abs=1)
    assert s["coupled"] is True
    assert s["probe_suspect"] is False
    assert s["peak_psi"] > 1.0
    assert f.top.cause == "combustion_instability"
    assert f.top.action == "combustion_tuning"
    # Escalate once dynamics sit above the alarm limit for a sustained period
    assert f.escalate == (s["minutes_above_alarm"] >= 30)


def test_dynamics_severity_increases_with_amplitude():
    peaks = [combustion_dynamics.analyze(case_for("combustion_dynamics", i), NO_NOTES).signature["peak_psi"] for i in (1, 5, 10)]
    assert peaks == sorted(peaks)
    assert combustion_dynamics.analyze(case_for("combustion_dynamics", 1), NO_NOTES).severity == "medium"
    assert combustion_dynamics.analyze(case_for("combustion_dynamics", 10), NO_NOTES).severity in ("high", "critical")


@pytest.mark.parametrize("i", range(1, 6))
def test_hets_trip_sequence(i):
    f = hets_trip.analyze(case_for("hets_trip", i), parse_notes("first-out: high exhaust spread"))
    t = hets_truth(i)
    s = f.signature
    assert f.detected
    assert f.events[0]["trip_i"] == t["trip_i"]
    assert s["opposing_hot_cold"] is True
    assert s["hot_zones"] == ["EGT_Z1_C"] and s["cold_zones"] == ["EGT_Z3_C"]
    assert s["step_rise"] is False
    assert 60 <= s["rise_minutes"] <= 240
    assert s["peak_spread_C"] >= t["peak"]
    assert f.top.cause == "combustion_hardware"
    assert f.top.action == "borescope"
    assert f.escalate is True
    assert f.severity == "high"


def test_hets_flags_implausibly_fast_return_to_load():
    # Sample 3 is generated without a recovery ramp: load is back 10 min after the trip flag
    f = hets_trip.analyze(case_for("hets_trip", 3), NO_NOTES)
    assert any("runback" in fl for fl in f.review_flags)


def test_hets_first_out_conflict_flagged():
    f = hets_trip.analyze(case_for("hets_trip", 1), parse_notes("First-out was flame detector loss"))
    assert any("first-out" in fl.lower() for fl in f.review_flags)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Exhaust Cold Spot", "cold_spot"),
        ("cold_spot", "cold_spot"),
        ("TTXD cold sector alarm", "cold_spot"),
        ("Combustion Dynamics High", "combustion_dynamics"),
        ("comb dyn", "combustion_dynamics"),
        ("CDM pulsation alert", "combustion_dynamics"),
        ("HETS Trip", "hets_trip"),
        ("High Exhaust Temperature Spread trip", "hets_trip"),
        ("Bearing vibration high", "generic"),
        ("", "generic"),
    ],
)
def test_resolve_analytic_names(name, expected):
    assert resolve(name).analytic == expected


def test_legacy_roles_are_what_the_analyzer_sees_and_the_case_is_not_mutated():
    """A profile resolution can point the rule analyzer at its own tags without editing the case."""
    from sentinel.analytics import run_analyzer
    from sentinel.profiles import resolve_alert_type, resolve_tags

    case = case_for("cold_spot", 1)
    original_ring = list(case.ring)
    choice = resolve_alert_type("Exhaust Cold Spot")
    assert choice.profile is not None and choice.profile.analyzer == "cold_spot"
    resolved = resolve_tags(choice.profile, list(case.df.columns))
    assert cold_spot.REQUIRES["load"] is True
    res = resolve("Exhaust Cold Spot", columns=list(case.df.columns))
    assert res.analytic == "cold_spot"
    findings = run_analyzer(res, case, NO_NOTES, legacy=resolved.legacy)
    assert findings.detected
    assert case.ring == original_ring
