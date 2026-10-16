"""Instrument faults injected into healthy data must route to sensor fix/replace, not hardware."""

from __future__ import annotations

import io

import numpy as np

from conftest import sample
from sentinel.analytics import cold_spot, combustion_dynamics, hets_trip
from sentinel.data import load_csv
from sentinel.notes import parse_notes

import pandas as pd


def _healthy(kind: str, i: int, frac: float) -> pd.DataFrame:
    df = pd.read_csv(sample(kind, i))
    return df.iloc[: int(len(df) * frac)].reset_index(drop=True)


def _case(df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return load_csv(buf.getvalue().encode(), name="injected.csv")


def _recompute_ring(df: pd.DataFrame, cols) -> None:
    df["EGT_spread_C"] = df[cols].max(axis=1) - df[cols].min(axis=1)
    df["EGT_avg_C"] = df[cols].mean(axis=1)


def test_single_tc_step_is_instrumentation():
    df = _healthy("cold_spot", 1, 0.30)  # cold event starts at 38%: this slice is healthy
    tcs = [f"TC{k}" for k in range(1, 28)]
    start = int(len(df) * 0.6)
    df.loc[start:, "TC12"] -= 40.0  # open-circuit style step on one TC
    _recompute_ring(df, tcs)
    f = cold_spot.analyze(_case(df), parse_notes(""))
    assert f.detected
    assert f.signature["primary_tc"] == "TC12"
    assert f.signature["sector_size"] == 1
    assert f.signature["step_onset"] is True
    assert f.top.cause == "instrumentation"
    assert f.top.action == "sensor_fix_replace"


def test_flatlined_tc_is_instrumentation():
    df = _healthy("cold_spot", 2, 0.30)
    tcs = [f"TC{k}" for k in range(1, 28)]
    start = int(len(df) * 0.5)
    df.loc[start:, "TC20"] = 515.0  # frozen reading well below the ring
    _recompute_ring(df, tcs)
    f = cold_spot.analyze(_case(df), parse_notes(""))
    assert f.top.cause == "instrumentation"
    assert f.top.action == "sensor_fix_replace"


def test_saturated_dynamics_probe_is_instrumentation():
    df = _healthy("combustion_dynamics", 1, 0.20)  # dynamics event starts at 29%
    start = int(len(df) * 0.5)
    df.loc[start : start + 240, "comb_dyn_psi"] = 2.5  # stuck / saturated for 4 h, nothing else moves
    f = combustion_dynamics.analyze(_case(df), parse_notes(""))
    assert f.detected
    assert f.signature["coupled"] is False
    assert f.signature["probe_suspect"] is True
    assert f.top.cause == "instrumentation"
    assert f.top.action == "sensor_fix_replace"


def test_single_zone_step_trip_is_instrumentation():
    df = _healthy("hets_trip", 1, 0.50)  # trip is at 57%: this slice has none
    zones = [f"EGT_Z{k}_C" for k in range(1, 5)]
    k = int(len(df) * 0.8)
    df.loc[k : k + 2, "EGT_Z2_C"] -= 60.0  # one zone drops in a single step
    df.loc[k + 2 : k + 5, "trip_active"] = 1
    df.loc[k + 2 : k + 5, "load_MW"] = 7.0
    df.loc[k + 2 : k + 5, "fuel_flow_kg_s"] = 0.6
    _recompute_ring(df, zones)
    f = hets_trip.analyze(_case(df), parse_notes(""))
    assert f.detected
    assert f.signature["single_zone"] is True
    assert f.signature["step_rise"] is True
    assert f.top.cause == "instrumentation"
    assert f.top.action == "sensor_fix_replace"


def test_healthy_data_does_not_confirm_alerts():
    df = _healthy("cold_spot", 3, 0.30)
    f = cold_spot.analyze(_case(df), parse_notes(""))
    assert not f.detected
    assert f.top.cause == "not_confirmed"
    assert f.top.action == "monitor_only"
    d = _healthy("combustion_dynamics", 2, 0.20)
    g = combustion_dynamics.analyze(_case(d), parse_notes(""))
    assert g.top.action == "monitor_only"
    assert np.isfinite(g.signature["baseline_psi"])
