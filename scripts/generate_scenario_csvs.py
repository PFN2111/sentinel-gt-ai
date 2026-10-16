#!/usr/bin/env python3
"""
Generate synthetic gas-turbine historian CSVs.

1. Scenario samples (default, unchanged — the rule-path test ground truth):
   - 10× combustion cold spot   — 2 months, 10-minute resolution
   - 5×  HETS trips             — 2 months, 10-minute resolution
   - 10× combustion dynamics    — 10 days, 1-minute resolution

2. Labeled training/evaluation dataset for the end-to-end learned path (``--dataset``):
   8 signatures (normal, cold spot hardware, TC sensor fault, combustion instability,
   dynamics probe fault, HETS trip, IGV LVDT drift, fuel control oscillation) with varied
   frames, resolutions, record lengths, tag naming, operating profiles, missing data,
   ambiguous cases and noisy notes; splits train / val / test (iid held-out), variant
   (shifted can, load, magnitude) and unseen (a fault type never used in training).
   See ``generate_dataset``.

       python scripts/generate_scenario_csvs.py                       # scenario samples
       python scripts/generate_scenario_csvs.py --dataset data/e2e/dataset [--scale 1.0]

Timestamps use real calendar dates (timezone-naive local plant time).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
OUT_COLD = SAMPLES / "cold_spot"
OUT_HETS = SAMPLES / "hets_trip"
OUT_DYN = SAMPLES / "combustion_dynamics"

# Exhaust TC ring size for cold-spot cases (plant-style circumferential array)
N_EXHAUST_TC = 27
TC_COLS = [f"TC{i}" for i in range(1, N_EXHAUST_TC + 1)]

# Column set for HETS / dynamics (4 zone summary + process)
COLUMNS = [
    "timestamp",
    "load_MW",
    "CDP_bar",
    "CDT_C",
    "EGT_avg_C",
    "EGT_spread_C",
    "EGT_Z1_C",
    "EGT_Z2_C",
    "EGT_Z3_C",
    "EGT_Z4_C",
    "fuel_flow_kg_s",
    "IGV_deg",
    "vib_DE_mm_s",
    "vib_NDE_mm_s",
    "lube_oil_C",
    "comb_dyn_psi",
    "trip_active",
]

# Cold-spot columns: process + full TC1..TC27 ring (no EGT_Z*)
COLUMNS_COLD = (
    [
        "timestamp",
        "load_MW",
        "CDP_bar",
        "CDT_C",
        "EGT_avg_C",
        "EGT_spread_C",
    ]
    + TC_COLS
    + [
        "fuel_flow_kg_s",
        "IGV_deg",
        "vib_DE_mm_s",
        "vib_NDE_mm_s",
        "lube_oil_C",
        "comb_dyn_psi",
        "trip_active",
    ]
)


def _baseload_core(
    index: pd.DatetimeIndex, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Shared process signals; returns (partial_df_without_TCs, load, egt_mean)."""
    n = len(index)
    hour = np.asarray(index.hour, dtype=float) + np.asarray(index.minute, dtype=float) / 60.0
    load = 48.0 + 1.2 * np.sin(2 * np.pi * (hour - 8) / 24.0) + rng.normal(0, 0.15, n)
    load = np.asarray(np.clip(load, 35.0, 55.0), dtype=float)

    cdp = 11.1 + 0.04 * (load - 48) + rng.normal(0, 0.03, n)
    cdt = 345.0 + 0.8 * (load - 48) + rng.normal(0, 0.25, n)
    egt_mean = np.asarray(
        540.0 + 1.1 * (load - 48) + rng.normal(0, 0.4, n), dtype=float
    )

    fuel = 2.85 + 0.035 * (load - 48) + rng.normal(0, 0.01, n)
    igv = 62.0 + 0.25 * (load - 48) + rng.normal(0, 0.15, n)
    vib_de = 3.05 + rng.normal(0, 0.08, n)
    vib_nde = 2.95 + rng.normal(0, 0.08, n)
    lube = 58.0 + 0.05 * (load - 48) + rng.normal(0, 0.12, n)
    dyn = np.clip(0.35 + rng.normal(0, 0.05, n), 0.05, 1.2)

    base = pd.DataFrame(
        {
            "timestamp": index,
            "load_MW": np.round(load, 3),
            "CDP_bar": np.round(cdp, 3),
            "CDT_C": np.round(cdt, 3),
            "fuel_flow_kg_s": np.round(fuel, 4),
            "IGV_deg": np.round(igv, 3),
            "vib_DE_mm_s": np.round(vib_de, 3),
            "vib_NDE_mm_s": np.round(vib_nde, 3),
            "lube_oil_C": np.round(lube, 3),
            "comb_dyn_psi": np.round(dyn, 3),
            "trip_active": 0,
        }
    )
    return base, load, egt_mean


def _baseload_frame(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Healthy baseload with 4 zone EGTs (HETS / dynamics samples)."""
    base, load, egt_mean = _baseload_core(index, rng)
    n = len(index)
    z = np.column_stack([egt_mean + rng.normal(0, 1.2, n) for _ in range(4)])
    base["EGT_avg_C"] = np.round(z.mean(axis=1), 3)
    base["EGT_spread_C"] = np.round(z.max(axis=1) - z.min(axis=1), 3)
    for i in range(4):
        base[f"EGT_Z{i + 1}_C"] = np.round(z[:, i], 3)
    return base[COLUMNS]


def _baseload_frame_cold(
    index: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """Healthy baseload with full exhaust TC ring TC1..TC27."""
    base, _load, egt_mean = _baseload_core(index, rng)
    n = len(index)
    # Circumferential pattern: small spatial sine + independent noise
    angles = 2 * np.pi * np.arange(N_EXHAUST_TC) / N_EXHAUST_TC
    spatial = 1.5 * np.sin(angles)[None, :]  # mild fixed pattern
    tc = egt_mean[:, None] + spatial + rng.normal(0, 1.0, (n, N_EXHAUST_TC))
    for i, col in enumerate(TC_COLS):
        base[col] = np.round(tc[:, i], 3)
    base["EGT_avg_C"] = np.round(tc.mean(axis=1), 3)
    base["EGT_spread_C"] = np.round(tc.max(axis=1) - tc.min(axis=1), 3)
    return base[COLUMNS_COLD]


def _inject_cold_spot(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    cold_tc: int,
    onset_frac: float,
    depth_C: float,
    duration_frac: float,
    load_dip: bool,
    sector_width: int = 3,
) -> pd.DataFrame:
    """
    Inject a circumferential cold sector on the TC1..TC27 ring.

    cold_tc : 1..27 primary cold thermocouple (neighbors also cool slightly).
    sector_width : odd-ish count of adjacent TCs in the cold sector (wraps ring).
    """
    out = df.copy()
    n = len(out)
    start = int(n * onset_frac)
    end = min(n, start + max(12, int(n * duration_frac)))
    width = max(1, int(sector_width))
    # Adjacent indices on a 27-point ring (0-based)
    center = (int(cold_tc) - 1) % N_EXHAUST_TC
    half = width // 2
    sector = [((center + k) % N_EXHAUST_TC) for k in range(-half, half + 1)]
    # Center deepest; neighbors 40–70% of depth
    weights = []
    for idx in sector:
        dist = min((idx - center) % N_EXHAUST_TC, (center - idx) % N_EXHAUST_TC)
        weights.append(1.0 if dist == 0 else max(0.35, 1.0 - 0.28 * dist))

    ramp_n = min(36, end - start)
    ramp = np.linspace(0, 1.0, ramp_n)
    hold = np.ones(max(0, end - start - ramp_n))
    envelope = np.concatenate([ramp, hold])[: end - start]

    tc = out[TC_COLS].to_numpy(copy=True)
    for w, idx in zip(weights, sector):
        bias = depth_C * w * envelope + rng.normal(0, 0.35, len(envelope))
        tc[start:end, idx] = tc[start:end, idx] - bias

    for i, col in enumerate(TC_COLS):
        out[col] = np.round(tc[:, i], 3)
    out["EGT_avg_C"] = np.round(tc.mean(axis=1), 3)
    out["EGT_spread_C"] = np.round(tc.max(axis=1) - tc.min(axis=1), 3)

    # Mild fuel / performance coupling
    out.loc[out.index[start:end], "fuel_flow_kg_s"] = np.round(
        out.loc[out.index[start:end], "fuel_flow_kg_s"].to_numpy()
        + 0.02
        + rng.normal(0, 0.005, end - start),
        4,
    )
    if load_dip:
        out.loc[out.index[start:end], "load_MW"] = np.round(
            out.loc[out.index[start:end], "load_MW"].to_numpy() - 0.4 - rng.uniform(0, 0.6),
            3,
        )
    if depth_C >= 28:
        out.loc[out.index[start:end], "vib_DE_mm_s"] = np.round(
            out.loc[out.index[start:end], "vib_DE_mm_s"].to_numpy() + 0.4, 3
        )
    return out


def _inject_hets_trip(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    trip_frac: float,
    peak_spread_C: float,
    recovery: bool,
) -> pd.DataFrame:
    """
    High Exhaust Temperature Spread (HETS) trip signature:
    rapid spread rise → protection trip → load collapse → optional recovery.
    """
    out = df.copy()
    n = len(out)
    trip_i = int(n * trip_frac)
    pre = max(0, trip_i - 18)  # ~3 hours at 10-min (18*10m)
    # Build-up: sectors diverge
    build = np.linspace(0, 1.0, trip_i - pre)
    z = out[[f"EGT_Z{i}_C" for i in range(1, 5)]].to_numpy(copy=True)
    # Make Z1 hot, Z3 cold for true spread (not single TC)
    for j, idx in enumerate(range(pre, trip_i)):
        f = build[j]
        z[idx, 0] += f * (peak_spread_C * 0.55) + rng.normal(0, 0.5)
        z[idx, 2] -= f * (peak_spread_C * 0.45) + rng.normal(0, 0.5)
        z[idx, 1] += f * 2.0
        z[idx, 3] -= f * 1.5

    # Trip window: ~40 minutes
    trip_end = min(n, trip_i + 4)
    for idx in range(trip_i, trip_end):
        z[idx, 0] += peak_spread_C * 0.6 + rng.normal(0, 1.0)
        z[idx, 2] -= peak_spread_C * 0.5 + rng.normal(0, 1.0)
        out.loc[out.index[idx], "trip_active"] = 1
        out.loc[out.index[idx], "load_MW"] = max(0.0, float(out.loc[out.index[idx], "load_MW"]) * 0.15)
        out.loc[out.index[idx], "fuel_flow_kg_s"] = max(
            0.2, float(out.loc[out.index[idx], "fuel_flow_kg_s"]) * 0.2
        )
        out.loc[out.index[idx], "IGV_deg"] = 25.0 + rng.normal(0, 1.0)
        out.loc[out.index[idx], "vib_DE_mm_s"] = 6.5 + rng.normal(0, 0.3)
        out.loc[out.index[idx], "vib_NDE_mm_s"] = 5.8 + rng.normal(0, 0.3)
        out.loc[out.index[idx], "comb_dyn_psi"] = 1.8 + rng.normal(0, 0.2)

    if recovery and trip_end < n:
        # Coast / restart to baseload over ~1 day
        rec_end = min(n, trip_end + 144)  # 144*10min = 24h
        for k, idx in enumerate(range(trip_end, rec_end)):
            f = min(1.0, k / max(1, rec_end - trip_end - 1))
            # zones re-blend toward mean
            mean = z[idx].mean()
            z[idx] = mean + (z[idx] - mean) * (1.0 - 0.85 * f)
            out.loc[out.index[idx], "load_MW"] = 20.0 + f * 28.0 + rng.normal(0, 0.2)
            out.loc[out.index[idx], "trip_active"] = 0

    out[[f"EGT_Z{i}_C" for i in range(1, 5)]] = np.round(z, 3)
    out["EGT_avg_C"] = np.round(z.mean(axis=1), 3)
    out["EGT_spread_C"] = np.round(z.max(axis=1) - z.min(axis=1), 3)
    # Clip physical-ish bounds
    out["load_MW"] = np.clip(out["load_MW"], 0, 60)
    out["vib_DE_mm_s"] = np.round(np.clip(out["vib_DE_mm_s"], 0.5, 15), 3)
    out["vib_NDE_mm_s"] = np.round(np.clip(out["vib_NDE_mm_s"], 0.5, 15), 3)
    out["comb_dyn_psi"] = np.round(np.clip(out["comb_dyn_psi"], 0.05, 8), 3)
    return out


def _inject_combustion_dynamics(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    onset_frac: float,
    severity: float,
    tone_period_min: float,
) -> pd.DataFrame:
    """
    Combustion dynamics (pulsation) burst at 1-minute resolution.
    severity: 0.5 mild → 3.0 severe (psi peak)
    """
    out = df.copy()
    n = len(out)
    start = int(n * onset_frac)
    # Event lasts 6–18 hours
    dur = int(rng.integers(6 * 60, 18 * 60))
    end = min(n, start + dur)
    t = np.arange(end - start, dtype=float)
    period = max(2.0, tone_period_min)
    osc = severity * (0.55 + 0.45 * np.sin(2 * np.pi * t / period))
    osc += 0.15 * severity * np.sin(2 * np.pi * t / (period * 0.37))
    osc += rng.normal(0, 0.08 * severity, len(t))
    osc = np.clip(osc, 0.1, 8.0)

    base = out.loc[out.index[start:end], "comb_dyn_psi"].to_numpy()
    out.loc[out.index[start:end], "comb_dyn_psi"] = np.round(base + osc, 3)

    # Couple to EGT spread flutter and mild vib
    zcols = [f"EGT_Z{i}_C" for i in range(1, 5)]
    z = out.loc[out.index[start:end], zcols].to_numpy(copy=True)
    flutter = (osc / max(severity, 1e-6)) * rng.uniform(3.0, 8.0)
    z[:, 0] += flutter * 0.6
    z[:, 2] -= flutter * 0.5
    out.loc[out.index[start:end], zcols] = np.round(z, 3)
    out.loc[out.index[start:end], "EGT_avg_C"] = np.round(z.mean(axis=1), 3)
    out.loc[out.index[start:end], "EGT_spread_C"] = np.round(z.max(axis=1) - z.min(axis=1), 3)
    out.loc[out.index[start:end], "vib_DE_mm_s"] = np.round(
        out.loc[out.index[start:end], "vib_DE_mm_s"].to_numpy() + 0.25 * osc, 3
    )
    out.loc[out.index[start:end], "fuel_flow_kg_s"] = np.round(
        out.loc[out.index[start:end], "fuel_flow_kg_s"].to_numpy()
        + 0.01 * (osc / max(severity, 1e-6)),
        4,
    )
    return out


def _write(df: pd.DataFrame, path: Path, meta: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ISO-8601 timestamps without timezone (plant local)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}  rows={len(out)}  {meta}")


def generate_all() -> None:
    OUT_COLD.mkdir(parents=True, exist_ok=True)
    OUT_HETS.mkdir(parents=True, exist_ok=True)
    OUT_DYN.mkdir(parents=True, exist_ok=True)

    # --- Cold spots: 2 months @ 10 min, staggered start dates, TC1..TC27 ---
    print("Combustion cold spots (10 files, ~2 months @ 10 min, TC1–TC27)…")
    cold_starts = [
        "2025-11-01",
        "2025-12-01",
        "2026-01-05",
        "2026-02-01",
        "2026-03-01",
        "2026-04-01",
        "2026-05-01",
        "2026-06-01",
        "2025-09-15",
        "2025-10-10",
    ]
    for i, start in enumerate(cold_starts, start=1):
        rng = np.random.default_rng(1000 + i)
        idx = pd.date_range(start=start, periods=60 * 24 * 6, freq="10min")  # 60 days
        df = _baseload_frame_cold(idx, rng)
        # Spread cold centers around the 27-TC ring
        cold_tc = 1 + ((i - 1) * 3) % N_EXHAUST_TC  # TC1,4,7,..., then wrap
        depth = 18 + (i % 5) * 4  # 18..34 C center depth
        onset = 0.35 + 0.03 * (i % 4)
        dur = 0.08 + 0.02 * (i % 3)  # multi-day event
        sector_w = 3 if i % 3 else 5  # 3- or 5-TC cold sector
        df = _inject_cold_spot(
            df,
            rng,
            cold_tc=cold_tc,
            onset_frac=onset,
            depth_C=float(depth),
            duration_frac=dur,
            load_dip=(i % 2 == 0),
            sector_width=sector_w,
        )
        _write(
            df,
            OUT_COLD / f"cold_spot_{i:02d}.csv",
            f"cold_center=TC{cold_tc} sector={sector_w} depth≈{depth}C start={start}",
        )

    # --- HETS trips: 2 months @ 10 min ---
    print("HETS trips (5 files, ~2 months @ 10 min)…")
    hets_starts = [
        "2025-08-01",
        "2025-11-15",
        "2026-01-10",
        "2026-03-20",
        "2026-05-12",
    ]
    for i, start in enumerate(hets_starts, start=1):
        rng = np.random.default_rng(2000 + i)
        idx = pd.date_range(start=start, periods=60 * 24 * 6, freq="10min")
        df = _baseload_frame(idx, rng)
        peak = 48 + i * 4  # trip-class spreads
        trip_frac = 0.55 + 0.02 * i
        df = _inject_hets_trip(
            df,
            rng,
            trip_frac=trip_frac,
            peak_spread_C=float(peak),
            recovery=(i != 3),  # one case leaves unit offline-ish longer
        )
        _write(
            df,
            OUT_HETS / f"hets_trip_{i:02d}.csv",
            f"peak_spread≈{peak}C trip≈{trip_frac:.2f} start={start}",
        )

    # --- Combustion dynamics: 10 days @ 1 min ---
    print("Combustion dynamics (10 files, 10 days @ 1 min)…")
    dyn_starts = [
        "2026-01-08",
        "2026-01-22",
        "2026-02-05",
        "2026-02-18",
        "2026-03-03",
        "2026-03-17",
        "2026-04-02",
        "2026-04-20",
        "2026-05-06",
        "2026-05-25",
    ]
    for i, start in enumerate(dyn_starts, start=1):
        rng = np.random.default_rng(3000 + i)
        idx = pd.date_range(start=start, periods=10 * 24 * 60, freq="1min")
        df = _baseload_frame(idx, rng)
        severity = 0.8 + 0.25 * i  # psi-class peaks
        period = 3.0 + (i % 5)  # minutes between pulsation peaks in envelope
        onset = 0.25 + 0.04 * (i % 5)
        df = _inject_combustion_dynamics(
            df,
            rng,
            onset_frac=onset,
            severity=float(severity),
            tone_period_min=float(period),
        )
        _write(
            df,
            OUT_DYN / f"comb_dyn_{i:02d}.csv",
            f"severity≈{severity:.2f}psi period≈{period}min start={start}",
        )

    # Manifest
    manifest = SAMPLES / "SCENARIO_README.md"
    manifest.write_text(
        """# Scenario sample CSVs

Generated by `scripts/generate_scenario_csvs.py`.

| Folder | Cases | Duration | Resolution | Signature |
|--------|------:|----------|------------|-----------|
| `cold_spot/` | 10 | ~60 days | 10 min | Cold sector on **TC1–TC27** ring → elevated `EGT_spread_C` |
| `hets_trip/` | 5 | ~60 days | 10 min | Rapid spread rise, `trip_active=1`, load/fuel collapse |
| `combustion_dynamics/` | 10 | 10 days | 1 min | Elevated `comb_dyn_psi` with EGT flutter / mild vib |

## Columns
**Cold spot:** `timestamp, load_MW, CDP_bar, CDT_C, EGT_avg_C, EGT_spread_C, TC1..TC27,
fuel_flow_kg_s, IGV_deg, vib_DE_mm_s, vib_NDE_mm_s, lube_oil_C, comb_dyn_psi, trip_active`

**HETS / dynamics:** same process channels with `EGT_Z1_C..EGT_Z4_C` instead of TC1–TC27

Timestamps are real calendar datetimes (`YYYY-MM-DD HH:MM:SS`).

## Process maps (RAG)
See `knowledge/process_map_cold_spot.md`, `process_map_hets_trip.md`,
`process_map_combustion_dynamics.md`.

## Suggested TUI tests
1. **Cold spot** — `samples/cold_spot/cold_spot_01.csv`, mode **Alerts**, paste cold-spot process map notes.
2. **HETS** — `samples/hets_trip/hets_trip_01.csv`, mode **Trips/Event**, note trip first-out HETS.
3. **Dynamics** — `samples/combustion_dynamics/comb_dyn_01.csv`, mode **Alerts** or **Trips/Event**.

Regenerate:
```bash
python scripts/generate_scenario_csvs.py
```
""",
        encoding="utf-8",
    )
    print(f"  wrote {manifest.relative_to(ROOT)}")
    print("Done.")


# =========================================================================== labeled dataset
# Larger, varied, labeled synthetic set for the end-to-end learned path (``--dataset``).

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentinel.e2e.taxonomy import BASE_SIGNATURES, UNSEEN_SIGNATURES  # noqa: E402

# The shipped training set stays these eight. The newer taxonomy ids are real
# signatures; the generator learns to write them in a later step. Counting
# them here would change the dataset size the current bundle was trained on.
_ORIGINAL_SEEN = (
    "normal",
    "cold_spot_hardware",
    "tc_sensor_fault",
    "combustion_instability",
    "dynamics_probe_fault",
    "hets_trip_hardware",
    "igv_lvdt_drift",
    "fuel_control_oscillation",
)
SEEN = [s for s in _ORIGINAL_SEEN if s in BASE_SIGNATURES]
UNSEEN = list(UNSEEN_SIGNATURES)
SPLIT_COUNTS = {"train": 150, "val": 25, "test": 40}  # per seen signature (scale=1)
VARIANT_CLASSES = ["cold_spot_hardware", "tc_sensor_fault", "combustion_instability", "hets_trip_hardware", "igv_lvdt_drift"]
VARIANT_KINDS = ["shifted", "load", "magnitude"]
VARIANT_COUNT = 8  # per (class, variant kind)
UNSEEN_COUNT = 40
# Held-out sensor-fault set (V2): one TC steps down by a cold-spot-sized amount, alone, on a ring,
# raised as a cold spot. Truth: tc_sensor_fault (sensor repair). Tests the residual cross-check.
SENSOR_FAULT_COUNT = 40

# OEM-style historian tags, used for ~30 % of files (exercises channel-role mapping)
ALT_TAGS = {
    "load_MW": "DWATT", "CDP_bar": "CPD", "CDT_C": "CTD", "EGT_avg_C": "TTXM", "EGT_spread_C": "TTXSP1",
    "fuel_flow_kg_s": "FQG", "IGV_deg": "CSGV", "vib_DE_mm_s": "BB1", "vib_NDE_mm_s": "BB2",
    "lube_oil_C": "LTOT", "comb_dyn_psi": "CDM_AMP", "trip_active": "TRIP_FLAG",
}

ANALYTIC_NAMES = {
    "normal": ["Exhaust Cold Spot", "Combustion Dynamics High", "High exhaust spread advisory", "Exhaust TC deviation",
               "Performance deviation", "comb dyn high", "IGV position deviation", "Load hunting"],
    "cold_spot_hardware": ["Exhaust Cold Spot", "Exhaust temperature cold spot", "Cold spot TC ring", "TTXD cold spot", "exh cold spot alarm"],
    "tc_sensor_fault": ["Exhaust Cold Spot", "Exhaust TC deviation", "High exhaust spread advisory", "TC failure advisory"],
    "combustion_instability": ["Combustion Dynamics High", "comb dyn high", "CDM pulsation alarm", "Combustion humming"],
    "dynamics_probe_fault": ["Combustion Dynamics High", "comb dyn high", "CDM channel alarm"],
    "hets_trip_hardware": ["HETS Trip", "High exhaust spread trip", "Unit trip - exhaust spread"],
    "igv_lvdt_drift": ["IGV position deviation", "IGV LVDT mismatch", "Compressor IGV advisory"],
    "fuel_control_oscillation": ["Fuel control oscillation", "Load hunting", "Fuel valve oscillation advisory"],
    "bearing_vibration": ["Bearing vibration high", "Vibration advisory BB1", "High vibration alarm"],
    "tc_harness_fault": ["Exhaust Cold Spot", "High exhaust spread advisory", "TTXD cold spot", "Exhaust TC deviation"],
}
GENERIC_ANALYTICS = ["GT alarm review", "Tier 1 escalation", "Exhaust temperature deviation advisory", "Unit advisory"]

HINTS = {
    "normal": ["Alert cleared on its own after a few hours.", "No operator complaints.", "Alert may be a false positive."],
    "cold_spot_hardware": ["Operator noticed exhaust spread creeping up over a few days.", "Fuel nozzle work done at last CI.",
                           "Spread has been trending up since last week."],
    "tc_sensor_fault": ["TC{tc} has been erratic before.", "I&C replaced the harness near TC{tc} last outage.",
                        "Reading on TC{tc} looks odd to the operator."],
    "combustion_instability": ["Operator reports humming from the combustor area.", "Unit was re-tuned last month.",
                               "Dynamics alarms after the load change this morning."],
    "dynamics_probe_fault": ["Dynamics probe cable was disturbed during maintenance.", "CDM channel shows strange values.",
                             "No humming heard on site."],
    "hets_trip_hardware": ["Unit tripped, first-out: high exhaust spread.", "Trip on HETS, unit restarted after checks.",
                           "Control room reports exhaust spread trip."],
    "igv_lvdt_drift": ["IGV calibration last done two years ago.", "Operator noticed IGV indication differs from local gauge.",
                       "Servo work done recently on IGV actuator."],
    "fuel_control_oscillation": ["Control room reports load hunting.", "Fuel valve was replaced last outage.",
                                 "Megawatts swinging slightly at baseload."],
    "bearing_vibration": ["Operator reports noise near bearing 1.", "Lube oil temperature alarm came in later.",
                          "Vibration trending up for days."],
    "tc_harness_fault": ["I&C was working in the turbine compartment last week.", "Several TCs on one side dropped out briefly.",
                         "Harness junction box was opened during the outage."],
}
FILLER = ["Please review asap.", "Ambient about 35 C today.", "Tier 1: alert auto-generated.", "Unit on AGC.",
          "Customer asked for a quick answer.", "Night shift, no further info.", "See attached data.", ""]


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _noisy(text: str, rng: np.random.Generator) -> str:
    """Typos, lower case and dropped punctuation, as in real hand-typed notes."""
    if rng.random() < 0.3:
        text = text.lower()
    if rng.random() < 0.3:
        text = text.replace(".", "").replace(",", "")
    chars = list(text)
    for _ in range(int(rng.integers(0, 3))):
        if len(chars) > 4:
            i = int(rng.integers(1, len(chars) - 1))
            chars[i], chars[i - 1] = chars[i - 1], chars[i]
    return "".join(chars)


def _notes(sig: str, rng: np.random.Generator, tc: Optional[int]) -> Tuple[str, Dict[str, Any]]:
    parts: List[str] = []
    meta: Dict[str, Any] = {"hint": "none"}
    if rng.random() < 0.8:
        unit = str(rng.choice([f"Unit {rng.integers(1, 7)}", f"GT{rng.integers(1, 7)}", f"CT-{rng.integers(1, 5)}"]))
        parts.append(unit + "," + str(rng.choice([" 7FA.03", " 7F.05", " 9FA", " 6B", " 7HA.02", ""])) + ".")
    r = rng.random()
    if r < 0.45:  # helpful hint
        book = HINTS.get(sig) or HINTS["normal"]
        parts.append(str(rng.choice(book)).format(tc=tc or int(rng.integers(1, 25))))
        meta["hint"] = "helpful"
    elif r < 0.55:  # misleading hint from another signature
        other = str(rng.choice([s for s in SEEN if s != sig]))
        parts.append(str(rng.choice(HINTS[other])).format(tc=int(rng.integers(1, 25))))
        meta["hint"] = f"misleading:{other}"
    parts.append(str(rng.choice(FILLER)))
    text = " ".join(p for p in parts if p).strip()
    if rng.random() < 0.25:
        text = _noisy(text, rng)
    return text, meta


def _frame_params(rng: np.random.Generator, load_variant: bool) -> Dict[str, float]:
    if load_variant:  # a much larger frame: different absolute levels and noise
        return {"rated": rng.uniform(150, 260), "egt0": rng.uniform(590, 640), "cdp0": rng.uniform(15, 20),
                "cdt0": rng.uniform(390, 430), "fuel0": rng.uniform(8, 14), "dyn0": rng.uniform(0.6, 0.9),
                "vib0": rng.uniform(2.0, 4.5), "spatial": rng.uniform(1.0, 4.0), "tc_noise": rng.uniform(1.0, 2.2)}
    return {"rated": rng.uniform(40, 60), "egt0": rng.uniform(525, 560), "cdp0": rng.uniform(10.5, 12),
            "cdt0": rng.uniform(335, 355), "fuel0": rng.uniform(2.5, 3.2), "dyn0": rng.uniform(0.25, 0.45),
            "vib0": rng.uniform(2.5, 3.5), "spatial": rng.uniform(0.5, 2.5), "tc_noise": rng.uniform(0.6, 1.4)}


def _load_profile(rng: np.random.Generator, n: int, freq: float, kind: str) -> np.ndarray:
    hours = np.arange(n) * freq / 60.0
    f = 0.95 + 0.02 * np.sin(2 * np.pi * (hours - rng.uniform(0, 24)) / 24.0) + rng.normal(0, 0.003, n)
    if kind == "steps":
        level = np.empty(n)
        i = 0
        while i < n:
            seg = int(rng.uniform(6, 48) * 60 / freq)
            level[i : i + seg] = rng.uniform(0.65, 1.0)
            i += seg
        ramp = max(1, int(30 / freq))
        level = pd.Series(level).rolling(ramp, min_periods=1).mean().to_numpy()
        f = f - 0.95 + level
    elif kind == "cycling":
        day = (hours // 24).astype(int)
        hod = hours % 24
        off_days = [d for d in range(int(day.max()) + 1) if rng.random() < 0.6]
        off_start, off_len = rng.uniform(21, 24), rng.uniform(5, 9)
        rel = (hod - off_start) % 24
        off = np.isin(day, off_days) & (rel < off_len)
        ramp = max(1, int(45 / freq))
        avail = pd.Series((~off).astype(float)).rolling(ramp, min_periods=1).mean().to_numpy()
        f = f * avail
    return np.clip(f, 0.0, 1.02)


def _base_arrays(rng: np.random.Generator, n: int, freq: float, layout: str, n_tc: int, fp: Dict[str, float], profile: str) -> Dict[str, Any]:
    f = _load_profile(rng, n, freq, profile)
    online = f > 0.2
    k = fp["rated"] / 48.0
    load = np.where(online, fp["rated"] * f + rng.normal(0, 0.15 * k, n), np.clip(fp["rated"] * f, 0, None))
    egt = np.where(online, fp["egt0"] - 25 * (1 - f) + rng.normal(0, 0.4, n), 140 + 30 * f + rng.normal(0, 1.0, n))
    dyn_noise = 0.05 * max(1.0, fp["dyn0"] / 0.35)
    a: Dict[str, Any] = {
        "f": f,
        "online": online,
        "load_MW": load,
        "CDP_bar": fp["cdp0"] * (0.55 + 0.45 * f) + rng.normal(0, 0.03, n),
        "CDT_C": fp["cdt0"] * (0.8 + 0.2 * f) + rng.normal(0, 0.25, n),
        "fuel_flow_kg_s": np.where(online, fp["fuel0"] * f, 0.0) + rng.normal(0, 0.004 * fp["fuel0"], n) * online,
        "IGV_deg": np.where(online, 30 + 32 * np.clip((f - 0.3) / 0.6, 0, 1), 28.0) + rng.normal(0, 0.15, n),
        "vib_DE_mm_s": np.where(online, fp["vib0"], 0.6) + rng.normal(0, 0.08, n),
        "vib_NDE_mm_s": np.where(online, fp["vib0"] * 0.95, 0.55) + rng.normal(0, 0.08, n),
        "lube_oil_C": 56 + 2.5 * f + rng.normal(0, 0.12, n),
        "comb_dyn_psi": np.where(online, fp["dyn0"] + np.abs(rng.normal(0, dyn_noise, n)), 0.05),
        "trip_active": np.zeros(n),
    }
    m = n_tc if layout == "ring" else 4
    angles = 2 * np.pi * np.arange(m) / m
    spatial = fp["spatial"] * np.sin(angles + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 0.6, m)
    noise = fp["tc_noise"] if layout == "ring" else fp["tc_noise"] * 1.2
    a["temps"] = egt[:, None] + np.where(online[:, None], spatial[None, :], 0.0) + rng.normal(0, noise, (n, m))
    a["egt_mean"] = egt
    return a


def _event_window(rng: np.random.Generator, n: int, freq: float, dur_h: Tuple[float, float], ambiguous: bool) -> Tuple[int, int]:
    # Tier 2 exports put most of the record before the alert: onset at 50-85 % of the record
    min_base = int(24 * 60 / freq)
    lo = max(min_base, int(0.5 * n))
    if ambiguous and rng.random() < 0.5:
        s = int(n * rng.uniform(0.9, 0.96))  # event only just starting at the end of the record
    else:
        s = int(rng.uniform(lo, max(lo + 1, 0.85 * n)))
    s = min(max(s, min_base), n - 2)
    e = min(n, s + max(2, int(rng.uniform(*dur_h) * 60 / freq)))
    return s, e


def _ramp_env(n: int, s: int, e: int, ramp: int, persist: bool) -> np.ndarray:
    env = np.zeros(n)
    ramp = max(1, min(ramp, e - s))
    env[s : s + ramp] = np.linspace(0, 1, ramp)
    env[s + ramp : e] = 1.0
    if persist:
        env[e:] = 1.0
    elif e < n:
        r2 = min(ramp, n - e)
        env[e : e + r2] = np.linspace(1, 0, r2)
    return env


def _inj_cold_spot(a, rng, p, layout, freq, variant, ambiguous):
    n, m = a["temps"].shape
    depth = rng.uniform(12, 35)
    if variant == "magnitude":
        depth = rng.uniform(5, 9) if rng.random() < 0.5 else rng.uniform(40, 60)
    elif ambiguous:
        depth = rng.uniform(7, 12)
    s, e = _event_window(rng, n, freq, (24, 24 * 8), ambiguous)
    env = _ramp_env(n, s, e, int(rng.uniform(2, 12) * 60 / freq), persist=rng.random() < 0.7) * a["online"]
    if layout == "ring":
        width = int(rng.choice([2, 3, 4, 5])) if not ambiguous else 2
        center = int(rng.integers(0, m))
        if variant == "shifted":
            center = int(rng.choice([0, m - 1]))  # sector straddles the ring wrap-around (TCn | TC1)
        members = [(center + d) % m for d in range(-(width // 2), width - width // 2)]
        for idx in members:
            dist = min((idx - center) % m, (center - idx) % m)
            w = 1.0 if dist == 0 else rng.uniform(0.45, 0.75) / dist
            a["temps"][:, idx] -= depth * w * env + rng.normal(0, 0.3, n) * (env > 0)
        p.update(center_tc=center + 1, sector_width=width)
    else:
        z = int(rng.integers(0, 4))
        a["temps"][:, z] -= 0.55 * depth * env
        a["temps"][:, (z + 1) % 4] -= 0.15 * depth * env
        p.update(cold_zone=z + 1)
    a["fuel_flow_kg_s"] *= 1 + rng.uniform(0.004, 0.01) * env
    if depth > 28:
        a["vib_DE_mm_s"] += 0.3 * env
    p.update(depth_C=round(float(depth), 1))
    return s, e


def _inj_tc_sensor(a, rng, p, layout, freq, variant, ambiguous):
    n, m = a["temps"].shape
    j = int(rng.integers(0, m))
    if variant == "shifted":
        j = int(rng.choice([0, m - 1]))
    mode = str(rng.choice(["step", "open", "flatline", "erratic", "drift"]))
    if ambiguous:
        mode = "drift"
    s, e = _event_window(rng, n, freq, (12, 24 * 10), ambiguous)
    x = a["temps"][:, j]
    big = variant == "magnitude"
    if mode == "step":
        off = rng.uniform(15, 80) * (1 if rng.random() < 0.25 else -1)
        if big:
            off = float(rng.choice([-1, 1])) * (rng.uniform(8, 12) if rng.random() < 0.5 else rng.uniform(150, 300))
        x[s:] += off
    elif mode == "open":
        x[s:] = (rng.uniform(-20, 5) if rng.random() < 0.6 else rng.uniform(1200, 1370)) + rng.normal(0, 0.5, n - s)
    elif mode == "flatline":
        x[s:] = round(float(x[s]), 1)
    elif mode == "erratic":
        k = max(3, int((n - s) * rng.uniform(0.02, 0.08)))
        idx = rng.integers(s, n, k)
        x[idx] += rng.choice([-1, 1], k) * rng.uniform(30, 150, k) * (2.0 if big else 1.0)
    else:  # drift
        d = rng.uniform(10, 30) if not big else rng.uniform(40, 70)
        if ambiguous:
            d = rng.uniform(8, 14)
        ramp = int(rng.uniform(24, 24 * 5) * 60 / freq)
        x[:] -= d * _ramp_env(n, s, n, ramp, persist=True)
    a["temps"][:, j] = x
    p.update(sensor_index=j + 1, mode=mode)
    return s, n


def _inj_instability(a, rng, p, layout, freq, variant, ambiguous):
    n = len(a["f"])
    sev = rng.uniform(1.0, 3.5)
    if variant == "magnitude":
        sev = rng.uniform(0.6, 0.9) if rng.random() < 0.5 else rng.uniform(4.5, 7.0)
    elif ambiguous:
        sev = rng.uniform(0.8, 1.1)
    period = rng.uniform(2, 8) if variant != "shifted" else rng.uniform(12, 30)
    s, e = _event_window(rng, n, freq, (3, 18), ambiguous)
    t = np.arange(e - s) * freq
    osc = sev * (0.55 + 0.45 * np.sin(2 * np.pi * t / period)) + 0.15 * sev * np.sin(2 * np.pi * t / (period * 0.37))
    osc = np.clip(osc + rng.normal(0, 0.08 * sev, e - s), 0.1, 12.0) * a["online"][s:e]
    scale = max(1.0, float(a["comb_dyn_psi"][:s].mean()) / 0.35) if s else 1.0
    a["comb_dyn_psi"][s:e] += osc * scale
    flutter = osc / sev * rng.uniform(3.0, 8.0)
    m = a["temps"].shape[1]
    hot = int(rng.integers(0, m))
    a["temps"][s:e, hot] += flutter * 0.6
    a["temps"][s:e, (hot + m // 2) % m] -= flutter * 0.5
    a["vib_DE_mm_s"][s:e] += 0.25 * osc
    a["fuel_flow_kg_s"][s:e] *= 1 + 0.003 * osc / sev
    p.update(severity_psi=round(float(sev), 2), period_min=round(float(period), 1))
    return s, e


def _inj_probe(a, rng, p, layout, freq, variant, ambiguous):
    n = len(a["f"])
    mode = str(rng.choice(["flatline", "saturation", "spikes", "noisy"]))
    s, e = _event_window(rng, n, freq, (6, 24 * 4), ambiguous)
    d = a["comb_dyn_psi"]
    if mode == "flatline":
        d[s:e] = float(rng.choice([0.0, round(float(d[s]), 2)]))
    elif mode == "saturation":
        on = rng.random(e - s) < rng.uniform(0.3, 1.0)
        d[s:e] = np.where(on, float(rng.choice([9.99, 15.0, 20.0])), d[s:e])
    elif mode == "spikes":
        k = max(4, int((e - s) * rng.uniform(0.01, 0.05)))
        d[rng.integers(s, e, k)] = rng.uniform(2, 9, k)
    else:
        d[s:e] = d[s:e] + rng.normal(0, rng.uniform(0.5, 1.5), e - s)
    p.update(mode=mode)
    return s, e


def _go_offline(a, rng, s: int, e: int) -> None:
    """Unit offline (trip / coast-down): no load, no fuel, exhaust cools."""
    n = len(a["f"])
    e = min(e, n)
    a["online"][s:e] = False
    a["f"][s:e] = 0.0
    a["load_MW"][s:e] = np.abs(rng.normal(0, 0.1, e - s))
    a["fuel_flow_kg_s"][s:e] = 0.0
    a["IGV_deg"][s:e] = 28.0 + rng.normal(0, 0.1, e - s)
    a["comb_dyn_psi"][s:e] = 0.05
    cool = 150 + (a["egt_mean"][s] - 150) * np.exp(-np.arange(e - s) / max(1.0, 0.1 * (e - s)))
    a["temps"][s:e] = cool[:, None] + rng.normal(0, 1.0, (e - s, a["temps"].shape[1]))
    a["CDP_bar"][s:e] = 1.0 + rng.normal(0, 0.01, e - s)


def _inj_hets(a, rng, p, layout, freq, variant, ambiguous):
    n, m = a["temps"].shape
    peak = rng.uniform(45, 80)
    if variant == "magnitude":
        peak = rng.uniform(25, 38) if rng.random() < 0.5 else rng.uniform(90, 130)
    build = max(2, int(rng.uniform(1, 6) * 60 / freq))
    min_base = int(24 * 60 / freq) + build
    if ambiguous:
        t = int(n * rng.uniform(0.9, 0.97))
    else:
        t = int(rng.uniform(max(min_base, 0.5 * n), 0.85 * n))
    t = min(max(t, min_base), n - 2)
    hot = int(rng.integers(0, m))
    cold = (hot + m // 2) % m
    ramp = np.linspace(0, 1, build)
    a["temps"][t - build : t, hot] += 0.55 * peak * ramp
    a["temps"][t - build : t, cold] -= 0.45 * peak * ramp
    if layout == "ring":
        for d in (-1, 1):
            a["temps"][t - build : t, (hot + d) % m] += 0.3 * peak * ramp
            a["temps"][t - build : t, (cold + d) % m] -= 0.25 * peak * ramp
    trip_len = max(2, int(rng.uniform(20, 60) / freq))
    a["trip_active"][t : t + trip_len] = 1
    off_end = min(n, t + int(rng.uniform(4, 72) * 60 / freq))
    pre_load = float(a["load_MW"][:t].mean())
    _go_offline(a, rng, t, off_end)
    k = min(n - t, max(1, int(30 / freq)))
    a["vib_DE_mm_s"][t : t + k] = 6.5 + rng.normal(0, 0.3, k)
    if off_end < n:  # restart: ramp back to load
        r = min(n - off_end, int(2 * 60 / freq))
        a["load_MW"][off_end : off_end + r] = np.linspace(0, pre_load, r)
    p.update(peak_spread_C=round(float(peak), 1), build_h=round(build * freq / 60, 1), restarted=bool(off_end < n))
    return t - build, t + trip_len


def _inj_igv(a, rng, p, layout, freq, variant, ambiguous):
    n = len(a["f"])
    delta = rng.uniform(2.5, 8) * float(rng.choice([-1, 1]))
    if variant == "magnitude":
        delta = np.sign(delta) * (rng.uniform(1.2, 2.2) if rng.random() < 0.5 else rng.uniform(10, 15))
    s, _ = _event_window(rng, n, freq, (24, 48), ambiguous)
    step = rng.random() < 0.3
    ramp = 1 if step else int(rng.uniform(24, 24 * 5) * 60 / freq)
    a["IGV_deg"] += delta * _ramp_env(n, s, n, ramp, persist=True) * a["online"]
    p.update(offset_deg=round(float(delta), 2), step=bool(step))
    return s, n


def _inj_fuel_osc(a, rng, p, layout, freq, variant, ambiguous):
    n = len(a["f"])
    period = rng.uniform(2, 10) if freq <= 1 else rng.uniform(40, 120)
    s, e = _event_window(rng, n, freq, (4, 72), ambiguous)
    t = np.arange(e - s) * freq
    grow = np.clip(t / max(1.0, t[-1] * 0.3), 0, 1)
    wave = np.sin(2 * np.pi * t / period) * grow * a["online"][s:e]
    af, al, ae = rng.uniform(0.02, 0.06), rng.uniform(0.008, 0.025), rng.uniform(3, 8)
    a["fuel_flow_kg_s"][s:e] *= 1 + af * wave
    a["load_MW"][s:e] += al * a["load_MW"][s:e] * np.roll(wave, 1)
    a["temps"][s:e] += (ae * wave)[:, None]
    a["comb_dyn_psi"][s:e] += 0.05 * np.abs(wave)
    p.update(period_min=round(float(period), 1), fuel_amp_pct=round(100 * float(af), 1))
    return s, e


def _inj_harness(a, rng, p, layout, freq, variant, ambiguous):
    """Damaged harness / connector: a group of adjacent TCs drops out together, intermittently, in steps."""
    n, m = a["temps"].shape
    width = int(rng.integers(3, 7)) if m > 6 else 2
    first = int(rng.integers(0, m))
    members = [(first + k) % m for k in range(width)]
    s, _ = _event_window(rng, n, freq, (24, 48), ambiguous)
    on = np.zeros(n, dtype=bool)
    i = s
    while i < n:
        dur = max(1, int(rng.uniform(0.5, 6) * 60 / freq))
        on[i : i + dur] = True
        i += dur + max(1, int(rng.uniform(0.5, 12) * 60 / freq))
    on &= a["online"]
    drop = rng.uniform(60, 300)
    for idx in members:
        a["temps"][on, idx] -= drop * rng.uniform(0.85, 1.0) + rng.normal(0, 3, int(on.sum()))
    p.update(first_tc=first + 1, width=width, drop_C=round(float(drop), 1))
    return s, n


def _inj_tc_step_lookalike(a, rng, p, layout, freq, variant, ambiguous):
    """One exhaust TC steps down 12-40 C (cold-spot sized) within minutes and stays there; neighbours unchanged."""
    n, m = a["temps"].shape
    j = int(rng.integers(0, m))
    s, _ = _event_window(rng, n, freq, (24, 48), False)
    off = rng.uniform(12, 40)
    ramp = 1 if rng.random() < 0.6 else max(1, int(rng.uniform(10, 40) / freq))
    a["temps"][:, j] -= off * _ramp_env(n, s, n, ramp, persist=True)
    p.update(sensor_index=j + 1, mode="step_lookalike", step_C=round(float(-off), 1))
    return s, n


def _inj_bearing(a, rng, p, layout, freq, variant, ambiguous):
    n = len(a["f"])
    s, _ = _event_window(rng, n, freq, (24, 48), ambiguous)
    rise = rng.uniform(3, 9)
    ramp = int(rng.uniform(24, 24 * 5) * 60 / freq)
    env = _ramp_env(n, s, n, ramp, persist=True) ** 2 * a["online"]
    a["vib_DE_mm_s"] += rise * env + rng.normal(0, 0.15, n) * env
    a["vib_NDE_mm_s"] += 0.3 * rise * env
    a["lube_oil_C"] += rng.uniform(2, 6) * env
    p.update(vib_rise_mm_s=round(float(rise), 2))
    return s, n


INJECTORS = {
    "normal": None,
    "cold_spot_hardware": _inj_cold_spot,
    "tc_sensor_fault": _inj_tc_sensor,
    "combustion_instability": _inj_instability,
    "dynamics_probe_fault": _inj_probe,
    "hets_trip_hardware": _inj_hets,
    "igv_lvdt_drift": _inj_igv,
    "fuel_control_oscillation": _inj_fuel_osc,
    "bearing_vibration": _inj_bearing,
    "tc_harness_fault": _inj_harness,
}

FREQ_CHOICES = {
    "combustion_instability": ([1, 5, 10], [0.7, 0.2, 0.1]),
    "dynamics_probe_fault": ([1, 5, 10], [0.6, 0.25, 0.15]),
    "fuel_control_oscillation": ([1, 5, 10], [0.6, 0.2, 0.2]),
}
ZONE_SHARE = {"hets_trip_hardware": 0.65, "combustion_instability": 0.65, "dynamics_probe_fault": 0.65,
              "fuel_control_oscillation": 0.65, "igv_lvdt_drift": 0.65, "bearing_vibration": 0.65,
              "normal": 0.4, "tc_sensor_fault": 0.25, "cold_spot_hardware": 0.2, "tc_harness_fault": 0.0}


def make_case(spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build one labeled case from a spec {split, signature, variant, index, ambiguous}."""
    sig, split, variant, i = spec["signature"], spec["split"], spec.get("variant"), spec["index"]
    rng = np.random.default_rng(_seed(split, sig, variant, i))
    ambiguous = bool(spec.get("ambiguous"))
    freqs, probs = FREQ_CHOICES.get(sig, ([10, 5, 1], [0.55, 0.25, 0.2]))
    freq = float(rng.choice(freqs, p=probs))
    days = {1.0: (2, 5), 5.0: (5, 14), 10.0: (10, 30)}[freq]
    n = int(rng.uniform(*days) * 24 * 60 / freq)
    layout = "zones" if rng.random() < ZONE_SHARE.get(sig, 0.3) else "ring"
    if variant == "step_lookalike":
        layout = "ring"
    n_tc = int(rng.choice([24, 27, 30]))
    if variant == "shifted":
        n_tc = int(rng.choice([16, 36]))
        if sig in ("hets_trip_hardware", "cold_spot_hardware", "tc_sensor_fault"):
            layout = "ring"
    profile = str(rng.choice(["baseload", "steps", "cycling"], p=[0.55, 0.3, 0.15]))
    if sig == "hets_trip_hardware":
        profile = str(rng.choice(["baseload", "steps"], p=[0.7, 0.3]))
    fp = _frame_params(rng, variant == "load")
    a = _base_arrays(rng, n, freq, layout, n_tc, fp, profile)
    params: Dict[str, Any] = {}
    ev: Optional[Tuple[int, int]] = None
    # A signature with no injector yet is written as a healthy case. The alert-type
    # path below has its own injectors; this eight-signature path stays put.
    inj = _inj_tc_step_lookalike if variant == "step_lookalike" else INJECTORS.get(sig)
    if inj is not None:
        ev = inj(a, rng, params, layout, freq, variant, ambiguous)

    temps = a["temps"]
    egt_avg = temps.mean(axis=1)
    spread = temps.max(axis=1) - temps.min(axis=1)
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)), hours=int(rng.integers(0, 24)))
    idx = pd.date_range(start=start, periods=n, freq=f"{int(freq)}min")
    alt = rng.random() < 0.3
    cols: Dict[str, Any] = {"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S")}

    def put(name: str, x: np.ndarray, nd: int) -> None:
        cols[ALT_TAGS.get(name, name) if alt else name] = np.round(x, nd)

    put("load_MW", np.clip(a["load_MW"], 0, None), 2)
    put("CDP_bar", a["CDP_bar"], 3)
    put("CDT_C", a["CDT_C"], 2)
    put("EGT_avg_C", egt_avg, 2)
    put("EGT_spread_C", spread, 2)
    if layout == "ring":
        for j in range(temps.shape[1]):
            cols[(f"TTXD_{j + 1}" if alt else f"TC{j + 1}")] = np.round(temps[:, j], 1)
    else:
        for j in range(4):
            cols[(f"EGT_ZONE{j + 1}" if alt else f"EGT_Z{j + 1}_C")] = np.round(temps[:, j], 2)
    put("fuel_flow_kg_s", np.clip(a["fuel_flow_kg_s"], 0, None), 4)
    put("IGV_deg", a["IGV_deg"], 2)
    put("vib_DE_mm_s", np.clip(a["vib_DE_mm_s"], 0.1, None), 3)
    put("vib_NDE_mm_s", np.clip(a["vib_NDE_mm_s"], 0.1, None), 3)
    put("lube_oil_C", a["lube_oil_C"], 2)
    put("comb_dyn_psi", a["comb_dyn_psi"], 3)
    cols[ALT_TAGS["trip_active"] if alt else "trip_active"] = a["trip_active"].astype(int)
    if rng.random() < 0.3:
        hours = np.arange(n) * freq / 60
        cols["ambient_C"] = np.round(22 + 8 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 0.3, n), 2)
    df = pd.DataFrame(cols)
    data_cols = [c for c in df.columns if c != "timestamp" and "trip" not in c.lower()]
    if rng.random() < 0.08:  # historian gaps
        for c in rng.choice(data_cols, size=int(rng.integers(1, 4)), replace=False):
            g0 = int(rng.integers(0, n - 10))
            df.loc[g0 : g0 + int(rng.uniform(1, 6) * 60 / freq), c] = np.nan
    if sig == "normal" and rng.random() < 0.15:  # single-sample historian glitches
        for c in rng.choice(data_cols, size=2, replace=False):
            j = int(rng.integers(0, n))
            if pd.notna(df.loc[j, c]):
                df.loc[j, c] = float(df.loc[j, c]) * rng.uniform(1.5, 3.0)

    tc_hint = params.get("sensor_index") or params.get("center_tc")
    notes, note_meta = _notes(sig, rng, tc_hint if layout == "ring" else None)
    name_choices = ANALYTIC_NAMES.get(sig) or GENERIC_ANALYTICS
    analytic = str(rng.choice(GENERIC_ANALYTICS)) if rng.random() < 0.1 else str(rng.choice(name_choices))
    if variant == "step_lookalike":  # raised as a cold spot; half the notes point at combustion hardware
        analytic = str(rng.choice(ANALYTIC_NAMES["cold_spot_hardware"]))
        if rng.random() < 0.5:
            notes = (notes + " " + str(rng.choice(HINTS["cold_spot_hardware"]))).strip()
            note_meta = {"hint": "misleading:cold_spot_hardware"}
    info = BASE_SIGNATURES.get(sig) or UNSEEN_SIGNATURES[sig]
    meta = {
        "case_id": f"{split}-{sig}-{variant or 'base'}-{i:03d}",
        "split": split,
        "signature": sig,
        "action": info["action"],
        "cause": info["cause"],
        "escalate": info["escalate"],
        "variant": variant,
        "ambiguous": ambiguous,
        "analytic": analytic,
        "notes": notes,
        "notes_hint": note_meta["hint"],
        "layout": layout,
        "n_sensors": int(temps.shape[1]),
        "freq_min": freq,
        "rows": n,
        "profile": profile,
        "frame_rated_MW": round(float(fp["rated"]), 1),
        "alt_tags": bool(alt),
        # Frame model the trainer conditions on. Historian tags are 7FA (ring) and 9HA (zones);
        # the friendly-tag twins are 7EA and 7HA. Older manifests omit this and derive it from layout + alt_tags.
        "frame": ("9HA" if alt else "7HA") if layout == "zones" else ("7FA" if alt else "7EA"),
        "event_start": idx[ev[0]].strftime("%Y-%m-%d %H:%M") if ev else None,
        "event_end": idx[min(ev[1], n - 1)].strftime("%Y-%m-%d %H:%M") if ev else None,
        "params": params,
    }
    return df, meta


def dataset_specs(scale: float = 1.0) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for split, count in SPLIT_COUNTS.items():
        k = max(1, int(round(count * scale)))
        for sig in SEEN:
            for i in range(k):
                specs.append({"split": split, "signature": sig, "variant": None, "index": i, "ambiguous": i % 10 == 9})
    kv = max(1, int(round(VARIANT_COUNT * scale)))
    for sig in VARIANT_CLASSES:
        for kind in VARIANT_KINDS:
            for i in range(kv):
                specs.append({"split": "variant", "signature": sig, "variant": kind, "index": i})
    for i in range(kv):
        specs.append({"split": "variant", "signature": "normal", "variant": "load", "index": i})
    for sig in UNSEEN:
        for i in range(max(1, int(round(UNSEEN_COUNT * scale)))):
            specs.append({"split": "unseen", "signature": sig, "variant": None, "index": i})
    specs += sensor_fault_specs(scale)
    return specs


def sensor_fault_specs(scale: float = 1.0) -> List[Dict[str, Any]]:
    return [{"split": "sensorfault", "signature": "tc_sensor_fault", "variant": "step_lookalike", "index": i}
            for i in range(max(1, int(round(SENSOR_FAULT_COUNT * scale))))]


def _write_one(job: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    spec, out = job
    df, meta = make_case(spec)
    path = Path(out) / meta["split"] / f"{meta['case_id']}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    meta["file"] = f"{meta['split']}/{path.name}"
    return meta


def generate_dataset(out: Path, scale: float = 1.0, workers: int = 0, only_split: Optional[str] = None,
                      profiles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Write the labeled dataset (gzip CSVs) and manifest.jsonl; returns the manifest rows.

    ``only_split``: (re)write just that split and merge it into an existing manifest.jsonl.
    ``profiles``: alert-type ids, or ``ALL_SYNTH_PROFILES``. Omit it to keep today's
    eight-signature dataset (no profile subdirectory, same counts).
    """
    if profiles is not None:
        return _generate_profile_dataset(out, scale, workers, only_split, profiles)
    import multiprocessing as mp

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs(scale)
    keep: List[Dict[str, Any]] = []
    if only_split:
        specs = [s for s in specs if s["split"] == only_split]
        mf = out / "manifest.jsonl"
        if mf.is_file():
            keep = [r for r in (json.loads(x) for x in mf.read_text(encoding="utf-8").splitlines() if x.strip()) if r["split"] != only_split]
    jobs = [(s, str(out)) for s in specs]
    workers = workers or max(1, (mp.cpu_count() or 2) - 2)
    if workers > 1 and len(jobs) > 8:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_write_one, jobs, chunksize=4)
    else:
        rows = [_write_one(j) for j in jobs]
    rows = keep + rows
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"Dataset: {len(rows)} cases in {out}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return rows


# --- alert-type profiles (kept out of the original eight-signature dataset)
# The original SEEN list stays eight names so dataset_specs(1.0) is still
# 1200/200/320/128/80/40. These profiles are a second manifest: pass
# profiles=ALL_SYNTH_PROFILES. fuel_gas_heater is the coldstart split only.

# One slot other than the context slot (load) is left out of this fraction of files.
# That is the mask the scorer has to handle. There is no optional-slot list.
MISSING_TAG_P = 0.05
# An alert-type case carries its real name. The trainer reads the alert type from the folder.
GENERIC_NAME_P = 0
EXTRA_TAG_MAX = 4
# scale 0.1 -> 6 cases per signature. The original split counts are not reused.
PROFILE_SPLIT_COUNTS = {"train": 60, "val": 10, "test": 20}
PROFILE_VARIANT_COUNT = 4
PROFILE_UNSEEN_COUNT = 10
COLDSTART_PER_SIGNATURE = 20  # two signatures -> 40 cases at scale 1

ALL_SYNTH_PROFILES = (
    "exhaust_cold_spot",
    "combustion_dynamics",
    "exhaust_spread_trip",
    "igv_fuel_control",
    "bearing_vibration",
    "inlet_filter_dp",
    "lube_oil_system",
    "fuel_gas_heater",
)
# Shipped bundles are the seven built-ins. The heater split is never one of them.
SHIPPED_SYNTH_PROFILES = tuple(p for p in ALL_SYNTH_PROFILES if p != "fuel_gas_heater")

PROFILE_SIGNATURES = {
    "exhaust_cold_spot": ["normal", "cold_spot_hardware", "tc_sensor_fault"],
    "combustion_dynamics": ["normal", "combustion_instability", "dynamics_probe_fault"],
    "exhaust_spread_trip": ["normal", "hets_trip_hardware", "tc_sensor_fault"],
    "igv_fuel_control": ["normal", "igv_lvdt_drift", "fuel_control_oscillation"],
    "bearing_vibration": ["normal", "bearing_wear_trend", "vibration_probe_fault"],
    "inlet_filter_dp": ["normal", "inlet_filter_fouling", "dp_transmitter_fault"],
    "lube_oil_system": ["normal", "lube_cooler_fouling", "drain_tc_fault"],
    "fuel_gas_heater": ["normal", "heater_fault"],
}
PROFILE_UNSEEN = {"exhaust_cold_spot": ["tc_harness_fault"]}
PROFILE_NAMES = {
    "exhaust_cold_spot": ["Exhaust Cold Spot", "Exhaust temperature cold spot", "TTXD cold spot"],
    "combustion_dynamics": ["Combustion Dynamics High", "comb dyn high", "CDM pulsation alarm"],
    "exhaust_spread_trip": ["HETS Trip", "High exhaust spread trip", "Unit trip - exhaust spread"],
    "igv_fuel_control": ["IGV position deviation", "Fuel control oscillation", "Load hunting"],
    "bearing_vibration": ["Bearing vibration high", "Vibration advisory", "High vibration alarm"],
    "inlet_filter_dp": ["Inlet filter high dp", "Inlet dp high", "Filter house dp advisory"],
    "lube_oil_system": ["Lube oil temperature high", "Lube cooler advisory", "Drain temperature alarm"],
    "fuel_gas_heater": ["Fuel gas heater outlet low", "Heater performance advisory"],
}
# The built-in tag list, in file order. A case writes every one of these, minus at most
# one non-context slot. igv_ref is not here: it is an extra column, not a slot.
PROFILE_SLOTS = {
    "exhaust_cold_spot": ["load", "exh_tc", "spread", "egt_avg", "fuel", "igv"],
    "combustion_dynamics": ["load", "dyn", "vib", "fuel", "exh_tc"],
    "exhaust_spread_trip": ["load", "exh_tc", "spread", "trip", "egt_avg", "fuel", "igv"],
    "igv_fuel_control": ["load", "igv", "fuel", "egt_avg", "cdp", "exh_tc"],
    "bearing_vibration": ["load", "speed", "brg_vib", "lube_supply"],
    "inlet_filter_dp": ["load", "inlet_dp", "ambient_t", "cdp"],
    "lube_oil_system": ["load", "ambient_t", "oil_hdr_p", "oil_supply_t", "drain_t"],
    "fuel_gas_heater": ["load", "heater_out", "fuel_gas", "fuel"],
}
_EXTRA_POOL = (
    "seal_air_bar", "wheelspace_C", "cooling_flow_kg_s", "exhaust_duct_dp_mbar",
    "comp_bleed_pct", "flame_int_pct", "nacelle_temp_C", "pad_temp_C",
)
_RING_SIZES = (16, 18, 24, 27, 31, 36)


def _scaled_count(count: int, scale: float) -> int:
    """At least one case. scale 0.1 of 60 is 6."""
    return max(1, int(round(count * scale)))


def profile_specs(scale: float = 1.0, profiles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """One spec per synthetic case. The heater profile is the coldstart split only."""
    wanted = list(profiles) if profiles is not None else list(ALL_SYNTH_PROFILES)
    unknown = [p for p in wanted if p not in PROFILE_SIGNATURES]
    if unknown:
        raise ValueError(f"unknown synthetic profile: {unknown[0]}")
    specs: List[Dict[str, Any]] = []
    for pid in wanted:
        if pid == "fuel_gas_heater":
            for sig in PROFILE_SIGNATURES[pid]:
                for i in range(_scaled_count(COLDSTART_PER_SIGNATURE, scale)):
                    specs.append({"split": "coldstart", "profile": pid, "signature": sig, "variant": None, "index": i})
            continue
        for split, count in PROFILE_SPLIT_COUNTS.items():
            for sig in PROFILE_SIGNATURES[pid]:
                for i in range(_scaled_count(count, scale)):
                    specs.append({
                        "split": split, "profile": pid, "signature": sig, "variant": None, "index": i,
                        "ambiguous": i % 10 == 9,
                    })
        kv = _scaled_count(PROFILE_VARIANT_COUNT, scale)
        for sig in PROFILE_SIGNATURES[pid]:
            if sig == "normal":
                continue
            for i in range(kv):
                specs.append({"split": "variant", "profile": pid, "signature": sig, "variant": "magnitude", "index": i})
        for sig in PROFILE_UNSEEN.get(pid, []):
            for i in range(_scaled_count(PROFILE_UNSEEN_COUNT, scale)):
                specs.append({"split": "unseen", "profile": pid, "signature": sig, "variant": None, "index": i})
    return specs


def _dialect_name(dialect: str, friendly: str, historian: str, site: str) -> str:
    return {"friendly": friendly, "historian": historian, "site": site}[dialect]


def _ring_names(dialect: str, n: int) -> List[str]:
    if dialect == "historian":
        return [f"TTXD_{j}" for j in range(1, n + 1)]
    if dialect == "site":
        return [f"exh_tc_{j}" for j in range(1, n + 1)]
    return [f"TC{j}" for j in range(1, n + 1)]


def _zone_names(dialect: str) -> List[str]:
    if dialect == "historian":
        return [f"EGT_ZONE{j}" for j in range(1, 5)]
    if dialect == "site":
        return [f"egt_z{j}" for j in range(1, 5)]
    return [f"EGT_Z{j}_C" for j in range(1, 5)]


def _profile_signals(a: Dict[str, Any], rng: np.random.Generator, freq: float, n_brg: int, n_drain: int) -> None:
    """Columns the new profiles read. Built on top of ``_base_arrays``."""
    n = len(a["f"])
    f = a["f"]
    online = a["online"]
    hours = np.arange(n) * freq / 60.0
    a["ambient"] = 22 + 8 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 0.3, n)
    a["speed_rpm"] = np.where(online, 3000 + 8 * (f - 0.9) + rng.normal(0, 1.5, n), rng.normal(0, 0.2, n))
    a["inlet_dp"] = np.where(online, 2.4 + 1.4 * f + rng.normal(0, 0.05, n), 0.3)
    a["oil_hdr"] = np.where(online, 2.3 - 0.2 * f + rng.normal(0, 0.02, n), 0.4)
    a["igv_ref"] = a["IGV_deg"] + rng.normal(0, 0.08, n)
    a["heater_out"] = np.where(online, 70 + 45 * f + rng.normal(0, 0.4, n), a["ambient"])
    a["fuel_gas"] = a["heater_out"] - rng.uniform(12, 20) + rng.normal(0, 0.3, n)
    base_vib = np.where(online, 2.8 + 0.4 * f, 0.5)
    a["brg"] = base_vib[:, None] + rng.normal(0, 0.08, (n, n_brg))
    a["drains"] = (a["lube_oil_C"] + 6)[:, None] + rng.normal(0, 0.3, (n, n_drain))


def _inj_inlet_fouling(a, rng, p, freq):
    n = len(a["f"])
    s, e = _event_window(rng, n, freq, (24, 72), False)
    rise = rng.uniform(3.0, 7.0)
    env = _ramp_env(n, s, n, int(rng.uniform(12, 48) * 60 / freq), persist=True)
    a["inlet_dp"] = a["inlet_dp"] + rise * env
    p.update(dp_rise_mbar=round(float(rise), 2))
    return s, e


def _inj_dp_transmitter(a, rng, p, freq):
    n = len(a["f"])
    s, _ = _event_window(rng, n, freq, (6, 24), False)
    if rng.random() < 0.5:
        a["inlet_dp"][s:] = a["inlet_dp"][s] + rng.normal(0, 0.01, n - s)
        p["mode"] = "stuck"
    else:
        a["inlet_dp"][s:] += rng.choice([-1.0, 1.0]) * rng.uniform(4, 8)
        p["mode"] = "step"
    return s, n


def _inj_lube_fouling(a, rng, p, freq):
    n = len(a["f"])
    s, e = _event_window(rng, n, freq, (24, 72), False)
    rise = rng.uniform(6, 14)
    env = _ramp_env(n, s, n, int(rng.uniform(12, 36) * 60 / freq), persist=True)
    a["lube_oil_C"] = a["lube_oil_C"] + rise * env
    a["drains"] = a["drains"] + (0.6 * rise * env)[:, None]
    p.update(supply_rise_C=round(float(rise), 1))
    return s, e


def _inj_drain_tc(a, rng, p, freq):
    n, m = a["drains"].shape
    j = int(rng.integers(0, m))
    s, _ = _event_window(rng, n, freq, (2, 12), False)
    step = rng.uniform(25, 60)
    a["drains"][s:, j] += step
    p.update(drain_index=j + 1, step_C=round(float(step), 1))
    return s, n


def _inj_bearing_wear(a, rng, p, freq):
    n = len(a["f"])
    s, _ = _event_window(rng, n, freq, (24, 72), False)
    rise = rng.uniform(2.5, 6.0)
    env = _ramp_env(n, s, n, int(rng.uniform(24, 96) * 60 / freq), persist=True)
    a["brg"][:, 0] += rise * env
    if a["brg"].shape[1] > 1:
        a["brg"][:, 1] += 0.35 * rise * env
    a["lube_oil_C"] = a["lube_oil_C"] + rng.uniform(2, 5) * env
    p.update(vib_rise_mm_s=round(float(rise), 2))
    return s, n


def _inj_vib_probe(a, rng, p, freq):
    n, m = a["brg"].shape
    j = int(rng.integers(0, m))
    s, _ = _event_window(rng, n, freq, (2, 12), False)
    if rng.random() < 0.5:
        a["brg"][s:, j] = 0.05
        p["mode"] = "flat"
    else:
        a["brg"][s:, j] += rng.uniform(6, 12)
        p["mode"] = "step"
    p["probe"] = j + 1
    return s, n


def _inj_heater(a, rng, p, freq):
    n = len(a["f"])
    s, _ = _event_window(rng, n, freq, (6, 36), False)
    # Outlet falls toward the fuel-gas temperature. Flow and load stay put.
    gap = a["heater_out"] - a["fuel_gas"]
    env = _ramp_env(n, s, n, max(1, int(rng.uniform(1, 6) * 60 / freq)), persist=True)
    a["heater_out"] = a["heater_out"] - gap * rng.uniform(0.7, 1.0) * env
    p.update(outlet_drop_C=round(float(np.nanmax(gap) * 0.85), 1))
    return s, n


_PROFILE_INJECTORS = {
    "inlet_filter_fouling": _inj_inlet_fouling,
    "dp_transmitter_fault": _inj_dp_transmitter,
    "lube_cooler_fouling": _inj_lube_fouling,
    "drain_tc_fault": _inj_drain_tc,
    "bearing_wear_trend": _inj_bearing_wear,
    "vibration_probe_fault": _inj_vib_probe,
    "heater_fault": _inj_heater,
}


def _apply_profile_fault(a, rng, params, sig, layout, freq, variant, ambiguous):
    """Existing combustion injectors, then the alert types that have their own."""
    if sig in _PROFILE_INJECTORS:
        return _PROFILE_INJECTORS[sig](a, rng, params, freq)
    inj = INJECTORS.get(sig)
    if inj is None:
        return None
    return inj(a, rng, params, layout, freq, variant, ambiguous)


def make_profile_case(spec: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """One labeled case for an alert type. Tag dialect and slot count vary; the seed does not."""
    pid, sig = spec["profile"], spec["signature"]
    split, variant, i = spec["split"], spec.get("variant"), spec["index"]
    rng = np.random.default_rng(_seed("profile", pid, split, sig, variant, i))
    freq = float(rng.choice([10, 5, 1], p=[0.6, 0.25, 0.15]))
    days = {1.0: (2, 4), 5.0: (5, 10), 10.0: (8, 20)}[freq]
    n = int(rng.uniform(*days) * 24 * 60 / freq)
    dialect = str(rng.choice(["friendly", "historian", "site"]))
    n_tc = int(rng.choice(_RING_SIZES))
    n_probe = int(rng.integers(1, 4))
    n_brg = int(rng.integers(2, 9))
    n_drain = int(rng.integers(3, 7))
    # Cold spot and combustion dynamics write the exhaust slot as a ring, or as four
    # zones on a quarter of cases. The slot id stays exh_tc either way.
    use_zones = pid in ("exhaust_cold_spot", "combustion_dynamics") and rng.random() < 0.25
    layout = "zones" if use_zones else "ring"
    op = str(rng.choice(["baseload", "steps", "cycling"], p=[0.6, 0.25, 0.15]))
    fp = _frame_params(rng, variant == "load")
    a = _base_arrays(rng, n, freq, layout, n_tc if layout == "ring" else 4, fp, op)
    _profile_signals(a, rng, freq, n_brg, n_drain)
    params: Dict[str, Any] = {}
    ev = _apply_profile_fault(a, rng, params, sig, layout, freq, variant, bool(spec.get("ambiguous")))

    # Full tag list. The context slot stays. At most one other slot is left out.
    slots = list(PROFILE_SLOTS[pid])
    missing: List[str] = []
    droppable = [s for s in slots if s != "load"]
    if droppable and rng.random() < MISSING_TAG_P:
        drop = str(rng.choice(droppable))
        slots = [s for s in slots if s != drop]
        missing.append(drop)

    temps = a["temps"]
    egt_avg = temps.mean(axis=1)
    spread = temps.max(axis=1) - temps.min(axis=1)
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
    idx = pd.date_range(start=start, periods=n, freq=f"{int(freq)}min")
    cols: Dict[str, Any] = {"timestamp": idx.strftime("%Y-%m-%d %H:%M:%S")}

    def put(name: str, values: np.ndarray, nd: int) -> None:
        cols[name] = np.round(np.asarray(values, dtype=float), nd)

    if "load" in slots:
        put(_dialect_name(dialect, "load_MW", "DWATT", "power_MW"), np.clip(a["load_MW"], 0, None), 2)
    if "fuel" in slots:
        put(_dialect_name(dialect, "fuel_flow_kg_s", "FQG", "fuel_kg_s"), np.clip(a["fuel_flow_kg_s"], 0, None), 4)
    if "cdp" in slots:
        put(_dialect_name(dialect, "CDP_bar", "CPD", "CDP"), a["CDP_bar"], 3)
    if "exh_tc" in slots and use_zones:
        for j, name in enumerate(_zone_names(dialect)):
            cols[name] = np.round(temps[:, j], 2)
    elif "exh_tc" in slots:
        for j, name in enumerate(_ring_names(dialect, temps.shape[1])):
            cols[name] = np.round(temps[:, j], 1)
    if "spread" in slots:
        put(_dialect_name(dialect, "EGT_spread_C", "TTXSP1", "exh_spread_C"), spread, 2)
    if "egt_avg" in slots:
        put(_dialect_name(dialect, "EGT_avg_C", "TTXM", "exh_avg_C"), egt_avg, 2)
    if "trip" in slots:
        cols[_dialect_name(dialect, "trip_active", "TRIP_FLAG", "trip_flag")] = a["trip_active"].astype(int)
    if "dyn" in slots:
        for j in range(n_probe):
            name = _dialect_name(dialect, f"comb_dyn_{j + 1}_psi", f"CDM_AMP_{j + 1}", f"dyn_probe_{j + 1}")
            wave = a["comb_dyn_psi"] + rng.normal(0, 0.02, n)
            cols[name] = np.round(wave, 3)
    if "vib" in slots:
        put(_dialect_name(dialect, "vib_DE_mm_s", "BB1", "vib_de"), np.clip(a["vib_DE_mm_s"], 0.05, None), 3)
        put(_dialect_name(dialect, "vib_NDE_mm_s", "BB2", "vib_nde"), np.clip(a["vib_NDE_mm_s"], 0.05, None), 3)
    if "igv" in slots:
        put(_dialect_name(dialect, "IGV_deg", "CSGV", "IGV"), a["IGV_deg"], 2)
    if "speed" in slots:
        put(_dialect_name(dialect, "speed_rpm", "TNH", "shaft_rpm"), a["speed_rpm"], 1)
    if "brg_vib" in slots:
        for j in range(n_brg):
            name = _dialect_name(dialect, f"BRG{j + 1}_VIB_mm_s", f"BV{j + 1}", f"vib_brg_{j + 1}")
            cols[name] = np.round(np.clip(a["brg"][:, j], 0.05, None), 3)
    if "lube_supply" in slots or "oil_supply_t" in slots:
        put(_dialect_name(dialect, "lube_oil_C", "LTOT", "oil_supply_C"), a["lube_oil_C"], 2)
    if "oil_hdr_p" in slots:
        put(_dialect_name(dialect, "oil_hdr_bar", "lube_hdr_pressure", "oil_hdr_bar"), a["oil_hdr"], 3)
    if "ambient_t" in slots:
        put(_dialect_name(dialect, "ambient_C", "AFTAMB", "ambient_temp"), a["ambient"], 2)
    if "inlet_dp" in slots:
        put(_dialect_name(dialect, "inlet_dp_mbar", "AFPAP", "filter_house_dp"), a["inlet_dp"], 3)
    if "drain_t" in slots:
        for j in range(n_drain):
            name = _dialect_name(dialect, f"drain_T{j + 1}_C", f"LTB{j + 1}D", f"drain_t_{j + 1}")
            cols[name] = np.round(a["drains"][:, j], 2)
    if "heater_out" in slots:
        put("heater_out_C", a["heater_out"], 2)
    if "fuel_gas" in slots:
        put("fuel_gas_C", a["fuel_gas"], 2)

    # igv_ref is not a slot of igv_fuel_control. Writing it as an extra tag puts it in _other.
    extra_tags: List[str] = []
    if pid == "igv_fuel_control":
        ref_name = _dialect_name(dialect, "IGV_ref_deg", "CSRGV", "IGV_REF")
        put(ref_name, a["igv_ref"], 2)
        extra_tags.append(ref_name)
    n_extra = int(rng.integers(0, EXTRA_TAG_MAX + 1))
    if n_extra:
        pool = [name for name in _EXTRA_POOL if name not in cols]
        pick = rng.choice(pool, size=min(n_extra, len(pool)), replace=False)
        for k, name in enumerate(np.atleast_1d(pick)):
            cols[str(name)] = np.round(rng.normal(10 + k, 0.4, n), 3)
            extra_tags.append(str(name))
    df = pd.DataFrame(cols)
    if rng.random() < GENERIC_NAME_P:
        analytic = str(rng.choice(GENERIC_ANALYTICS))
    else:
        analytic = str(rng.choice(PROFILE_NAMES[pid]))
    notes, note_meta = _notes(sig, rng, params.get("sensor_index") or params.get("center_tc"))
    info = BASE_SIGNATURES.get(sig) or UNSEEN_SIGNATURES.get(sig) or {}
    meta = {
        "case_id": f"{split}-{pid}-{sig}-{variant or 'base'}-{i:03d}",
        "split": split,
        "profile": pid,
        "signature": sig,
        "action": info.get("action", "escalate"),
        "cause": info.get("cause", "unknown"),
        "escalate": bool(info.get("escalate", False)),
        "variant": variant,
        "ambiguous": bool(spec.get("ambiguous")),
        "analytic": analytic,
        "notes": notes,
        "notes_hint": note_meta["hint"],
        "dialect": dialect,
        "slots_present": slots,
        "missing_tags": missing,
        "extra_tags": extra_tags,
        "layout": ("zones" if use_zones else "ring") if "exh_tc" in slots else "none",
        "n_sensors": int(temps.shape[1]) if "exh_tc" in slots else 0,
        "freq_min": freq,
        "rows": n,
        "operating_profile": op,
        "frame_rated_MW": round(float(fp["rated"]), 1),
        "event_start": idx[ev[0]].strftime("%Y-%m-%d %H:%M") if ev else None,
        "event_end": idx[min(ev[1], n - 1)].strftime("%Y-%m-%d %H:%M") if ev else None,
        "params": params,
    }
    return df, meta


def _write_profile_one(job: Tuple[Dict[str, Any], str]) -> Dict[str, Any]:
    """Write ``split/profile/case.csv.gz`` and return the manifest row."""
    spec, out = job
    df, meta = make_profile_case(spec)
    path = Path(out) / meta["split"] / meta["profile"] / f"{meta['case_id']}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    meta["file"] = f"{meta['split']}/{meta['profile']}/{path.name}"
    return meta


def _generate_profile_dataset(out: Path, scale: float, workers: int, only_split: Optional[str],
                              profiles: List[str]) -> List[Dict[str, Any]]:
    """Alert-type dataset. The coldstart profile is not in train and is not a shipped id."""
    import multiprocessing as mp

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    specs = profile_specs(scale, list(profiles))
    keep: List[Dict[str, Any]] = []
    if only_split:
        specs = [s for s in specs if s["split"] == only_split]
        mf = out / "manifest.jsonl"
        if mf.is_file():
            keep = [r for r in (json.loads(x) for x in mf.read_text(encoding="utf-8").splitlines() if x.strip())
                    if r.get("split") != only_split]
    jobs = [(s, str(out)) for s in specs]
    workers = workers or max(1, (mp.cpu_count() or 2) - 2)
    if workers > 1 and len(jobs) > 8:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_write_profile_one, jobs, chunksize=4)
    else:
        rows = [_write_profile_one(j) for j in jobs]
    rows = keep + rows
    rows.sort(key=lambda r: (r.get("split", ""), r.get("profile", ""), r.get("case_id", "")))
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"Profile dataset: {len(rows)} cases in {out}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return rows


# --- end alert-type profiles


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic gas-turbine historian CSVs (samples or labeled e2e dataset).")
    ap.add_argument("--dataset", help="write the labeled e2e dataset into this folder (e.g. data/e2e/dataset)")
    ap.add_argument("--scale", type=float, default=1.0, help="dataset size multiplier (1.0 = ~1900 cases)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--only-split", help="write only this split (e.g. sensorfault) and merge it into the existing manifest")
    ap.add_argument("--profiles", action="store_true", help="write the seven alert-type profiles plus the coldstart heater split")
    a = ap.parse_args(argv)
    if a.dataset:
        chosen = list(ALL_SYNTH_PROFILES) if a.profiles else None
        generate_dataset(Path(a.dataset), scale=a.scale, workers=a.workers, only_split=a.only_split, profiles=chosen)
    else:
        generate_all()


if __name__ == "__main__":
    main()
