"""V2 residual feed: extraction, onset ordering, neighbour coherence, load tracking, TS Pulse / fallback,
deterministic cross-check, residual novelty and its calibration."""

from __future__ import annotations

import importlib.util
import io
import re

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from sentinel.config import load_settings
from sentinel.data import load_csv
from sentinel.e2e import residual
from sentinel.e2e.model import Calibration, SimilarityIndex, calibrate_residual, decide, train_head
from sentinel.e2e.taxonomy import NOVEL

AD_WEIGHTS = ROOT / "models" / "tspulse" / "ad" / "model.safetensors"
needs_ad = pytest.mark.skipif(not AD_WEIGHTS.is_file(), reason="TS Pulse anomaly weights not in models/tspulse/ad")


def _gen():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_scenario_csvs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _gen()


def _case_from(df: pd.DataFrame):
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return load_csv(b.getvalue(), name="t.csv")


def _ring_frame(n: int = 2000, m: int = 12, seed: int = 0) -> pd.DataFrame:
    """1-min record: steady load, a ring of m TCs, spread, dynamics, IGV, fuel."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-03-01 00:00", periods=n, freq="1min")
    load = 50 + rng.normal(0, 0.1, n)
    egt = 550 + rng.normal(0, 0.3, n)
    spatial = 2 * np.sin(2 * np.pi * np.arange(m) / m)
    T = egt[:, None] + spatial[None, :] + rng.normal(0, 0.8, (n, m))
    df = pd.DataFrame({"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"), "load_MW": load})
    for j in range(m):
        df[f"TC{j + 1}"] = T[:, j]
    df["comb_dyn_psi"] = 0.35 + np.abs(rng.normal(0, 0.05, n))
    df["IGV_deg"] = 60 + rng.normal(0, 0.1, n)
    df["fuel_flow_kg_s"] = 3.0 + rng.normal(0, 0.01, n)
    return df


def _finish(df: pd.DataFrame, m: int = 12) -> pd.DataFrame:
    T = df[[f"TC{j + 1}" for j in range(m)]].to_numpy()
    df["EGT_spread_C"] = T.max(axis=1) - T.min(axis=1)
    return df


@pytest.fixture(params=["robust_z", "tspulse"])
def res_settings(request, tmp_path):
    if request.param == "tspulse" and not AD_WEIGHTS.is_file():
        pytest.skip("TS Pulse anomaly weights missing")
    return load_settings({"data_dir": tmp_path / "d", "residual_mode": "auto" if request.param == "tspulse" else "robust_z"})


def test_single_tc_step_onset_size_and_neighbours(res_settings):
    df = _ring_frame()
    df.loc[1200:, "TC3"] -= 30.0  # one TC steps down alone
    res = residual.score(_case_from(_finish(df)), res_settings)
    assert res["mode"] == ("robust_z" if res_settings.residual_mode == "robust_z" else "tspulse")
    ev = {e["channel"]: e for e in res["events"]}
    e = ev["TC3"]
    assert e["direction"] == "down" and abs(abs(e["delta"]) - 30.0) < 3.0
    assert e["onset"] == "2025-03-01 20:00" and e["shape"] == "step" and e["ongoing"]
    nb = e["neighbours"]
    assert nb["left"]["channel"] == "TC2" and nb["right"]["channel"] == "TC4"
    assert nb["left"]["label"] == "flat" and nb["right"]["label"] == "flat" and "single sensor" in nb["summary"]
    assert res["pattern"]["ring"] == "single_sensor" and res["pattern"]["top_sensor"] == "TC3"
    ctx = res["context"]
    assert ctx["load"]["steady"] and abs(ctx["load"]["at_onset"] - 50) < 1 and "trip" not in ctx
    xc = residual.cross_check("cold_spot_hardware", res)
    assert xc["verdict"] == "contradicts" and xc["points_to"] == "tc_sensor_fault" and "TC3" in xc["reason"]
    assert residual.cross_check("tc_sensor_fault", res)["verdict"] == "supports"
    d = res["description"]
    assert re.search(r"TC3 stepped down (29|30)\.\d C", d) and "neighbours TC2 flat" in d and "load steady at" in d and len(d.splitlines()) <= 10


def test_onset_order_across_channels_and_sector_coherence(res_settings):
    df = _ring_frame(seed=1)
    for j, k in ((5, 1.0), (6, 0.7), (4, 0.6)):  # a cold sector TC6 (+ TC5, TC7), ramping over 20 min
        df.loc[1000:, f"TC{j + 1}"] -= 25.0 * k * np.clip((np.arange(len(df) - 1000)) / 20.0, 0, 1)
    df.loc[1300:, "comb_dyn_psi"] += 2.0  # dynamics rise 300 min later
    res = residual.score(_case_from(_finish(df)), res_settings)
    order = [e["channel"] for e in res["events"]]
    assert order.index("TC6") < order.index("comb_dyn_psi")
    dyn = next(e for e in res["events"] if e["channel"] == "comb_dyn_psi")
    first = res["events"][0]
    assert 1000 <= first["onset_idx"] <= 1008 and abs(dyn["minutes_after_first"] - 300) <= 10  # onset = 25 % of the final size
    tc6 = next(e for e in res["events"] if e["channel"] == "TC6")
    assert "followed" in tc6["neighbours"]["left"]["label"] and "followed" in tc6["neighbours"]["right"]["label"]
    assert res["pattern"]["ring"] == "sector" and res["pattern"]["adjacent_abnormal"] >= 2
    assert residual.cross_check("cold_spot_hardware", res)["verdict"] == "supports"
    assert residual.cross_check("tc_sensor_fault", res)["verdict"] == "contradicts"


def test_flatline_spikes_and_ttxd_ring_order(res_settings):
    df = _ring_frame(seed=2)
    df.loc[1100:, "TC8"] = round(float(df.loc[1100, "TC8"]), 1)  # frozen sensor
    res = residual.score(_case_from(_finish(df)), res_settings)
    e = next(e for e in res["events"] if e["channel"] == "TC8")
    assert e["shape"] == "flatline" and res["pattern"]["sensor_like"]
    # OEM-style tags: neighbours by sensor number (TTXD_10 next to TTXD_9 / TTXD_11), not by last digit
    df = _ring_frame(seed=3)
    df.loc[1200:, "TC10"] += 150 * (np.random.default_rng(0).random(len(df) - 1200) < 0.05)
    df = _finish(df).rename(columns={f"TC{j + 1}": f"TTXD_{j + 1}" for j in range(12)})
    res = residual.score(_case_from(df), res_settings)
    e = next(e for e in res["events"] if e["channel"] == "TTXD_10")
    assert e["shape"] == "spikes" and {e["neighbours"]["left"]["channel"], e["neighbours"]["right"]["channel"]} == {"TTXD_9", "TTXD_11"}


def test_normal_record_has_no_events_and_load_ramps_are_not_faults(tmp_path):
    s = load_settings({"data_dir": tmp_path, "residual_mode": "robust_z"})
    df = _ring_frame(seed=4)
    df.loc[1000:1100, "load_MW"] = np.linspace(50, 40, 101)  # dispatch: load and IGV move together
    df.loc[1101:, "load_MW"] = 40 + np.random.default_rng(1).normal(0, 0.1, len(df) - 1101)
    df.loc[1000:, "IGV_deg"] = 60 - (50 - df.loc[1000:, "load_MW"]) * 0.8
    res = residual.score(_case_from(_finish(df)), s)
    assert not [e for e in res["events"] if e["group"] in ("igv", "exhaust", "dynamics")], res["description"]
    assert residual.cross_check("normal", res)["verdict"] in ("supports", "inconclusive")


def test_status_and_fallback_mode(tmp_path):
    s = load_settings({"data_dir": tmp_path, "residual_mode": "robust_z"})
    st = residual.status(s)
    assert st["mode"] == "robust_z" and "disabled" in st["detail"]
    s2 = load_settings({"data_dir": tmp_path, "tspulse_dir": str(tmp_path / "nowhere")})
    st2 = residual.status(s2)
    assert st2["mode"] == "robust_z" and "not found" in st2["detail"]
    if AD_WEIGHTS.is_file():
        st3 = residual.status(load_settings({"data_dir": tmp_path}))
        assert st3["mode"] == "tspulse" and "TS Pulse" in st3["detail"]


@needs_ad
def test_sensor_fault_lookalike_from_generator(gen, tmp_path):
    """Held-out sensorfault split: one TC steps by a cold-spot-sized amount; the residuals isolate it."""
    df, meta = gen.make_case({"split": "sensorfault", "signature": "tc_sensor_fault", "variant": "step_lookalike", "index": 8})
    assert meta["params"]["mode"] == "step_lookalike" and meta["layout"] == "ring" and "cold" in meta["analytic"].lower()
    res = residual.score(_case_from(df), load_settings({"data_dir": tmp_path}))
    p = res["pattern"]
    assert p["ring"] == "single_sensor" and p["top_sensor"].endswith(str(meta["params"]["sensor_index"]))
    top = next(e for e in res["events"] if e["channel"] == p["top_sensor"])
    lag = (pd.Timestamp(top["onset"]) - pd.Timestamp(meta["event_start"])) / pd.Timedelta(minutes=meta["freq_min"])
    assert abs(abs(top["delta"]) - abs(meta["params"]["step_C"])) < 4 and 0 <= lag <= 2  # a ramped step crosses 25 % within 2 samples
    assert residual.cross_check("cold_spot_hardware", res)["verdict"] == "contradicts"


# ------------------------------------------------------------------ novelty with residual evidence


def _toy():
    rng = np.random.default_rng(0)
    centers = {"normal": np.eye(8)[0], "combustion_instability": np.eye(8)[1], "cold_spot_hardware": np.eye(8)[2]}
    Z, y = [], []
    for k, c in centers.items():
        for _ in range(20):
            v = c + rng.normal(0, 0.05, 8)
            Z.append(v / np.linalg.norm(v))
            y.append(k)
    return np.array(Z, np.float32), y


def test_residual_novelty_rule():
    Z, y = _toy()
    head = train_head(Z, y)
    idx = SimilarityIndex(Z, [{"id": f"s{i}", "signature": s, "action": "monitor_only", "source": "synthetic"} for i, s in enumerate(y)])
    cal = Calibration(novelty_threshold=0.9, match_threshold=0.1, residual_novelty=True, novelty_border=0.9, novelty_conf=0.99,
                      residual_thresholds=dict(residual.THRESHOLDS))
    v = np.eye(8)[1] + 0.6 * np.eye(8)[3]  # instability-ish, not too close
    z = (v / np.linalg.norm(v)).astype(np.float32)
    bearing = {"groups_abnormal": {"vibration": 60.0, "other": 40.0}, "max_score": 60.0, "thresholds": residual.THRESHOLDS}
    d = decide(z, head, idx, cal, residual=bearing)
    assert d["novel"] and d["signature"] == NOVEL and d["basis"] == "residual_novelty" and d["residual_novelty_from"] == "combustion_instability"
    explained = {"groups_abnormal": {"dynamics": 80.0, "vibration": 10.0}, "max_score": 80.0, "thresholds": residual.THRESHOLDS}
    assert decide(z, head, idx, cal, residual=explained)["signature"] == "combustion_instability"
    assert decide(z, head, idx, cal)["signature"] == "combustion_instability"  # no residual -> V1 behaviour
    cal.residual_novelty = False
    assert decide(z, head, idx, cal, residual=bearing)["signature"] == "combustion_instability"
    cal.residual_novelty, cal.novelty_conf, cal.novelty_border = True, 0.0, 0.99  # confident and close -> keep the class
    assert decide(z, head, idx, cal, residual=bearing)["signature"] == "combustion_instability"


def test_explained_and_thresholds():
    t = residual.THRESHOLDS
    assert not residual.explained("combustion_instability", {"groups_abnormal": {"vibration": 30.0}, "thresholds": t})
    assert residual.explained("combustion_instability", {"groups_abnormal": {"dynamics": 30.0, "vibration": 8.0}, "thresholds": t})
    assert not residual.explained("normal", {"groups_abnormal": {"exhaust": 9.0}, "thresholds": t})
    assert residual.explained("normal", {"groups_abnormal": {}, "thresholds": t})
    assert residual.explained("my_custom_signature", {"groups_abnormal": {"vibration": 30.0}, "thresholds": t})
    res = [{"channels": [{"group": "igv", "score": 5.0 + i * 0.1}, {"group": "dynamics", "score": 4.0}]} for i in range(30)]
    out = residual.calibrate_thresholds(res, ["cold_spot_hardware"] * 30)
    assert out["thresholds"]["igv"] == round(1.1 * float(np.quantile([5.0 + i * 0.1 for i in range(30)], 0.99)), 1)
    assert out["thresholds"]["dynamics"] == residual.ABNORMAL  # floor


def _profile_settings(tmp_path):
    return load_settings({"data_dir": tmp_path / "d", "residual_mode": "robust_z", "profiles_dir": tmp_path / "profiles"})


def test_builtin_profile_keeps_the_cold_spot_rule(tmp_path):
    """legacy_role ring still runs today's single-sensor contradiction."""
    from sentinel.profiles import load_profiles, resolve_tags

    df = _ring_frame()
    df.loc[1200:, "TC3"] -= 30.0
    case = _case_from(_finish(df))
    prof = load_profiles()["exhaust_cold_spot"]
    resolved = resolve_tags(prof, case)
    res = residual.score(case, _profile_settings(tmp_path), profile=prof, resolved=resolved)
    event = next(e for e in res["events"] if e["channel"] == "TC3")
    assert event["group"] == "exhaust" and event["slot"] == "exh_tc" and event["direction"] == "down"
    assert event["neighbours"]["left"]["channel"] == "TC2" and event["neighbours"]["right"]["channel"] == "TC4"
    xc = residual.cross_check("cold_spot_hardware", res, prof)
    assert xc["verdict"] == "contradicts" and xc["points_to"] == "tc_sensor_fault"
    assert "exhaust temperature" in res["description"] and "TC3" in res["description"]


def test_generic_residual_expect_supports_contradicts_and_inconclusive(tmp_path):
    """No legacy_role: residual_expect decides, and a missing entry never contradicts."""
    from sentinel.profiles import Profile, Slot, resolve_tags

    prof = Profile(
        id="inlet_demo",
        name="Inlet demo",
        slots=[
            Slot(id="load", role="context", label="unit load", unit="MW", required=True, aliases={"*": ["load_MW"]}),
            Slot(id="dp", role="process", label="inlet differential pressure", unit="mbar", required=True, aliases={"*": ["inlet_dp"]}),
            Slot(id="extra", role="process", label="extra temperature", unit="C", aliases={"*": ["extra_t"]}),
        ],
        signatures=["normal", "inlet_filter_fouling", "extra_fault"],
        residual_expect={
            "inlet_filter_fouling": {"primary": ["dp"], "allowed": ["dp"]},
            "extra_fault": {"primary": ["extra"], "allowed": ["extra"]},
        },
        include_other=False,
    )
    settings = _profile_settings(tmp_path)
    n = 2000
    t = pd.date_range("2025-03-01 00:00", periods=n, freq="1min")

    def frame(dp_step: float, extra_step: float, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        load = 50 + rng.normal(0, 0.1, n)
        dp = 20 + rng.normal(0, 0.05, n)
        extra = 30 + rng.normal(0, 0.05, n)
        dp[1200:] += dp_step
        extra[1200:] += extra_step
        return pd.DataFrame({"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"), "load_MW": load, "inlet_dp": dp, "extra_t": extra})

    fouled_df = frame(40.0, 0.0, 0)
    other_df = frame(0.0, 80.0, 1)
    fouled = residual.score(_case_from(fouled_df), settings, profile=prof, resolved=resolve_tags(prof, _case_from(fouled_df)))
    other = residual.score(_case_from(other_df), settings, profile=prof, resolved=resolve_tags(prof, _case_from(other_df)))
    assert "dp" in fouled["slots_abnormal"]
    assert residual.cross_check("inlet_filter_fouling", fouled, prof)["verdict"] == "supports"
    assert residual.cross_check("normal", fouled, prof)["verdict"] == "inconclusive"
    assert residual.cross_check("not_a_signature", fouled, prof)["verdict"] == "inconclusive"
    contradicted = residual.cross_check("inlet_filter_fouling", other, prof)
    assert contradicted["verdict"] == "contradicts" and contradicted["points_to"] == "extra_fault"
    assert "inlet differential pressure" in fouled["description"]


def test_ring_neighbours_stay_inside_their_slot(tmp_path):
    """Two rings both number from zero. A TC's neighbours are the adjacent TCs, not the other ring."""
    from sentinel.profiles import Profile, Slot, resolve_tags

    prof = Profile(
        id="two_rings",
        name="Two rings",
        slots=[
            Slot(id="load", role="context", label="unit load", unit="MW", aliases={"*": ["load_MW"]}),
            Slot(id="tc", role="ring", label="thermocouples", unit="C", health=True, aliases={"*": [{"prefix": "TC"}]}),
            Slot(id="ttxd", role="ring", label="wheelspace", unit="C", health=True, aliases={"*": [{"prefix": "TTXD"}]}),
        ],
        include_other=False,
    )
    df = _ring_frame(m=6)
    df.loc[1200:, "TC3"] -= 30.0
    for j in range(6):
        df[f"TTXD_{j + 1}"] = 400 + np.random.default_rng(1).normal(0, 0.4, len(df))
    case = _case_from(df)
    res = residual.score(case, _profile_settings(tmp_path), profile=prof, resolved=resolve_tags(prof, case))
    event = next(e for e in res["events"] if e["channel"] == "TC3")
    assert {event["neighbours"]["left"]["channel"], event["neighbours"]["right"]["channel"]} == {"TC2", "TC4"}
    assert res["pattern"]["slot_patterns"]["tc"]["pattern"] == "single_sensor"


def test_profile_threshold_calibration_uses_role_defaults_then_quantile():
    from sentinel.profiles import Profile, Slot

    prof = Profile(
        id="demo",
        name="Demo",
        slots=[
            Slot(id="dp", role="process", label="inlet dp"),
            Slot(id="ring", role="ring", label="ring"),
            Slot(id="igv", role="process", label="IGV", legacy_role="igv"),
        ],
    )
    few = [{"channels": [{"slot": "dp", "score": 4.0}]}] * 5
    out = residual.calibrate_profile_thresholds(few, prof)
    assert out["source"] == "role_default" and out["n"] == 5
    assert out["thresholds"]["dp"] == 7.4
    assert out["thresholds"]["ring"] == 6.0
    assert out["thresholds"]["igv"] == residual.THRESHOLDS["igv"]
    scores = [5.0 + i * 0.1 for i in range(30)]
    many = [{"channels": [{"slot": "dp", "score": s}, {"slot": "ring", "score": 1.0}, {"slot": "igv", "score": 1.0}]} for s in scores]
    out2 = residual.calibrate_profile_thresholds(many, prof)
    expect = round(max(7.4, residual.ABNORMAL, 1.1 * float(np.quantile(scores, 0.99))), 1)
    assert out2["source"] == "calibrated"
    assert out2["thresholds"]["dp"] == expect
    assert out2["thresholds"]["ring"] == 6.0


def test_calibrate_residual_respects_false_novelty_budget():
    cal = Calibration(novelty_threshold=0.9, match_threshold=0.1, residual_thresholds=dict(residual.THRESHOLDS))
    dec, res, truth = [], [], []
    for i in range(100):
        wrong = i < 6
        dec.append({"signature": "combustion_instability", "head_confidence": 0.4 if wrong else 0.95, "nearest_distance": 0.3, "novel": False})
        res.append({"groups_abnormal": {"vibration": 50.0} if (wrong or i in (50, 51)) else {"dynamics": 50.0}, "thresholds": residual.THRESHOLDS})
        truth.append("bearing" if wrong else "combustion_instability")
    out = calibrate_residual(cal, dec, res, truth, max_extra_false_novelty=0.01)
    ch = out["chosen"]
    assert cal.residual_novelty and ch["flag_correct"] <= 1 and ch["flag_wrong"] == 6
    assert 0.4 < cal.novelty_conf <= 0.95
