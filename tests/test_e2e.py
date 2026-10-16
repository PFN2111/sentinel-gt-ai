"""End-to-end learned path: dataset generator, features, head/index/novelty, engine, pipeline,
labeling loop (instant index, background retrain, versions, rollback), LLM runtime planning and
fallback, writer checks, CLI, dashboard and the drop-in UI."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, sample
from sentinel.config import load_settings
from sentinel.data import load_csv
from sentinel.e2e.features import CHANNELS, WINDOW, canonical
from sentinel.e2e.model import Calibration, Head, SimilarityIndex, calibrate, decide, train_head
from sentinel.e2e.taxonomy import BASE_SIGNATURES, NOVEL

MODEL_DIR = ROOT / "models" / "e2e"
needs_model = pytest.mark.skipif(
    not ((MODEL_DIR / "legacy" / "base" / "head.npz").is_file() or (MODEL_DIR / "base" / "head.npz").is_file()),
    reason="learned model not trained (scripts/train_e2e.py)",
)


def _gen():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_scenario_csvs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _gen()


def _write(gen, tmp_path, sig, variant=None, i=0, split="t"):
    df, meta = gen.make_case({"split": split, "signature": sig, "variant": variant, "index": i})
    p = tmp_path / f"{meta['case_id']}.csv"
    df.to_csv(p, index=False)
    return p, meta


# ------------------------------------------------------------------ generator


def test_dataset_specs_and_determinism(gen, tmp_path):
    specs = gen.dataset_specs(1.0)
    by = {}
    for s in specs:
        by[s["split"]] = by.get(s["split"], 0) + 1
    assert by == {"train": 1200, "val": 200, "test": 320, "variant": 128, "unseen": 80, "sensorfault": 40}
    a, ma = gen.make_case({"split": "x", "signature": "cold_spot_hardware", "variant": None, "index": 3})
    b, mb = gen.make_case({"split": "x", "signature": "cold_spot_hardware", "variant": None, "index": 3})
    pd.testing.assert_frame_equal(a, b)
    assert ma == mb and ma["action"] == "borescope"
    for sig in list(BASE_SIGNATURES) + ["bearing_vibration", "tc_harness_fault"]:
        for v in (None, "shifted", "load", "magnitude"):
            df, m = gen.make_case({"split": "x", "signature": sig, "variant": v, "index": 1})
            assert len(df) == m["rows"] > 200 and m["signature"] == sig
    # small written dataset round trip (gzip CSV readable by the loader)
    rows = gen.generate_dataset(tmp_path / "ds", scale=0.01, workers=1)
    assert (tmp_path / "ds" / "manifest.jsonl").is_file()
    case = load_csv(tmp_path / "ds" / rows[0]["file"])
    assert case.n == rows[0]["rows"]


def test_variants_change_what_they_claim(gen):
    _, base = gen.make_case({"split": "x", "signature": "cold_spot_hardware", "variant": None, "index": 2})
    _, load = gen.make_case({"split": "x", "signature": "cold_spot_hardware", "variant": "load", "index": 2})
    _, shift = gen.make_case({"split": "x", "signature": "cold_spot_hardware", "variant": "shifted", "index": 2})
    _, mag = gen.make_case({"split": "x", "signature": "cold_spot_hardware", "variant": "magnitude", "index": 2})
    assert load["frame_rated_MW"] > 140 > base["frame_rated_MW"]
    assert shift["n_sensors"] in (16, 36) and shift["params"]["center_tc"] in (1, shift["n_sensors"])
    assert mag["params"]["depth_C"] < 10 or mag["params"]["depth_C"] > 39


# ------------------------------------------------------------------ features


def test_canonical_shape_and_rotation_invariance(gen, tmp_path):
    p, meta = _write(gen, tmp_path, "cold_spot_hardware", i=4)
    df = pd.read_csv(p)
    f1 = canonical(load_csv(p))
    assert f1.x.shape == (WINDOW, len(CHANNELS)) and np.isfinite(f1.x).all()
    assert f1.excursions and f1.excursions[0]["channel"] in ("cold_dev", "spread", "cold_pair", "hot_dev", "egt_resid")
    ring = [c for c in df.columns if c.startswith("TC") or c.startswith("TTXD_")]
    if ring:  # rotate the ring by 7 positions: the canonical view must not change
        rot = df.copy()
        vals = df[ring].to_numpy()
        rot[ring] = np.roll(vals, 7, axis=1)
        q = tmp_path / "rot.csv"
        rot.to_csv(q, index=False)
        f2 = canonical(load_csv(q))
        assert np.allclose(f1.x, f2.x, atol=1e-4)


def test_historian_tag_names_map_to_same_channels(gen, tmp_path):
    df, _ = gen.make_case({"split": "x", "signature": "combustion_instability", "variant": None, "index": 5})
    if any(c in df.columns for c in gen.ALT_TAGS.values()):
        inv = {v: k for k, v in gen.ALT_TAGS.items()}
        plain = df.rename(columns=inv)
    else:
        plain = df
    tagged = plain.rename(columns=gen.ALT_TAGS).rename(columns=lambda c: c.replace("EGT_Z", "EGT_ZONE").replace("_C", "") if c.startswith("EGT_Z") else c)
    a, b = tmp_path / "plain.csv", tmp_path / "tagged.csv"
    plain.to_csv(a, index=False)
    tagged.to_csv(b, index=False)
    fa, fb = canonical(load_csv(a)), canonical(load_csv(b))
    assert np.allclose(fa.x, fb.x, atol=1e-4)


def test_short_and_channel_poor_csv_still_featurizes():
    f = canonical(load_csv(ROOT / "samples" / "gt_sensors_demo.csv"))
    assert f.x.shape == (WINDOW, len(CHANNELS)) and np.isfinite(f.x).all()


# ------------------------------------------------------------------ head / index / novelty / one-shot


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    centers = {"a": np.eye(8)[0], "b": np.eye(8)[1], "c": np.eye(8)[2]}
    Z, y = [], []
    for k, c in centers.items():
        for _ in range(20):
            v = c + rng.normal(0, 0.05, 8)
            Z.append(v / np.linalg.norm(v))
            y.append(k)
    return np.array(Z, np.float32), y


def test_head_index_novelty_and_one_shot():
    Z, y = _toy()
    head = train_head(Z, y)
    assert head.predict(Z[:1]) == ["a"] and set(head.classes) == {"a", "b", "c"}
    idx = SimilarityIndex(Z, [{"id": f"s{i}", "signature": s, "action": "monitor_only"} for i, s in enumerate(y)])
    Zv, yv = _toy(1)
    cal = calibrate(idx, Zv, yv)
    assert 0 < cal.match_threshold < cal.novelty_threshold
    d = decide(Zv[0], head, idx, cal)
    assert d["signature"] == "a" and d["basis"] == "classifier" and not d["novel"]
    far = np.eye(8)[5].astype(np.float32)
    dn = decide(far, head, idx, cal)
    assert dn["novel"] and dn["signature"] == NOVEL
    # one labeled example of a new signature near `far` -> one-shot match, cited
    idx.add(far * 0.999 + np.eye(8)[6] * 0.04, {"id": "eng-1", "signature": "new_thing", "action": "escalate"})
    d1 = decide(far, head, idx, cal)
    assert d1["one_shot"] and d1["signature"] == "new_thing" and d1["similar"][0]["id"] == "eng-1"
    # head save / load round trip (no pickle)
    head2 = Head(head.classes, head.W, head.b)
    assert np.allclose(head2.proba(Z[:3]), head.proba(Z[:3]))


def test_index_replace_and_remove():
    idx = SimilarityIndex(dim=3)
    idx.add(np.array([1, 0, 0]), {"id": "x", "signature": "a"})
    idx.add(np.array([0, 1, 0]), {"id": "x", "signature": "b"})
    assert len(idx) == 1 and idx.records[0]["signature"] == "b"
    idx.remove("x")
    assert len(idx) == 0


# ------------------------------------------------------------------ LLM runtime: device planning + fallback


def test_device_plan_gpu_auto_layers_and_cpu_fallback(monkeypatch):
    from sentinel import llm_runtime as rt

    monkeypatch.setattr(rt, "llama_cpp_info", lambda: {"installed": True, "version": "x", "gpu_offload": True})
    s = load_settings({"device": "auto"})
    gpu6 = rt.GpuInfo(nvidia=True, name="RTX 3050 6GB Laptop", vram_total_mb=6144, vram_free_mb=5600, driver="555")
    p = rt.plan(s, gpu=gpu6)
    if p.model_found:
        assert p.device == "gpu" and 22 <= p.n_gpu_layers <= 34, p.reason
    assert rt.plan(load_settings({"device": "cpu"}), gpu=gpu6).device == "cpu"
    p3 = rt.plan(load_settings({"device": "gpu"}), gpu=rt.GpuInfo(error="nvidia-smi not found"))
    assert p3.device == "cpu" and "falling back to CPU" in p3.reason
    p4 = rt.plan(load_settings({"device": "auto", "n_gpu_layers": "12"}), gpu=gpu6)
    assert p4.n_gpu_layers == 12


def test_gpu_failure_falls_back_to_cpu_and_is_reported(monkeypatch):
    from sentinel import llm_runtime as rt

    s = load_settings({"device": "gpu"})
    if not rt.resolve_model(s)["found"]:
        pytest.skip("GGUF not present")
    monkeypatch.setattr(rt, "plan", lambda settings, gpu=None: rt.Plan("quality", str(rt.resolve_model(s)["path"]), "fake", True, "gpu", "gpu", 28, 48, 4, "test"))

    class FakeLlama:
        def create_chat_completion(self, **kw):
            return {"choices": [{"message": {"content": '{"reply": "OK"}'}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    calls = []

    def fake_load(path, layers, ctx, threads):
        calls.append(layers)
        if layers > 0:
            raise RuntimeError("CUDA error: out of memory")
        return FakeLlama()

    monkeypatch.setattr(rt, "_load", fake_load)
    out = rt.chat_json(s, "sys", "user", {"type": "object"})
    assert calls == [28, 0] and out["device"] == "cpu"
    assert "fell back to CPU" in out["fallback"] and "VRAM" in out["fallback"]


# ------------------------------------------------------------------ engine / pipeline / loop (needs the trained model)


@pytest.fixture
def e2e_settings(tmp_path):
    return load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False, "e2e_retrain_every": 2})


@needs_model
def test_analyze_case_report_and_shadow(e2e_settings):
    from sentinel.e2e.pipeline import analyze_case

    r = analyze_case(sample("cold_spot", 1), "Exhaust Cold Spot", "Unit 2, 7FA.03", settings=e2e_settings)
    d, L, sh = r["decision"], r["learned"], r["shadow"]
    assert r["schema"] == "sentinel.e2e.report/1" and r["review"]["status"] == "DRAFT"
    assert sh["rules"]["primary_action"] == "borescope"
    assert L["similar"] and all("distance" in s for s in L["similar"])
    assert d["primary_action"] in ("borescope", "sensor_fix_replace", "escalate", "monitor_only")
    assert d["needs_human"] == bool(d["review_flags"])
    if d["primary_action"] != "borescope":
        assert not sh["agree"] and d["needs_human"]
    cdir = e2e_settings.cases_dir / r["case_id"]
    md = (cdir / "report.md").read_text(encoding="utf-8")
    for sec in ("## 1. Summary", "## 2. Recommended corrective action", "## 5. Evidence", "## 7. Similar labeled cases", "## 8. Shadow mode", "## 9. Audit trail", "## 10. Engineer review"):
        assert sec in md
    assert "Similar to case" in md and (cdir / "input.csv").is_file()
    assert (e2e_settings.data_dir / "e2e" / "emb" / f"{r['case_id']}.npy").is_file()
    assert "profile" not in r and e2e_settings.e2e_mode == "legacy"


@needs_model
def test_profiles_mode_resolves_embeds_and_stores_the_label(e2e_settings):
    """Per-alert mode decides from the alert name. Legacy mode is the test above."""
    from sentinel.e2e.pipeline import analyze_case, label_case
    from sentinel.profiles import load_profiles

    e2e_settings.e2e_mode = "peralert"
    e2e_settings.embedder = "stats"
    e2e_settings.profiles_dir = e2e_settings.data_dir / "e2e" / "profiles"
    csv = sample("cold_spot", 1)
    r = analyze_case(csv, "Exhaust Cold Spot", "Unit 2, 7FA.03", settings=e2e_settings)
    block, learned, decision = r["profile"], r["learned"], r["decision"]
    assert block["id"] == "exhaust_cold_spot"
    assert block["selected_by"] == "name"
    # The shipped stats bundle is head tier. Served with the scaler it was built with, this
    # cold spot sample sits inside the radius and the head names it; the rule path agrees.
    # (With the identity scaler every stats query looked far and this case was novel.)
    assert block["tier"] == "head" and learned["basis"] == "classifier" and learned["signature"] == "cold_spot_hardware"
    assert learned["nearest_distance"] < learned["novelty_threshold"]
    assert block["channels_hash"] and block["frame_key"] and block["embedder"] == "stats"
    assert block["slots"] and block["coverage"]["required"]
    assert r["review"]["status"] == "DRAFT"
    assert decision["primary_action"] == "borescope" == r["shadow"]["rules"]["primary_action"]
    emb = e2e_settings.data_dir / "e2e" / "emb2" / "exhaust_cold_spot" / f"{r['case_id']}.npz"
    assert emb.is_file()
    assert not (e2e_settings.data_dir / "e2e" / "emb" / f"{r['case_id']}.npy").is_file()
    md = (e2e_settings.cases_dir / r["case_id"] / "report.md").read_text(encoding="utf-8")
    assert "## 1. Summary" in md and "## 10. Engineer review" in md

    res = label_case(e2e_settings, r["case_id"], "Site specific cold", "borescope", note="sector confirmed", reviewer="pytest")
    lab = res["label"]
    assert lab["signature"] == "site_specific_cold"
    assert lab["profile"] == "exhaust_cold_spot"
    assert lab["profile_rev"] == block["rev"] + 1
    assert lab["channels_hash"] == block["channels_hash"] and lab["frame_key"] == block["frame_key"]
    assert res["profile_note"] and res["retrain_started"] is False
    prof = load_profiles(settings=e2e_settings)["exhaust_cold_spot"]
    assert "site_specific_cold" in prof.signatures and prof.rev == block["rev"] + 1

    r2 = analyze_case(csv, "Exhaust Cold Spot", "", settings=e2e_settings, run_rules=False)
    assert r2["profile"]["tier"] == "head"
    assert any(row["id"] == r["case_id"] for row in r2["learned"]["similar"])
    assert r2["learned"]["signature"] == "site_specific_cold"
    assert r2["learned"]["basis"] == "one_shot"


def test_peralert_far_case_with_a_confident_head_is_flagged_not_novel(e2e_settings):
    """Beyond the radius but above the confidence floor: the head answers and a person confirms."""
    from sentinel.e2e.pipeline import analyze_case, get_engine

    e2e_settings.e2e_mode = "peralert"
    e2e_settings.embedder = "stats"
    e2e_settings.profiles_dir = e2e_settings.data_dir / "e2e" / "profiles"
    csv = sample("cold_spot", 1)
    first = analyze_case(csv, "Exhaust Cold Spot", "", settings=e2e_settings, run_rules=False, save=False)
    assert first["profile"]["tier"] == "head"
    model = get_engine(e2e_settings).profile_cache.get("exhaust_cold_spot")
    assert model is not None
    # Every case is beyond this radius. A floor of 0 means the head is never unsure.
    model.calibration = {"profile": {"n": 600, "radius": 1e-6, "match": 1e-6, "confidence": 0.0}, "prior": {"n": 0, "radius": 0.5, "match": 0.4}}
    kept = analyze_case(csv, "Exhaust Cold Spot", "", settings=e2e_settings, run_rules=False, save=False)
    learned, decision = kept["learned"], kept["decision"]
    assert learned["novel"] is False and learned["basis"] == "classifier"
    assert learned["signature"] == learned["head_signature"] != "novel_pattern"
    assert learned["beyond_radius"] is True and learned["confidence_floor"] == 0.0
    assert decision["needs_human"] is True
    assert any(flag.startswith("Unfamiliar case:") for flag in decision["review_flags"])
    # A floor above every probability: the head is unsure, so both pieces of evidence agree.
    model.calibration["profile"]["confidence"] = 1.01
    novel = analyze_case(csv, "Exhaust Cold Spot", "", settings=e2e_settings, run_rules=False, save=False)
    assert novel["learned"]["novel"] is True and novel["learned"]["signature"] == "novel_pattern"
    assert not any(flag.startswith("Unfamiliar case:") for flag in novel["decision"]["review_flags"])


@needs_model
def test_peralert_unknown_name_keeps_the_rule_action_and_does_not_embed(e2e_settings):
    """An unknown name is never guessed. A keyword phrase still has a rule action."""
    from sentinel.config import MISSING_TAG_CONFIDENCE_CAP
    from sentinel.e2e.pipeline import analyze_case, label_case

    e2e_settings.e2e_mode = "peralert"
    e2e_settings.embedder = "stats"
    csv = sample("cold_spot", 1)
    unknown = analyze_case(csv, "Unit advisory", "", settings=e2e_settings, use_llm=False)
    assert unknown["learned"]["signature"] == "novel_pattern"
    assert unknown["learned"]["basis"] == "unknown_alert_type"
    assert unknown["decision"]["needs_human"] is True
    assert unknown["decision"]["primary_action"] == "escalate"
    assert unknown["profile"]["id"] is None
    assert "Unknown alert name" in unknown["decision"]["review_flags"][0]
    assert not (e2e_settings.data_dir / "e2e" / "emb2").exists() or not any(
        (e2e_settings.data_dir / "e2e" / "emb2").rglob(f"{unknown['case_id']}.npz")
    )
    with pytest.raises(ValueError, match="unknown alert name"):
        label_case(e2e_settings, unknown["case_id"], "normal", "monitor_only", reviewer="pytest")

    ruled = analyze_case(csv, "cold sector on tc 12", "", settings=e2e_settings, use_llm=False)
    assert ruled["learned"]["basis"] == "unknown_alert_type"
    assert ruled["profile"]["id"] is None
    assert ruled["decision"]["needs_human"] is True
    assert ruled["decision"]["primary_action"] == ruled["shadow"]["rules"]["primary_action"] == "borescope"

    alias = analyze_case(csv, "TTXD cold spot", "", settings=e2e_settings, use_llm=False, save=False)
    assert alias["profile"]["id"] == "exhaust_cold_spot"
    assert alias["profile"]["selected_by"] == "alias"

    explicit = analyze_case(
        csv, "HETS Trip", "", settings=e2e_settings, use_llm=False, alert_type="combustion_dynamics"
    )
    assert explicit["profile"]["id"] == "combustion_dynamics"
    assert explicit["profile"]["selected_by"] == "explicit"

    short = analyze_case(csv, "Inlet filter high differential pressure", "", settings=e2e_settings, use_llm=False)
    assert short["profile"]["id"] == "inlet_filter_dp"
    assert any("is not in this file" in flag for flag in short["decision"]["review_flags"])
    assert short["decision"]["needs_human"] is True
    assert short["decision"]["confidence"] <= MISSING_TAG_CONFIDENCE_CAP


def test_missing_tag_caps_confidence_at_the_constant():
    """The cap is the config constant, including when the model was confident."""
    from sentinel.config import MISSING_TAG_CONFIDENCE_CAP
    from sentinel.e2e.model import ProfileModel

    model = ProfileModel.empty("exhaust_cold_spot", channels=2)
    capped = model._finish(
        {"confidence": 0.95, "signature": "normal", "novel": False},
        missing_required=True,
        fallback=False,
    )
    assert capped["needs_human"] is True
    assert capped["confidence"] == MISSING_TAG_CONFIDENCE_CAP


@needs_model
def test_label_loop_instant_index_retrain_versions_rollback(gen, tmp_path, e2e_settings):
    from sentinel.e2e.pipeline import analyze_case, get_engine, label_case

    eng = get_engine(e2e_settings)
    n0 = len(eng.index)
    p1, _ = _write(gen, tmp_path, "bearing_vibration", split="u", i=1)
    p2, _ = _write(gen, tmp_path, "bearing_vibration", split="u", i=2)
    r1 = analyze_case(p1, "Bearing vibration high", "", settings=e2e_settings, engine=eng)
    # a brand-new signature, created while labeling
    res = label_case(e2e_settings, r1["case_id"], "Bearing vibration rise", "escalate", note="BB1 trending up", reviewer="pytest", engine=eng)
    assert res["label"]["signature"] == "bearing_vibration_rise"
    assert len(eng.index) == n0 + 1 and "bearing_vibration_rise" in eng.taxonomy()
    rep = json.loads((e2e_settings.cases_dir / r1["case_id"] / "report.json").read_text(encoding="utf-8"))
    assert rep["review"]["status"] in ("APPROVED", "REJECTED") and rep["review"]["feedback"]["signature"] == "bearing_vibration_rise"
    # one-shot: the next similar case cites the labeled one
    r2 = analyze_case(p2, "Bearing vibration high", "", settings=e2e_settings, engine=eng)
    assert any(s["id"] == r1["case_id"] for s in r2["learned"]["similar"])
    # second label reaches N=2 -> background retrain -> new version, then rollback
    out = label_case(e2e_settings, r2["case_id"], "bearing_vibration_rise", "escalate", engine=eng)
    eng.worker.join(60)
    assert out["retrain_started"] and not eng.worker.error, eng.worker.error
    assert eng.head_version == "v0001" and "bearing_vibration_rise" in eng.head.classes
    assert [v["version"] for v in eng.registry.versions()] == ["v0000", "v0001"]
    assert eng.rollback() == "v0000" and eng.head_version == "v0000"
    eng.activate("v0001")
    assert eng.registry.current() == "v0001"
    ev = [e["type"] for e in eng.metrics.events()]
    assert ev.count("analysis") == 2 and ev.count("label") == 2 and "retrain" in ev


@needs_model
def test_dashboard_series(e2e_settings, gen, tmp_path):
    from sentinel.e2e import dashboard
    from sentinel.e2e.pipeline import analyze_case, get_engine, label_case

    eng = get_engine(e2e_settings)
    for i in range(2):
        p, meta = _write(gen, tmp_path, "tc_sensor_fault", i=10 + i)
        r = analyze_case(p, "Exhaust TC deviation", "", settings=e2e_settings, engine=eng)
        label_case(e2e_settings, r["case_id"], "tc_sensor_fault", engine=eng)
    eng.worker.join(60)
    m = dashboard.metrics(eng)
    assert m["n_labels"] == 2 and m["n_analyses"] == 2
    assert len(dashboard.rolling_series(eng)) and len(dashboard.label_counts(eng)) == 1
    assert len(dashboard.heldout_series(eng)) >= 2


@needs_model
def test_novel_unseen_fault_is_not_forced(gen, tmp_path, e2e_settings):
    from sentinel.e2e.pipeline import analyze_case

    novel = 0
    for i in range(3):
        p, _ = _write(gen, tmp_path, "bearing_vibration", split="unseen", i=i)
        r = analyze_case(p, "Bearing vibration high", "", settings=e2e_settings, run_rules=False)
        if r["learned"]["novel"]:
            novel += 1
            assert r["decision"]["primary_action"] == "escalate" and r["decision"]["needs_human"]
            assert any("Novel pattern" in f for f in r["decision"]["review_flags"])
    assert novel >= 2


@needs_model
def test_writer_apply_guards(e2e_settings):
    from sentinel.e2e import writer
    from sentinel.e2e.pipeline import analyze_case, get_engine

    r = analyze_case(sample("hets_trip", 2), "HETS trip", "first-out high spread", settings=e2e_settings, save=False)
    tax = get_engine(e2e_settings).taxonomy()
    prompt = writer.build_prompt(r, tax)
    assert "MOST SIMILAR LABELED CASES" in prompt and "SIGNATURE CATALOGUE" in prompt and "RULE-BASED ANALYSIS" in prompt
    obj = {"signature": "normal", "primary_action": "monitor_only", "urgency": "routine", "escalate": False,
           "what_happened": "x" * 50, "why": "y" * 50, "action_rationale": "Monitor only, trend the spread.", "concerns": "notes say benign"}
    r["learned"]["novel"] = True
    narrative, flags = writer.apply(r, {"obj": obj, "checks": {}, "flags": []}, tax)
    assert r["decision"]["primary_action"] == "escalate" and any("novel" in f for f in flags)
    sch = writer.schema(list(tax))
    assert NOVEL in sch["properties"]["signature"]["enum"]


@needs_model
def test_cli_e2e_and_doctor(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", SENTINEL_USE_TSPULSE="0")
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "app.py"), "--data-dir", str(tmp_path / "d"), *a], capture_output=True, text=True,
                                    encoding="utf-8", env=env, cwd=str(ROOT), timeout=600)
    p = run("e2e", "run", "--csv", str(sample("combustion_dynamics", 3)), "--analytic", "comb dyn high")
    assert p.returncode == 0, p.stderr[-2000:]
    line = next(l for l in p.stdout.splitlines() if l.startswith("CASE\t"))
    cid = line.split("\t")[1]
    assert "RULES\tcombustion_tuning" in line
    p = run("e2e", "label", "--case", cid, "--signature", "combustion_instability", "--reviewer", "cli")
    assert p.returncode == 0 and "LABEL" in p.stdout, p.stderr[-2000:]
    p = run("--device", "cpu", "doctor", "--no-generate")
    assert "ACTIVE:" in p.stdout and "CPU" in p.stdout and "RESULT: READY" in p.stdout, p.stdout + p.stderr[-1000:]
    p = run("e2e", "versions")
    assert p.returncode == 0 and "v0000" in p.stdout


@needs_model
def test_export_import_labeled_folder(gen, tmp_path, e2e_settings):
    from sentinel.e2e.dataset import export_labeled, read_labeled_folder
    from sentinel.e2e.pipeline import analyze_case, get_engine, label_case

    eng = get_engine(e2e_settings)
    p, _ = _write(gen, tmp_path, "igv_lvdt_drift", i=3)
    r = analyze_case(p, "IGV position deviation", "", settings=e2e_settings, engine=eng)
    label_case(e2e_settings, r["case_id"], "igv_lvdt_drift", engine=eng)
    n = export_labeled(e2e_settings, tmp_path / "export")
    assert n == 1
    labs = read_labeled_folder(tmp_path / "export")
    assert labs[0]["signature"] == "igv_lvdt_drift" and labs[0]["action"] == "lvdt_calibration"


@needs_model
def test_site_adaptation_simulation_small(e2e_settings):
    from sentinel.e2e.engine import Engine
    from sentinel.e2e.simulate import site_adaptation

    eng = Engine(e2e_settings)
    rng = np.random.default_rng(0)
    d = eng.base_index.Z.shape[1]
    base = rng.normal(0, 1, d)
    items = [{"case_id": f"sim{i}", "emb": (base + rng.normal(0, 0.05, d)).astype(np.float32) * eng.scaler.std + eng.scaler.mean,
              "signature": "weird_new", "action": "escalate", "rules_action": "monitor_only"} for i in range(8)]
    out = site_adaptation(e2e_settings, items[:6], items[6:], every=3, new_signature_actions={"weird_new": "escalate"})
    assert out["curve"][0]["labels"] == 0 and out["curve"][-1]["labels"] == 6
    assert out["curve"][-1]["learned"] >= out["curve"][0]["learned"]
    assert out["versions"] >= 3


@needs_model
def test_dropin_ui_confirm_labels_case(tmp_path, monkeypatch):
    from sentinel.e2e.pipeline import analyze_case
    from ui_support import browser, run

    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SENTINEL_USE_LLM", "0")
    s = load_settings({"use_tspulse": False})
    r = analyze_case(sample("hets_trip", 3), "HETS trip", "", settings=s)

    async def body():
        async with browser() as user:
            await user.open(f"/?case={r['case_id']}")
            await user.should_see("DRAFT")
            user.find(marker="approve").click()
            await user.should_see("APPROVED")
            await user.open("/teach")
            await user.should_see("Alert types")
            await user.open("/models")
            await user.should_see("Model versions")
            await user.open("/settings")
            await user.should_see("Runtime device")

    run(body)
    rep = json.loads((s.cases_dir / r["case_id"] / "report.json").read_text(encoding="utf-8"))
    assert rep["review"]["status"] == "APPROVED"


def test_v2_explainer_is_self_contained():
    import re

    page_path = ROOT / "docs" / "sentinel_v2_explainer.html"
    if not page_path.is_file():
        pytest.skip("v2 explainer not built yet")
    page = page_path.read_text(encoding="utf-8")
    assert not re.findall(r"%%[A-Z_]+%%", page)
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", page)
    assert "<link" not in page and "@import" not in page and len(re.findall(r"<script[^>]*\bsrc=", page)) == 0
    assert "Qwen" not in page and "llm_profile" not in page
    for sec in ("north", "pipeline", "prompt", "teach", "dashboard", "reports", "eval", "deploy", "map"):
        assert f'id="{sec}"' in page
    assert "Gemma" in page and "sentinel_gt_ai.py" in page and "RESIDUAL VIEW" in page


def test_e2e_explainer_is_self_contained():
    import re

    page_path = ROOT / "docs" / "e2e_explainer.html"
    if not page_path.is_file():
        pytest.skip("explainer not built yet")
    page = page_path.read_text(encoding="utf-8")
    assert not re.findall(r"%%[A-Z_]+%%", page)
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", page)
    assert "<link" not in page and "@import" not in page and len(re.findall(r"<script[^>]*\bsrc=", page)) == 0
    for sec in ("what", "pipeline", "loop", "dashboard", "reports", "eval", "run", "limits"):
        assert f'<section id="{sec}"' in page
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1).replace("<\\/", "</"))
    assert data["results"]["test"]["n"] == 320 and data["results"]["sim_site"]["curve"]
    assert "Tier 2 Site Report" in page


def test_export_includes_reviewed_rule_path_cases(tmp_path):
    """Good / Not good reviews from the rule-path app become labeled training cases."""
    from sentinel import store
    from sentinel.e2e.dataset import export_labeled, read_labeled_folder
    from sentinel.pipeline import run_case

    s = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    a = run_case(sample("cold_spot", 2), "Exhaust Cold Spot", "", settings=s)
    b = run_case(sample("cold_spot", 3), "Exhaust Cold Spot", "", settings=s)
    store.record_feedback(s, a["case_id"], "good", reviewer="t")
    store.record_feedback(s, b["case_id"], "not_good", corrected_action="sensor_fix_replace", reason="TC harness", reviewer="t")
    n = export_labeled(s, tmp_path / "export")
    labs = {r["file"]: r for r in read_labeled_folder(tmp_path / "export")}
    assert n == 2
    assert labs[f"{a['case_id']}.csv"]["signature"] == "cold_spot_hardware"
    assert labs[f"{b['case_id']}.csv"]["signature"] == "tc_sensor_fault"


def test_llm_schemas_compile_to_grammars():
    """Length-bounded strings crash llama.cpp natively; our schemas must not use them."""
    from sentinel import llm, llm_runtime
    from sentinel.e2e import writer

    llm_runtime.check_schema(writer.schema(list(BASE_SIGNATURES)))
    llm_runtime.check_schema(llm.SCHEMA)
    with pytest.raises(ValueError):
        llm_runtime.check_schema({"type": "object", "properties": {"why": {"type": "string", "maxLength": 900}}})
    long = writer._fit("First sentence here. " * 80, 300)
    assert len(long) <= 301 and long.endswith(".")
