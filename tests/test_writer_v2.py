"""V2 learned-path report writer: residual view + process knowledge in the prompt, token budget,
validation of the new output fields (residual_consistency, process_steps), number tracing over them,
cross-check merge with the deterministic check, and report / pipeline integration."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import shutil

import pytest

from conftest import ROOT
from sentinel.config import load_settings
from sentinel.knowledge import approx_tokens, load_sections, retrieve, retrieve_learned

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
def lookalike_csv():
    """Held-out sensor-fault case 008: one TC steps 29.5 C alone; the classifier calls it a cold spot."""
    df, meta = _gen().make_case({"split": "sensorfault", "signature": "tc_sensor_fault", "variant": "step_lookalike", "index": 8})
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue(), meta


@pytest.fixture
def kdir(tmp_path):
    d = tmp_path / "knowledge"
    d.mkdir()
    for p in (ROOT / "knowledge").glob("*.md"):
        shutil.copy(p, d / p.name)
    return d


@pytest.fixture
def v2_settings(tmp_path, kdir):
    return load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False, "knowledge_dir": str(kdir)})


# ------------------------------------------------------------------ knowledge retrieval


def test_retrieve_learned_signature_docs_cap_and_user_tags(kdir):
    k = retrieve_learned("Exhaust Cold Spot", "spread trending up", "cold_spot_hardware", token_cap=700, folder=kdir)
    refs = [x["ref"] for x in k]
    assert refs[0] == "process_map_cold_spot.md § Decision flow" and "process_map_exhaust_spread.md § Decision flow" in refs
    assert not any("Sample CSVs" in r or "combustion_dynamics" in r for r in refs)
    assert sum(approx_tokens(x["ref"]) + 4 + approx_tokens(x["text"]) for x in k) <= 700
    small = retrieve_learned("Exhaust Cold Spot", "", "cold_spot_hardware", token_cap=120, folder=kdir)
    assert len(small) == 1 and small[0]["text"].endswith("…")
    (kdir / "user").mkdir()
    (kdir / "user" / "site_tc.md").write_text('<!-- sentinel-doc: {"title": "Site TC check", "analytics": ["cold_spot"], "signatures": ["tc_sensor_fault"]} -->\n'
                                              "# Site TC check\n\n## Loop check first\n1. Megger the TC loop.\n2. Compare with the neighbour TCs.\n", encoding="utf-8")
    k2 = retrieve_learned("Exhaust Cold Spot", "", "cold_spot_hardware", extra_signatures=["tc_sensor_fault"], folder=kdir)
    assert k2[0]["ref"] == "user/site_tc.md § Loop check first" and "tagged for tc_sensor_fault" in k2[0]["why"]
    # rule path: the analytic's tagged user document comes first
    r = retrieve("cold_spot", "cold spot", folder=kdir)
    assert r[0]["ref"].startswith("user/site_tc.md") and any(x["ref"].startswith("process_map_cold_spot.md") for x in r)
    assert any(s.user for s in load_sections(kdir))


def test_deleted_user_document_is_not_retrieved(kdir):
    (kdir / "user").mkdir()
    (kdir / "user" / "old.md").write_text('<!-- sentinel-doc: {"title": "Old", "analytics": ["cold_spot"], "deleted": true} -->\n## Step\ntext\n', encoding="utf-8")
    assert not [s for s in load_sections(kdir) if s.user]


# ------------------------------------------------------------------ prompt, budget, validation


@needs_model
def test_prompt_has_residual_view_knowledge_and_why_within_budget(v2_settings, lookalike_csv):
    from sentinel.e2e import writer
    from sentinel.e2e.pipeline import analyze_case, get_engine

    raw, meta = lookalike_csv
    r = analyze_case(raw, meta["analytic"], meta["notes"], settings=v2_settings, source_name="sf008.csv")
    tax = get_engine(v2_settings).taxonomy()
    p = writer.build_prompt(r, tax)
    for part in ("RESIDUAL VIEW", "Onset order:", "neighbours", "PROCESS STEPS AND BACKGROUND", "[process_map_cold_spot.md § Decision flow]",
                 "residuals: one exhaust TC / zone moves alone", "MOST SIMILAR LABELED CASES"):
        assert part in p, part
    st = writer.prompt_stats(r, tax)
    assert st["total"] <= writer.PROMPT_TOKEN_BUDGET and 0 < st["residual_part"] < 600 and 0 < st["knowledge_part"] <= writer.KNOWLEDGE_TOKENS + 50
    assert st["added_v2"] < 3000
    # a tight budget trims the process knowledge first, then the residual detail
    tight = writer.build_prompt(r, tax, budget=approx_tokens(writer.SYSTEM) + approx_tokens(p) - 150)
    assert tight.count("§") < p.count("§") and "RESIDUAL VIEW" in tight
    tiny = writer.build_prompt(r, tax, budget=500)
    assert "PROCESS STEPS AND BACKGROUND" not in tiny and "RESIDUAL VIEW" in tiny


def _fake_chat(obj):
    def chat(settings, system, user, schema, max_tokens=None):
        from sentinel.llm_runtime import check_schema

        check_schema(schema)
        return {"text": json.dumps(obj), "usage": {"prompt_tokens": 2400, "completion_tokens": 500}, "load_s": 1.0, "gen_s": 2.0,
                "device": "cpu", "model_label": "fake", "model_file": "fake.gguf"}
    return chat


@needs_model
def test_writer_validates_new_fields_and_merges_cross_check(v2_settings, lookalike_csv, monkeypatch):
    from sentinel import llm_runtime
    from sentinel.e2e import writer
    from sentinel.e2e.pipeline import analyze_case, get_engine

    raw, meta = lookalike_csv
    r = analyze_case(raw, meta["analytic"], meta["notes"], settings=v2_settings, source_name="sf008.csv", save=False)
    tax = get_engine(v2_settings).taxonomy()
    L = r["learned"]
    assert L["signature"] == "cold_spot_hardware" and L["residual_check"]["verdict"] == "contradicts"
    sensor = r["residual"]["pattern"]["top_sensor"]
    good = {"signature": "tc_sensor_fault", "primary_action": "sensor_fix_replace", "urgency": "prompt", "escalate": False,
            "what_happened": f"{sensor} stepped down alone while its neighbours stayed flat; load was steady.",
            "why": f"The residual view shows {sensor} moved alone, which contradicts a cold sector; nearest case cited.",
            "action_rationale": "Loop-check and repair or replace the thermocouple before any borescope.",
            "residual_consistency": "contradicts", "residual_reason": f"{sensor} stepped alone; neighbours flat.",
            "process_steps": [{"step": "Validate instrumentation", "finding": f"{sensor} stepped alone: points to the sensor."},
                              {"step": "Characterize the cold sector", "finding": "No sector: neighbours flat, onset was a step of 77.7 C."}],
            "concerns": "Residuals contradict the classifier's cold sector."}
    monkeypatch.setattr(llm_runtime, "chat_json", _fake_chat(good))
    w = writer.write(r, v2_settings, tax)
    assert w["ok"] and w["checks"]["residual cross-check (LLM)"] == "contradicts" and "process steps walked" in w["checks"]
    assert "77.7" in w["checks"].get("untraced numbers", "")  # numbers in process steps are traced too
    conf0 = r["decision"]["confidence"]
    nar, flags = writer.apply(r, w, tax)
    d = r["decision"]
    assert d["residual_consistency"]["final"] == "contradicts" and d["residual_consistency"]["llm"] == "contradicts"
    assert d["primary_action"] == "sensor_fix_replace" and any("changed the signature" in f for f in flags)
    assert nar["process_steps"][0]["step"] == "Validate instrumentation" and d["confidence"] <= conf0 < 0.6
    # LLM disagrees with the deterministic contradiction -> explicit flag for a human
    r2 = analyze_case(raw, meta["analytic"], meta["notes"], settings=v2_settings, source_name="sf008.csv", save=False)
    lax = {**good, "signature": "cold_spot_hardware", "primary_action": "borescope", "residual_consistency": "supports", "residual_reason": "", "process_steps": []}
    monkeypatch.setattr(llm_runtime, "chat_json", _fake_chat(lax))
    w2 = writer.write(r2, v2_settings, tax)
    assert any("did not walk" in f for f in w2["flags"])
    _, flags2 = writer.apply(r2, w2, tax)
    assert any("disagrees with the deterministic residual check" in f for f in flags2)
    # invalid verdict -> rejected
    monkeypatch.setattr(llm_runtime, "chat_json", _fake_chat({**good, "residual_consistency": "maybe"}))
    assert not writer.write(r2, v2_settings, tax)["ok"]
    # schema stays grammar-safe (no length / item bounds)
    llm_runtime.check_schema(writer.schema(list(tax)))


@needs_model
def test_pipeline_report_carries_residual_evidence_and_lowered_confidence(v2_settings, lookalike_csv):
    from pathlib import Path

    from sentinel.e2e.pipeline import analyze_case

    raw, meta = lookalike_csv
    r = analyze_case(raw, meta["analytic"], meta["notes"], settings=v2_settings, source_name="sf008.csv")
    d = r["decision"]
    assert d["residual_consistency"]["deterministic"] == "contradicts" and d["needs_human"]
    assert d["confidence"] < d["confidence_before_residual"] and d["confidence"] < 0.6
    assert any(f.startswith("Residual cross-check contradicts cold_spot_hardware") for f in d["review_flags"])
    assert r["residual"]["profile"]["channels"] and r["audit"]["residual_mode"] in ("tspulse", "robust_z") and r["audit"]["knowledge"]
    assert "residual view contradicts" in r["narrative"]["concerns"]
    md = (Path(r["_path"]) / "report.md").read_text(encoding="utf-8")
    for part in ("Residual: top abnormal channels", "Residual: onset order", "Residual: neighbour coherence", "Residual cross-check vs cold_spot_hardware",
                 "**Residual cross-check**", "Process knowledge in the LLM prompt"):
        assert part in md, part
    assert re.search(r"Lowered from 0\.\d\d", md)


def test_residual_chart_builds(lookalike_csv, tmp_path):
    from sentinel.charts import residual_chart
    from sentinel.data import load_csv
    from sentinel.e2e import residual

    raw, _ = lookalike_csv
    res = residual.score(load_csv(raw), load_settings({"data_dir": tmp_path, "residual_mode": "robust_z"}))
    ch = residual_chart(res)
    spec = ch.to_dict()
    assert spec["layer"][0]["mark"]["type"] == "rect" and any(lay["mark"]["type"] == "point" for lay in spec["layer"])
    assert residual_chart({}) is None


# ------------------------------------------------------------------ alert-type prompt (profiles)


def _learned(**over):
    row = {
        "features": {"layout": "ring", "n_sensors": 12, "zoom": "event", "missing_roles": []},
        "head_probs": {"inlet_filter_fouling": 0.8},
        "head_version": "v0000",
        "signature": "inlet_filter_fouling",
        "signature_title": "Inlet filter fouling",
        "basis": "knn",
        "confidence": 0.8,
        "novel": False,
        "nearest_distance": 0.2,
        "novelty_threshold": 0.5,
        "match_threshold": 0.15,
        "index_size": 6,
        "similar": [],
        "excursions": [{"channel": "inlet_dp", "label": "inlet differential pressure", "score": 4.0, "time": "t0"}],
        "low_confidence": 0.6,
    }
    row.update(over)
    return row


def _report(profile=None, **learned_over):
    r = {
        "learned": _learned(**learned_over),
        "input": {"rows": 100, "start": "t0", "end": "t1", "interval_minutes": 5, "source_name": "case.csv",
                  "sha256": "a" * 64, "columns": 8, "frame_type": "auto"},
        "analytic": {"entered": "Inlet filter DP high"},
        "case": {"unit": "U3", "notes": "none"},
        "shadow": {"rules": {}, "agree": True, "comparison": ""},
        "knowledge": [],
        "guardrails": [],
    }
    if profile is not None:
        r["profile"] = profile
    return r


def _filter_profile():
    return {
        "id": "inlet_filter_dp",
        "name": "Inlet filter high differential pressure",
        "rev": 3,
        "frame_key": "site_unit3",
        "tier": "knn",
        "signatures": ["inlet_filter_fouling", "dp_transmitter_fault"],
        "docs": ["process_map_hets_trip.md"],
        "other_tags": ["LTOT", "BB1"],
        "residual_expect": {
            "inlet_filter_fouling": {"primary": ["inlet_dp"], "allowed": ["inlet_dp", "cdp"]},
        },
        "slots": [
            {"slot": "load", "role": "context", "label": "load", "unit": "MW", "load_fit": False, "tags": ["U3_GROSS_MW"], "status": "present"},
            {"slot": "inlet_dp", "role": "process", "label": "inlet differential pressure", "unit": "mbar", "load_fit": True,
             "tags": ["U3_INL_DP"], "status": "present"},
            {"slot": "ambient_t", "role": "process", "label": "ambient temperature", "unit": "C", "load_fit": False, "tags": [], "status": "missing"},
            {"slot": "cdp", "role": "process", "label": "compressor discharge pressure", "unit": "bar", "load_fit": False, "tags": ["CPD"], "status": "masked"},
            {"slot": "exh_tc", "role": "ring", "label": "exhaust temperature", "unit": "C", "load_fit": False,
             "tags": ["TC12", "TC1", "TC3"], "status": "present"},
        ],
    }


_TAX = {
    "cold_spot_hardware": {"title": "Cold sector", "action": "borescope", "urgency": "next_outage", "residual_pattern": "sector cools"},
    "inlet_filter_fouling": {"title": "Inlet filter fouling", "action": "monitor_only", "urgency": "next_opportunity"},
    "dp_transmitter_fault": {"title": "DP transmitter fault", "action": "sensor_fix_replace", "urgency": "prompt"},
    "novel_pattern": {"title": "Novel", "action": "escalate", "urgency": "prompt"},
}


def test_legacy_prompt_keeps_the_exhaust_layout_and_full_catalogue():
    from sentinel.e2e import writer

    p = writer.build_prompt(_report(), _TAX)
    assert "ALERT TYPE AND TAGS" not in p
    assert "exhaust layout ring (12 sensors)" in p
    assert "cold_spot_hardware:" in p and "inlet_filter_fouling:" in p
    assert "residuals: two or more adjacent exhaust TCs" in p or "residuals: sector cools" in p
    names = writer._prompt_signature_names(_report(), _TAX)
    assert names == list(_TAX)
    assert "Alert type:" not in writer_md(_report())


def writer_md(r):
    """Minimal report markdown. The slot table is the only new block."""
    from sentinel.e2e.report import render_markdown

    r = {
        **r,
        "decision": {
            "primary_action": "monitor_only", "urgency": "next_opportunity", "severity": "low",
            "confidence": 0.8, "confidence_label": "High", "escalate": False, "needs_human": False,
            "hypothesis": "Inlet filter fouling", "signature": r["learned"]["signature"],
            "cause_title": "process", "review_flags": [],
        },
        "narrative": {"action_rationale": "Watch the trend.", "what_happened": "DP rose.", "why": "Slow rise.", "site_instructions": []},
        "review": {"status": "DRAFT"},
        "bottom_line": "Watch it.",
        "case_id": "c1",
        "created": "2026-01-01",
        "audit": {"encoder": "stats", "narrative_engine": "template", "elapsed_s": 1},
    }
    return render_markdown(r)


def test_profile_prompt_lists_real_tags_and_cuts_the_catalogue():
    from sentinel.e2e import writer
    from sentinel.e2e.taxonomy import NOVEL

    prof = _filter_profile()
    r = _report(prof)
    p = writer.build_prompt(r, _TAX)
    assert p.startswith("ALERT TYPE AND TAGS")
    assert "Alert type: Inlet filter high differential pressure (profile inlet_filter_dp rev 3, frame site_unit3, tier knn: 6 labeled cases)" in p
    assert "- load (context, MW): U3_GROSS_MW — present" in p
    assert "- inlet_dp (process, mbar, load-fitted): U3_INL_DP — present" in p
    assert "- ambient_t (process, C): not in this file" in p
    assert "- cdp (process, bar): mostly empty" in p
    assert "- exh_tc (ring, C): TC1 .. TC12 (3 tags) — present" in p
    assert "Other numeric tags (not in the profile): LTOT, BB1" in p
    assert "exhaust layout" not in p
    assert "inlet differential pressure score +4.0 at t0" in p
    assert "cold_spot_hardware:" not in p
    assert "- inlet_filter_fouling: Inlet filter fouling -> monitor_only, next_opportunity; residuals: primary inlet differential pressure; allowed inlet differential pressure, compressor discharge pressure" in p
    assert "- dp_transmitter_fault:" in p
    assert f"- {NOVEL}: unlike any labeled case" in p
    enum = writer.schema(writer._prompt_signature_names(r, _TAX))["properties"]["signature"]["enum"]
    assert enum == ["inlet_filter_fouling", "dp_transmitter_fault", NOVEL]
    md = writer_md(r)
    assert "## 1. Summary" in md and "| Slot | Role | Unit | Tags | Status |" in md
    assert "| inlet_dp | process | mbar | U3_INL_DP | present |" in md
    assert "**Alert type:** Inlet filter high differential pressure" in md
    assert "Other numeric tags (not in the alert type): LTOT, BB1." in md


def test_many_slots_keep_the_twenty_largest_excursions():
    from sentinel.e2e import writer

    slots = []
    excursions = []
    for i in range(32):
        sid = f"p{i:02d}"
        slots.append({"slot": sid, "role": "process", "label": sid, "unit": "C", "load_fit": False, "tags": [f"T{i}"], "status": "present"})
        excursions.append({"channel": sid, "label": sid, "score": float(32 - i), "time": "t0"})
    slots.append({"slot": "gone", "role": "process", "label": "gone", "unit": "", "tags": [], "status": "missing"})
    prof = {**_filter_profile(), "slots": slots, "other_tags": []}
    p = writer.build_prompt(_report(prof, excursions=excursions, index_size=0), _TAX)
    assert "and 12 more present" in p
    assert "- p00 " in p and "- p19 " in p
    assert "- p20 " not in p and "- gone " not in p


def test_cold_novel_prompt_does_not_format_a_missing_distance():
    from sentinel.e2e import writer

    r = _report(_filter_profile(), novel=True, nearest_distance=None, novelty_threshold=None, basis="cold_start",
                note="NOVEL: no labeled cases for this alert type yet.")
    p = writer.build_prompt(r, _TAX)
    assert "NOVEL: no labeled cases for this alert type yet." in p
    assert "novelty threshold None" not in p


def test_retrieve_learned_prefers_the_profile_documents(kdir):
    k = retrieve_learned(
        "Exhaust Cold Spot", "spread trending up", "cold_spot_hardware",
        token_cap=700, folder=kdir,
        profile={"id": "exhaust_spread_trip", "docs": ["process_map_hets_trip.md"]},
    )
    refs = [x["ref"] for x in k]
    assert refs[0] == "process_map_hets_trip.md § Decision flow"
    assert "process map for exhaust_spread_trip" in k[0]["why"]
    assert not any("process_map_cold_spot.md" in ref for ref in refs)
    (kdir / "user").mkdir()
    (kdir / "user" / "site_filter.md").write_text(
        '<!-- sentinel-doc: {"title": "Site filter", "analytics": ["inlet_filter_dp"], "signatures": ["inlet_filter_fouling"]} -->\n'
        "# Site filter\n\n## Check the DP\nCompare the inlet differential pressure with the clean baseline.\n",
        encoding="utf-8",
    )
    k2 = retrieve_learned(
        "Inlet filter DP high", "", "inlet_filter_fouling",
        folder=kdir, profile={"id": "inlet_filter_dp", "docs": ["process_map_hets_trip.md"]},
    )
    refs2 = [x["ref"] for x in k2]
    assert refs2[0].startswith("process_map_hets_trip.md")
    user = next(i for i, ref in enumerate(refs2) if ref.startswith("user/site_filter.md"))
    assert user > 0 and "tagged for inlet_filter_dp" in k2[user]["why"]
    assert not any("process_map_cold_spot.md" in ref for ref in refs2)


def test_unused_gguf_files_are_listed_and_the_active_file_is_skipped(tmp_path):
    from sentinel.llm_runtime import unused_ggufs

    models = tmp_path / "models"
    active = models / "gemma-4-12b-it-Q4_K_M.gguf"
    active.parent.mkdir()
    active.write_bytes(b"active")
    other = models / "other.gguf"
    other.write_bytes(b"nope")
    nested = models / "extra" / "llama.gguf"
    nested.parent.mkdir()
    nested.write_bytes(b"x")
    found = {p.name for p in unused_ggufs(models, active)}
    assert found == {"other.gguf", "llama.gguf"}
    assert unused_ggufs(tmp_path / "missing", None) == []


def test_writer_rejects_a_signature_outside_the_profile(v2_settings, monkeypatch):
    from sentinel import llm_runtime
    from sentinel.e2e import writer

    seen = {}

    def chat(settings, system, user, schema, max_tokens=None):
        seen["enum"] = list(schema["properties"]["signature"]["enum"])
        llm_runtime.check_schema(schema)
        obj = {
            "signature": "cold_spot_hardware", "primary_action": "borescope", "urgency": "next_outage", "escalate": False,
            "what_happened": "The inlet differential pressure rose and stayed above its normal band.",
            "why": "The alert type only covers the inlet filter, so a cold sector is not in this catalogue.",
            "action_rationale": "Send the case to a human because the signature is outside this alert type.",
            "residual_consistency": "inconclusive", "residual_reason": "No expected pattern was scored.",
            "process_steps": [], "concerns": "",
        }
        return {"text": json.dumps(obj), "usage": {"prompt_tokens": 100, "completion_tokens": 40}, "load_s": 0.0, "gen_s": 0.0,
                "device": "cpu", "model_label": "fake", "model_file": "fake.gguf"}

    monkeypatch.setattr(llm_runtime, "chat_json", chat)
    out = writer.write(_report(_filter_profile()), v2_settings, _TAX)
    assert seen["enum"] == ["inlet_filter_fouling", "dp_transmitter_fault", "novel_pattern"]
    assert not out["ok"] and "outside the allowed enums" in out["error"]
