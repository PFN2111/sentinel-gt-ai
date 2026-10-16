"""LLM layer: prompt, parsing and output checks (offline), plus an opt-in real-model test."""

from __future__ import annotations

import os

import pytest

from conftest import sample
from sentinel.config import load_settings
from sentinel.llm import SCHEMA, build_user_prompt, check_output, parse_json
from sentinel.pipeline import run_case


def test_parse_json_valid_and_invalid():
    ok = '{"what_happened": "a", "why": "b", "action_rationale": "c", "supports_action": true, "concerns": ""}'
    assert parse_json("noise " + ok + " trailing")["why"] == "b"
    with pytest.raises(ValueError):
        parse_json("no json here")
    with pytest.raises(ValueError):
        parse_json('{"what_happened": "a"}')
    with pytest.raises(ValueError):
        parse_json(ok.replace("true", '"yes"'))


@pytest.mark.parametrize("text,is_null", [("", True), ("No concerns; the facts fully support it.", True), ("None", True), ("N/A", True), ("The facts support the action.", True), ("Only zone data, sector is coarse.", False)])
def test_null_concern_detection(text, is_null):
    from sentinel.llm import _NULL_CONCERN

    assert bool(_NULL_CONCERN.match(text)) is is_null or (text == "" and is_null)


def test_check_output_flags_untraced_numbers_and_missing_action():
    prompt = "Peak 26.3 C at 2025-12-28 01:00, spread 29.7 C, fuel +1.76%"
    good = {"what_happened": "TC4 ran 26.3 C cold; spread 29.7 C.", "why": "fuel +1.76%", "action_rationale": "A borescope inspection is needed."}
    checks, flags = check_output(good, prompt, "borescope")
    assert checks["action named in rationale"] == "yes" and not flags
    bad = {"what_happened": "Spread hit 88.4 C and 71.2 C.", "why": "", "action_rationale": "Do something."}
    checks, flags = check_output(bad, prompt, "borescope")
    assert checks["action named in rationale"] == "NO"
    assert any("numbers not found" in f for f in flags)


def test_prompt_contains_facts_and_decision(settings):
    r = run_case(sample("hets_trip", 1), "HETS trip", "GT1 first-out HETS", settings=settings)
    user = build_user_prompt(r)
    assert "DECISION (rules engine" in user and "Borescope" in user
    assert "Evidence:" in user and "process map" in user.lower()
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"])


@pytest.mark.skipif(os.environ.get("SENTINEL_RUN_LLM_TESTS") != "1", reason="set SENTINEL_RUN_LLM_TESTS=1 to run the real local LLM (minutes on CPU)")
def test_real_llm_narrative(tmp_path):
    s = load_settings({"data_dir": tmp_path, "use_tspulse": False})
    if not s.llm_ready():
        pytest.skip("GGUF / llama-cpp-python not available")
    r = run_case(sample("cold_spot", 2), "Exhaust Cold Spot", "Unit 2", settings=s, use_llm=True)
    n = r["narrative"]
    assert n["engine"] == "llm", r["audit"]["narrative_engine"]
    assert n["checks"]["json"].startswith("valid")
    assert r["decision"]["primary_action"] == "borescope"  # LLM cannot change the rule action
