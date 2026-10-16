"""Human review gate and the feedback loop: stored locally, changes future reports."""

from __future__ import annotations

import json

import pytest

from conftest import sample
from sentinel import store
from sentinel.llm import build_user_prompt
from sentinel.pipeline import run_case


def test_not_good_with_correction_changes_next_report(settings):
    first = run_case(sample("cold_spot", 1), "Exhaust Cold Spot", "Unit 2", settings=settings)
    assert first["decision"]["primary_action"] == "borescope"
    fb = store.record_feedback(
        settings,
        first["case_id"],
        "not_good",
        reason="Site found fuel gas control valve LVDT drift; cold sector cleared after calibration.",
        corrected_action="lvdt_calibration",
        reviewer="Reviewer",
    )
    assert fb["final_action"] == "lvdt_calibration"
    rep = store.load_report(settings, first["case_id"])
    assert rep["review"]["status"] == "REJECTED"
    md = (settings.cases_dir / first["case_id"] / "report.md").read_text(encoding="utf-8")
    assert "**REJECTED**" in md and "LVDT calibration" in md

    lines = settings.feedback_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["corrected_action"] == "lvdt_calibration"

    second = run_case(sample("cold_spot", 1), "Exhaust Cold Spot", "Unit 2", settings=settings)
    d = second["decision"]
    assert d["primary_action"] == "lvdt_calibration"
    assert d["rule_action"] == "borescope"
    assert "borescope" in d["secondary_actions"]
    assert "FB-OVERRIDE" in d["rules_fired"]
    assert d["needs_human"] is True
    sim = second["similar_cases"][0]
    assert sim["case_id"] == first["case_id"] and sim["similarity"] > 0.95
    # The correction and its reason reach the LLM prompt
    prompt = build_user_prompt(second)
    assert "LVDT drift" in prompt and "corrected action" in prompt


def test_good_feedback_confirms_and_raises_confidence(settings):
    first = run_case(sample("combustion_dynamics", 3), "combustion dynamics", "", settings=settings)
    store.record_feedback(settings, first["case_id"], "good", reviewer="Reviewer")
    assert store.load_report(settings, first["case_id"])["review"]["status"] == "APPROVED"
    second = run_case(sample("combustion_dynamics", 3), "combustion dynamics", "", settings=settings)
    d = second["decision"]
    assert d["primary_action"] == "combustion_tuning"
    assert "FB-CONFIRM" in d["rules_fired"]
    assert d["confidence"] >= first["decision"]["confidence"]
    assert any("endorsed by engineers" in f for f, _ in d["confidence_factors"])


def test_feedback_does_not_leak_to_dissimilar_cases(settings):
    first = run_case(sample("cold_spot", 1), "cold spot", "", settings=settings)
    store.record_feedback(settings, first["case_id"], "not_good", corrected_action="lvdt_calibration")
    other = run_case(sample("cold_spot", 4), "cold spot", "", settings=settings)  # much deeper sector
    assert other["decision"]["primary_action"] == "borescope"
    assert "FB-OVERRIDE" not in other["decision"]["rules_fired"]


def test_feedback_is_scoped_to_the_analytic(settings):
    first = run_case(sample("hets_trip", 1), "HETS trip", "", settings=settings)
    store.record_feedback(settings, first["case_id"], "not_good", corrected_action="sensor_fix_replace")
    other = run_case(sample("cold_spot", 1), "cold spot", "", settings=settings)
    assert other["similar_cases"] == []


def test_rejection_without_correction_lowers_confidence(settings):
    first = run_case(sample("hets_trip", 4), "HETS trip", "", settings=settings)
    store.record_feedback(settings, first["case_id"], "not_good", reason="Report too generic for site.")
    second = run_case(sample("hets_trip", 4), "HETS trip", "", settings=settings)
    assert "FB-REJECT" in second["decision"]["rules_fired"]
    assert second["decision"]["confidence"] < first["decision"]["confidence"]


def test_invalid_feedback_rejected(settings):
    r = run_case(sample("cold_spot", 2), "cold spot", "", settings=settings)
    with pytest.raises(ValueError):
        store.record_feedback(settings, r["case_id"], "maybe")
    with pytest.raises(ValueError):
        store.record_feedback(settings, r["case_id"], "not_good", corrected_action="replace_everything")
    assert store.load_report(settings, r["case_id"])["review"]["status"] == "DRAFT"


def test_export_preferences(settings, tmp_path):
    a = run_case(sample("cold_spot", 2), "cold spot", "", settings=settings)
    b = run_case(sample("hets_trip", 2), "HETS trip", "", settings=settings)
    run_case(sample("hets_trip", 3), "HETS trip", "", settings=settings)  # left as DRAFT
    store.record_feedback(settings, a["case_id"], "good")
    store.record_feedback(settings, b["case_id"], "not_good", corrected_action="sensor_fix_replace", reason="TC harness")
    out = tmp_path / "prefs.jsonl"
    assert store.export_preferences(settings, out) == 2
    recs = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert {r["label"] for r in recs} == {"chosen", "rejected"}
    assert all(r["response"] for r in recs)
