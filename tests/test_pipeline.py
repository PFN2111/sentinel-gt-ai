"""End-to-end pipeline in offline mode (no GGUF): reports, gates, uncertainty handling."""

from __future__ import annotations

import json

import pytest

from conftest import ROOT, sample
from sentinel.config import load_settings
from sentinel.pipeline import run_case

SECTIONS = [
    "## 1. Summary",
    "## 2. Recommended corrective action",
    "## 3. What happened",
    "## 4. Why — engineering assessment",
    "## 5. Evidence",
    "## 6. Confidence and uncertainty",
    "## 7. Similar past cases",
    "## 8. Audit trail",
    "## 9. Engineer review",
]


@pytest.mark.parametrize(
    "kind,i,analytic,action",
    [
        ("cold_spot", 1, "Exhaust Cold Spot", "borescope"),
        ("combustion_dynamics", 4, "Combustion Dynamics High", "combustion_tuning"),
        ("hets_trip", 2, "HETS Trip", "borescope"),
    ],
)
def test_offline_report_complete(settings, kind, i, analytic, action):
    r = run_case(sample(kind, i), analytic, "Unit 2 7FA.03, no recent work.", settings=settings)
    d = r["decision"]
    assert r["analytic"]["resolved"] == kind
    assert d["primary_action"] == action
    assert r["review"]["status"] == "DRAFT"
    assert r["narrative"]["engine"] == "template"
    assert r["audit"]["narrative_engine"].startswith("template")
    assert d["confidence"] >= 0.75 and d["needs_human"] is False
    assert r["case"]["unit"] == "Unit 2"
    cdir = settings.cases_dir / r["case_id"]
    for name in ("report.md", "report.json", "case.json", "input.csv"):
        assert (cdir / name).exists()
    md = (cdir / "report.md").read_text(encoding="utf-8")
    for s in SECTIONS:
        assert s in md
    assert "DRAFT — pending Tier 2 engineer review" in md
    assert "SHA-256" in md and d["rules_fired"]
    # every evidence line carries a rule / model source
    assert all(e["source"] for e in r["findings"]["evidence"])
    assert json.loads((cdir / "report.json").read_text(encoding="utf-8"))["case_id"] == r["case_id"]


def test_hets_escalates_and_gives_restart_guidance(settings):
    r = run_case(sample("hets_trip", 1), "HETS trip", "First-out: high exhaust spread.", settings=settings)
    d = r["decision"]
    assert d["escalate"] is True
    assert d["severity"] == "high"
    assert any("restart" in g.lower() for g in r["findings"]["operating_guidance"])


def test_missing_gguf_falls_back_to_template(tmp_path):
    s = load_settings({"data_dir": tmp_path, "gguf_path": tmp_path / "missing.gguf", "use_llm": True, "use_tspulse": False})
    r = run_case(sample("cold_spot", 3), "cold spot", "", settings=s)
    assert r["narrative"]["engine"] == "template"
    assert "LLM not available" in r["audit"]["narrative_engine"]
    assert r["decision"]["primary_action"] == "borescope"


def test_unknown_analytic_matching_known_signature(settings):
    r = run_case(sample("cold_spot", 5), "Exhaust temperature deviation advisory", "", settings=settings)
    d = r["decision"]
    assert r["analytic"]["resolved"] == "generic" and r["analytic"]["known"] is False
    assert r["findings"]["signature"]["matched_signature"] == "cold_spot"
    assert d["needs_human"] is True
    assert d["confidence"] <= 0.40
    assert d["primary_action"] == "borescope"  # tentative, from the matched signature
    assert any("not one Sentinel has rules for" in f for f in d["review_flags"])


def test_unknown_analytic_short_data_is_flagged(settings):
    r = run_case(ROOT / "samples" / "gt_sensors_demo.csv", "Bearing vibration high", "", settings=settings)
    d = r["decision"]
    assert d["needs_human"] is True
    assert d["confidence"] < 0.5
    assert any("short data" in f.lower() for f in d["review_flags"])
    assert d["primary_action"] in ("monitor_only", "escalate")


def test_wrong_analytic_for_data_is_cross_checked(settings):
    r = run_case(sample("combustion_dynamics", 6), "Exhaust cold spot", "", settings=settings)
    d = r["decision"]
    assert d["needs_human"] is True
    assert "TR-XCHK" in d["rules_fired"]
    assert any("combustion_dynamics" in f for f in d["review_flags"])


def test_required_channels_missing_escalates(settings):
    r = run_case(ROOT / "samples" / "gt_sensors_demo.csv", "cold spot", "", settings=settings)
    d = r["decision"]
    assert d["primary_action"] == "escalate"
    assert d["needs_human"] is True
    assert any("Required channels missing" in f for f in d["review_flags"])


def test_novel_anomaly_is_flagged(settings, tmp_path):
    import pandas as pd

    df = pd.read_csv(sample("cold_spot", 7))
    n = len(df)
    df.loc[int(n * 0.8) : int(n * 0.85), "lube_oil_C"] += 6.0  # unrelated lube-oil excursion
    p = tmp_path / "cold_spot_plus_lube.csv"
    df.to_csv(p, index=False)
    r = run_case(p, "cold spot", "", settings=settings)
    d = r["decision"]
    assert d["primary_action"] == "borescope"
    assert "AN-NOVEL" in d["rules_fired"]
    assert d["needs_human"] is True


def test_notes_history_changes_dynamics_decision(settings):
    r = run_case(sample("combustion_dynamics", 2), "combustion dynamics", "Unit 4 was re-tuned last month; dynamics are back again.", settings=settings)
    d = r["decision"]
    assert d["escalate"] is True
    assert any("recently tuned" in f for f in d["review_flags"])


def test_tspulse_corroborates_when_available(tmp_path):
    s = load_settings({"data_dir": tmp_path, "use_llm": False, "use_tspulse": True})
    from sentinel.anomaly import load_tspulse

    if load_tspulse(s.tspulse_dir) is None:
        pytest.skip("TS Pulse weights / granite-tsfm not available on this machine")
    r = run_case(sample("combustion_dynamics", 1), "combustion dynamics", "", settings=s)
    assert r["anomaly"]["tspulse_status"] == "ok"
    assert "AN-TSP" in r["decision"]["rules_fired"]
