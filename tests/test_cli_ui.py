"""CLI round trip (subprocess) and the NiceGUI review form."""

from __future__ import annotations

import os
import subprocess
import sys

from conftest import ROOT, sample
from ui_support import browser, run, set_value


def _cli(*args, data_dir):
    env = dict(os.environ, SENTINEL_USE_TSPULSE="0", PYTHONIOENCODING="utf-8")
    p = subprocess.run(
        [sys.executable, str(ROOT / "app.py"), "--data-dir", str(data_dir), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(ROOT),
        timeout=300,
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


def _field(out, key):
    for line in out.splitlines():
        if line.startswith(key + "\t"):
            return line.split("\t")[1:]
    raise AssertionError(f"{key} not in output:\n{out}")


def test_cli_run_feedback_cases(tmp_path):
    out = _cli("run", "--csv", str(sample("hets_trip", 5)), "--analytic", "HETS trip", "--notes", "GT2 first-out exhaust spread", "--no-llm", "--out", str(tmp_path / "out"), data_dir=tmp_path / "data")
    case_id = _field(out, "CASE")[0]
    assert _field(out, "ACTION")[0] == "borescope"
    assert _field(out, "STATUS")[0] == "DRAFT"
    assert (tmp_path / "out" / "hets_trip_hets_trip_05.md").exists()
    out = _cli("feedback", "--case", case_id, "--verdict", "good", "--reviewer", "tester", data_dir=tmp_path / "data")
    assert "APPROVED" in out
    out = _cli("cases", data_dir=tmp_path / "data")
    assert case_id in out and "APPROVED" in out
    assert "borescope" in _cli("actions", data_dir=tmp_path / "data")


def test_ui_renders_case_and_takes_review(tmp_path, monkeypatch):
    from sentinel import store
    from sentinel.config import load_settings
    from sentinel.pipeline import run_case

    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SENTINEL_USE_LLM", "0")
    monkeypatch.setenv("SENTINEL_USE_TSPULSE", "0")
    s = load_settings({"use_tspulse": False})
    r = run_case(sample("cold_spot", 6), "Exhaust Cold Spot", "Unit 5", settings=s)

    async def body():
        async with browser() as user:
            await user.open(f"/?case={r['case_id']}")
            await user.should_see("DRAFT")
            await user.should_see("Borescope inspection")
            user.find("Edit").click()
            set_value(user, "verdict", "Not good")
            set_value(user, "corrected", "sensor_fix_replace")
            set_value(user, "reason", "TC harness damage found")
            set_value(user, "reviewer", "UI tester")
            user.find(marker="submit").click()

    run(body)
    rep = store.load_report(s, r["case_id"])
    assert rep["review"]["status"] == "REJECTED"
    assert rep["review"]["feedback"]["corrected_action"] == "sensor_fix_replace"
    assert rep["review"]["feedback"]["reviewer"] == "UI tester"
