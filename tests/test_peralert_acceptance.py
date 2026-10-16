"""Per-alert acceptance gate on a 12-case fixture.

Decisions are fake. Nothing here loads a channel-independent model or scores
the 320 held-out rows. ``score_peralert_acceptance`` is the real scorer and
is not called.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT
from sentinel.e2e.model import Calibration, Head, SimilarityIndex, decide
from sentinel.e2e.taxonomy import NOVEL


def _load():
    spec = importlib.util.spec_from_file_location("eval_e2e_peralert", ROOT / "scripts" / "eval_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load()

G3 = {
    "sidecar_sha": True,
    "rows_match_manifest": True,
    "no_sidecar_id_in_bundle": True,
    "encoder_loaded": True,
    "head_tier": True,
    "both_runs": True,
}


def _cases():
    """Four normal, four cold-spot, four instability. One instability row is rule B."""
    rows = []
    for i in range(4):
        rows.append({
            "case_id": f"c{i:02d}",
            "alert_type": "exhaust_cold_spot",
            "truth": "normal",
            "frame": "7EA",
            "rule": "A",
        })
    for i in range(4, 8):
        rows.append({
            "case_id": f"c{i:02d}",
            "alert_type": "exhaust_cold_spot",
            "truth": "cold_spot_hardware",
            "frame": "7FA",
            "rule": "A",
        })
    for i in range(8, 12):
        rows.append({
            "case_id": f"c{i:02d}",
            "alert_type": "combustion_dynamics",
            "truth": "combustion_instability",
            "frame": "7EA",
            "rule": "B-owned" if i == 11 else "A",
        })
    return rows


def _correct(rows, **overrides):
    out = {row["case_id"]: row["truth"] for row in rows}
    out.update(overrides)
    return out


def _pair(run):
    return [dict(run), dict(run)]


def _sources(shipped, active_status="v0000", active_runs=None, reason="active legacy version is v0000; the bar is the shipped bundle"):
    return {
        "shipped": {"status": "scored", "runs": shipped},
        "active": {
            "status": active_status,
            "version": "v0000" if active_status == "v0000" else "v0002",
            "reason": reason,
            "runs": list(active_runs or []),
        },
    }


def _evaluate(rows, shipped, peralert, **kwargs):
    return MOD.evaluate_peralert(rows, _sources(shipped, **{k: v for k, v in kwargs.items() if k in {"active_status", "active_runs", "reason"}}), peralert, g3=dict(G3), keep_g2=True)


def test_counts_and_ties_hold():
    rows = _cases()
    run = _correct(rows)
    payload = _evaluate(rows, _pair(run), _pair(run))
    assert payload["passed"] is True
    assert payload["reasons"] == []
    assert payload["d1"] == "strict"
    assert payload["e2e_mode_switched"] is False
    cold = payload["g1"]["exhaust_cold_spot"]
    assert cold["n"] == 8
    assert cold["metrics"]["signature"]["verdict"] == "hold"
    assert cold["metrics"]["false_novelty"]["legacy"] == 0
    assert payload["cases"][-1]["original_name_signature"] == NOVEL
    assert payload["cases"][-1]["original_name_needs_human"] is True
    assert payload["cases"][0]["original_name_signature"] == "normal"
    assert payload["reported"]["rule_b"]["combustion_dynamics"]["n"] == 1


def test_a_signature_miss_fails_that_alert_type_only():
    rows = _cases()
    legacy = _correct(rows)
    per = _correct(rows, c00="cold_spot_hardware")
    payload = _evaluate(rows, _pair(legacy), _pair(per))
    assert payload["passed"] is False
    assert any(reason.startswith("G1 exhaust_cold_spot signature") for reason in payload["reasons"])
    assert not any(reason.startswith("G1 combustion_dynamics") for reason in payload["reasons"])
    assert payload["g1"]["combustion_dynamics"]["metrics"]["signature"]["verdict"] == "hold"


def test_legacy_best_of_uses_the_harder_count():
    rows = _cases()
    perfect = _correct(rows)
    shipped = _correct(rows, c08="normal")
    # Active is right everywhere, so the signature bar is 4, not the shipped 3.
    hold = _evaluate(
        rows,
        _pair(shipped),
        _pair(perfect),
        active_status="scored",
        active_runs=_pair(perfect),
        reason="active version v0002 is a second legacy source",
    )
    assert hold["passed"] is True
    assert hold["g1"]["combustion_dynamics"]["metrics"]["signature"]["legacy"] == 4
    short = _evaluate(
        rows,
        _pair(shipped),
        _pair(shipped),
        active_status="scored",
        active_runs=_pair(perfect),
        reason="active version v0002 is a second legacy source",
    )
    assert short["passed"] is False
    assert any(reason.startswith("G1 combustion_dynamics signature") for reason in short["reasons"])
    # False novelty: the harder bar is the source with fewer novel calls.
    shipped_novel = _correct(rows, c08=NOVEL)
    novel_hold = _evaluate(
        rows,
        _pair(shipped_novel),
        _pair(perfect),
        active_status="scored",
        active_runs=_pair(perfect),
        reason="active version v0002 is a second legacy source",
    )
    assert novel_hold["g1"]["combustion_dynamics"]["metrics"]["false_novelty"]["legacy"] == 0
    assert novel_hold["passed"] is True
    novel_fail = _evaluate(
        rows,
        _pair(shipped_novel),
        _pair(shipped_novel),
        active_status="scored",
        active_runs=_pair(perfect),
        reason="active version v0002 is a second legacy source",
    )
    assert novel_fail["passed"] is False
    assert any("false_novelty" in reason and reason.startswith("G1 combustion_dynamics") for reason in novel_fail["reasons"])


def test_unscored_active_version_fails_and_v0000_does_not():
    rows = _cases()
    run = _correct(rows)
    failed = _evaluate(
        rows,
        _pair(run),
        _pair(run),
        active_status="unscored",
        reason="active version v0002 cannot be scored",
    )
    assert failed["passed"] is False
    assert any("cannot be scored" in reason for reason in failed["reasons"])
    held = _evaluate(rows, _pair(run), _pair(run), active_status="v0000")
    assert held["passed"] is True
    assert held["legacy_active_used"] is False
    assert "v0000" in held["legacy_active"]["reason"]


def test_engineer_overlap_excludes_the_active_version():
    rows = _cases()
    # Shipped and per-alert both call one case novel. An excluded active run
    # that calls none novel must not tighten the bar.
    novel = _correct(rows, c08=NOVEL)
    perfect = _correct(rows)
    payload = _evaluate(
        rows,
        _pair(novel),
        _pair(novel),
        active_status="excluded",
        active_runs=_pair(perfect),
        reason="active version v0002 has engineer labels and overlaps the side-car (c08); left out of the bar",
    )
    assert payload["passed"] is True
    assert payload["legacy_active_used"] is False
    assert payload["g1"]["combustion_dynamics"]["metrics"]["false_novelty"]["legacy"] == 1
    status = MOD.legacy_active_status(
        {"version": "v0002"},
        {"n_engineer": 2},
        [{"id": "c08", "source": "engineer", "file": "deadbeef"}],
        {"c08"},
        {"deadbeef"},
    )
    assert status["status"] == "excluded"
    assert "c08" in status["reason"]
    clean = MOD.legacy_active_status(
        {"version": "v0002"},
        {"n_engineer": 0},
        [{"id": "train-row", "source": "synthetic"}],
        {"c08"},
        set(),
    )
    assert clean["status"] == "scored"
    base = MOD.legacy_active_status({"version": "v0000"}, None, [], set(), set())
    assert base["status"] == "v0000"
    absent = MOD.legacy_active_status(None, None, [], set(), set())
    assert absent["status"] == "absent"
    missing = MOD.legacy_active_status({"version": "v0002"}, None, [], set(), set())
    assert missing["status"] == "unscored"


def test_two_runs_use_the_worse_count_and_list_ids():
    rows = _cases()
    good = _correct(rows)
    bad = _correct(rows, c08=NOVEL)
    payload = _evaluate(rows, [good, bad], [good, bad])
    assert payload["passed"] is False
    block = payload["g1"]["combustion_dynamics"]["metrics"]
    # Legacy keeps the higher signature count and the lower false-novelty count.
    assert block["signature"]["legacy"] == 4
    assert block["signature"]["peralert"] == 3
    assert block["false_novelty"]["legacy"] == 0
    assert block["false_novelty"]["peralert"] == 1
    assert "c08" in payload["disagreeing_ids"]["legacy_shipped"]
    assert "c08" in payload["disagreeing_ids"]["peralert"]
    agreed = _evaluate(rows, _pair(good), _pair(good))
    assert agreed["passed"] is True
    assert agreed["disagreeing_ids"]["peralert"] == []


def test_g2_slice_can_fail_while_the_alert_type_ties():
    rows = _cases()
    legacy = _correct(rows, c00="cold_spot_hardware")
    # hets shares the cold-spot action, so the signature drops and the action does not.
    per = _correct(rows, c04="hets_trip_hardware")
    payload = _evaluate(rows, _pair(legacy), _pair(per))
    assert payload["passed"] is False
    assert payload["g1"]["exhaust_cold_spot"]["metrics"]["signature"]["verdict"] == "hold"
    assert payload["g1"]["exhaust_cold_spot"]["metrics"]["action"]["verdict"] == "better"
    assert not any(reason.startswith("G1 ") for reason in payload["reasons"])
    assert any(reason.startswith("G2 truth cold_spot_hardware signature") for reason in payload["reasons"])
    assert payload["g2"]["overall"]["metrics"]["signature"]["verdict"] == "hold"
    cold = payload["g2"]["by_truth"]["cold_spot_hardware"]
    assert cold["metrics"]["signature"]["verdict"] == "fail"
    assert cold["metrics"]["action"]["verdict"] == "hold"


def test_each_g3_flag_fails_the_gate():
    rows = _cases()
    run = _pair(_correct(rows))
    expected = dict(MOD._G3_REASONS)
    for key in G3:
        flags = dict(G3)
        flags[key] = False
        payload = MOD.evaluate_peralert(rows, _sources(run), run, g3=flags, keep_g2=True)
        assert payload["passed"] is False
        assert expected[key] in payload["reasons"]
        assert payload["g3"]["flags"][key] is False
        assert not any(reason.startswith("G1 ") for reason in payload["reasons"])


def test_switch_helper_requires_a_passing_match():
    hashes = {"exhaust_cold_spot": {"head.npz": "abc"}}
    sha = "ab" * 32
    results = {"passed": False, "bundle_hashes": hashes, "decision_path_sha": sha}
    assert MOD.acceptance_allows_switch(results, hashes, sha) is False
    results["passed"] = True
    assert MOD.acceptance_allows_switch(results, hashes, sha) is True
    assert MOD.acceptance_allows_switch(results, {"exhaust_cold_spot": {"head.npz": "def"}}, sha) is False
    assert MOD.acceptance_allows_switch(results, hashes, "cd" * 32) is False
    assert MOD.acceptance_allows_switch(results, hashes, "") is False


def test_d1_options_are_reported_and_do_not_gate():
    rows = _cases()
    # One extra novel call on a case legacy already got wrong. Accuracy ties.
    # Strict false novelty fails. Option (b) allows +1. Option (c) sees p = 0.5.
    legacy = _correct(rows, c00="cold_spot_hardware")
    per = _correct(rows, c00=NOVEL)
    payload = _evaluate(rows, _pair(legacy), _pair(per))
    assert payload["passed"] is False
    assert any("false_novelty" in reason for reason in payload["reasons"])
    assert payload["d1_options"]["applied"] == "strict"
    assert payload["d1_options"]["b_false_novelty_plus_one"]["passed"] is True
    assert payload["d1_options"]["c_sign_test"]["passed"] is True
    assert "not applied" in payload["d1_options"]["note"]
    cell = next(item for item in payload["d1_options"]["c_sign_test"]["cells"] if item["alert_type"] == "exhaust_cold_spot" and item["metric"] == "false_novelty")
    assert cell["n_worse"] == 1
    assert cell["p"] == 0.5
    assert cell["worse"] is False
    assert MOD.sign_test_p(1, 0) == 0.5
    assert MOD.sign_test_p(0, 4) == 1.0
    assert MOD.sign_test_p(5, 0) == pytest.approx(1 / 32)
    assert MOD.sign_test_p(4, 0) == pytest.approx(1 / 16)
    # Five discordant signature losses on one alert type. (c) calls that cell worse.
    wide = _correct(rows)
    for cid in ("c00", "c01", "c02", "c03", "c04"):
        wide[cid] = NOVEL
    signed = _evaluate(rows, _pair(_correct(rows)), _pair(wide))
    assert signed["d1_options"]["c_sign_test"]["passed"] is False
    sig_cell = next(item for item in signed["d1_options"]["c_sign_test"]["cells"] if item["alert_type"] == "exhaust_cold_spot" and item["metric"] == "signature")
    assert sig_cell["n_worse"] == 5
    assert sig_cell["worse"] is True


def test_legacy_decision_does_not_take_a_name():
    assert "name" not in inspect.signature(decide).parameters
    assert "analytic" not in inspect.signature(decide).parameters
    vector = np.array([1.0, 0.0], dtype=np.float32)
    head = Head(
        ["normal", "cold_spot_hardware"],
        np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    index = SimilarityIndex()
    index.add(vector, {"id": "near", "signature": "normal", "source": "synthetic"})
    cal = Calibration(novelty_threshold=0.5, match_threshold=0.2)
    first = decide(vector, head, index, cal)
    second = decide(vector, head, index, cal)
    assert first["signature"] == second["signature"] == "normal"
    assert MOD.legacy_name_columns(first["signature"]) == {"original_name": "normal", "sidecar_name": "normal"}
    assert MOD.original_name_signature("A", "normal") == "normal"
    assert MOD.original_name_signature("B-hash", "normal") == NOVEL
    assert MOD.act(NOVEL) == "escalate"


def test_legacy_decisions_score_a_frame_conditioned_version():
    """A retrained legacy version (head and index 672 + layout one-hot) is scored as the app serves it."""
    from sentinel.e2e.model import condition_vector

    layouts = ["ring_tc", "ring_ttxd", "zone_egt", "zone_named"]
    # Columns 0-1 are the embedding, 2-5 the layout one-hot. Only the one-hot separates the classes:
    # 7EA lights ring_tc (column 2), 7HA lights zone_egt (column 4).
    W = np.zeros((2, 6), dtype=np.float32)
    W[0, 2] = 10.0
    W[1, 4] = 10.0
    head = Head(["normal", "cold_spot_hardware"], W, np.zeros(2, dtype=np.float32),
                {"uses_frame": True, "frame_key": "layout", "frames": layouts})
    Z = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    index = SimilarityIndex(dim=6)
    for frame, sig in (("7EA", "normal"), ("7HA", "cold_spot_hardware")):
        index.add(condition_vector(Z[0], frame, layouts), {"id": frame, "signature": sig, "source": "synthetic"})
    cal = Calibration(novelty_threshold=0.5, match_threshold=0.2)
    with pytest.raises(ValueError):
        decide(Z[0], head, index, cal)
    got = MOD.legacy_decisions(Z, ["7EA", "7HA"], head, index, cal)
    assert [d["signature"] for d in got] == ["normal", "cold_spot_hardware"]
    assert not any(d["novel"] for d in got)

    # A head without a frame column is decided exactly as before.
    plain = Head(["normal", "cold_spot_hardware"], np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32), np.zeros(2, dtype=np.float32))
    flat = SimilarityIndex()
    flat.add(Z[0], {"id": "near", "signature": "normal", "source": "synthetic"})
    same = MOD.legacy_decisions(Z, ["7EA", "7HA"], plain, flat, cal)
    assert [d["signature"] for d in same] == [decide(z, plain, flat, cal)["signature"] for z in Z]


def test_decision_path_sha_is_stable_and_a_missing_name_fails():
    first = MOD.decision_path_sha()
    second = MOD.decision_path_sha()
    assert first == second
    assert len(first) == 64
    int(first, 16)
    with pytest.raises(ValueError, match="resolve_alert_type"):
        MOD.decision_function_sources("def alert_type_for_name():\n    return 1\n")


def test_bundle_scanner_sees_index_stats_and_heldout():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle = root / "profiles" / "exhaust_cold_spot"
        (bundle / "stats").mkdir(parents=True)
        (bundle / "index.jsonl").write_text('{"id": "side-id", "signature": "normal"}\n', encoding="utf-8")
        (bundle / "stats" / "index.jsonl").write_text('{"id": "stats-id", "signature": "normal"}\n', encoding="utf-8")
        np.savez_compressed(bundle / "heldout.npz", case_id=np.array(["held-id"]))
        hits = MOD.sidecar_ids_in_bundles(root, {"side-id", "stats-id", "held-id", "other-id"})
        assert "exhaust_cold_spot:side-id" in hits
        assert "exhaust_cold_spot:stats-id" in hits
        assert "exhaust_cold_spot:held-id" in hits
        assert not any("other-id" in hit for hit in hits)
        assert MOD.legacy_version_readable(bundle) is False
        for name in ("scaler.npz", "head.npz", "head.json", "index.npz", "index.jsonl", "calibration.json"):
            path = bundle / name
            if not path.exists():
                path.write_bytes(b"x")
        (bundle / "encoder").mkdir()
        (bundle / "encoder" / "config.json").write_text("{}", encoding="utf-8")
        assert MOD.legacy_version_readable(bundle) is True


def test_report_writer_stays_in_its_folder(tmp_path):
    rows = _cases()
    run = _correct(rows)
    payload = _evaluate(rows, _pair(run), _pair(run))
    payload["git_head"] = "abc"
    payload["bundle_hashes"] = {"exhaust_cold_spot": {"head.npz": "abc"}}
    payload["decision_path_sha"] = "ab" * 32
    dynamic = ROOT / "eval" / "EVAL_DYNAMIC.md"
    results = ROOT / "eval" / "results_dynamic.json"
    sidecar = ROOT / "eval" / "acceptance_peralert.jsonl"
    before = {
        "dynamic": dynamic.read_bytes() if dynamic.is_file() else None,
        "results": results.read_bytes() if results.is_file() else None,
        "sidecar": sidecar.read_bytes() if sidecar.is_file() else None,
    }
    MOD.write_peralert_outputs(payload, tmp_path)
    text = (tmp_path / "EVAL_PERALERT.md").read_text(encoding="utf-8")
    assert text.startswith("# Per-alert acceptance\n\n**PASS**\n")
    saved = json.loads((tmp_path / "results_peralert.json").read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["n"] == 12
    run_line = (tmp_path / "acceptance_runs.jsonl").read_text(encoding="utf-8").strip()
    logged = json.loads(run_line)
    assert logged["passed"] is True
    assert logged["git_head"] == "abc"
    assert not (tmp_path / "EVAL_DYNAMIC.md").exists()
    assert not (tmp_path / "results_dynamic.json").exists()
    assert (dynamic.read_bytes() if dynamic.is_file() else None) == before["dynamic"]
    assert (results.read_bytes() if results.is_file() else None) == before["results"]
    assert (sidecar.read_bytes() if sidecar.is_file() else None) == before["sidecar"]


def test_peralert_flag_is_on_the_eval_script():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_e2e.py"), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--peralert" in proc.stdout
    assert "--write-sidecar" in proc.stdout
    assert "--profiles" in proc.stdout
