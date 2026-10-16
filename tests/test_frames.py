"""Part E: frame layouts, natural tag order, Teach-added alerts, and the slice promotion gate.

The full TS Pulse fine-tune is not run here. These tests cover the tag map, the one-hot that a
frame-conditioned head appends, Teach rollback of a new frame and a new alert, the report path
label, and the rule that a large slice regression blocks promotion while a tiny slice does not.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT, sample
from sentinel.config import frame_of_row, load_settings, load_taught_frames
from sentinel.data import apply_frame, map_roles
from sentinel.e2e.model import Head, novelty_limit, project_embedding
from sentinel.e2e.train import not_worse

MODEL_DIR = ROOT / "models" / "e2e"
needs_model = pytest.mark.skipif(
    not ((MODEL_DIR / "legacy" / "base" / "head.npz").is_file() or (MODEL_DIR / "base" / "head.npz").is_file()),
    reason="learned model not trained (scripts/train_e2e.py)",
)

# Synthetic historian files land on these four models so each trained one-hot slot stays distinct.
SYNTH_FRAMES = {"7EA", "7FA", "7HA", "9HA"}


def _script_generator():
    spec = importlib.util.spec_from_file_location("gen_script", ROOT / "scripts" / "generate_scenario_csvs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ttxd_tags_sort_by_the_full_sensor_number():
    """TTXD_10 used to sort as if its number were 0, which scrambled ring neighbours."""
    _roles, ring, _zones = map_roles(["TTXD_10", "TTXD_2", "TTXD_1", "TTXD_20"])
    assert ring == ["TTXD_1", "TTXD_2", "TTXD_10", "TTXD_20"]
    _roles, tc_ring, _zones = map_roles(["TC10", "TC2", "TC1"])
    assert tc_ring == ["TC1", "TC2", "TC10"]


def test_bank_form_ttxd_tags_stay_on_the_ring_in_sensor_order():
    """TTXD_1_05 / TTXD2_05 were ring tags before the natural-sort fix and must stay ring tags."""
    _roles, ring, _zones = map_roles(["TTXD_1_10", "TTXD_1_02", "TTXD_1_01", "DWATT"])
    assert ring == ["TTXD_1_01", "TTXD_1_02", "TTXD_1_10"]
    _roles, ring2, _zones = map_roles(["TTXD2_12", "TTXD2_3"])
    assert ring2 == ["TTXD2_3", "TTXD2_12"]


def test_frame_map_puts_historian_tags_on_the_canonical_roles():
    from sentinel.config import CUSTOM_FRAME, frame_maps_for

    names = ["TTXD_10", "TTXD_2", "DWATT", "FQG", "TTXD_1", "CSGV", "TTXD_20"]
    roles, ring, zones, fid = apply_frame(names, "ring_ttxd")
    assert fid == "7FA"
    assert ring == ["TTXD_1", "TTXD_2", "TTXD_10", "TTXD_20"]
    assert zones == []
    assert roles["load"] == ["DWATT"] and roles["fuel"] == ["FQG"] and roles["igv"] == ["CSGV"]
    same = apply_frame(names, "7FA")
    assert same[1] == ring and same[3] == "7FA"

    zroles, zring, zzones, zid = apply_frame(["EGT_Z10_C", "EGT_Z2_C", "load_MW", "EGT_Z1_C"], "zone_egt")
    assert zid == "7HA" and zring == []
    assert zzones == ["EGT_Z1_C", "EGT_Z2_C", "EGT_Z10_C"]
    assert zroles["load"] == ["load_MW"]
    named = apply_frame(["EGT_ZONE1", "EGT_ZONE10", "DWATT", "EGT_ZONE2"], "zone_named")
    assert named[3] == "9HA" and named[2] == ["EGT_ZONE1", "EGT_ZONE2", "EGT_ZONE10"]

    custom = frame_maps_for(None, CUSTOM_FRAME, "ring_ttxd")
    croles, cring, _cz, cid = apply_frame(names, CUSTOM_FRAME, custom)
    assert cid == CUSTOM_FRAME and cring == ring and croles["load"] == ["DWATT"]


def test_taught_frame_keeps_the_order_written_in_the_table():
    taught = {"site_unit3": {"description": "site", "ring": ["TTXD_2", "TTXD_1"], "zones": [], "roles": {"load": ["DWATT"]}}}
    roles, ring, _zones, fid = apply_frame(["TTXD_1", "DWATT", "TTXD_2"], "site_unit3", taught)
    assert fid == "site_unit3" and ring == ["TTXD_2", "TTXD_1"] and roles["load"] == ["DWATT"]


def test_notes_machine_model_is_a_frame_id_when_it_is_one():
    """7FA is a frame model, so it wins over layout. A name that is not a model still uses layout + alt_tags."""
    assert frame_of_row({"frame": "7FA", "layout": "ring", "alt_tags": True}) == "7FA"
    assert frame_of_row({"frame": "7FA", "layout": "zones", "alt_tags": False}) == "7FA"
    assert frame_of_row({"frame": "ring_tc", "layout": "zones"}) == "7EA"
    assert frame_of_row({"frame_type": "zone_named", "frame": "ring_tc"}) == "9HA"
    assert frame_of_row({"frame": "LM2500", "layout": "ring", "alt_tags": False}) == "7EA"
    assert frame_of_row({"frame": "LM2500"}) == "LM2500"


def test_project_embedding_conditions_only_a_frame_head_and_does_not_double_apply():
    bare = np.ones(4, np.float32)
    bare = bare / np.linalg.norm(bare)
    shipped = Head(["a"], np.zeros((1, 4), np.float32), np.zeros(1), {})
    assert project_embedding(bare, "ring_ttxd", shipped).shape == (4,)
    cond = Head(["a"], np.zeros((1, 6), np.float32), np.zeros(1), {"uses_frame": True, "frames": ["ring_tc", "ring_ttxd"]})
    out = project_embedding(bare, "ring_ttxd", cond)
    assert out.shape == (6,)
    assert np.allclose(project_embedding(bare, "7FA", cond), out)
    assert np.allclose(project_embedding(out, "ring_tc", cond), out)
    cal_frames = type("C", (), {"novelty_threshold": 0.5, "frame_thresholds": {"ring_ttxd": 0.2}})()
    assert novelty_limit(cal_frames, "ring_ttxd") == 0.2
    assert novelty_limit(cal_frames, "7FA") == 0.2
    assert novelty_limit(cal_frames, "zone_egt") == 0.5
    assert novelty_limit(cal_frames, None) == 0.5


def test_slice_regression_blocks_and_a_small_slice_does_not():
    overall = {"action_acc": 0.9, "signature_acc": 0.9}
    big = {"n": 20, "action_acc": 0.9, "signature_acc": 0.9}
    before = {**overall, "by_frame": {"ring_tc": big}, "by_alert": {"normal": big}}
    dropped = {**overall, "by_frame": {"ring_tc": {**big, "action_acc": 0.8}}, "by_alert": {"normal": big}}
    assert not not_worse(before, dropped)
    alert_drop = {**overall, "by_frame": {"ring_tc": big}, "by_alert": {"normal": {**big, "signature_acc": 0.7}}}
    assert not not_worse(before, alert_drop)
    small_before = {**overall, "by_frame": {"ring_ttxd": {"n": 3, "action_acc": 0.9, "signature_acc": 0.9}}}
    small_after = {**overall, "by_frame": {"ring_ttxd": {"n": 3, "action_acc": 0.1, "signature_acc": 0.1}}}
    assert not_worse(small_before, small_after)
    # A slice present on only one side does not block.
    assert not_worse(overall, {**overall, "by_frame": {"zone_egt": {**big, "action_acc": 0.1}}})


def test_force_overrides_a_failed_slice_gate_and_names_the_reason():
    from sentinel.e2e.engine import Engine

    before = {"action_acc": 0.9, "signature_acc": 0.9, "by_frame": {"ring_tc": {"n": 12, "action_acc": 0.9, "signature_acc": 0.9}}}
    after = {"action_acc": 0.9, "signature_acc": 0.9, "by_frame": {"ring_tc": {"n": 12, "action_acc": 0.5, "signature_acc": 0.9}}}
    ok, reason = Engine._gate(None, before, after, False)
    assert ok is False and reason == ""
    forced, why = Engine._gate(None, before, after, True)
    assert forced is True and "slice" in why and "--force" in why


def test_run_help_exposes_frame_on_both_paths():
    exe = [sys.executable, str(ROOT / "app.py")]
    rule = subprocess.run(exe + ["run", "--help"], capture_output=True, text=True, encoding="utf-8")
    learned = subprocess.run(exe + ["e2e", "run", "--help"], capture_output=True, text=True, encoding="utf-8")
    assert rule.returncode == 0 and "--frame" in rule.stdout and "7FA" in rule.stdout
    assert learned.returncode == 0 and "--frame" in learned.stdout and "7FA" in learned.stdout


def test_rule_report_states_the_generic_fallback_and_the_frame(settings):
    from sentinel.pipeline import run_case
    from sentinel.report import render_markdown

    report = run_case(sample("cold_spot", 1), "Inlet filter high dp", "Unit 3", settings=settings, use_llm=False, save=False)
    path = report["audit"]["analyzer_path"]
    # The rule path does not run the learned model, so its label must not claim it.
    assert path == "generic fallback (no built-in analyzer; generic analyzer + process knowledge)"
    assert report["case"]["frame_type"]
    text = render_markdown(report)
    assert "generic fallback (no built-in analyzer; generic analyzer + process knowledge)" in text
    assert "Frame:" in text
    known = run_case(sample("cold_spot", 1), "Exhaust Cold Spot", "", settings=settings, use_llm=False, save=False, frame="ring_tc")
    assert known["audit"]["analyzer_path"].startswith("built-in analyzer (")
    assert known["case"]["frame_type"] == "7EA"


def test_teach_adds_a_frame_and_rollback_removes_it(tmp_path):
    from sentinel.e2e import teach

    ts = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    fid, version = teach.save_frame(ts, "Site unit 3", "18 exhaust TCs", ["TTXD_2", "TTXD_1"], [], {"load": ["DWATT"]})
    assert fid == "site_unit_3" and version == "k0001"
    stored = load_taught_frames(ts.data_dir)
    assert stored[fid]["ring"] == ["TTXD_2", "TTXD_1"] and stored[fid]["roles"]["load"] == ["DWATT"]
    with pytest.raises(ValueError):
        teach.save_frame(ts, "empty", "", [], [], {})
    # A misspelt role would otherwise be stored and silently ignored by the tag map.
    with pytest.raises(ValueError, match="fuel_flow"):
        teach.save_frame(ts, "typo", "", [], [], {"fuel_flow": ["FQG"]})
    fid2, _v = teach.save_frame(ts, "caps", "", [], [], {"Load": ["DWATT"]})
    assert load_taught_frames(ts.data_dir)[fid2]["roles"] == {"load": ["DWATT"]}
    teach.TeachRegistry(ts).rollback()
    reg = teach.TeachRegistry(ts)
    assert reg.rollback() == "k0000"
    assert load_taught_frames(ts.data_dir) == {}
    reg.activate(version)
    assert fid in load_taught_frames(ts.data_dir)


def test_teach_adds_an_alert_type_with_a_process_document(tmp_path):
    from sentinel.e2e import teach
    from sentinel.e2e.taxonomy import load_taxonomy
    from sentinel.knowledge import retrieve

    kd = tmp_path / "knowledge"
    kd.mkdir()
    ts = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False, "knowledge_dir": str(kd)})
    with pytest.raises(ValueError):
        teach.add_alert_type(ts, "Inlet filter high dp", "dp climbed", "monitor_only", "  ")
    out = teach.add_alert_type(
        ts, "Inlet filter high dp", "Inlet differential pressure climbed at steady load", "monitor_only",
        "Check the inlet dp transmitter, then inspect the filters.",
    )
    sid = out["signature"]
    assert sid == "inlet_filter_high_dp"
    assert load_taxonomy(ts.data_dir)[sid]["action"] == "monitor_only"
    _meta, body = teach.read_document(ts, out["document"])
    assert "inspect the filters" in body
    hits = retrieve("inlet_filter_high_dp", "inlet differential", folder=ts.knowledge_dir)
    assert hits and hits[0]["ref"].startswith("user/")
    reg = teach.TeachRegistry(ts)
    reg.rollback()
    assert teach.list_documents(ts) == []
    assert sid in load_taxonomy(ts.data_dir)
    reg.rollback()
    assert sid not in load_taxonomy(ts.data_dir)


def test_generic_fallback_retrieves_the_new_alerts_process_document(tmp_path):
    """A case raised under a Teach-added alert name must see that alert's document on both paths,
    even though the classifier predicts some other signature and no built-in analyzer matches."""
    from sentinel.e2e import teach
    from sentinel.knowledge import retrieve, retrieve_learned
    from sentinel.pipeline import run_case

    kd = tmp_path / "knowledge"
    shutil.copytree(ROOT / "knowledge", kd, ignore=shutil.ignore_patterns("user"))
    ts = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False, "knowledge_dir": str(kd)})
    teach.add_alert_type(ts, "Inlet filter high dp", "Inlet differential pressure climbed at steady load", "monitor_only",
                         "## Inlet filter dp\nCheck the inlet dp transmitter first, then inspect the filter elements.")
    learned = retrieve_learned("Inlet filter high dp", "Unit 3", "cold_spot_hardware", folder=kd)
    assert learned and learned[0]["ref"].startswith("user/")
    rule = retrieve("", "Inlet filter high dp exhaust cold sector", folder=kd, alert_name="Inlet filter high dp")
    assert rule and rule[0]["ref"].startswith("user/")
    report = run_case(sample("cold_spot", 1), "Inlet filter high dp", "Unit 3", settings=ts, use_llm=False, save=False)
    assert any(k["ref"].startswith("user/") for k in report["knowledge"])


@needs_model
def test_heldout_without_a_frame_column_gets_frame_slices_from_the_manifest(tmp_path):
    """The shipped held-out file has no frame column. Without the manifest lookup the 'before' side of the
    first full retrain had no frame slices, so the gate could not block a frame regression."""
    from sentinel.e2e.engine import Engine

    s = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    eng = Engine(s)
    if "frame" in eng.heldout:
        pytest.skip("shipped held-out already carries frames")
    assert eng.heldout_eval()["by_frame"] == {}
    ids = [str(c) for c in eng.heldout["case_id"]]
    root = tmp_path / "data" / "e2e" / "dataset"
    root.mkdir(parents=True)
    rows = [{"case_id": c, "layout": "ring" if i % 2 else "zones", "alt_tags": i % 3 == 0} for i, c in enumerate(ids)]
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ev = eng.heldout_eval()
    assert set(ev["by_frame"]) == SYNTH_FRAMES
    assert sum(v["n"] for v in ev["by_frame"].values()) == len(ids)


def test_synthetic_cases_cover_ring_and_zone_layouts_and_match_the_embedded_copy():
    from sentinel_gt_ai import exec_section

    script = _script_generator()
    embedded = exec_section("sentinel.generator")
    layouts, frames = set(), set()
    for i in range(24):
        job = {"split": "train", "signature": "normal", "variant": None, "index": i}
        _df, meta = script.make_case(job)
        _edf, emeta = embedded.make_case(job)
        assert meta["frame"] == emeta["frame"]
        assert meta["frame"] in SYNTH_FRAMES
        if meta["layout"] == "zones":
            assert meta["frame"] == ("9HA" if meta["alt_tags"] else "7HA")
        else:
            assert meta["layout"] == "ring"
            assert meta["frame"] == ("7FA" if meta["alt_tags"] else "7EA")
        layouts.add(meta["layout"])
        frames.add(meta["frame"])
    assert layouts == {"ring", "zones"}
    assert len(frames) >= 2


@needs_model
def test_learned_report_states_the_generic_fallback(settings):
    from sentinel.e2e.pipeline import analyze_case
    from sentinel.e2e.report import render_markdown

    unknown = analyze_case(sample("cold_spot", 1), "Inlet filter high dp", "Unit 3", settings=settings, use_llm=False, save=False)
    path = unknown["audit"]["analyzer_path"]
    assert path == "generic fallback (no built-in analyzer; learned model + process knowledge)"
    assert unknown["audit"]["frame_type"]
    assert "slices" in unknown["audit"]
    text = render_markdown(unknown)
    assert "generic fallback (no built-in analyzer; learned model + process knowledge)" in text
    assert "Held-out slices:" in text
    known = analyze_case(sample("cold_spot", 1), "Exhaust Cold Spot", "", settings=settings, use_llm=False, save=False, frame="ring_tc")
    assert known["audit"]["analyzer_path"].startswith("built-in analyzer (")
    assert known["case"]["frame_type"] == "7EA" and known["input"]["frame_type"] == "7EA"


def test_shipped_frame_list_is_the_model_dropdown():
    from sentinel.config import CUSTOM_FRAME, FRAME_MODELS, LAYOUTS, frame_vocab, normalize_frame_arg

    shipped = [name for name in frame_vocab() if name in FRAME_MODELS]
    assert shipped == ["6B", "7EA", "7FA", "7HA", "9E", "9FA", "9HA", CUSTOM_FRAME]
    for name, spec in FRAME_MODELS.items():
        if name == CUSTOM_FRAME:
            assert spec["layout"] == ""
            continue
        assert spec["layout"] in LAYOUTS
        assert spec["exhaust_tc"] > 0
    assert FRAME_MODELS["6B"]["layout"] == FRAME_MODELS["7EA"]["layout"] == FRAME_MODELS["9E"]["layout"] == "ring_tc"
    assert FRAME_MODELS["7FA"]["layout"] == FRAME_MODELS["9FA"]["layout"] == "ring_ttxd"
    assert FRAME_MODELS["7HA"]["layout"] == "zone_egt" and FRAME_MODELS["9HA"]["layout"] == "zone_named"
    assert FRAME_MODELS["6B"]["exhaust_tc"] < FRAME_MODELS["7FA"]["exhaust_tc"] < FRAME_MODELS["7HA"]["exhaust_tc"]
    assert FRAME_MODELS["9FA"]["exhaust_tc"] < FRAME_MODELS["9HA"]["exhaust_tc"]
    assert normalize_frame_arg("7fa") == "7FA"
    assert normalize_frame_arg("ring_tc") == "7EA"
    assert normalize_frame_arg("") is None
    assert normalize_frame_arg("other") == CUSTOM_FRAME
    with pytest.raises(ValueError, match="7FA"):
        normalize_frame_arg("no-such-frame")


def test_old_layout_ids_migrate_on_load_and_the_files_stay(tmp_path):
    """labels.jsonl, frames.json, the similarity index and a head all rename old ids in memory only."""
    from sentinel.config import load_settings, load_taught_frames, migrate_frame_table
    from sentinel.e2e.labels import LabelStore
    from sentinel.e2e.model import Head, SimilarityIndex

    s = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    lab = LabelStore(s)
    lab.file.parent.mkdir(parents=True, exist_ok=True)
    lab.file.write_text(json.dumps({"case_id": "c1", "frame_type": "ring_ttxd", "signature": "normal", "note": "keep"}) + "\n", encoding="utf-8")
    rows = lab.all()
    assert rows[0]["frame_type"] == "7FA" and rows[0]["note"] == "keep" and rows[0]["signature"] == "normal"
    assert "ring_ttxd" in lab.file.read_text(encoding="utf-8")

    frames = lab.file.parent / "frames.json"
    frames.write_text(json.dumps({
        "ring_tc": {"ring": ["TC1", "TC2"], "roles": {"load": ["DWATT"]}},
        "site_unit_3": {"ring": ["TTXD_1"]},
    }), encoding="utf-8")
    taught = load_taught_frames(s.data_dir)
    assert taught["7EA"]["ring"] == ["TC1", "TC2"]
    assert taught["site_unit_3"]["ring"] == ["TTXD_1"]
    assert "ring_tc" in frames.read_text(encoding="utf-8")

    folder = tmp_path / "idx"
    SimilarityIndex(np.zeros((1, 2), np.float32), [{"id": "a", "frame_type": "zone_named", "signature": "normal"}]).save(folder, "index")
    loaded = SimilarityIndex.load(folder, "index")
    assert loaded.records[0]["frame_type"] == "9HA" and loaded.records[0]["signature"] == "normal"
    assert "zone_named" in (folder / "index.jsonl").read_text(encoding="utf-8")

    head_dir = tmp_path / "head"
    head_dir.mkdir()
    np.savez_compressed(head_dir / "head.npz", W=np.zeros((1, 4), np.float32), b=np.zeros(1, np.float32))
    (head_dir / "head.json").write_text(json.dumps({
        "classes": ["a"],
        "meta": {"uses_frame": True, "frames": ["ring_tc", "ring_ttxd", "zone_egt", "zone_named"], "kept": 1},
    }), encoding="utf-8")
    head = Head.load(head_dir)
    assert head.meta["frames"] == ["7EA", "7FA", "7HA", "9HA"]
    assert head.meta["kept"] == 1
    assert "ring_tc" in (head_dir / "head.json").read_text(encoding="utf-8")
    # The first stored name for a frame wins, so an old id is not overwritten by a later alias.
    assert migrate_frame_table({"ring_tc": 1, "7EA": 2})["7EA"] == 1


def test_layout_slot_is_shared_and_does_not_flag_a_sibling_novel():
    """6B, 9E and 9FA have no training cases. They use their layout's column, not an empty one.

    The same embedding on that column stays inside the layout radius. A different layout
    is farther away, which is the distance an empty per-model slot used to add.
    """
    from sentinel.config import CUSTOM_FRAME
    from sentinel.e2e.model import Calibration, Head, SimilarityIndex, condition_vector, decide, novelty_limit
    from sentinel.e2e.train import per_frame_thresholds

    vocab = ["ring_tc", "ring_ttxd", "zone_egt", "zone_named"]
    base = np.ones(8, np.float32)
    base = base / np.linalg.norm(base)
    z_7ea = condition_vector(base, "7EA", vocab)
    assert np.allclose(z_7ea, condition_vector(base, "6B", vocab))
    assert np.allclose(z_7ea, condition_vector(base, "9E", vocab))
    assert np.allclose(z_7ea, condition_vector(base, CUSTOM_FRAME, vocab, layout="ring_tc"))
    z_7fa = condition_vector(base, "7FA", vocab)
    assert np.allclose(z_7fa, condition_vector(base, "9FA", vocab))
    assert not np.allclose(z_7ea, z_7fa)
    # A head saved before layout keys: Head.load renamed each layout to its default model.
    migrated = ["7EA", "7FA", "7HA", "9HA"]
    assert np.allclose(condition_vector(base, "6B", migrated), condition_vector(base, "7EA", migrated))
    assert np.allclose(condition_vector(base, "9E", migrated), condition_vector(base, "7EA", migrated))
    assert np.allclose(condition_vector(base, "9FA", migrated), condition_vector(base, "7FA", migrated))

    idx = SimilarityIndex(z_7ea.reshape(1, -1), [{"id": "t", "signature": "normal", "source": "synthetic"}])
    d_same = float(idx.distances(condition_vector(base, "6B", vocab))[0])
    d_other = float(idx.distances(z_7fa)[0])
    assert d_same < 1e-5
    assert d_other > 0.05
    head = Head(["normal"], np.zeros((1, z_7ea.shape[0]), np.float32), np.zeros(1),
                {"uses_frame": True, "frame_key": "layout", "frames": vocab})
    # Between the same-layout distance and the other-layout distance.
    radius = round(d_other / 2, 4)
    cal = Calibration(novelty_threshold=radius, match_threshold=radius / 2, frame_thresholds={"ring_tc": radius})
    same = decide(condition_vector(base, "6B", vocab), head, idx, cal, frame="6B")
    assert same["novel"] is False and same["frame_layout"] == "ring_tc"
    other = decide(z_7fa, head, idx, cal, frame="9FA")
    assert other["novel"] is True
    # ring_tc's radius covers 6B and 9E. It does not cover 9FA, which falls back to the shared radius.
    assert novelty_limit(cal, "6B") == radius
    assert novelty_limit(cal, "9E") == radius
    split = Calibration(novelty_threshold=0.9, match_threshold=0.45, frame_thresholds={"ring_tc": radius})
    assert novelty_limit(split, "9FA") == 0.9
    assert novelty_limit(split, CUSTOM_FRAME, layout="zone_named") == 0.9
    both = Calibration(novelty_threshold=0.5, match_threshold=0.25, frame_thresholds={"ring_ttxd": 0.2, "7FA": 0.11, "9FA": 0.08})
    assert novelty_limit(both, "9FA") == 0.08  # enough cases of its own
    assert novelty_limit(both, "7FA") == 0.11
    sparse = Calibration(novelty_threshold=0.5, match_threshold=0.25, frame_thresholds={"ring_ttxd": 0.2, "7FA": 0.11})
    assert novelty_limit(sparse, "9FA") == 0.2  # too few 9FA cases: the layout radius
    assert novelty_limit(Calibration(novelty_threshold=0.5, match_threshold=0.25, frame_thresholds={"7FA": 0.11}), "9FA") == 0.11

    # 25 labeled 7EA cases and 25 labeled 9FA cases earn their own keys. 3 of 6B do not.
    frames = ["7EA"] * 25 + ["6B"] * 3 + ["9FA"] * 25
    z = np.ones(4, np.float32)
    z = z / np.linalg.norm(z)
    pool = SimilarityIndex(np.stack([z, z]), [{"id": "a", "signature": "normal"}, {"id": "b", "signature": "normal"}])
    got = per_frame_thresholds(pool, np.stack([z] * len(frames)), frames, 0.98)
    assert "ring_tc" in got and "7EA" in got and "6B" not in got
    assert "ring_ttxd" in got and "9FA" in got and "7FA" not in got
    cal2 = Calibration(novelty_threshold=0.9, match_threshold=0.45, frame_thresholds=got)
    assert novelty_limit(cal2, "6B") == got["ring_tc"]
    assert novelty_limit(cal2, "9E") == got["ring_tc"]
    assert novelty_limit(cal2, CUSTOM_FRAME, layout="ring_tc") == got["ring_tc"]
    assert novelty_limit(cal2, "7EA") == got["7EA"]


def test_layout_keyed_head_keeps_layout_ids_on_load(tmp_path):
    """A head trained on layouts must not be renamed to model ids, or 6B would miss the column."""
    from sentinel.e2e.model import Head

    head_dir = tmp_path / "head"
    head_dir.mkdir()
    np.savez_compressed(head_dir / "head.npz", W=np.zeros((1, 4), np.float32), b=np.zeros(1, np.float32))
    (head_dir / "head.json").write_text(json.dumps({
        "classes": ["a"],
        "meta": {"uses_frame": True, "frame_key": "layout", "frames": ["ring_tc", "ring_ttxd", "zone_egt", "zone_named"]},
    }), encoding="utf-8")
    head = Head.load(head_dir)
    assert head.meta["frames"] == ["ring_tc", "ring_ttxd", "zone_egt", "zone_named"]
    assert head.meta["frame_key"] == "layout"


def test_alias_slice_is_the_same_gate_as_the_new_frame_id():
    overall = {"action_acc": 0.9, "signature_acc": 0.9}
    big = {"n": 20, "action_acc": 0.9, "signature_acc": 0.9}
    before = {**overall, "by_frame": {"ring_tc": big}}
    after = {**overall, "by_frame": {"7EA": {**big, "action_acc": 0.5}}}
    assert not not_worse(before, after)


def test_setup_theme_file_is_local_and_plots_share_the_palette(tmp_path):
    from sentinel.bootstrap import write_streamlit_config
    from sentinel.charts import PLOT_PALETTE, onset_order_figure, residual_chart, residual_heatmap_figure, similar_cases_figure

    path = write_streamlit_config(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "config.toml"
    assert 'font = "sans serif"' in text
    assert "[theme.light]" in text and "[theme.dark]" in text
    assert "http" not in text.lower()
    assert "#2f6fed" in text
    assert residual_chart({}) is None
    res = {
        "profile": {"channels": ["TC1", "TC2"], "times": ["t0", "t1"], "matrix": [[1.0, 4.0], [2.0, 9.0]]},
        "events": [{"channel": "TC2", "onset_idx": 1, "minutes_after_first": 5, "shape": "step", "direction": "down"}],
        "channels": [{"channel": "TC1", "group": "exhaust", "score": 3.0}, {"channel": "spread", "group": "spread", "score": 8.0}],
    }
    assert residual_heatmap_figure(res) is not None
    assert onset_order_figure(res) is not None
    assert similar_cases_figure([{"id": "A", "distance": 0.2}, {"id": "B", "distance": 0.4}]) is not None
    assert PLOT_PALETTE[0] == "#2f6fed" and len(PLOT_PALETTE) >= 4
