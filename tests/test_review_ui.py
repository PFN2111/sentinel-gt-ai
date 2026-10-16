"""Review UI fixes: Reject does not teach the rejected signature, Other / custom tags survive a
reload, setup keeps an existing Streamlit config, and the setup dependency probe needs plotly."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from conftest import ROOT, sample
from sentinel.config import load_settings

MODEL_DIR = ROOT / "models" / "e2e"
needs_model = pytest.mark.skipif(
    not ((MODEL_DIR / "legacy" / "base" / "head.npz").is_file() or (MODEL_DIR / "base" / "head.npz").is_file()),
    reason="learned model not trained (scripts/train_e2e.py)",
)


@pytest.fixture
def review_settings(tmp_path):
    # No background retrain: these tests look at the index and labels only.
    return load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False, "e2e_retrain_every": 1000})


@needs_model
def test_reject_logs_the_case_without_learning_the_rejected_signature(review_settings):
    from sentinel import store
    from sentinel.e2e.pipeline import analyze_case, get_engine, reject_case
    from sentinel.e2e.taxonomy import NOVEL

    eng = get_engine(review_settings)
    r = analyze_case(sample("cold_spot", 1), "Exhaust Cold Spot", "", settings=review_settings, engine=eng, run_rules=False)
    n0 = len(eng.index)
    out = reject_case(review_settings, r["case_id"], reviewer="pytest", engine=eng)

    # Nothing enters the similarity index, so no later case can one-shot match the rejected answer.
    assert len(eng.index) == n0
    lab = eng.labels.latest()[r["case_id"]]
    assert lab["signature"] == NOVEL and lab["verdict"] == "not_good" and lab["action"] == "escalate"
    assert lab["predicted_signature"] == r["decision"]["signature"]
    assert out["feedback"]["learned_signature"] == r["decision"]["signature"]
    assert store.load_report(review_settings, r["case_id"])["review"]["status"] == "REJECTED"
    # A restart rebuilds the site index from the labels; the rejected case stays out.
    eng._load_site_index()
    assert len(eng.index) == n0


@needs_model
def test_custom_tags_are_stored_with_the_label_and_used_on_reembed(review_settings):
    from sentinel.config import CUSTOM_FRAME, frame_maps_for
    from sentinel.data import load_csv
    from sentinel.e2e.features import canonical
    from sentinel.e2e.pipeline import analyze_case, get_engine, label_case

    typed = ["TC1", "TC2", "TC3", "TC4"]
    overlay = frame_maps_for(review_settings.data_dir, CUSTOM_FRAME, None)  # what the Frame picker builds
    overlay[CUSTOM_FRAME] = {**overlay[CUSTOM_FRAME], "ring": typed}

    eng = get_engine(review_settings)
    path = sample("cold_spot", 1)
    r = analyze_case(path, "Exhaust Cold Spot", "", settings=review_settings, engine=eng, run_rules=False,
                     frame=CUSTOM_FRAME, frame_maps=overlay)
    assert r["input"]["frame_type"] == CUSTOM_FRAME
    assert r["input"]["frame_spec"]["ring"] == typed

    label_case(review_settings, r["case_id"], "tc_sensor_fault", engine=eng)
    lab = eng.labels.latest()[r["case_id"]]
    assert lab["frame_spec"]["ring"] == typed

    # Without the stored tags the empty Other / custom row would map the full TC1..TC27 ring instead.
    bare = load_csv(path, frame=CUSTOM_FRAME, frame_maps=frame_maps_for(review_settings.data_dir, CUSTOM_FRAME))
    kept = load_csv(path, frame=CUSTOM_FRAME, frame_maps=frame_maps_for(review_settings.data_dir, CUSTOM_FRAME, spec=lab["frame_spec"]))
    assert kept.ring == typed and len(bare.ring) > len(typed)
    assert not np.allclose(np.nan_to_num(canonical(bare).x), np.nan_to_num(canonical(kept).x))

    # Re-embedding (a new encoder version does this) reproduces the embedding stored at analysis time.
    before = np.array(eng.labels.embedding(r["case_id"]))
    assert eng._reembed_site() == 1
    assert np.allclose(before, eng.labels.embedding(r["case_id"]), atol=1e-4)


def test_setup_keeps_an_existing_streamlit_config(tmp_path):
    from sentinel.bootstrap import STREAMLIT_CONFIG, write_streamlit_config

    fresh = write_streamlit_config(tmp_path / "a")
    assert fresh.read_text(encoding="utf-8") == STREAMLIT_CONFIG

    mine = tmp_path / "b" / ".streamlit" / "config.toml"
    mine.parent.mkdir(parents=True)
    mine.write_text('[server]\nport = 8600\naddress = "localhost"\n', encoding="utf-8")
    assert write_streamlit_config(tmp_path / "b") == mine
    assert "port = 8600" in mine.read_text(encoding="utf-8")


def test_novel_label_toast_does_not_claim_the_index():
    from sentinel.e2e.taxonomy import NOVEL
    from sentinel.ui import label_saved_message

    text = label_saved_message("SGT-20261012-155038-56a4", NOVEL, "escalate", False)
    assert "Added to the similarity index" not in text
    assert "Not added" in text
    kept = label_saved_message("SGT-20261012-155038-56a4", "normal", "monitor_only", False)
    assert "Added to the similarity index" in kept
    assert "Not added" not in kept


def test_edit_flag_clears_after_save_and_cancel():
    import inspect

    from sentinel.ui import _case_card, clear_edit_flag, show_case

    state = {"editing_card": True, "editing_rules_c1": True, "other": 1}
    clear_edit_flag(state, "card")
    assert "editing_card" not in state and state["other"] == 1
    clear_edit_flag(state, "c1")
    assert "editing_rules_c1" not in state
    # Save and Cancel on both cards go through the same helper, so a rerun does not keep the banner.
    learned = inspect.getsource(_case_card)
    rules = inspect.getsource(show_case)
    assert learned.count("clear_edit_flag") >= 2 and 'key=f"cancel_{key}"' in learned
    assert rules.count("clear_edit_flag") >= 2 and 'key=f"cancel_rules_{cid}"' in rules


def test_layout_flag_warns_and_is_ignored_when_it_conflicts(capsys):
    import argparse

    from sentinel.cli import _frame_choice
    from sentinel.config import CUSTOM_FRAME, load_settings

    settings = load_settings({"data_dir": "data", "use_llm": False, "use_tspulse": False})
    frame, layout, code = _frame_choice(argparse.Namespace(frame="7FA", layout="ring_tc"), settings)
    err = capsys.readouterr().err
    assert code == 0 and frame == "7FA" and layout is None
    assert "ignored" in err and "ring_tc" in err and "7FA" in err

    frame, layout, code = _frame_choice(argparse.Namespace(frame="7FA", layout="ring_ttxd"), settings)
    err = capsys.readouterr().err
    assert code == 0 and frame == "7FA" and layout is None and err == ""

    frame, layout, code = _frame_choice(argparse.Namespace(frame="Other / custom", layout="ring_tc"), settings)
    assert code == 0 and frame == CUSTOM_FRAME and layout == "ring_tc"

    frame, layout, code = _frame_choice(argparse.Namespace(frame="", layout="zone_egt"), settings)
    err = capsys.readouterr().err
    assert code == 0 and frame is None and layout is None
    assert "ignored" in err and "zone_egt" in err


def test_setup_dependency_probe_requires_plotly(monkeypatch):
    from sentinel import bootstrap

    monkeypatch.setitem(sys.modules, "plotly", types.ModuleType("plotly"))
    if not bootstrap._deps_importable():
        pytest.skip("another core dependency is missing from this interpreter")
    monkeypatch.setitem(sys.modules, "plotly", None)  # import plotly now raises ImportError
    assert not bootstrap._deps_importable()
