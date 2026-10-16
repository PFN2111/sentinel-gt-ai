"""Assign stored labels and cases to alert-type profiles without rewriting the legacy logs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from conftest import sample
from sentinel.config import load_settings
from sentinel.e2e.labels import LabelStore
from sentinel.e2e.migrate import ensure_migrated, migrate_profiles
from sentinel.e2e.model import MaskedIndex, SimilarityIndex
from sentinel.e2e.teach import TeachRegistry, add_alert_type, save_document, save_frame
from sentinel.e2e.train import load_profile_model
from sentinel.profiles import load_profiles
from sentinel.store import list_cases


def _tree(root: Path) -> dict:
    """File bytes under ``root``. Missing directory is an empty tree."""
    out = {}
    if not root.exists():
        return out
    for path in root.rglob("*"):
        if path.is_file():
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


def _settings(tmp_path: Path):
    settings = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    settings.embedder = "stats"
    return settings


def _label(settings, case_id: str, signature: str, analytic: str = "", csv: str = "", frame: str = "") -> None:
    LabelStore(settings).add(
        {
            "case_id": case_id,
            "signature": signature,
            "action": "monitor_only",
            "analytic": analytic,
            "frame_type": frame,
            "csv_path": csv,
            "verdict": "good",
        }
    )


def _tiny_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp,DWATT,TTXD1_1,TTXD1_2,BB1_VIB,widget_housing\n"
        "2024-01-01 00:00:00,100,500,510,1.2,3\n"
        "2024-01-01 00:10:00,101,501,509,1.1,3\n"
        "2024-01-01 00:20:00,102,499,511,1.3,4\n"
        f"2024-01-01 00:30:00,103,498,512,1.2,3\n",
        encoding="utf-8",
    )


def test_dry_run_writes_nothing(tmp_path, capsys):
    from sentinel.cli import main

    settings = _settings(tmp_path)
    csv = sample("cold_spot", 1)
    _label(settings, "c-name", "cold_spot_hardware", analytic="Exhaust Cold Spot", csv=str(csv))
    before = _tree(settings.data_dir)
    plan = migrate_profiles(settings, dry_run=True)
    assert plan["dry_run"] is True
    assert plan["assignments"]["c-name"] == {"profile": "exhaust_cold_spot", "rule": "name", "rev": 1}
    assert _tree(settings.data_dir) == before
    assert main(["migrate-profiles", "--dry-run", "--data-dir", str(settings.data_dir)]) == 0
    text = capsys.readouterr().out
    assert '"dry_run": true' in text
    assert _tree(settings.data_dir) == before


def test_ensure_migrated_respects_mode(tmp_path):
    settings = _settings(tmp_path)
    settings.e2e_mode = "legacy"
    assert ensure_migrated(settings) == ""
    assert not (settings.data_dir / "e2e" / "migrations" / "dynamic_v1.json").is_file()
    settings.e2e_mode = "profiles"
    assert ensure_migrated(settings) == ""
    settings.e2e_mode = "peralert"
    banner = ensure_migrated(settings)
    assert "No stored labels" in banner
    assert (settings.data_dir / "e2e" / "migrations" / "dynamic_v1.json").is_file()
    frozen = _tree(settings.data_dir)
    again = ensure_migrated(settings)
    assert again == banner
    assert _tree(settings.data_dir) == frozen


def test_migrate_assigns_embeds_and_rolls_back(tmp_path):
    settings = _settings(tmp_path)
    data = settings.data_dir
    emb = data / "e2e" / "emb" / "keep.npy"
    emb.parent.mkdir(parents=True)
    emb.write_bytes(b"legacy-emb")
    legacy_reg = data / "e2e" / "registry" / "v0003" / "marker.txt"
    legacy_reg.parent.mkdir(parents=True)
    legacy_reg.write_bytes(b"legacy-reg")

    cold = tmp_path / "cold.csv"
    shutil.copy(sample("cold_spot", 1), cold)
    auto_csv = tmp_path / "auto.csv"
    _tiny_csv(auto_csv)
    general_csv = tmp_path / "general.csv"
    _tiny_csv(general_csv)
    sig_csv = tmp_path / "sig.csv"
    _tiny_csv(sig_csv)

    save_frame(settings, "Site unit 3", "taught frame", ring=["TTXD1_1", "TTXD1_2"], roles={"load": ["DWATT"]})
    save_document(settings, "Widget note", "Look at the housing tag.", analytics=["Widget housing alarm"])
    added = add_alert_type(
        settings,
        "Seal oil pressure",
        "Seal oil pressure is high",
        "monitor_only",
        "Check the seal oil pressure transmitter before adjusting the setpoint.",
    )
    _label(settings, "c-name", "cold_spot_hardware", analytic="Exhaust Cold Spot", csv=str(cold), frame="7FA")
    _label(settings, "c-sig", "igv_lvdt_drift", analytic="unrelated note", csv=str(sig_csv), frame="7FA")
    _label(settings, "c-auto", "normal", analytic="Widget housing alarm", csv=str(auto_csv), frame="site_unit_3")
    _label(settings, "c-general", "tc_sensor_fault", analytic="", csv=str(general_csv))
    _label(settings, "c-nocsv", "bearing_wear_trend", analytic="")
    case_only = settings.cases_dir / "c-caseonly"
    case_only.mkdir(parents=True)
    (case_only / "case.json").write_text(
        json.dumps(
            {
                "case_id": "c-caseonly",
                "created": "2026-01-01T00:00:00",
                "analytic_entered": "",
                "signature": "tc_sensor_fault",
                "primary_action": "sensor_fix_replace",
            }
        ),
        encoding="utf-8",
    )
    labels_path = data / "e2e" / "labels.jsonl"
    labels_before = labels_path.read_bytes()
    assert b'"profile"' not in labels_before

    result = migrate_profiles(settings)
    assigned = result["assignments"]
    assert assigned["c-name"]["profile"] == "exhaust_cold_spot" and assigned["c-name"]["rule"] == "name"
    assert assigned["c-sig"]["profile"] == "igv_fuel_control" and assigned["c-sig"]["rule"] == "signature"
    assert assigned["c-auto"]["rule"] == "auto" and assigned["c-auto"]["profile"] == "auto_widget_housing_alarm"
    assert assigned["c-general"]["profile"] == "general_review" and assigned["c-general"]["rule"] == "general"
    assert assigned["c-caseonly"]["profile"] == "general_review"
    assert assigned["c-nocsv"]["profile"] == "bearing_vibration"
    assert "c-nocsv" in result["not_reembeddable"]
    assert "c-name" not in result["not_reembeddable"]
    assert result["counts"] == {"labels": 5, "cases": 1}
    assert result["encoder"] == "e0000"
    assert "general_review" in result["created_profiles"]
    assert "auto_widget_housing_alarm" in result["created_profiles"]
    assert f"alert_{added['signature']}" in result["created_profiles"]
    assert "exhaust_cold_spot" in result["frame_aliases_added"]["site_unit_3"]
    assert labels_path.read_bytes() == labels_before
    assert emb.read_bytes() == b"legacy-emb"
    assert legacy_reg.read_bytes() == b"legacy-reg"
    assert not (data / "e2e" / "registry" / "v0001").exists()

    book = load_profiles(settings=settings)
    cold_prof = book["exhaust_cold_spot"]
    assert cold_prof.source == "migration" and cold_prof.rev == 2
    assert any("site_unit_3" in (slot.aliases or {}) for slot in cold_prof.slots)
    auto = book["auto_widget_housing_alarm"]
    assert auto.source == "migration" and auto.needs_review and auto.analyzer == "generic"
    assert auto.slots and all(slot.required for slot in auto.slots)
    assert any("site_unit_3" in (slot.aliases or {}) and "*" in (slot.aliases or {}) for slot in auto.slots)
    assert any(doc.endswith(".md") for doc in auto.docs)
    stub = book[f"alert_{added['signature']}"]
    assert stub.slots == [] and stub.needs_review and stub.source == "migration"
    for pid in ("exhaust_cold_spot", "igv_fuel_control", "bearing_vibration"):
        assert not (data / "e2e" / "registry" / "profiles" / pid).exists()
    for pid in ("general_review", "auto_widget_housing_alarm"):
        folder = next((data / "e2e" / "registry" / "profiles" / pid).glob("v*/"))
        assert (folder / "index.npz").is_file()
        assert not (folder / "head.npz").is_file()
        model = load_profile_model(settings, pid)
        assert model is not None and model.head is None and model.tier() == "knn"
    assert (data / "e2e" / "emb2" / "exhaust_cold_spot" / "c-name.npz").is_file()
    assert not (data / "e2e" / "emb2" / "bearing_vibration" / "c-nocsv.npz").exists()
    assert not list((data / "e2e" / "emb2").glob("*/c-caseonly.npz"))

    filled = {row["case_id"]: row for row in LabelStore(settings).all() if not row.get("deleted")}
    assert filled["c-name"]["profile"] == "exhaust_cold_spot"
    assert b'"profile"' not in labels_path.read_bytes()
    listed = {row["case_id"]: row for row in list_cases(settings)}
    assert listed["c-caseonly"]["profile"] == "general_review"
    assert "profile" not in (case_only / "case.json").read_text(encoding="utf-8")

    scratch = data / "e2e" / "scratch"
    SimilarityIndex(np.zeros((1, 4), dtype=np.float32), [{"id": "c-name", "signature": "cold_spot_hardware"}]).save(scratch, "flat")
    loaded = SimilarityIndex.load(scratch, "flat")
    assert loaded.records[0]["profile"] == "exhaust_cold_spot"
    assert b'"profile"' not in (scratch / "flat.jsonl").read_bytes()
    MaskedIndex(
        np.zeros((1, 1, 48), dtype=np.float32),
        np.ones((1, 1), dtype=bool),
        [""],
        [{"id": "c-general", "signature": "tc_sensor_fault"}],
    ).save(scratch, "blocks")
    blocked = MaskedIndex.load(scratch, "blocks")
    assert blocked.records[0]["profile"] == "general_review"

    # The scratch index is new work relative to the migration map, not a new case.
    # Drop it before the no-op check so the data tree is only what migration wrote.
    shutil.rmtree(scratch)
    frozen = _tree(data)
    second = migrate_profiles(settings)
    assert second == result
    assert _tree(data) == frozen

    reg = TeachRegistry(settings)
    target = next(row["version"] for row in reg.versions() if row.get("action") == "before profile migration")
    reg.activate(target)
    assert not (data / "e2e" / "migrations" / "dynamic_v1.json").is_file()
    assert list(Path(settings.profiles_dir).glob("*.json")) == []
    from sentinel.config import load_taught_frames
    from sentinel.e2e.taxonomy import load_taxonomy

    assert "site_unit_3" in load_taught_frames(settings.data_dir)
    assert added["signature"] in load_taxonomy(settings.data_dir)
    assert labels_path.read_bytes() == labels_before


def test_old_teach_snapshot_does_not_delete_profiles(tmp_path):
    settings = _settings(tmp_path)
    profiles = Path(settings.profiles_dir)
    profiles.mkdir(parents=True)
    kept = profiles / "keep_me.json"
    kept.write_text('{"id": "keep_me"}\n', encoding="utf-8")
    version = settings.data_dir / "e2e" / "teach" / "versions" / "k0007"
    (version / "docs").mkdir(parents=True)
    (version / "labels.json").write_text("{}\n", encoding="utf-8")
    (version / "signatures.json").write_text("{}\n", encoding="utf-8")
    (version / "frames.json").write_text("{}\n", encoding="utf-8")
    (version / "meta.json").write_text("{}\n", encoding="utf-8")
    TeachRegistry(settings).activate("k0007")
    assert kept.read_text(encoding="utf-8") == '{"id": "keep_me"}\n'
