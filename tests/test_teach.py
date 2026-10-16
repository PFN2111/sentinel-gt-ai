"""Teach the app: labeled cases (index now + labeled set, "why" in the prompt), documents (paste / md /
txt / PDF, tags, retrieval in both paths), signatures, and versioned rollback that undoes deletions."""

from __future__ import annotations

import importlib.util
import io
import json
import shutil

import pytest

from conftest import ROOT
from sentinel.config import load_settings
from sentinel.knowledge import retrieve, retrieve_learned

MODEL_DIR = ROOT / "models" / "e2e"
needs_model = pytest.mark.skipif(
    not ((MODEL_DIR / "legacy" / "base" / "head.npz").is_file() or (MODEL_DIR / "base" / "head.npz").is_file()),
    reason="learned model not trained (scripts/train_e2e.py)",
)


def minimal_pdf(lines):
    """A valid one-page PDF with the given text lines (Helvetica), built by hand."""
    content = "BT /F1 12 Tf 72 720 Td 16 TL " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.fixture
def ts(tmp_path):
    kd = tmp_path / "knowledge"
    kd.mkdir()
    for p in (ROOT / "knowledge").glob("*.md"):
        shutil.copy(p, kd / p.name)
    return load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False, "knowledge_dir": str(kd), "e2e_retrain_every": 50})


def test_pdf_and_text_ingestion():
    from sentinel.e2e import teach

    pdf = minimal_pdf(["Exhaust TC loop check", "Step 1: megger the harness", "Step 2: compare with neighbour TCs"])
    txt = teach.pdf_text(pdf)
    assert "## page 1" in txt and "megger the harness" in txt
    assert teach.file_text("proc.pdf", pdf) == txt
    assert "Loop check" in teach.file_text("a.txt", "Loop check\n\nmore".encode("cp1252"))
    assert teach.file_text("a.md", "## Step\nx".encode("utf-8")).startswith("## Step")
    with pytest.raises(ValueError):
        teach.file_text("a.docx", b"PK..")
    with pytest.raises(ValueError):
        teach.pdf_text(minimal_pdf([]))


def test_documents_tags_retrieval_delete_and_rollback(ts):
    from sentinel.e2e import teach

    pdf_text = teach.pdf_text(minimal_pdf(["Site procedure for a cold TC", "1. Megger the TC harness before any borescope."]))
    did, v1 = teach.save_document(ts, "Site TC procedure", pdf_text, analytics=["cold_spot"], signatures=["tc_sensor_fault"], source="upload:p.pdf")
    did2, v2 = teach.save_document(ts, "Dynamics notes", "## Probe check\nCompare the two probes.\n\n## Tuning\nCall the tuner.", analytics=["combustion dynamics"])
    assert (v1, v2) == ("k0001", "k0002") and did == "site_tc_procedure"
    docs = {d["id"]: d for d in teach.list_documents(ts)}
    assert docs[did]["analytics"] == ["cold_spot"] and docs[did]["signatures"] == ["tc_sensor_fault"] and docs[did2]["sections"] == 2
    k = retrieve("cold_spot", "cold spot", folder=ts.knowledge_dir)
    assert k[0]["ref"].startswith("user/site_tc_procedure.md") and "Megger" in k[0]["text"]
    kl = retrieve_learned("Exhaust Cold Spot", "", "cold_spot_hardware", extra_signatures=["tc_sensor_fault"], folder=ts.knowledge_dir)
    assert kl[0]["ref"].startswith("user/site_tc_procedure.md")
    meta, body = teach.read_document(ts, did)
    did_e, v3 = teach.save_document(ts, "Site TC procedure", body + "\n\n## Afterwards\nRe-check spread.", ["cold_spot"], ["tc_sensor_fault"], doc_id=did)
    assert did_e == did and "Afterwards" in teach.read_document(ts, did)[1]
    v4 = teach.delete_document(ts, did)
    assert not (ts.knowledge_dir / "user" / f"{did}.md").exists()
    assert not retrieve("cold_spot", "", folder=ts.knowledge_dir)[0]["ref"].startswith("user/")
    reg = teach.TeachRegistry(ts)
    assert reg.current() == v4 and [v["version"] for v in reg.versions()] == ["k0000", "k0001", "k0002", "k0003", "k0004"]
    assert reg.rollback() == "k0003"
    assert "Afterwards" in teach.read_document(ts, did)[1]  # deletion undone
    reg.activate("k0000")
    assert teach.list_documents(ts) == []
    reg.activate("k0004")
    assert [d["id"] for d in teach.list_documents(ts)] == [did2]


def test_signatures_add_edit_delete_and_rollback(ts):
    from sentinel.e2e import teach
    from sentinel.e2e.taxonomy import load_taxonomy

    sid, v = teach.save_signature(ts, "TC harness fault", "Several adjacent TCs drop out together (harness / connector)", "sensor_fix_replace",
                                  urgency="next_opportunity", residual_pattern="3-6 adjacent TCs drop by > 100 C together")
    tax = load_taxonomy(ts.data_dir)
    assert sid == "tc_harness_fault" and tax[sid]["action"] == "sensor_fix_replace" and tax[sid]["urgency"] == "next_opportunity"
    assert tax[sid]["title"].startswith("Several adjacent") and tax[sid]["residual_pattern"]
    teach.save_signature(ts, "TC harness fault", "Harness / connector fault", "sensor_fix_replace", urgency="prompt")
    assert load_taxonomy(ts.data_dir)[sid]["urgency"] == "prompt"
    with pytest.raises(ValueError):
        teach.save_signature(ts, "x", "y", "not_an_action")
    with pytest.raises(ValueError):
        teach.remove_signature(ts, "cold_spot_hardware")
    teach.remove_signature(ts, sid)
    assert sid not in load_taxonomy(ts.data_dir)
    teach.TeachRegistry(ts).rollback()
    assert load_taxonomy(ts.data_dir)[sid]["title"] == "Harness / connector fault"


@pytest.fixture(scope="module")
def tc_step_csv():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "generate_scenario_csvs.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    df, meta = gen.make_case({"split": "teach", "signature": "tc_sensor_fault", "variant": "step_lookalike", "index": 3})
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue(), meta


@needs_model
def test_add_case_index_now_why_in_prompt_edit_delete_rollback(ts, tc_step_csv):
    from sentinel.e2e import teach, writer
    from sentinel.e2e.pipeline import analyze_case, get_engine

    raw, meta = tc_step_csv
    eng = get_engine(ts)
    why = "TC stepped alone by about 30 C; neighbours flat; harness connector found loose"
    out = teach.add_case(ts, eng, raw, "site_case.csv", "tc_sensor_fault", why=why, note="I&C confirmed", analytic=meta["analytic"], notes=meta["notes"])
    cid = out["case_id"]
    assert out["index_after"] == out["index_before"] + 1 and out["version"] == "k0001"
    assert out["label"]["why"] == why and out["nearest_now"] and all(s["id"] != cid for s in out["nearest_now"])
    labs = teach.list_cases(ts)
    assert labs[0]["case_id"] == cid and labs[0]["why"] == why and labs[0]["signature"] == "tc_sensor_fault"
    assert len(eng.site_index) == 1 and eng.site_index.records[0]["why"] == why
    # the same pattern again: the taught case is the nearest neighbour and its why reaches the LLM prompt
    r2 = analyze_case(raw, meta["analytic"], meta["notes"], settings=ts, engine=eng, source_name="again.csv")
    s0 = r2["learned"]["similar"][0]
    assert s0["id"] == cid and s0["why"] == why and s0["distance"] < 0.01
    assert f'engineer why: "{why}' in writer.build_prompt(r2, eng.taxonomy())
    assert any(c["case_id"] == r2["case_id"] for c in teach.now_cited_by(ts, eng, cid))
    # edit, delete (out of the index and the labeled set), rollback (deletion undone)
    teach.edit_case(ts, eng, cid, "cold_spot_hardware", why="re-labeled")
    assert eng.site_index.records[0]["signature"] == "cold_spot_hardware" and eng.site_index.records[0]["why"] == "re-labeled"
    n = len(eng.index)
    teach.delete_case(ts, eng, cid)
    assert len(eng.index) == n - 1 and cid not in teach.list_cases(ts) and not eng.site_index.records
    assert eng.training_set()["n_engineer"] == 0
    reg = teach.TeachRegistry(ts)
    assert reg.rollback(eng) == "k0002"
    assert len(eng.index) == n and eng.site_index.records[0]["why"] == "re-labeled"
    reg.activate("k0001", eng)
    assert eng.site_index.records[0]["signature"] == "tc_sensor_fault" and eng.site_index.records[0]["why"] == why
    reg.activate("k0000", eng)
    assert not eng.site_index.records and teach.list_cases(ts) == []
    # the label log is append-only: every step is still there
    log = [json.loads(x) for x in (ts.data_dir / "e2e" / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for x in log if x.get("deleted")) >= 2 and sum(1 for x in log if x.get("restored_by")) >= 2


@needs_model
def test_signature_in_use_cannot_be_deleted(ts, tc_step_csv):
    from sentinel.e2e import teach
    from sentinel.e2e.pipeline import get_engine

    raw, meta = tc_step_csv
    teach.save_signature(ts, "Loose TC connector", "Connector on one TC is loose", "sensor_fix_replace")
    teach.add_case(ts, get_engine(ts), raw, "c.csv", "loose_tc_connector", why="connector loose")
    with pytest.raises(ValueError, match="still use"):
        teach.remove_signature(ts, "loose_tc_connector")


def test_teach_page_renders(tmp_path, monkeypatch):
    from ui_support import browser, run

    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SENTINEL_USE_LLM", "0")
    monkeypatch.setenv("SENTINEL_USE_TSPULSE", "0")

    async def body():
        async with browser() as user:
            await user.open("/teach")
            await user.should_see("Alert types")
            await user.should_see("Frames")
            await user.should_see("Labels")
            await user.should_see("Add labeled case")
            await user.should_not_see("Retrain")

    run(body)
