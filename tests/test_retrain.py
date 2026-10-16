"""Retrain TS Pulse (Teach page, part B5): device fallback, promote-if-not-worse, bundles, background job.

The full fine-tune is not run here (it is the 15-minute CPU job). These tests cover the decision,
the GPU-to-CPU fallback, a quick update against the shipped model, and a full update whose
fine-tune is replaced with a copy of the shipped encoder so the bundle, rollback and progress
callback still execute.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT
from sentinel.config import load_settings
from sentinel.e2e import encoder, train_job
from sentinel.e2e.model import SimilarityIndex
from sentinel.e2e.train import loo_calibration, not_worse

MODEL_DIR = ROOT / "models" / "e2e"
needs_model = pytest.mark.skipif(
    not ((MODEL_DIR / "legacy" / "base" / "head.npz").is_file() or (MODEL_DIR / "base" / "head.npz").is_file()),
    reason="learned model not trained (scripts/train_e2e.py)",
)


def test_not_worse_gate():
    good = {"action_acc": 0.9, "signature_acc": 0.8}
    assert not_worse(good, {"action_acc": 0.9, "signature_acc": 0.8})
    assert not_worse(good, {"action_acc": 0.91, "signature_acc": 0.8})
    assert not not_worse(good, {"action_acc": 0.89, "signature_acc": 0.9})
    assert not not_worse(good, {"action_acc": 0.9, "signature_acc": 0.79})
    assert not not_worse(good, {})
    assert not_worse({}, {"action_acc": 0.5, "signature_acc": 0.5})
    assert not not_worse(None, None)


def test_loo_calibration_moves_with_the_index():
    z = np.eye(4, dtype=np.float32)
    idx = SimilarityIndex(z, [{"id": str(i), "signature": "a" if i < 3 else "b", "action": "monitor_only"} for i in range(4)])
    cal = loo_calibration(idx, quantile=0.5)
    assert 0 < cal.novelty_threshold < 1.5
    assert cal.match_threshold <= cal.novelty_threshold


def test_resolve_device_cpu_and_gpu_fallback(monkeypatch):
    import torch

    assert encoder.resolve_train_device("cpu")["device"] == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gpu = encoder.resolve_train_device("gpu")
    auto = encoder.resolve_train_device("auto")
    assert gpu["device"] == "cpu" and "falling back to CPU" in gpu["reason"]
    assert auto["device"] == "cpu" and "using CPU" in auto["reason"]
    with pytest.raises(ValueError):
        encoder.resolve_train_device("tpu")


def test_finetune_retries_on_cpu_when_the_gpu_runs_out_of_memory(monkeypatch, tmp_path):
    calls = []

    def fake(dev, base_dir, X_tr, y_tr, X_va, y_va, classes, out_dir, **kw):
        calls.append(dev)
        if dev == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {"best": {"val_acc": 0.5, "epoch": 1}, "history": [], "classes": list(classes)}

    monkeypatch.setattr(encoder, "_finetune_on_device", fake)
    monkeypatch.setattr(encoder, "resolve_train_device", lambda p: {"device": "cuda", "requested": "gpu", "reason": "CUDA GPU (test)"})
    seen = []
    info = encoder.finetune(
        tmp_path, np.zeros((2, 4, 2), np.float32), np.array([0, 1]), np.zeros((1, 4, 2), np.float32), np.array([0]),
        ["a", "b"], tmp_path / "out", epochs=1, device="gpu", progress=lambda f, m: seen.append((f, m)),
    )
    assert calls == ["cuda", "cpu"]
    assert info["device"] == "cpu" and "out of memory" in info["device_reason"]
    assert seen and seen[0][0] == 0.0
    assert (tmp_path / "out" / "training.json").is_file()


def test_train_cli_exposes_quick_full_device_and_force():
    p = subprocess.run([sys.executable, str(ROOT / "app.py"), "train", "--help"], capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    for flag in ("--quick", "--full", "--force", "--device", "--epochs"):
        assert flag in p.stdout, flag


def test_job_file_survives_a_reread_and_a_dead_process_is_an_error(tmp_path):
    s = load_settings({"data_dir": tmp_path / "data"})
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        train_job._write(train_job.job_file(s), {
            "status": "running", "mode": "quick", "device": "cpu", "pid": proc.pid,
            "frac": 0.4, "message": "epoch 1", "log": "", "result": None,
        })
        again = train_job.read_job(s)
        assert again["status"] == "running" and again["frac"] == 0.4 and again["pid"] == proc.pid
    finally:
        proc.kill()
        proc.wait(10)
    dead = train_job.read_job(s)
    assert dead["status"] == "error" and "stopped" in dead["message"]


@needs_model
def test_full_refuses_a_missing_dataset(tmp_path):
    from sentinel.e2e.engine import Engine

    s = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    eng = Engine(s)
    eng.heldout_eval = lambda *a, **k: {}
    with pytest.raises(FileNotFoundError, match="generate_scenario_csvs"):
        eng.train(mode="full", device="cpu", trigger="missing dataset")


@needs_model
def test_quick_saves_a_worse_model_and_force_promotes_it(tmp_path):
    from sentinel.e2e.engine import Engine

    s = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    eng = Engine(s)
    Z = eng.base_index.Z[:30].astype(np.float32)
    y = [r["signature"] for r in eng.base_index.records[:30]]
    eng.validation_embeddings = lambda progress=None: (Z, y)
    calls = {"n": 0}

    def fake(head=None, cal=None, index=None, heldout=None):
        calls["n"] += 1
        base = {"n": 4, "rules_action_acc": 0.5, "novelty_rate": 0.01, "one_shot_n": 2, "one_shot_acc": 0.5}
        acc = 0.9 if calls["n"] % 2 == 1 else 0.4
        return {**base, "signature_acc": acc, "action_acc": acc}

    eng.heldout_eval = fake
    notes = []
    out = eng.train(mode="quick", device="cpu", force=False, trigger="test quick", progress=lambda f, m: notes.append(m))
    assert out["promoted"] is False and eng.head_version == "v0000"
    folder = tmp_path / "data" / "e2e" / "registry" / out["version"]
    assert (folder / "head.npz").is_file() and (folder / "calibration.json").is_file()
    assert out["before"]["action_acc"] == 0.9 and out["after"]["action_acc"] == 0.4
    assert out["after"]["one_shot_n"] == 2
    assert notes and notes[-1].startswith(out["version"])
    forced = eng.train(mode="quick", device="cpu", force=True, trigger="test force")
    assert forced["promoted"] is True and eng.head_version == forced["version"]
    # The override is logged, not silent: one force_promote line naming this version.
    log = (tmp_path / "data" / "e2e" / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"type": "force_promote"' in log and forced["version"] in log and forced["force_reason"]
    assert eng.rollback() == out["version"]
    assert eng.head_version == out["version"]
    eng.activate("v0000")
    assert eng.head_version == "v0000" and eng.encoder_version == "shipped"


@needs_model
def test_full_bundle_promotes_with_progress_and_rolls_back_to_the_shipped_encoder(tmp_path, monkeypatch):
    from sentinel.e2e.engine import Engine
    from sentinel.e2e.features import CHANNELS, WINDOW

    s = load_settings({"data_dir": tmp_path / "data", "use_llm": False, "use_tspulse": False})
    eng = Engine(s)
    eng.heldout_eval = lambda *a, **k: {"n": 2, "signature_acc": 0.5, "action_acc": 0.5, "rules_action_acc": 0.2, "novelty_rate": 0.0, "one_shot_n": 0, "one_shot_acc": None}
    x = np.zeros((4, WINDOW, len(CHANNELS)), np.float32)

    def arrays(progress=None):
        return {
            "Xtr": x, "ytr": np.array([0, 0, 1, 1]),
            "Xva": x[:2], "yva": np.array([0, 1]), "yva_labels": ["normal", "normal"],
            "classes": ["normal", "tc_sensor_fault"],
            "train_records": [
                {"id": f"s{i}", "signature": "normal" if i < 2 else "tc_sensor_fault", "action": "monitor_only", "source": "synthetic"}
                for i in range(4)
            ],
            "engineer": [],
            "Xte": x[:2], "sig_te": np.array(["normal", "normal"]),
            "rules_te": np.array(["monitor_only", "escalate"]),
            "case_te": np.array(["c0", "c1"]),
        }

    eng._full_arrays = arrays
    seen = {}

    def fake_ft(base, Xtr, ytr, Xva, yva, classes, out, epochs=12, device="cpu", progress=None, log_fn=print, **kw):
        seen["device"] = device
        if progress:
            progress(1.0, "epoch 1/1 on cpu")
        enc_src = MODEL_DIR / "legacy" / "encoder"
        if not (enc_src / "classes.json").is_file():
            enc_src = MODEL_DIR / "encoder"
        shutil.copytree(enc_src, out)
        return {"best": {"val_acc": 0.5, "epoch": 1}, "history": [], "device": "cpu", "device_reason": "CPU requested", "classes": list(classes)}

    monkeypatch.setattr("sentinel.e2e.engine.finetune", fake_ft)
    out = eng.train(mode="full", device="gpu", force=True, epochs=1, trigger="test full")
    assert seen["device"] == "gpu"
    assert out["promoted"] is True and out["kind"] == "full"
    assert eng.head_version == out["version"] and eng.encoder_version == out["version"]
    folder = tmp_path / "data" / "e2e" / "registry" / out["version"]
    assert (folder / "encoder" / "config.json").is_file()
    assert (folder / "scaler.npz").is_file() and (folder / "index.npz").is_file() and (folder / "heldout.npz").is_file()
    assert "action_acc" in out["after"] and "novelty_rate" in out["after"]
    from sentinel.e2e.labels import legacy_model_dir

    shipped = legacy_model_dir(eng.mdir, "encoder")
    assert eng.rollback() == "v0000"
    assert eng.encoder_version == "shipped" and Path(eng.encoder.path).resolve() == shipped.resolve()
    eng.activate(out["version"])
    assert eng.encoder_version == out["version"] and (folder / "encoder").resolve() == Path(eng.encoder.path).resolve()
