#!/usr/bin/env python3
"""Train the end-to-end learned path (fully local, CPU is fine).

    python scripts/generate_scenario_csvs.py --dataset data/e2e/dataset     # once (~1 min)
    python scripts/train_e2e.py                                              # ~15-20 min on CPU
    python scripts/train_e2e.py --skip-finetune                              # reuse models/e2e/encoder
    python scripts/train_e2e.py --extra data/e2e/export                      # + real labeled cases
    python scripts/train_e2e.py --residual-only                              # V2: residual calibration only (~2 min)
    python scripts/train_e2e.py --profiles                                   # per-profile bundles, no 14-channel fine-tune
    python scripts/train_e2e.py --peralert                                   # per-alert bundles from both generators

Steps
1. Canonical features for every case (cached in data/e2e/cache).
2. Fine-tune TS Pulse (block-dualhead-512-p16, classification variant) on the train split;
   the validation split picks the best epoch.                        -> models/e2e/encoder
3. Embeddings -> scaler, base classifier head (train split), similarity index (train split
   + optional real labeled cases), novelty / one-shot calibration on the validation split.
4. Held-out test embeddings + rule-path actions for the learning dashboard. -> models/e2e/base
5. Residual feed (V2) on the validation split: per-group abnormal thresholds and the residual-novelty
   rule (residual.calibrate_thresholds, model.calibrate_residual)        -> base/calibration.json
6. Model card (models/e2e/card.json).

Nothing here touches the test, variant or unseen splits except to store the dashboard's
held-out set; the evaluation lives in scripts/eval_e2e.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sentinel.config import load_settings  # noqa: E402
from sentinel.e2e.dataset import cached_features, featurize_files, load_manifest, read_labeled_folder  # noqa: E402
from sentinel.e2e.labels import legacy_model_dir  # noqa: E402
from sentinel.e2e.encoder import Encoder, finetune  # noqa: E402
from sentinel.e2e.evaluation import rule_results  # noqa: E402
from sentinel.e2e.features import CHANNELS  # noqa: E402
from sentinel.e2e.model import Calibration, Head, Scaler, SimilarityIndex, calibrate, calibrate_residual, decide, train_head  # noqa: E402
from sentinel.e2e.taxonomy import BASE_SIGNATURES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default=str(ROOT / "data" / "e2e" / "dataset"))
    ap.add_argument("--out", default=None, help="model folder (default: settings e2e_model_dir = models/e2e)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="cpu", help="fine-tune device; gpu/auto fall back to CPU (default cpu)")
    ap.add_argument("--skip-finetune", action="store_true", help="reuse the existing fine-tuned encoder")
    ap.add_argument("--extra", action="append", default=[], help="labeled folder(s) of real cases (labels.csv + CSVs)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--residual-only", action="store_true", help="only (re)calibrate the residual feed on the validation split")
    ap.add_argument("--profiles", action="store_true", help="fit per-profile bundles with the channel-independent encoder; does not fine-tune the 14-channel head")
    ap.add_argument("--peralert", action="store_true", help="fit per-alert bundles from both generators; does not fine-tune the 14-channel head")
    ap.add_argument("--profile-dataset", default=str(ROOT / "data" / "e2e" / "dataset_profiles"), help="alert-type CSVs (not the 320-case acceptance set)")
    a = ap.parse_args()

    settings = load_settings()
    out = Path(a.out) if a.out else Path(settings.e2e_model_dir)
    if a.peralert:
        from sentinel.e2e.train import train_alert_bundles

        # Both generators. The acceptance side-car is read inside the trainer and those rows stay out.
        train_alert_bundles(settings, Path(a.dataset), Path(a.profile_dataset), out, device=a.device)
        return 0
    if a.profiles:
        from sentinel.e2e.train import train_profile_bundles

        train_profile_bundles(settings, Path(a.profile_dataset), out, device=a.device)
        return 0
    # The 14-channel write follows the moved bundle so it cannot replace the channel-independent encoder.
    base = legacy_model_dir(out, "base")
    base.mkdir(parents=True, exist_ok=True)
    root = Path(a.dataset)
    cache = ROOT / "data" / "e2e" / "cache"
    t0 = time.monotonic()

    rows = load_manifest(root)
    if a.residual_only:
        X = cached_features(root, rows, cache, workers=a.workers)
        va = np.flatnonzero(np.array([r["split"] for r in rows]) == "val")
        Z = Scaler.load(base / "scaler.npz")(Encoder(legacy_model_dir(out, "encoder")).embed(X[va]))
        info = residual_calibration(settings, root, [rows[i] for i in va], Z, base, cache)
        card_file = out / "card.json"
        card = json.loads(card_file.read_text(encoding="utf-8"))
        card["calibration"] = json.loads((base / "calibration.json").read_text(encoding="utf-8"))
        card["residual"] = info
        card_file.write_text(json.dumps(card, indent=2), encoding="utf-8")
        print(f"residual calibration written to {base / 'calibration.json'} in {(time.monotonic() - t0) / 60:.1f} min")
        return 0
    print(f"[1/5] canonical features for {len(rows)} cases")
    X = cached_features(root, rows, cache, workers=a.workers)
    classes = list(BASE_SIGNATURES)
    split = np.array([r["split"] for r in rows])
    sig = np.array([r["signature"] for r in rows])
    tr, va, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
    y = np.array([classes.index(s) if s in classes else -1 for s in sig])

    enc_dir = legacy_model_dir(out, "encoder")
    if a.skip_finetune and (enc_dir / "classes.json").is_file():
        print("[2/5] reusing fine-tuned encoder", enc_dir)
        ft = json.loads((enc_dir / "training.json").read_text(encoding="utf-8"))
    else:
        print(f"[2/5] fine-tuning TS Pulse on {len(tr)} training cases ({a.epochs} epochs, device {a.device})")
        ft = finetune(settings.tspulse_clf_dir, X[tr], y[tr], X[va], y[va], classes, enc_dir, epochs=a.epochs, device=a.device)
    enc = Encoder(enc_dir)

    print("[3/5] embeddings, scaler, base head, similarity index, calibration")
    E = enc.embed(X)
    scaler = Scaler.fit(E[tr])
    Z = scaler(E)
    scaler.save(base / "scaler.npz")
    records = [{"id": rows[i]["case_id"], "signature": rows[i]["signature"], "action": rows[i]["action"], "source": "synthetic"} for i in tr]
    Zi = Z[tr]
    for folder in a.extra:
        lab = read_labeled_folder(Path(folder))
        if lab:
            Xe, _ = featurize_files([r["path"] for r in lab])
            Ze = scaler(enc.embed(Xe))
            Zi = np.vstack([Zi, Ze])
            records += [{"id": f"real:{r['file']}", "signature": r["signature"], "action": r["action"], "source": "real", "note": r.get("notes", "")} for r in lab]
            print(f"      + {len(lab)} real labeled cases from {folder}")
    index = SimilarityIndex(Zi, records)
    index.save(base, "index")
    w = np.array([1.0 if r["source"] == "synthetic" else 3.0 for r in records])
    head = train_head(Zi, [r["signature"] for r in records], w)
    head.meta.update({"trained": datetime.now().isoformat(timespec="seconds"), "data": "synthetic train split" + (" + real labeled cases" if a.extra else "")})
    head.save(base)
    (base / "meta.json").write_text(json.dumps({"created": head.meta["trained"], "trigger": "shipped base model (scripts/train_e2e.py)"}, indent=2), encoding="utf-8")
    cal = calibrate(index, Z[va], list(sig[va]))
    cal.save(base / "calibration.json")
    print(f"      novelty threshold {cal.novelty_threshold}  one-shot threshold {cal.match_threshold}  ({cal.notes})")

    print("[4/5] held-out set for the learning dashboard (test split + rule-path actions)")
    eval_rows = [r for r in rows if r["split"] != "train"]
    rr = rule_results(root, eval_rows, cache, workers=a.workers)
    rules_action = np.array([rr.get(rows[i]["case_id"], {}).get("action", "error") for i in te])
    np.savez_compressed(base / "heldout.npz", Z=Z[te].astype(np.float16), signature=sig[te], rules_action=rules_action, case_id=np.array([rows[i]["case_id"] for i in te]))

    print("[5/6] residual feed calibration on the validation split")
    rinfo = residual_calibration(settings, root, [rows[i] for i in va], Z[va], base, cache)
    head_val = float(np.mean(np.array(head.predict(Z[va])) == sig[va]))
    card = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "encoder": enc.description,
        "base_weights": "ibm-granite/granite-timeseries-tspulse-r1 @ tspulse-block-dualhead-512-p16-r1",
        "channels": CHANNELS,
        "signatures": classes,
        "data": {k: int((split == k).sum()) for k in ("train", "val", "test", "variant", "unseen")},
        "finetune": {"best": ft.get("best"), "epochs": len(ft.get("history", []))},
        "head_val_acc": round(head_val, 4),
        "calibration": json.loads((base / "calibration.json").read_text(encoding="utf-8")),
        "residual": rinfo,
        "extra_labeled_folders": a.extra,
        "train_minutes": round((time.monotonic() - t0) / 60, 1),
    }
    (out / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"[6/6] done in {card['train_minutes']} min: fine-tune best val acc {ft.get('best')}, head val acc {head_val:.3f}")
    return 0


def residual_calibration(settings, root: Path, val_rows, Zva: np.ndarray, base: Path, cache: Path):
    """Per-group residual thresholds + residual-novelty rule from the validation split -> calibration.json."""
    from sentinel.e2e import residual
    from sentinel.e2e.dataset import cached_residuals

    truth = [r["signature"] for r in val_rows]
    pre = cached_residuals(root, val_rows, cache, settings)  # default thresholds: channel scores do not depend on them
    th = residual.calibrate_thresholds([pre[r["case_id"]] for r in val_rows], truth)
    res = cached_residuals(root, val_rows, cache, settings, thresholds=th["thresholds"])
    cal = Calibration.load(base / "calibration.json")
    cal.residual_thresholds = th["thresholds"]
    head = Head.load(base)
    index = SimilarityIndex.load(base, "index")
    cal.residual_novelty = False
    dec = [decide(z, head, index, cal) for z in Zva]
    grid = calibrate_residual(cal, dec, [res[r["case_id"]] for r in val_rows], truth)
    cal.save(base / "calibration.json")
    mode = next(iter(res.values()))["mode"] if res else "?"
    print(f"      residual mode {mode}; thresholds {th['thresholds']}")
    print(f"      residual novelty: {cal.residual_notes}")
    return {"mode": mode, "thresholds": th, "novelty": grid.get("chosen"), "notes": cal.residual_notes}


if __name__ == "__main__":
    raise SystemExit(main())
