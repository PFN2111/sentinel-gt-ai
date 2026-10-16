#!/usr/bin/env python3
"""Build docs/e2e_explainer.html — single self-contained offline explainer of the end-to-end AI path.

Inputs (all real): eval/results.json (scripts/eval_e2e.py), eval/examples/*.md (real LLM reports),
models/e2e/encoder/training.json, and live feature/decision runs on a few dataset cases.

    python scripts/build_e2e_explainer.py
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from build_overview import md_to_html  # noqa: E402
from sentinel.config import load_settings  # noqa: E402
from sentinel.e2e.features import CHANNEL_LABELS, CHANNELS, featurize_path  # noqa: E402

TEMPLATE = ROOT / "scripts" / "e2e_explainer_template.html"
OUT = ROOT / "docs" / "e2e_explainer.html"
SHORT = {
    "normal": "normal", "cold_spot_hardware": "cold spot", "tc_sensor_fault": "TC fault", "combustion_instability": "instability",
    "dynamics_probe_fault": "probe fault", "hets_trip_hardware": "HETS", "igv_lvdt_drift": "IGV drift", "fuel_control_oscillation": "fuel osc",
    "bearing_vibration": "bearing (unseen)", "tc_harness_fault": "TC harness (unseen)", "novel_pattern": "NOVEL",
}
CHANNEL_SHORT = {
    "load": "load", "egt_resid": "EGT vs load", "spread": "exhaust spread", "cold_dev": "coldest TC", "cold_pair": "coldest adjacent pair",
    "hot_dev": "hottest TC", "ring_flat": "frozen TC", "dyn": "dynamics", "dyn_flat": "dynamics probe health", "vib": "vibration",
    "fuel_resid": "fuel vs load", "igv_resid": "IGV vs load", "trip": "trip flag", "other": "other channels",
}
SAMPLE_CASES = [
    ("test", "cold_spot_hardware", "Cold sector (combustion hardware)"),
    ("test", "tc_sensor_fault", "Single TC fault (instrumentation)"),
    ("test", "hets_trip_hardware", "HETS trip"),
    ("test", "dynamics_probe_fault", "Dynamics probe fault"),
    ("unseen", "bearing_vibration", "Unseen: bearing vibration"),
    ("unseen", "tc_harness_fault", "Unseen look-alike: TC harness fault"),
]


def _pool(x: np.ndarray, n: int) -> List[List[float]]:
    edges = np.linspace(0, len(x), n + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[a:max(b, a + 1)]
        i = np.argmax(np.abs(seg), axis=0)
        out.append([round(float(v), 2) for v in seg[i, np.arange(seg.shape[1])]])
    return out


def samples() -> List[Dict[str, Any]]:
    from sentinel.e2e.dataset import load_manifest
    from sentinel.e2e.engine import Engine

    root = ROOT / "data" / "e2e" / "dataset"
    if not (root / "manifest.jsonl").is_file():
        return []
    rows = load_manifest(root)
    eng = Engine(load_settings({"data_dir": ROOT / ".scratch" / "explainer_data"}))
    out = []
    for split, sig, title in SAMPLE_CASES:
        r = next((r for r in rows if r["split"] == split and r["signature"] == sig), None)
        if r is None:
            continue
        f = featurize_path(str(root / r["file"]))
        d = eng.decide(eng.scaler(eng.embed(f.x))[0])
        half = len(f.x) // 2
        x = _pool(f.x[:half], 64) + _pool(f.x[half:], 64)
        out.append({"title": f"{title} — {r['case_id']}", "truth": sig, "learned": f"{d['signature']} ({d['basis']}, nearest distance {d['nearest_distance']})", "x": x})
    return out


def report_tabs() -> Dict[str, str]:
    ex = sorted((ROOT / "eval" / "examples").glob("*.md"))
    tabs, panels = [], []
    for i, p in enumerate(ex):
        key = f"r{i}"
        label = p.stem.replace("test-", "").replace("unseen-", "unseen: ").replace("-base-", " #").replace("_", " ")
        tabs.append(f'<button role="tab" data-tab="{key}">{html.escape(label)}</button>')
        panels.append(f'<div role="tabpanel" data-panel="{key}"><div class="report-frame">{md_to_html(p.read_text(encoding="utf-8"))}</div></div>')
    if not ex:
        panels.append('<p class="note">No example reports yet — run python scripts/eval_e2e.py --llm-only --llm-cases 8.</p>')
    return {"tabs": "".join(tabs), "panels": "".join(panels)}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


def build() -> str:
    res = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    tr = ROOT / "models" / "e2e" / "legacy" / "encoder" / "training.json"
    if not tr.is_file():
        tr = ROOT / "models" / "e2e" / "encoder" / "training.json"
    ft = json.loads(tr.read_text(encoding="utf-8")).get("history", []) if tr.is_file() else []
    for key in ("unseen", "lookalike"):
        (res.get(key) or {}).pop("members", None)
    data = {"results": res, "finetune": ft, "samples": samples(), "channels": CHANNELS, "channel_labels": CHANNEL_LABELS,
            "channel_short": CHANNEL_SHORT, "short": SHORT}
    rt = report_tabs()
    page = TEMPLATE.read_text(encoding="utf-8")
    for k, v in {
        "%%DATA%%": json.dumps(data, default=str).replace("</", "<\\/"),
        "%%REPORT_TABS%%": rt["tabs"],
        "%%REPORT_PANELS%%": rt["panels"],
        "%%GENERATED%%": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "%%COMMIT%%": git_commit(),
    }.items():
        page = page.replace(k, v)
    return page


def main() -> int:
    page = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
