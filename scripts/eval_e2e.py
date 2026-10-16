#!/usr/bin/env python3
"""Evaluate the end-to-end learned path against the rules and the ground truth -> eval/EVAL.md.

    python scripts/eval_e2e.py                    # all offline metrics + simulations (~5 min)
    python scripts/eval_e2e.py --llm-cases 8      # + real local LLM reports on 8 held-out cases (CPU: ~5 min each)
    python scripts/eval_e2e.py --llm-only         # only (re)run the LLM subset, merge into results.json
    python scripts/eval_e2e.py --profiles         # design 8.4: original 320 test rows, profile system vs legacy
    python scripts/eval_e2e.py --peralert         # design section 7: per-alert acceptance. Does not change e2e_mode.

Held-out data only: the iid test split, the variant split (shifted can, other load/frame,
out-of-range magnitude) and the unseen split (a fault type never in training).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from sentinel.config import load_settings  # noqa: E402
from sentinel.e2e.dataset import cached_features, load_manifest  # noqa: E402
from sentinel.e2e.encoder import Encoder  # noqa: E402
from sentinel.e2e.labels import legacy_model_dir  # noqa: E402
from sentinel.e2e.evaluation import accuracy, confusion, md_confusion, per_class, rule_results  # noqa: E402
from sentinel.e2e.model import Calibration, Head, Scaler, SimilarityIndex, calibrate, decide, train_head  # noqa: E402
from sentinel.e2e.simulate import cold_start, site_adaptation  # noqa: E402
from sentinel.e2e.taxonomy import BASE_SIGNATURES, NOVEL, UNSEEN_SIGNATURES  # noqa: E402

EVAL = ROOT / "eval"
RULE_CLASSES = ["normal", "cold_spot_hardware", "tc_sensor_fault", "combustion_instability", "dynamics_probe_fault", "hets_trip_hardware"]
SHORT = {
    "normal": "normal", "cold_spot_hardware": "cold spot", "tc_sensor_fault": "TC fault", "combustion_instability": "instability",
    "dynamics_probe_fault": "probe fault", "hets_trip_hardware": "HETS", "igv_lvdt_drift": "IGV drift", "fuel_control_oscillation": "fuel osc",
    "bearing_vibration": "bearing (unseen)", "tc_harness_fault": "TC harness (unseen)", NOVEL: "NOVEL", "sensor_fix_replace": "sensor", "borescope": "borescope", "lvdt_calibration": "LVDT",
    "combustion_tuning": "tuning", "control_adjustment": "control", "monitor_only": "monitor", "escalate": "escalate", "error": "error",
}


def act(sig: str) -> str:
    if sig == NOVEL:
        return "escalate"
    return (BASE_SIGNATURES.get(sig) or UNSEEN_SIGNATURES.get(sig) or {}).get("action", "escalate")


def auroc(pos: List[float], neg: List[float]) -> float:
    """P(score_pos > score_neg)."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos), np.asarray(neg)
    return float(((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()))


def _pct(value: Any) -> str:
    """Four-decimal accuracy as a percent, matching the 89.69% baseline line."""
    if value is None:
        return "—"
    return f"{100.0 * float(value):.2f}%"


def _bundle_metrics(truth: List[str], pred: List[str], novel: List[bool]) -> Dict[str, Any]:
    """Action is the taxonomy action of the signature. NOVEL counts as escalate.

    That is the same ``act`` the legacy test block uses. The profile decider's
    ``primary_action`` is the neighbor's action or the rule action, and it is
    not this number.
    """
    n = len(truth)
    if n == 0:
        return {"n": 0, "signature_acc": None, "action_acc": None, "false_novelty": None}
    sig_ok = sum(a == b for a, b in zip(truth, pred))
    act_ok = sum(act(a) == act(b) for a, b in zip(truth, pred))
    return {
        "n": n,
        "signature_acc": round(sig_ok / n, 4),
        "action_acc": round(act_ok / n, 4),
        "false_novelty": round(sum(bool(flag) for flag in novel) / n, 4),
    }


def _by_key(keys: List[str], truth: List[str], pred: List[str], novel: List[bool]) -> Dict[str, Dict[str, Any]]:
    """One metrics block per key. Empty keys are dropped, same as the legacy slice table."""
    grouped: Dict[str, List[int]] = {}
    for i, key in enumerate(keys):
        if key:
            grouped.setdefault(str(key), []).append(i)
    return {
        key: _bundle_metrics([truth[i] for i in idx], [pred[i] for i in idx], [novel[i] for i in idx])
        for key, idx in grouped.items()
    }


def _shipped_embedders(model_dir: Path) -> Dict[str, str]:
    """Recorded embedder on each shipped profile bundle. Missing meta is omitted."""
    out: Dict[str, str] = {}
    root = Path(model_dir) / "profiles"
    if not root.is_dir():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict):
            out[folder.name] = str(meta.get("embedder") or "")
    return out


def _acceptance_gate(
    legacy: Dict[str, Any],
    profile: Dict[str, Any],
    legacy_slices: Dict[str, Dict[str, Any]],
    profile_slices: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Design 8.2, applied to the 8.4 comparison.

    A tie is not worse. Alert slices are the truth signatures. A slice counts
    only with at least ``SLICE_MIN_CASES`` on both sides. False novelty is the
    overall rate and may rise by at most ``NOVELTY_GATE_TOL``. Frame slices are
    reported and do not gate: 8.4 names alert slices.
    """
    from sentinel.config import NOVELTY_GATE_TOL, SLICE_MIN_CASES

    reasons: List[str] = []
    if (
        legacy.get("action_acc") is None
        or legacy.get("signature_acc") is None
        or profile.get("action_acc") is None
        or profile.get("signature_acc") is None
        or float(profile["action_acc"]) + 1e-9 < float(legacy["action_acc"])
        or float(profile["signature_acc"]) + 1e-9 < float(legacy["signature_acc"])
    ):
        reasons.append(
            "overall action or signature accuracy dropped "
            f"(action {legacy.get('action_acc')} -> {profile.get('action_acc')}, "
            f"signature {legacy.get('signature_acc')} -> {profile.get('signature_acc')})"
        )
    for name, brow in legacy_slices.items():
        if int(brow.get("n") or 0) < int(SLICE_MIN_CASES):
            continue
        arow = profile_slices.get(name)
        if not isinstance(arow, dict) or int(arow.get("n") or 0) < int(SLICE_MIN_CASES):
            reasons.append(f"slice {name} is missing on the profile side")
            continue
        if float(arow["action_acc"]) + 1e-9 < float(brow["action_acc"]) or float(arow["signature_acc"]) + 1e-9 < float(brow["signature_acc"]):
            reasons.append(
                f"slice {name} dropped (n={arow['n']}, action {_pct(brow['action_acc'])} -> {_pct(arow['action_acc'])}, "
                f"signature {_pct(brow['signature_acc'])} -> {_pct(arow['signature_acc'])})"
            )
    before_rate = legacy.get("false_novelty")
    after_rate = profile.get("false_novelty")
    if before_rate is not None and after_rate is not None and float(after_rate) > float(before_rate) + float(NOVELTY_GATE_TOL) + 1e-12:
        reasons.append(f"false novelty rose from {_pct(before_rate)} to {_pct(after_rate)} (tolerance {float(NOVELTY_GATE_TOL):.2f})")
    return reasons


def legacy_decisions(Z: np.ndarray, frames: Sequence[str], head: Head, index: SimilarityIndex, cal: Calibration) -> List[Dict[str, Any]]:
    """Legacy decision per scaled embedding, as the app serves that version.

    A retrained version can have a frame-conditioned head (``meta.uses_frame``):
    its head and index are 672 wide plus one layout one-hot column per frame.
    ``Engine.decide`` and ``score_heldout`` append the case's layout one-hot and
    pass the frame so a per-frame radius applies; this does the same. The
    shipped head has no frame column and its calibration no frame radius, so
    both calls leave the shipped decisions unchanged.
    """
    from sentinel.e2e.model import project_embedding

    return [
        decide(project_embedding(Z[i], frames[i], head), head, index, cal, frame=frames[i] or None)
        for i in range(len(Z))
    ]


def _legacy_folder_scores(
    enc_dir: Path,
    base_dir: Path,
    root: Path,
    rows: List[Dict[str, Any]],
    indices: np.ndarray,
    announce: str,
) -> Dict[str, Any]:
    """Score manifest rows with one 14-channel legacy bundle.

    ``decide`` takes the embedding, the head, the index and the calibration.
    It does not take the alert name, so the same vectors are the decision for
    the original name and the side-car name.

    A retrained version can have a frame-conditioned head; see ``legacy_decisions``.
    """
    from sentinel.config import frame_of_row

    print(f"{announce}: loading the feature cache and the 14-channel bundle ...", flush=True)
    X = cached_features(root, rows, ROOT / "data" / "e2e" / "cache")
    enc = Encoder(enc_dir)
    print(f"{announce}: embedding {len(indices)} rows ...", flush=True)
    Z = Scaler.load(Path(base_dir) / "scaler.npz")(enc.embed(X[indices]))
    del enc
    head = Head.load(base_dir)
    index = SimilarityIndex.load(base_dir, "index")
    cal = Calibration.load(Path(base_dir) / "calibration.json")
    frames = [frame_of_row(rows[int(i)]) for i in indices]
    decisions = legacy_decisions(Z, frames, head, index, cal)
    truth = [str(rows[int(i)]["signature"]) for i in indices]
    pred = [str(d["signature"]) for d in decisions]
    novel = [bool(d["novel"]) for d in decisions]
    ids = [str(rows[int(i)]["case_id"]) for i in indices]
    return {
        "metrics": _bundle_metrics(truth, pred, novel),
        "pred": pred,
        "novel": novel,
        "truth": truth,
        "case_id": ids,
    }


def _legacy_test_scores(root: Path, rows: List[Dict[str, Any]], te: np.ndarray, settings: Any) -> Dict[str, Any]:
    """Fresh legacy rescore of the test rows. The feature cache is the full manifest key."""
    mdir = Path(settings.e2e_model_dir)
    legacy_enc = legacy_model_dir(mdir, "encoder")
    legacy_base = legacy_model_dir(mdir, "base")
    # The announce text stays "legacy:" so the shipped-bundle log line is unchanged.
    return _legacy_folder_scores(legacy_enc, legacy_base, root, rows, te, "legacy")


def _prepare_profile_cases(root: Path, rows: List[Dict[str, Any]], te: np.ndarray, settings: Any, shipped: Dict[str, str]) -> List[Dict[str, Any]]:
    """Resolve each test CSV. Windows are built only when a shipped bundle can score them.

    A cold decision (no bundle, channel-count mismatch, unknown name, or ``_adhoc``)
    is ``novel_pattern`` whether or not the window is embedded, so those files
    are not embedded. The alert name selects the alert type. Tags do not.
    ``get_engine`` is not called: a profiles-mode engine would migrate the plant label store.
    """
    from sentinel.data import load_csv
    from sentinel.e2e.features import profile_windows
    from sentinel.e2e.train import load_profile_model
    from sentinel.profiles import compile_channels, resolve_alert_type, resolve_tags

    bundles: Dict[tuple, Any] = {}

    def bundle(pid: str, embedder: str):
        key = (pid, embedder)
        if key not in bundles:
            if not pid or pid in {"_adhoc", "_legacy"}:
                bundles[key] = None
            else:
                bundles[key] = load_profile_model(settings, pid, channels=None, embedder=embedder)
        return bundles[key]

    prepared: List[Dict[str, Any]] = []
    n = len(te)
    t0 = time.monotonic()
    for done, idx in enumerate(te, start=1):
        row = rows[int(idx)]
        path = root / row["file"]
        case = load_csv(path)
        analytic = str(row.get("analytic") or "")
        choice = resolve_alert_type(analytic, settings=settings)
        if choice.profile is None:
            # An unknown name is not routed by tags. The learned path does not run.
            prepared.append(
                {
                    "case_id": row["case_id"],
                    "truth": str(row["signature"]),
                    "analytic": analytic,
                    "profile": "",
                    "why": "unknown alert name",
                    "frame_key": "",
                    "channels": 0,
                    "missing_required": False,
                    "usable": {},
                    "notes": {},
                    "x": None,
                    "mask": None,
                }
            )
            if done == 1 or done % 20 == 0 or done == n:
                elapsed = time.monotonic() - t0
                print(f"profiles: resolved {done}/{n} in {elapsed:.0f}s (last {row['case_id']} -> unknown)", flush=True)
            continue
        profile = choice.profile
        resolved = resolve_tags(profile, case, frame=case.frame_type)
        why = f"alert type {profile.id} ({choice.how})"
        pid = str(profile.id)
        specs = compile_channels(profile, resolved)
        channels = len(specs)
        missing_required = any(hit.required and hit.status != "present" for hit in resolved.slots)
        choices = ["tspulse_ci"]
        recorded = shipped.get(pid) or ""
        if recorded and recorded not in choices:
            choices.append(recorded)
        usable: Dict[str, bool] = {}
        notes: Dict[str, str] = {}
        for embedder in choices:
            model = bundle(pid, embedder)
            if model is None:
                usable[embedder] = False
                notes[embedder] = "no bundle for this embedder"
            elif int(model.index.channels) != channels:
                usable[embedder] = False
                notes[embedder] = f"channel count {channels} != shipped {int(model.index.channels)}"
            else:
                usable[embedder] = True
                notes[embedder] = "shipped bundle"
        item: Dict[str, Any] = {
            "case_id": row["case_id"],
            "truth": str(row["signature"]),
            "analytic": str(row.get("analytic") or ""),
            "profile": pid,
            "why": why,
            "frame_key": str(resolved.frame_key or ""),
            "channels": channels,
            "missing_required": missing_required,
            "usable": usable,
            "notes": notes,
            "x": None,
            "mask": None,
        }
        if any(usable.values()):
            feats = profile_windows(case, profile, resolved)
            item["x"] = np.asarray(feats.x, dtype=np.float32)
            mask = feats.mask if feats.mask is not None else np.ones(item["x"].shape[1], dtype=bool)
            item["mask"] = np.asarray(mask, dtype=bool)
        prepared.append(item)
        if done == 1 or done % 20 == 0 or done == n:
            elapsed = time.monotonic() - t0
            print(f"profiles: resolved {done}/{n} in {elapsed:.0f}s (last {row['case_id']} -> {pid})", flush=True)
    return prepared


def _embed_groups(prepared: List[Dict[str, Any]], embedder_name: str, settings: Any) -> tuple:
    """Embed every case this embedder can score. Returns vectors aligned with ``prepared`` and the fallback flag."""
    from sentinel.e2e.encoder import get_embedder
    from sentinel.e2e.labels import Registry
    from sentinel.e2e.model import ChannelScaler

    active = get_embedder(embedder_name, settings=settings)
    fallback = bool(getattr(active, "fallback", False))
    used_name = str(getattr(active, "name", embedder_name) or embedder_name)
    if fallback or used_name != embedder_name:
        # A silent stats run is not the tspulse_ci column. Callers treat fallback as a failed load.
        return [None] * len(prepared), True, str(getattr(active, "fallback_reason", "") or f"embedder {embedder_name} was not used")
    identity = ChannelScaler(np.zeros(48, dtype=np.float64), np.ones(48, dtype=np.float64))
    scaler = identity
    if used_name == "tspulse_ci":
        folder = Registry(settings).encoder_folder(Registry(settings).current_encoder())
        path = Path(folder) / "channel_scaler.npz"
        if path.is_file():
            scaler = ChannelScaler.load(path)
    elif used_name == "stats":
        from sentinel.e2e.train import stats_scaler

        scaler = stats_scaler(settings)
    groups: Dict[int, List[int]] = {}
    for i, item in enumerate(prepared):
        if item["x"] is None or not item["usable"].get(embedder_name):
            continue
        groups.setdefault(int(item["channels"]), []).append(i)
    vectors: List[Any] = [None] * len(prepared)
    for channels, idxs in groups.items():
        print(f"profiles: embedding {len(idxs)} cases x {channels} channels with {used_name} ...", flush=True)
        stacked = np.stack([prepared[i]["x"] for i in idxs], axis=0)
        masks = np.stack([prepared[i]["mask"] for i in idxs], axis=0)
        raw = np.asarray(active.embed_channels(stacked, masks), dtype=np.float32)
        scaled = scaler.transform(raw, masks)
        for row, i in enumerate(idxs):
            vectors[i] = scaled[row]
    return vectors, False, ""


def _decide_prepared(prepared: List[Dict[str, Any]], vectors: List[Any], embedder_name: str, settings: Any, fallback: bool, fallback_reason: str) -> List[Dict[str, Any]]:
    """One decision per case. Cold, or a failed embedder load, is novel_pattern."""
    from sentinel.e2e.model import ProfileModel
    from sentinel.e2e.taxonomy import NOVEL
    from sentinel.e2e.train import load_profile_model

    models: Dict[str, Any] = {}

    def model_for(pid: str):
        if pid not in models:
            if not pid or pid in {"_adhoc", "_legacy"}:
                models[pid] = None
            else:
                models[pid] = load_profile_model(settings, pid, channels=None, embedder=embedder_name)
        return models[pid]

    out: List[Dict[str, Any]] = []
    for item, vector in zip(prepared, vectors):
        pid = item["profile"]
        model = model_for(pid)
        loaded = (
            not fallback
            and model is not None
            and vector is not None
            and int(model.index.channels) == int(item["channels"])
        )
        if not loaded:
            decision = ProfileModel.empty(pid, channels=int(item["channels"])).decide_profile(
                np.zeros((int(item["channels"]), 48), dtype=np.float32),
                np.zeros(int(item["channels"]), dtype=bool),
                item["frame_key"],
                missing_required=bool(item["missing_required"]),
                fallback=fallback,
            )
            tier = "cold"
            signature = NOVEL
            novel = True
        else:
            mask = item["mask"]
            stats_blocks = None
            if model.head is not None and (model.head.meta or {}).get("stats_view") and item.get("x") is not None:
                from sentinel.e2e.encoder import StatsEmbedder
                from sentinel.e2e.train import stats_scaler

                raw = StatsEmbedder().embed_channels(item["x"], mask)
                stats_blocks = stats_scaler(settings).transform(raw, mask)
            decision = model.decide_profile(
                vector,
                mask,
                item["frame_key"],
                missing_required=bool(item["missing_required"]),
                fallback=False,
                stats=stats_blocks,
            )
            tier = str(decision.get("tier") or model.tier())
            signature = str(decision.get("signature") or NOVEL)
            novel = bool(decision.get("novel"))
        out.append(
            {
                "case_id": item["case_id"],
                "truth": item["truth"],
                "analytic": item["analytic"],
                "profile": pid,
                "why": item["why"],
                "frame_key": item["frame_key"],
                "channels": item["channels"],
                "tier": tier,
                "signature": signature,
                "novel": novel,
                "embedder": embedder_name if not fallback else "stats",
                "fallback": fallback,
                "fallback_reason": fallback_reason if fallback else "",
                "model_note": item["notes"].get(embedder_name) or ("embedder fallback" if fallback else ""),
            }
        )
    return out


def _score_column(prepared: List[Dict[str, Any]], embedder_name: str, settings: Any) -> Dict[str, Any]:
    """Metrics for one embedder column, plus the routing counts."""
    vectors, fallback, reason = _embed_groups(prepared, embedder_name, settings)
    rows = _decide_prepared(prepared, vectors, embedder_name, settings, fallback, reason)
    truth = [row["truth"] for row in rows]
    pred = [row["signature"] for row in rows]
    novel = [bool(row["novel"]) for row in rows]
    routing: Dict[str, int] = {}
    tiers: Dict[str, int] = {}
    by_sig: Dict[str, Dict[str, int]] = {}
    for row in rows:
        routing[row["profile"]] = routing.get(row["profile"], 0) + 1
        tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
        by_sig.setdefault(row["truth"], {})
        by_sig[row["truth"]][row["profile"]] = by_sig[row["truth"]].get(row["profile"], 0) + 1
    return {
        "embedder": embedder_name,
        "fallback": fallback,
        "fallback_reason": reason,
        "metrics": _bundle_metrics(truth, pred, novel),
        "by_alert": _by_key(truth, truth, pred, novel),
        "by_frame": _by_key([row["frame_key"] for row in rows], truth, pred, novel),
        "routing": routing,
        "routing_by_signature": by_sig,
        "tiers": tiers,
        "cases": rows,
    }


def _md_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _write_dynamic(payload: Dict[str, Any]) -> None:
    """``eval/EVAL_DYNAMIC.md`` and ``eval/results_dynamic.json``. The legacy ``results.json`` is left alone."""
    EVAL.mkdir(exist_ok=True)
    legacy = payload["legacy"]
    column = payload["tspulse_ci"]
    profile = column["metrics"]
    reasons = payload["reasons"]
    verdict = "PASS" if payload["passed"] else "FAIL"
    lines: List[str] = []
    lines.append("# Alert-type profiles — acceptance (design 8.4)")
    lines.append("")
    lines.append(f"Created {payload['created']}. Dataset `{payload['dataset']}`, split `test`, n={payload['n']}.")
    lines.append("")
    lines.append(
        "The profile system routes each case by alert name and tags, embeds with the channel-independent "
        "encoder, and decides with that alert type's shipped bundle. Action accuracy is the taxonomy action "
        "of the predicted signature (a novel call is escalate). That is the same definition as the legacy rescore."
    )
    lines.append("")
    lines.append(
        "Recorded baseline for this machine's active legacy v0002: 89.69% action and signature, 2.19% false novelty. "
        "The gate below uses a fresh rescore of these same rows."
    )
    lines.append("")
    lines.append(f"**8.4: {verdict}.**")
    lines.append("")
    if reasons:
        lines.append("Reasons:")
        lines.append("")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")
    else:
        lines.append("Not worse overall and on every alert slice with at least 8 cases. False novelty did not rise by more than 0.01.")
        lines.append("")
    if payload["embedder_note"]:
        lines.append(payload["embedder_note"])
        lines.append("")
    if column.get("fallback"):
        lines.append(f"The `tspulse_ci` embedder did not load ({column.get('fallback_reason') or 'fallback'}). This column is not a `tspulse_ci` score.")
        lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.extend(
        _md_table(
            ["", "n", "signature", "action", "false novelty"],
            [
                ["Legacy rescore", str(legacy["n"]), _pct(legacy["signature_acc"]), _pct(legacy["action_acc"]), _pct(legacy["false_novelty"])],
                ["Profile system (`tspulse_ci`)", str(profile["n"]), _pct(profile["signature_acc"]), _pct(profile["action_acc"]), _pct(profile["false_novelty"])],
            ],
        )
    )
    lines.append("")
    lines.append("## Alert slices (truth signature)")
    lines.append("")
    lines.append("A slice gates when both sides have at least 8 cases. A tie holds.")
    lines.append("")
    slice_rows = []
    names = list(payload["slice_order"])
    for name in sorted(set(payload["legacy_by_alert"]) | set(column["by_alert"])):
        if name not in names:
            names.append(name)
    for name in names:
        brow = payload["legacy_by_alert"].get(name)
        arow = column["by_alert"].get(name)
        if not brow or not arow:
            continue
        holds = (
            float(arow["action_acc"]) + 1e-9 >= float(brow["action_acc"])
            and float(arow["signature_acc"]) + 1e-9 >= float(brow["signature_acc"])
        )
        slice_rows.append(
            [
                name,
                str(brow["n"]),
                _pct(brow["signature_acc"]),
                _pct(arow["signature_acc"]),
                _pct(brow["action_acc"]),
                _pct(arow["action_acc"]),
                _pct(brow["false_novelty"]),
                _pct(arow["false_novelty"]),
                "hold" if holds else "drop",
            ]
        )
    lines.extend(
        _md_table(
            ["signature", "n", "legacy sig", "profile sig", "legacy action", "profile action", "legacy false novelty", "profile false novelty", "8.4"],
            slice_rows,
        )
    )
    lines.append("")
    lines.append("## Where the cases routed")
    lines.append("")
    lines.append("Resolved alert type, then the truth signature that landed there. `_adhoc` is a cold start.")
    lines.append("")
    route_rows = [[pid, str(n)] for pid, n in sorted(column["routing"].items(), key=lambda kv: (-kv[1], kv[0]))]
    lines.extend(_md_table(["alert type", "n"], route_rows))
    lines.append("")
    lines.append("Tiers: " + ", ".join(f"{name} {n}" for name, n in sorted(column["tiers"].items())) + ".")
    lines.append("")
    cross = []
    for sig in names:
        counts = column["routing_by_signature"].get(sig) or {}
        if not counts:
            continue
        shown = ", ".join(f"{pid} {n}" for pid, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        cross.append([sig, shown])
    lines.extend(_md_table(["truth signature", "resolved alert types"], cross))
    lines.append("")
    lines.append("## Frame keys (not part of the 8.4 gate)")
    lines.append("")
    frame_rows = []
    for name, arow in sorted(column["by_frame"].items()):
        frame_rows.append([name or "(blank)", str(arow["n"]), _pct(arow["signature_acc"]), _pct(arow["action_acc"]), _pct(arow["false_novelty"])])
    lines.extend(_md_table(["frame key", "n", "signature", "action", "false novelty"], frame_rows))
    lines.append("")
    if payload.get("per_profile_column") is not None:
        other = payload["per_profile_column"]
        lines.append("## Per-profile embedder choice")
        lines.append("")
        lines.append("A shipped bundle whose recorded embedder is not `tspulse_ci` is scored again in that embedder. The other cases keep the `tspulse_ci` decision.")
        lines.append("")
        om = other["metrics"]
        lines.extend(
            _md_table(
                ["", "n", "signature", "action", "false novelty"],
                [["Per-profile embedder", str(om["n"]), _pct(om["signature_acc"]), _pct(om["action_acc"]), _pct(om["false_novelty"])]],
            )
        )
        lines.append("")
        if other.get("reasons"):
            for reason in other["reasons"]:
                lines.append(f"- {reason}")
            lines.append("")
    (EVAL / "EVAL_DYNAMIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sidecar = dict(payload)
    # The markdown is the review surface. The sidecar keeps one row per case for the slice audit.
    (EVAL / "results_dynamic.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")


def score_acceptance(dataset: Path) -> int:
    """Design 8.4 on the original held-out split. Writes the report and returns 0 either way.

    The pass/fail is in the report. This function does not change ``e2e_mode``.
    A temporary data dir keeps the registry on the shipped bundles and keeps
    the migration hook away from the plant label store.
    """
    import tempfile

    root = Path(dataset)
    rows = load_manifest(root)
    te = np.flatnonzero(np.array([r["split"] for r in rows]) == "test")
    if len(te) != 320:
        raise SystemExit(f"acceptance set must be the original 320 test rows, found {len(te)} in {root}")
    with tempfile.TemporaryDirectory(prefix="sentinel_accept_") as tmp:
        settings = load_settings(
            {
                "data_dir": tmp,
                "e2e_model_dir": str(ROOT / "models" / "e2e"),
                "use_llm": False,
                "use_tspulse": False,
                "embedder": "tspulse_ci",
                "e2e_mode": "legacy",
            }
        )
        shipped = _shipped_embedders(Path(settings.e2e_model_dir))
        legacy_block = _legacy_test_scores(root, rows, te, settings)
        print("profiles: resolving alert types ...", flush=True)
        prepared = _prepare_profile_cases(root, rows, te, settings, shipped)
        column = _score_column(prepared, "tspulse_ci", settings)
        truth = legacy_block["truth"]
        legacy_by_alert = _by_key(truth, truth, legacy_block["pred"], legacy_block["novel"])
        reasons = _acceptance_gate(legacy_block["metrics"], column["metrics"], legacy_by_alert, column["by_alert"])
        if column["fallback"]:
            reasons.insert(0, f"tspulse_ci did not load ({column['fallback_reason'] or 'fallback'})")
        same = bool(shipped) and all(value == "tspulse_ci" for value in shipped.values())
        per_profile = None
        note = ""
        if same:
            note = (
                "Every shipped bundle records embedder `tspulse_ci` ("
                + ", ".join(f"{pid}" for pid in sorted(shipped))
                + "). The per-profile embedder column is this same run."
            )
        else:
            differing = {pid: value for pid, value in shipped.items() if value != "tspulse_ci"}
            note = "Shipped embedders differ from `tspulse_ci`: " + ", ".join(f"{pid}={value or 'unset'}" for pid, value in sorted(differing.items()))
            # Cases whose bundle is tspulse_ci keep that decision. The rest are scored in the recorded embedder.
            mixed_rows = []
            by_embedder: Dict[str, List[int]] = {}
            for i, item in enumerate(prepared):
                recorded = shipped.get(item["profile"]) or "tspulse_ci"
                by_embedder.setdefault(recorded, []).append(i)
            decided = [None] * len(prepared)
            tspulse_rows = {row["case_id"]: row for row in column["cases"]}
            for embedder, idxs in by_embedder.items():
                if embedder == "tspulse_ci":
                    for i in idxs:
                        decided[i] = tspulse_rows[prepared[i]["case_id"]]
                    continue
                sub = [prepared[i] for i in idxs]
                sub_column = _score_column(sub, embedder, settings)
                for i, row in zip(idxs, sub_column["cases"]):
                    decided[i] = row
            assert all(row is not None for row in decided)
            pred = [row["signature"] for row in decided]
            novel = [bool(row["novel"]) for row in decided]
            per_metrics = _bundle_metrics(truth, pred, novel)
            per_slices = _by_key(truth, truth, pred, novel)
            per_profile = {
                "metrics": per_metrics,
                "by_alert": per_slices,
                "reasons": _acceptance_gate(legacy_block["metrics"], per_metrics, legacy_by_alert, per_slices),
            }
            reasons.extend(f"per-profile embedder: {reason}" for reason in per_profile["reasons"])
        passed = not reasons
        order = list(BASE_SIGNATURES)
        payload = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(root),
            "n": int(len(te)),
            "passed": passed,
            "reasons": reasons,
            "embedder_note": note,
            "shipped_embedders": shipped,
            "legacy": legacy_block["metrics"],
            "legacy_by_alert": legacy_by_alert,
            "tspulse_ci": {k: v for k, v in column.items() if k != "cases"} | {"cases": column["cases"]},
            "per_profile_column": per_profile,
            "slice_order": order,
            "e2e_mode_switched": False,
        }
        _write_dynamic(payload)
    verdict = "PASS" if passed else "FAIL"
    print(f"ACCEPTANCE: {verdict}", flush=True)
    print(f"wrote {EVAL / 'EVAL_DYNAMIC.md'}", flush=True)
    return 0


def write_peralert_sidecar(dataset: Path) -> int:
    """Write the frozen alert type of each of the 320 test rows. Does not score them."""
    from sentinel.e2e.dataset import acceptance_sidecar_rows

    rows = acceptance_sidecar_rows(dataset)
    dest = EVAL / "acceptance_peralert.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    dest.write_bytes(text.encode("utf-8"))
    print(f"wrote {dest} ({len(rows)} rows)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Design section 7. Per-alert acceptance. The 8.4 writer above is not used.
#
# Counts are integers. A tie holds. D1 is strict: false novelty has no +1 case
# and no sign test. Options (b) and (c) are computed for the report only.
#
# Two runs (7.6). The design sentence says "the legacy run with the higher
# count and the per-alert run with the lower count". That is the lucky
# direction for accuracy. False novelty is better when the count is lower, so
# the same rule taken so luck cannot pass is:
#   signature and action: legacy max, per-alert min
#   false novelty:        legacy min, per-alert max
# The count is one of the two runs. Disagreeing case ids are listed either way.
# ---------------------------------------------------------------------------

# Same bytes the side-car test pins. The scorer checks the file. It does not rewrite it.
PERALERT_SIDECAR_SHA256 = "85611457ad9bf527bb237ecc82f67b5c43c4fab3591ee91e4bb55cf5bb63b3c1"

_METRICS = ("signature", "action", "false_novelty")

# Source of the decision, hashed by decision_path_sha. Teach and the CLI are not in this list.
_DECISION_FUNCTIONS = (
    "alert_type_for_name",
    "resolve_alert_type",
    "resolve_tags",
    "compile_channels",
    "profile_windows",
    "CIEncoder._forward_chunk",
    "ChannelScaler.transform",
    "block_distances",
    "MaskedIndex.distances",
    "head_input",
    "head_vector",
    "ProfileModel.radius_for",
    "ProfileModel.confidence_floor",
    "ProfileModel.decide_profile",
    "ProfileModel._knn",
    "ProfileModel._head",
    "ProfileModel._finish",
    "decide_alert_case",
)

_G3_REASONS = (
    ("sidecar_sha", "G3 side-car sha256 does not match the pinned value"),
    ("rows_match_manifest", "G3 side-car rows do not match the manifest test rows"),
    ("no_sidecar_id_in_bundle", "G3 a side-car id is in a bundle"),
    ("encoder_loaded", "G3 the shared encoder did not load"),
    ("head_tier", "G3 a case did not reach a head-tier model of its alert type"),
    ("both_runs", "G3 both runs of each side did not complete"),
)

_BUNDLE_FILES = ("head.npz", "head.json", "index.npz", "index.jsonl", "calibration.json", "meta.json", "heldout.npz")


def _count_cell(count: int, n: int) -> str:
    """Integer count and a display percent. The gate compares the count."""
    if n <= 0:
        return "—"
    return f"{int(count)}/{int(n)} ({100.0 * float(count) / float(n):.2f}%)"


def _prediction(run: Dict[str, Any], case_id: str) -> str:
    """A run maps a case id to a signature, or to a dict that carries one."""
    if case_id not in run:
        raise ValueError(f"prediction missing for {case_id}")
    raw = run[case_id]
    if isinstance(raw, dict):
        return str(raw.get("signature") or NOVEL)
    return str(raw)


def _case_flags(truth: str, pred: str) -> Dict[str, int]:
    """One case. False novelty is the novel signature, including a cold or unknown decision."""
    return {
        "signature": int(pred == truth),
        "action": int(act(pred) == act(truth)),
        "false_novelty": int(pred == NOVEL),
    }


def _higher_is_better(metric: str) -> bool:
    return metric != "false_novelty"


def _collapse_value(values: Sequence[int], metric: str, role: str) -> int:
    """Worst run for this metric. ``role`` is ``legacy`` or ``peralert``."""
    if not values:
        raise ValueError("no runs to collapse")
    nums = [int(v) for v in values]
    if role == "legacy":
        return max(nums) if _higher_is_better(metric) else min(nums)
    if role != "peralert":
        raise ValueError(f"unknown role {role}")
    return min(nums) if _higher_is_better(metric) else max(nums)


def _counts_by(cases: Sequence[Dict[str, Any]], run: Dict[str, Any], key_of) -> Dict[str, Dict[str, int]]:
    """Sum integer flags for one run. ``key_of`` returns None to skip a case."""
    buckets: Dict[str, Dict[str, int]] = {}
    for case in cases:
        key = key_of(case)
        if key is None:
            continue
        bucket = buckets.setdefault(str(key), {"n": 0, "signature": 0, "action": 0, "false_novelty": 0})
        flags = _case_flags(str(case["truth"]), _prediction(run, str(case["case_id"])))
        bucket["n"] += 1
        for metric in _METRICS:
            bucket[metric] += flags[metric]
    return buckets


def _collapse_tables(tables: Sequence[Dict[str, Dict[str, int]]], role: str) -> Dict[str, Dict[str, int]]:
    """One table per run, collapsed per slice and metric."""
    keys: Set[str] = set()
    for table in tables:
        keys.update(table)
    out: Dict[str, Dict[str, int]] = {}
    for key in keys:
        present = [table[key] for table in tables if key in table]
        row: Dict[str, int] = {"n": int(present[0]["n"])}
        for metric in _METRICS:
            row[metric] = _collapse_value([block[metric] for block in present], metric, role)
        out[key] = row
    return out


def _best_of(shipped: Dict[str, Dict[str, int]], active: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Harder bar per slice and metric. Higher signature and action, lower false novelty."""
    out: Dict[str, Dict[str, int]] = {}
    for key in set(shipped) | set(active):
        left = shipped.get(key)
        right = active.get(key)
        if left is None or right is None:
            out[key] = dict(left or right or {})
            continue
        row = {"n": int(left["n"])}
        for metric in _METRICS:
            if _higher_is_better(metric):
                row[metric] = max(int(left[metric]), int(right[metric]))
            else:
                row[metric] = min(int(left[metric]), int(right[metric]))
        out[key] = row
    return out


def _verdict(metric: str, bar: int, cand: int) -> str:
    """``hold`` is a tie. ``better`` beats the bar. ``fail`` is worse."""
    if _higher_is_better(metric):
        if cand > bar:
            return "better"
        return "hold" if cand == bar else "fail"
    if cand < bar:
        return "better"
    return "hold" if cand == bar else "fail"


def _record_strings(record: Dict[str, Any]) -> Set[str]:
    """Every string stored on an index row. Ids and CSV hashes are both scanned."""
    found: Set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(record)
    return found


def legacy_active_status(
    current: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    sidecar_ids: Set[str],
    sidecar_hashes: Set[str],
) -> Dict[str, Any]:
    """Whether the active legacy version may raise the bar.

    ``v0000`` or a missing pointer uses the shipped bundle only. That is not a
    failure. Engineer labels are used only when none of the side-car case ids
    or CSV hashes appear in the index. Otherwise the version is left out and
    the reason says why. A non-base pointer with no readable meta is
    ``unscored``: the caller fails acceptance instead of dropping to an easier bar.
    This function does not load a model.
    """
    version = ""
    if isinstance(current, dict):
        version = str(current.get("version") or "")
    if not version:
        return {"status": "absent", "version": "", "reason": "no active legacy pointer; the bar is the shipped bundle"}
    if version == "v0000":
        return {"status": "v0000", "version": version, "reason": "active legacy version is v0000; the bar is the shipped bundle"}
    if not isinstance(meta, dict):
        return {"status": "unscored", "version": version, "reason": f"active version {version} cannot be scored"}
    n_engineer = int(meta.get("n_engineer") or 0)
    has_engineer = n_engineer > 0 or any(str(row.get("source") or "") == "engineer" for row in records)
    if has_engineer:
        overlap_ids = []
        overlap_hashes = []
        for row in records:
            texts = _record_strings(row if isinstance(row, dict) else {})
            overlap_ids.extend(sorted(texts & sidecar_ids))
            overlap_hashes.extend(sorted(texts & sidecar_hashes))
        if overlap_ids or overlap_hashes:
            shown = ", ".join((overlap_ids + overlap_hashes)[:5])
            return {
                "status": "excluded",
                "version": version,
                "reason": (
                    f"active version {version} has engineer labels and overlaps the side-car ({shown}); "
                    "left out of the bar"
                ),
            }
    return {"status": "scored", "version": version, "reason": f"active version {version} is a second legacy source"}


def legacy_version_readable(version_dir: Path) -> bool:
    """The files a legacy rescore needs. Missing any one of them means the version cannot be scored."""
    folder = Path(version_dir)
    needed = ("scaler.npz", "head.npz", "head.json", "index.npz", "index.jsonl", "calibration.json")
    if any(not (folder / name).is_file() for name in needed):
        return False
    enc = folder / "encoder" / "config.json"
    return enc.is_file() or (folder / "config.json").is_file()


def sign_test_p(n_worse: int, n_better: int) -> float:
    """One-sided exact test. P(X >= n_worse) for X binomial(n_worse + n_better, 0.5).

    No discordant cases, or no losses, is 1. A single loss is 0.5, so it does
    not count as worse at alpha 0.05. This is option (c). It does not gate.
    """
    worse = int(n_worse)
    better = int(n_better)
    n = worse + better
    if n <= 0 or worse <= 0:
        return 1.0
    total = sum(math.comb(n, k) for k in range(worse, n + 1))
    return total / float(2 ** n)


def _discordant(cases: Sequence[Dict[str, Any]], legacy_run: Dict[str, Any], per_run: Dict[str, Any], metric: str, alert_type: str) -> Tuple[int, int]:
    """Cases where the two systems differ. ``worse`` means per-alert is the worse one."""
    worse = 0
    better = 0
    for case in cases:
        if str(case.get("alert_type") or "") != alert_type:
            continue
        flags_l = _case_flags(str(case["truth"]), _prediction(legacy_run, str(case["case_id"])))
        flags_p = _case_flags(str(case["truth"]), _prediction(per_run, str(case["case_id"])))
        if metric == "false_novelty":
            if flags_p[metric] > flags_l[metric]:
                worse += 1
            elif flags_l[metric] > flags_p[metric]:
                better += 1
        elif flags_p[metric] < flags_l[metric]:
            worse += 1
        elif flags_p[metric] > flags_l[metric]:
            better += 1
    return worse, better


def _pessimistic_sign(
    cases: Sequence[Dict[str, Any]],
    legacy_runs: Sequence[Dict[str, Any]],
    per_runs: Sequence[Dict[str, Any]],
    metric: str,
    alert_type: str,
) -> Dict[str, Any]:
    """The legacy-run × per-alert-run pairing with the smallest p."""
    best: Optional[Dict[str, Any]] = None
    for legacy_run in legacy_runs:
        for per_run in per_runs:
            worse, better = _discordant(cases, legacy_run, per_run, metric, alert_type)
            p = sign_test_p(worse, better)
            row = {"n_worse": worse, "n_better": better, "p": p}
            if best is None or p < float(best["p"]) or (p == float(best["p"]) and worse > int(best["n_worse"])):
                best = row
    if best is None:
        best = {"n_worse": 0, "n_better": 0, "p": 1.0}
    best["worse"] = bool(float(best["p"]) < 0.05)
    return best


def original_name_signature(rule: str, gated: str) -> str:
    """Rule B is the unknown-name path: novel, no second embed. Rule A reuses the gated signature."""
    if str(rule or "").startswith("B"):
        return NOVEL
    return str(gated)


def legacy_name_columns(signature: str) -> Dict[str, str]:
    """Legacy is name-blind, so both name columns store the one decision."""
    text = str(signature)
    return {"original_name": text, "sidecar_name": text}


def _disagreeing(runs: Sequence[Dict[str, Any]], case_ids: Sequence[str]) -> List[str]:
    """Case ids whose signature is not the same on every run."""
    if len(runs) < 2:
        return []
    out = []
    for case_id in case_ids:
        values = {_prediction(run, case_id) for run in runs}
        if len(values) > 1:
            out.append(case_id)
    return out


def _g3_reasons(flags: Dict[str, Any]) -> List[str]:
    """A missing flag fails closed. G3 is integrity, not a tolerance."""
    return [text for key, text in _G3_REASONS if not flags.get(key)]


def _slice_row(bar: Dict[str, int], cand: Dict[str, int]) -> Dict[str, Any]:
    """Counts, display percents and a verdict for each metric."""
    n = int(bar["n"])
    metrics = {}
    for metric in _METRICS:
        verdict = _verdict(metric, int(bar[metric]), int(cand[metric]))
        metrics[metric] = {
            "legacy": int(bar[metric]),
            "peralert": int(cand[metric]),
            "legacy_pct": None if n == 0 else round(int(bar[metric]) / n, 4),
            "peralert_pct": None if n == 0 else round(int(cand[metric]) / n, 4),
            "verdict": verdict,
        }
    return {"n": n, "metrics": metrics}


def _failures(prefix: str, row: Dict[str, Any], metrics: Sequence[str]) -> List[str]:
    reasons = []
    for metric in metrics:
        block = row["metrics"][metric]
        if block["verdict"] != "fail":
            continue
        reasons.append(
            f"{prefix} {metric} {_count_cell(block['legacy'], row['n'])} legacy vs {_count_cell(block['peralert'], row['n'])} per-alert"
        )
    return reasons


def evaluate_peralert(
    cases: Sequence[Dict[str, Any]],
    legacy_sources: Dict[str, Any],
    peralert_runs: Sequence[Dict[str, Any]],
    *,
    g3: Optional[Dict[str, Any]] = None,
    keep_g2: bool = True,
    sidecar_sha256: str = "",
    bundle_hashes: Optional[Dict[str, Any]] = None,
    decision_sha: str = "",
    git_head: str = "",
    peralert_details: Optional[Sequence[Dict[str, Any]]] = None,
    unseen: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """G1, G2 and G3 from already-scored runs. No model and no held-out file is opened.

    ``legacy_sources['shipped']`` and ``['active']`` each have ``status`` and
    ``runs`` (a list of case-id → signature maps). Active status comes from
    ``legacy_active_status``. ``keep_g2`` is decision D2; the default keeps G2.
    """
    rows = [dict(case) for case in cases]
    case_ids = [str(case["case_id"]) for case in rows]
    shipped = legacy_sources.get("shipped") or {}
    active = legacy_sources.get("active") or {}
    shipped_runs = list(shipped.get("runs") or [])
    active_runs = list(active.get("runs") or [])
    per_runs = list(peralert_runs or [])
    active_info = {
        "status": str(active.get("status") or "absent"),
        "version": str(active.get("version") or ""),
        "reason": str(active.get("reason") or ""),
    }
    use_active = active_info["status"] == "scored" and len(active_runs) >= 1
    flags = dict(g3 or {})
    runs_ok = len(shipped_runs) == 2 and len(per_runs) == 2 and (active_info["status"] != "scored" or len(active_runs) == 2)
    flags["both_runs"] = bool(flags.get("both_runs")) and runs_ok

    def tables_for(runs: Sequence[Dict[str, Any]], key_of, role: str) -> Dict[str, Dict[str, int]]:
        if not runs:
            return {}
        return _collapse_tables([_counts_by(rows, run, key_of) for run in runs], role)

    by_type = lambda case: str(case.get("alert_type") or "")  # noqa: E731
    by_truth = lambda case: str(case.get("truth") or "")  # noqa: E731
    by_frame = lambda case: str(case.get("frame") or "")  # noqa: E731
    by_cell = lambda case: f"{case.get('alert_type')}|{case.get('truth')}"  # noqa: E731
    by_all = lambda case: "all"  # noqa: E731
    by_rule_b = lambda case: str(case.get("alert_type") or "") if str(case.get("rule") or "").startswith("B") else None  # noqa: E731

    shipped_type = tables_for(shipped_runs, by_type, "legacy")
    active_type = tables_for(active_runs, by_type, "legacy") if use_active else {}
    bar_type = _best_of(shipped_type, active_type) if use_active else shipped_type
    cand_type = tables_for(per_runs, by_type, "peralert")

    shipped_truth = tables_for(shipped_runs, by_truth, "legacy")
    active_truth = tables_for(active_runs, by_truth, "legacy") if use_active else {}
    bar_truth = _best_of(shipped_truth, active_truth) if use_active else shipped_truth
    cand_truth = tables_for(per_runs, by_truth, "peralert")

    shipped_all = tables_for(shipped_runs, by_all, "legacy")
    active_all = tables_for(active_runs, by_all, "legacy") if use_active else {}
    bar_all = _best_of(shipped_all, active_all) if use_active else shipped_all
    cand_all = tables_for(per_runs, by_all, "peralert")

    g1: Dict[str, Any] = {}
    reasons: List[str] = []
    if str(shipped.get("status") or "scored") != "scored" or not shipped_runs:
        reasons.append("legacy: shipped bundle was not scored")
    if active_info["status"] == "unscored":
        reasons.append(active_info["reason"] or "legacy: active version cannot be scored")
    for alert_type in sorted(set(bar_type) | set(cand_type)):
        bar = bar_type.get(alert_type)
        cand = cand_type.get(alert_type)
        if bar is None or cand is None:
            reasons.append(f"G1 {alert_type} is missing on one side")
            continue
        g1[alert_type] = _slice_row(bar, cand)
        reasons.extend(_failures(f"G1 {alert_type}", g1[alert_type], _METRICS))

    g2_overall = {}
    g2_truth: Dict[str, Any] = {}
    if bar_all.get("all") and cand_all.get("all"):
        g2_overall = _slice_row(bar_all["all"], cand_all["all"])
        if keep_g2:
            reasons.extend(_failures("G2 overall", g2_overall, ("signature", "action")))
    elif keep_g2 and rows:
        reasons.append("G2 overall is missing")
    for truth in sorted(set(bar_truth) | set(cand_truth)):
        bar = bar_truth.get(truth)
        cand = cand_truth.get(truth)
        if bar is None or cand is None:
            if keep_g2:
                reasons.append(f"G2 truth {truth} is missing on one side")
            continue
        g2_truth[truth] = _slice_row(bar, cand)
        if keep_g2:
            # No minimum size. False novelty is not a G2 metric.
            reasons.extend(_failures(f"G2 truth {truth}", g2_truth[truth], ("signature", "action")))

    reasons.extend(_g3_reasons(flags))

    # Option (b): false novelty may be one case over the bar, accuracy stays strict.
    b_failures = []
    for alert_type, row in g1.items():
        for metric in ("signature", "action"):
            block = row["metrics"][metric]
            if block["peralert"] < block["legacy"]:
                b_failures.append(f"{alert_type} {metric}")
        fn = row["metrics"]["false_novelty"]
        if fn["peralert"] > fn["legacy"] + 1:
            b_failures.append(f"{alert_type} false_novelty")
    # Option (c): pessimistic pairing across the runs that actually form the bar.
    # When the active version is in the bar, both sources are paired and the
    # smaller p is kept. A cell is worse only when p < 0.05.
    c_cells = []
    legacy_pairings: List[Dict[str, Any]] = []
    if shipped_runs:
        legacy_pairings.extend(shipped_runs)
    if use_active:
        legacy_pairings.extend(active_runs)
    for alert_type in sorted(g1):
        for metric in _METRICS:
            if not legacy_pairings or not per_runs:
                cell = {"alert_type": alert_type, "metric": metric, "n_worse": 0, "n_better": 0, "p": 1.0, "worse": False}
            else:
                cell = {"alert_type": alert_type, "metric": metric, **_pessimistic_sign(rows, legacy_pairings, per_runs, metric, alert_type)}
            c_cells.append(cell)
    c_failures = [f"{cell['alert_type']} {cell['metric']}" for cell in c_cells if cell.get("worse")]

    reported_frames = _reported_pair(rows, shipped_runs, active_runs, per_runs, use_active, by_frame)
    reported_cells = _reported_pair(rows, shipped_runs, active_runs, per_runs, use_active, by_cell)
    reported_rule_b = _reported_pair(rows, shipped_runs, active_runs, per_runs, use_active, by_rule_b)
    original_runs = []
    for run in per_runs:
        mapped = {}
        for case in rows:
            gated = _prediction(run, str(case["case_id"]))
            mapped[str(case["case_id"])] = original_name_signature(str(case.get("rule") or ""), gated)
        original_runs.append(mapped)
    reported_original = _reported_pair(rows, shipped_runs, active_runs, original_runs, use_active, by_type)

    case_rows = []
    for case in rows:
        cid = str(case["case_id"])
        detail = {}
        if peralert_details:
            for block in peralert_details:
                if isinstance(block, dict) and cid in block and isinstance(block[cid], dict):
                    detail = block[cid]
        gated = _prediction(per_runs[-1], cid) if per_runs else ""
        original = original_name_signature(str(case.get("rule") or ""), gated) if gated else ""
        case_rows.append(
            {
                "case_id": cid,
                "alert_type": str(case.get("alert_type") or ""),
                "truth": str(case.get("truth") or ""),
                "frame": str(case.get("frame") or ""),
                "rule": str(case.get("rule") or ""),
                "legacy_shipped": [_prediction(run, cid) for run in shipped_runs],
                "legacy_active": [_prediction(run, cid) for run in active_runs] if use_active else [],
                "peralert": [_prediction(run, cid) for run in per_runs],
                "tier": detail.get("tier"),
                "nearest_distance": detail.get("nearest_distance"),
                "radius": detail.get("radius"),
                "head_confidence": detail.get("head_confidence"),
                "confidence_floor": detail.get("confidence_floor"),
                "beyond_radius": detail.get("beyond_radius"),
                "novel": detail.get("novel"),
                "original_name_signature": original,
                "original_name_action": act(original) if original else "",
                "original_name_needs_human": bool(str(case.get("rule") or "").startswith("B") or original == NOVEL),
            }
        )

    passed = not reasons
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n": len(rows),
        "passed": passed,
        "reasons": reasons,
        "d1": "strict",
        "d2_g2": bool(keep_g2),
        "worst_case_rule": (
            "signature and action use the max legacy run and the min per-alert run; "
            "false novelty uses the min legacy run and the max per-alert run"
        ),
        "g1": g1,
        "g2": {"overall": g2_overall, "by_truth": g2_truth},
        "g3": {"flags": flags, "reasons": _g3_reasons(flags)},
        "legacy_active": active_info,
        "legacy_active_used": use_active,
        "disagreeing_ids": {
            "legacy_shipped": _disagreeing(shipped_runs, case_ids),
            "legacy_active": _disagreeing(active_runs, case_ids) if use_active else [],
            "peralert": _disagreeing(per_runs, case_ids),
        },
        "d1_options": {
            "applied": "strict",
            "note": (
                "Options (b) and (c) are not the gate. Either would also require unseen novel recall "
                "to hold against legacy, and that guard is not applied."
            ),
            "b_false_novelty_plus_one": {"passed": not b_failures, "failures": b_failures},
            "c_sign_test": {"passed": not c_failures, "alpha": 0.05, "cells": c_cells, "failures": c_failures},
        },
        "reported": {
            "by_frame": reported_frames,
            "by_type_and_truth": reported_cells,
            "rule_b": reported_rule_b,
            "original_name": reported_original,
            "unseen_novel_recall": list(unseen or []),
        },
        "cases": case_rows,
        "sidecar_sha256": sidecar_sha256,
        "bundle_hashes": bundle_hashes or {},
        "decision_path_sha": decision_sha,
        "git_head": git_head,
        "e2e_mode_switched": False,
    }


def _reported_pair(cases, shipped_runs, active_runs, per_runs, use_active, key_of) -> Dict[str, Any]:
    """Same worst-case counts as the gate, for a slice that does not gate."""
    shipped = _collapse_tables([_counts_by(cases, run, key_of) for run in shipped_runs], "legacy") if shipped_runs else {}
    active = _collapse_tables([_counts_by(cases, run, key_of) for run in active_runs], "legacy") if use_active and active_runs else {}
    bar = _best_of(shipped, active) if use_active else shipped
    cand = _collapse_tables([_counts_by(cases, run, key_of) for run in per_runs], "peralert") if per_runs else {}
    out = {}
    for key in sorted(set(bar) | set(cand)):
        if key not in bar or key not in cand:
            continue
        out[key] = _slice_row(bar[key], cand[key])
    return out


def render_peralert_md(payload: Dict[str, Any]) -> str:
    """Verdict first, then one table per alert type, then G2, G3 and the reported tables."""
    verdict = "PASS" if payload.get("passed") else "FAIL"
    lines = ["# Per-alert acceptance", "", f"**{verdict}**", ""]
    lines.append(
        f"n={payload.get('n')}  D1={payload.get('d1')}  G2={'on' if payload.get('d2_g2') else 'off'}  "
        "e2e_mode not switched."
    )
    lines.append("")
    lines.append(str(payload.get("worst_case_rule") or ""))
    lines.append("")
    reasons = list(payload.get("reasons") or [])
    if reasons:
        lines.append("Failed rules:")
        lines.append("")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")
    else:
        lines.append("Every gated count holds.")
        lines.append("")
    active = payload.get("legacy_active") or {}
    lines.append(f"Legacy active source: {active.get('status') or 'absent'}. {active.get('reason') or ''}")
    lines.append("")
    for alert_type, row in (payload.get("g1") or {}).items():
        lines.append(f"## {alert_type}")
        lines.append("")
        lines.extend(_metric_table(row))
        lines.append("")
    lines.append("## G2")
    lines.append("")
    overall = (payload.get("g2") or {}).get("overall") or {}
    if overall:
        lines.append("Overall (signature and action gate; false novelty does not).")
        lines.append("")
        lines.extend(_metric_table(overall))
        lines.append("")
    lines.append("Truth-signature slices (no minimum size).")
    lines.append("")
    lines.extend(_metric_table_group((payload.get("g2") or {}).get("by_truth") or {}))
    lines.append("")
    lines.append("## G3")
    lines.append("")
    flags = ((payload.get("g3") or {}).get("flags") or {})
    for key, _text in _G3_REASONS:
        mark = "yes" if flags.get(key) else "no"
        lines.append(f"- {key}: {mark}")
    lines.append("")
    lines.append("## Reported, not gating")
    lines.append("")
    reported = payload.get("reported") or {}
    lines.append("Frame slices")
    lines.append("")
    lines.extend(_metric_table_group(reported.get("by_frame") or {}))
    lines.append("")
    lines.append("Alert type and truth signature")
    lines.append("")
    lines.extend(_metric_table_group(reported.get("by_type_and_truth") or {}))
    lines.append("")
    lines.append("Rule B cases")
    lines.append("")
    lines.extend(_metric_table_group(reported.get("rule_b") or {}))
    lines.append("")
    lines.append("Original alert name (rule B is novel, escalate, needs a human; rule A reuses the gated signature).")
    lines.append("")
    lines.extend(_metric_table_group(reported.get("original_name") or {}))
    lines.append("")
    lines.append("Unseen novel recall (one pass, not gated).")
    lines.append("")
    unseen_rows = reported.get("unseen_novel_recall") or []
    if not unseen_rows:
        lines.append("No unseen rows in this report.")
        lines.append("")
    else:
        body = []
        for item in unseen_rows:
            body.append([
                str(item.get("alert_type") or ""),
                str(item.get("n") or 0),
                str(item.get("legacy_novel") if item.get("legacy_novel") is not None else "—"),
                str(item.get("peralert_novel") if item.get("peralert_novel") is not None else "—"),
            ])
        lines.extend(_md_table(["alert type", "n", "legacy novel", "per-alert novel"], body))
        lines.append("")
    options = payload.get("d1_options") or {}
    lines.append("## D1 options (not applied)")
    lines.append("")
    lines.append(str(options.get("note") or ""))
    lines.append("")
    option_b = options.get("b_false_novelty_plus_one") or {}
    option_c = options.get("c_sign_test") or {}
    lines.append(f"- (b) false novelty +1 and strict accuracy: {'pass' if option_b.get('passed') else 'fail'}")
    lines.append(f"- (c) one-sided sign test at 0.05: {'pass' if option_c.get('passed') else 'fail'}")
    lines.append("")
    disagreed = payload.get("disagreeing_ids") or {}
    lines.append("## Two-run disagreements")
    lines.append("")
    for name, ids in disagreed.items():
        shown = ", ".join(ids) if ids else "none"
        lines.append(f"- {name}: {shown}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _metric_table(row: Dict[str, Any]) -> List[str]:
    body = []
    n = int(row.get("n") or 0)
    for metric in _METRICS:
        block = (row.get("metrics") or {}).get(metric) or {}
        body.append([
            metric,
            str(n),
            _count_cell(int(block.get("legacy") or 0), n),
            _count_cell(int(block.get("peralert") or 0), n),
            str(block.get("verdict") or ""),
        ])
    return _md_table(["metric", "n", "legacy", "per-alert", "verdict"], body)


def _metric_table_group(rows: Dict[str, Any]) -> List[str]:
    if not rows:
        return ["None.", ""]
    body = []
    for name, row in rows.items():
        n = int(row.get("n") or 0)
        metrics = row.get("metrics") or {}
        sig = metrics.get("signature") or {}
        act_row = metrics.get("action") or {}
        nov = metrics.get("false_novelty") or {}
        body.append([
            str(name),
            str(n),
            _count_cell(int(sig.get("legacy") or 0), n),
            _count_cell(int(sig.get("peralert") or 0), n),
            str(sig.get("verdict") or ""),
            _count_cell(int(act_row.get("legacy") or 0), n),
            _count_cell(int(act_row.get("peralert") or 0), n),
            str(act_row.get("verdict") or ""),
            _count_cell(int(nov.get("legacy") or 0), n),
            _count_cell(int(nov.get("peralert") or 0), n),
        ])
    return _md_table(
        ["slice", "n", "legacy signature", "per-alert signature", "signature", "legacy action", "per-alert action", "action", "legacy false novelty", "per-alert false novelty"],
        body,
    )


def write_peralert_outputs(payload: Dict[str, Any], dest: Path) -> None:
    """Write the per-alert report, the JSON and one acceptance-run line.

    ``eval/EVAL_DYNAMIC.md`` and ``eval/results_dynamic.json`` are not written
    here. The side-car is not written here.
    """
    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)
    text = render_peralert_md(payload)
    (folder / "EVAL_PERALERT.md").write_bytes(text.encode("utf-8"))
    blob = json.dumps(payload, indent=2, default=str) + "\n"
    (folder / "results_peralert.json").write_bytes(blob.encode("utf-8"))
    _append_acceptance_run(folder / "acceptance_runs.jsonl", payload)


def _append_acceptance_run(path: Path, payload: Dict[str, Any]) -> None:
    """One JSON line per acceptance run. Pass or fail, appended, never rewritten."""
    row = {
        "time": payload.get("created"),
        "git_head": payload.get("git_head") or "",
        "bundle_hashes": payload.get("bundle_hashes") or {},
        "passed": bool(payload.get("passed")),
        "reasons": list(payload.get("reasons") or []),
        "decision_path_sha": payload.get("decision_path_sha") or "",
    }
    line = json.dumps(row, sort_keys=True, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(line.encode("utf-8"))


def decision_function_sources(text: str) -> Dict[str, str]:
    """Source text of each decision function. A missing or duplicated name raises."""
    tree = ast.parse(text)
    found: Dict[str, List[ast.AST]] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual = f"{prefix}.{child.name}" if prefix else child.name
                found.setdefault(qual, []).append(child)
                walk(child, qual if isinstance(child, ast.ClassDef) else qual)
            else:
                walk(child, prefix)

    walk(tree, "")
    sources: Dict[str, str] = {}
    for name in _DECISION_FUNCTIONS:
        nodes = found.get(name) or []
        if len(nodes) != 1:
            raise ValueError(f"decision_path_sha: missing {name}" if not nodes else f"decision_path_sha: duplicate {name}")
        segment = ast.get_source_segment(text, nodes[0])
        if not segment:
            raise ValueError(f"decision_path_sha: missing {name}")
        sources[name] = segment
    return sources


def _builtin_threshold_pin() -> Dict[str, Any]:
    """Built-in threshold values. A site file is not loaded, so the pin is the code."""
    from sentinel.config import FRAME_DIST_PENALTY, HEAD_MIN_LABELS, HEAD_MIN_PER_CLASS, MASK_PENALTY, NOVELTY_QUANTILE
    from sentinel.profiles import load_profiles

    book = load_profiles()
    thresholds = {}
    for pid in sorted(book):
        prof = book[pid]
        if str(getattr(prof, "source", "") or "") != "builtin":
            continue
        raw = prof.thresholds if isinstance(prof.thresholds, dict) else {}
        residual = raw.get("residual") if isinstance(raw.get("residual"), dict) else {}
        thresholds[pid] = {
            "novelty_quantile": raw.get("novelty_quantile", NOVELTY_QUANTILE),
            "knn_vote_min": raw.get("knn_vote_min"),
            "residual": {str(k): residual[k] for k in sorted(residual)},
        }
    return {
        "NOVELTY_QUANTILE": NOVELTY_QUANTILE,
        "HEAD_MIN_LABELS": HEAD_MIN_LABELS,
        "HEAD_MIN_PER_CLASS": HEAD_MIN_PER_CLASS,
        "MASK_PENALTY": MASK_PENALTY,
        "FRAME_DIST_PENALTY": FRAME_DIST_PENALTY,
        "builtin_thresholds": thresholds,
    }


def decision_path_sha(path: Optional[Path] = None) -> str:
    """sha256 of the decision-function source plus the pinned constants.

    Two calls on an unchanged tree return the same 64 hex characters. Teach,
    the CLI and this script are not part of the hash.
    """
    target = Path(path) if path is not None else ROOT / "sentinel_gt_ai.py"
    text = target.read_text(encoding="utf-8")
    sources = decision_function_sources(text)
    payload = {"functions": sources, "constants": _builtin_threshold_pin()}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def acceptance_allows_switch(results: Dict[str, Any], bundle_hashes: Dict[str, Any], decision_sha: str) -> bool:
    """True only when the recorded run passed and still matches the shipped bytes and the decision source.

    Step 16 does not change the default. Step 18 calls this before a switch.
    """
    if not isinstance(results, dict) or not results.get("passed"):
        return False
    recorded_sha = str(results.get("decision_path_sha") or "")
    if not recorded_sha or recorded_sha != str(decision_sha or ""):
        return False
    if results.get("bundle_hashes") != bundle_hashes:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_file_hashes(model_dir: Path) -> Dict[str, Dict[str, str]]:
    """sha256 of each shipped alert-type file, including the stats sub-bundle. Weights are not loaded."""
    root = Path(model_dir) / "profiles"
    out: Dict[str, Dict[str, str]] = {}
    if not root.is_dir():
        return out
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        files: Dict[str, str] = {}
        for name in _BUNDLE_FILES:
            path = folder / name
            if path.is_file():
                files[name] = _sha256_file(path)
        stats = folder / "stats"
        if stats.is_dir():
            for name in _BUNDLE_FILES:
                path = stats / name
                if path.is_file():
                    files[f"stats/{name}"] = _sha256_file(path)
        out[folder.name] = files
    return out


def bundle_case_ids(folder: Path) -> Set[str]:
    """Ids in ``index.jsonl`` and ``heldout.npz``. No torch."""
    found: Set[str] = set()
    index_text = Path(folder) / "index.jsonl"
    if index_text.is_file():
        for line in index_text.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("id"):
                found.add(str(row["id"]))
    held = Path(folder) / "heldout.npz"
    if held.is_file():
        blob = np.load(held, allow_pickle=False)
        try:
            if "case_id" in blob.files:
                for item in blob["case_id"]:
                    text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    if text:
                        found.add(text)
        finally:
            blob.close()
    return found


def sidecar_ids_in_bundles(model_dir: Path, sidecar_ids: Set[str]) -> List[str]:
    """Side-car ids that appear in any alert-type bundle, including ``stats/``."""
    root = Path(model_dir) / "profiles"
    hits: List[str] = []
    if not root.is_dir() or not sidecar_ids:
        return hits
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        ids = bundle_case_ids(folder) | bundle_case_ids(folder / "stats")
        for case_id in sorted(ids & sidecar_ids):
            hits.append(f"{folder.name}:{case_id}")
    return hits


def _git_head() -> str:
    """Current commit, or an empty string when git is not available. No identity is read."""
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _legacy_version_scores(version_dir: Path, root: Path, rows: List[Dict[str, Any]], indices: np.ndarray) -> Dict[str, Any]:
    """Same procedure as the shipped rescore, pointed at one registry version. Read-only."""
    folder = Path(version_dir)
    enc_dir = folder / "encoder" if (folder / "encoder" / "config.json").is_file() else folder
    return _legacy_folder_scores(enc_dir, folder, root, rows, indices, f"legacy {folder.name}")


def _read_index_records(folder: Path) -> List[Dict[str, Any]]:
    path = Path(folder) / "index.jsonl"
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def _sidecar_bytes() -> bytes:
    """Committed side-car bytes with newlines folded to LF. The file is not rewritten."""
    path = EVAL / "acceptance_peralert.jsonl"
    return path.read_bytes().replace(b"\r\n", b"\n")


def _load_sidecar_rows() -> List[Dict[str, Any]]:
    rows = []
    for line in _sidecar_bytes().decode("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class _PeralertEngine:
    """The pieces ``decide_alert_case`` reads. This is not ``Engine`` and does not load the 14-channel model."""

    def __init__(self, settings: Any):
        from sentinel.e2e.labels import Registry
        from sentinel.e2e.model import Calibration, ProfileCache

        self.settings = settings
        self.registry = Registry(settings)
        self.profile_cache = ProfileCache()
        self.cal = Calibration(novelty_threshold=0.5, match_threshold=0.4, low_confidence=0.6)


def _score_peralert_once(dataset: Path, sidecar_rows: Sequence[Dict[str, Any]], settings: Any, engine: Any) -> Dict[str, Any]:
    """One pass over the side-car, in side-car order. One case at a time."""
    from sentinel.data import load_csv
    from sentinel.e2e.pipeline import decide_alert_case
    from sentinel.profiles import resolve_alert_type, resolve_tags

    preds: Dict[str, str] = {}
    details: Dict[str, Any] = {}
    encoder_loaded = True
    head_ok = True
    for row in sidecar_rows:
        case_id = str(row["case_id"])
        case = load_csv(Path(dataset) / str(row["file"]))
        choice = resolve_alert_type(str(row["alert_name"]), settings=settings)
        profile = choice.profile
        if profile is None or str(profile.id) != str(row["alert_type"]):
            preds[case_id] = NOVEL
            details[case_id] = {"tier": "cold", "nearest_distance": None, "radius": None, "novel": True, "signature": NOVEL}
            head_ok = False
            continue
        resolved = resolve_tags(profile, case, frame=getattr(case, "frame_type", None))
        out = decide_alert_case(case, profile, resolved, settings, engine, case_id=case_id)
        if out.get("fallback"):
            encoder_loaded = False
        decision = out.get("decision") or {}
        signature = str(decision.get("signature") or NOVEL)
        tier = str(decision.get("tier") or "")
        if tier != "head" or str(decision.get("profile") or profile.id) != str(row["alert_type"]):
            head_ok = False
        preds[case_id] = signature
        details[case_id] = {
            "tier": tier,
            "nearest_distance": decision.get("nearest_distance"),
            "radius": decision.get("novelty_threshold"),
            "head_confidence": decision.get("head_confidence"),
            "confidence_floor": decision.get("confidence_floor"),
            "beyond_radius": bool(decision.get("beyond_radius")),
            "novel": bool(decision.get("novel") or signature == NOVEL),
            "signature": signature,
        }
    return {"preds": preds, "details": details, "encoder_loaded": encoder_loaded, "head_ok": head_ok}


def _unseen_novel_recall(dataset: Path, settings: Any, engine: Any, active_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Novel calls on the original unseen split, per alert type. Reported, not gated.

    The alert type is the alert name. The truth is not in the built-in list, so
    the side-car rule is not used. One pass. A failure here does not fail G1.
    """
    from sentinel.data import load_csv
    from sentinel.e2e.pipeline import decide_alert_case
    from sentinel.profiles import alert_type_for_name, load_profiles, resolve_alert_type, resolve_tags

    root = Path(dataset)
    rows = load_manifest(root)
    indices = np.flatnonzero(np.array([str(r.get("split") or "") for r in rows]) == "unseen")
    if len(indices) == 0:
        return []
    book = load_profiles()
    mdir = Path(settings.e2e_model_dir)
    shipped = _legacy_folder_scores(
        legacy_model_dir(mdir, "encoder"),
        legacy_model_dir(mdir, "base"),
        root,
        rows,
        indices,
        "legacy unseen",
    )
    shipped_pred = {cid: sig for cid, sig in zip(shipped["case_id"], shipped["pred"])}
    active_pred: Dict[str, str] = {}
    if active_dir is not None:
        active = _legacy_version_scores(active_dir, root, rows, indices)
        active_pred = {cid: sig for cid, sig in zip(active["case_id"], active["pred"])}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for idx in indices:
        row = rows[int(idx)]
        prof = alert_type_for_name(str(row.get("analytic") or ""), book)
        alert_type = prof.id if prof is not None else ""
        grouped.setdefault(alert_type or "(unassigned)", []).append(row)
    out = []
    for alert_type in sorted(grouped):
        group = grouped[alert_type]
        per_novel = 0
        shipped_n = sum(1 for row in group if shipped_pred.get(str(row["case_id"])) == NOVEL)
        active_n = sum(1 for row in group if active_pred.get(str(row["case_id"])) == NOVEL) if active_pred else None
        legacy_novel = shipped_n if active_n is None else max(shipped_n, active_n)
        if alert_type != "(unassigned)":
            for row in group:
                case = load_csv(root / str(row["file"]))
                choice = resolve_alert_type(str(row.get("analytic") or ""), settings=settings)
                if choice.profile is None:
                    per_novel += 1
                    continue
                resolved = resolve_tags(choice.profile, case, frame=getattr(case, "frame_type", None))
                decided = decide_alert_case(case, choice.profile, resolved, settings, engine, case_id=str(row["case_id"]))
                signature = str((decided.get("decision") or {}).get("signature") or NOVEL)
                if signature == NOVEL:
                    per_novel += 1
        out.append({
            "alert_type": alert_type,
            "n": len(group),
            "legacy_novel": legacy_novel,
            "peralert_novel": per_novel if alert_type != "(unassigned)" else None,
        })
    return out


def _manifest_matches_sidecar(dataset: Path, sidecar_rows: Sequence[Dict[str, Any]]) -> bool:
    """Side-car ids, in order, equal the manifest test rows, and the CSV hashes match the files."""
    from sentinel.e2e.dataset import decompressed_csv

    rows = load_manifest(Path(dataset))
    test_rows = [row for row in rows if str(row.get("split") or "") == "test"]
    if len(test_rows) != len(sidecar_rows):
        return False
    for manifest_row, side in zip(test_rows, sidecar_rows):
        if str(manifest_row.get("case_id") or "") != str(side.get("case_id") or ""):
            return False
        blob = decompressed_csv(Path(dataset) / str(side["file"]))
        digest = hashlib.sha256(blob).hexdigest()
        if digest != str(side.get("csv_sha256") or ""):
            return False
    return True


def score_peralert_acceptance(dataset: Path) -> int:
    """Design section 7. Writes the per-alert report and returns 0 either way.

    The pass or fail is in the report. This does not change ``e2e_mode``, does
    not rewrite the side-car, and does not rewrite the design-8.4 files.
    Unit tests must not call this: it embeds the 320 held-out rows.
    """
    import tempfile

    side_bytes = _sidecar_bytes()
    side_sha = hashlib.sha256(side_bytes).hexdigest()
    sidecar_rows = _load_sidecar_rows()
    sidecar_ids = {str(row["case_id"]) for row in sidecar_rows}
    sidecar_hashes = {str(row.get("csv_sha256") or "") for row in sidecar_rows if row.get("csv_sha256")}
    model_dir = ROOT / "models" / "e2e"
    hits = sidecar_ids_in_bundles(model_dir, sidecar_ids)
    rows_match = _manifest_matches_sidecar(dataset, sidecar_rows)

    real = load_settings({})
    registry = Path(real.data_dir) / "e2e" / "registry"
    current_path = registry / "current.json"
    current: Optional[Dict[str, Any]] = None
    if current_path.is_file():
        try:
            loaded = json.loads(current_path.read_text(encoding="utf-8"))
            current = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            current = None
    version = str((current or {}).get("version") or "")
    version_dir = registry / version if version else None
    meta = None
    records: List[Dict[str, Any]] = []
    if version_dir is not None and version not in {"", "v0000"} and version_dir.is_dir():
        meta_path = version_dir / "meta.json"
        if meta_path.is_file():
            try:
                loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = loaded_meta if isinstance(loaded_meta, dict) else None
            except (OSError, ValueError):
                meta = None
        records = _read_index_records(version_dir)
    active_info = legacy_active_status(current, meta, records, sidecar_ids, sidecar_hashes)
    if active_info["status"] == "scored" and (version_dir is None or not legacy_version_readable(version_dir)):
        active_info = {"status": "unscored", "version": version, "reason": f"active version {version} cannot be scored"}

    root = Path(dataset)
    manifest = load_manifest(root)
    te = np.flatnonzero(np.array([str(r.get("split") or "") for r in manifest]) == "test")
    shipped_runs = []
    shipped_status = "scored"
    try:
        for _ in range(2):
            block = _legacy_test_scores(root, manifest, te, load_settings({"e2e_model_dir": str(model_dir)}))
            shipped_runs.append({cid: sig for cid, sig in zip(block["case_id"], block["pred"])})
    except Exception as exc:  # noqa: BLE001 — a missing shipped bundle is a failed gate, not a crash
        print(f"shipped legacy score failed ({type(exc).__name__})", flush=True)
        shipped_runs = []
        shipped_status = "unscored"

    active_runs: List[Dict[str, str]] = []
    active_dir: Optional[Path] = None
    if active_info["status"] == "scored" and version_dir is not None:
        try:
            for _ in range(2):
                block = _legacy_version_scores(version_dir, root, manifest, te)
                active_runs.append({cid: sig for cid, sig in zip(block["case_id"], block["pred"])})
            active_dir = version_dir
        except Exception as exc:  # noqa: BLE001 — do not fall back to the easier shipped-only bar
            print(f"active legacy score failed ({type(exc).__name__})", flush=True)
            active_runs = []
            active_info = {"status": "unscored", "version": version, "reason": f"active version {version} cannot be scored"}

    per_runs: List[Dict[str, str]] = []
    per_details: List[Dict[str, Any]] = []
    encoder_loaded = True
    head_ok = True
    unseen: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="sentinel_peralert_") as tmp:
        settings = load_settings(
            {
                "data_dir": tmp,
                "e2e_model_dir": str(model_dir),
                "use_llm": False,
                "embedder": "tspulse_ci",
                "device": "cpu",
                "e2e_mode": "legacy",
            }
        )
        engine = _PeralertEngine(settings)
        try:
            for _ in range(2):
                once = _score_peralert_once(root, sidecar_rows, settings, engine)
                per_runs.append(once["preds"])
                per_details.append(once["details"])
                encoder_loaded = encoder_loaded and bool(once["encoder_loaded"])
                head_ok = head_ok and bool(once["head_ok"])
        except Exception as exc:  # noqa: BLE001 — record the failure instead of switching mode
            print(f"per-alert score failed ({type(exc).__name__})", flush=True)
            per_runs = []
            encoder_loaded = False
            head_ok = False
        try:
            unseen = _unseen_novel_recall(root, settings, engine, active_dir)
        except Exception as exc:  # noqa: BLE001 — unseen recall is reported, not gated
            print(f"unseen recall skipped ({type(exc).__name__})", flush=True)
            unseen = []

    g3 = {
        "sidecar_sha": side_sha == PERALERT_SIDECAR_SHA256,
        "rows_match_manifest": rows_match,
        "no_sidecar_id_in_bundle": not hits,
        "encoder_loaded": encoder_loaded and bool(per_runs),
        "head_tier": head_ok and bool(per_runs),
        "both_runs": len(shipped_runs) == 2 and len(per_runs) == 2 and (active_info["status"] != "scored" or len(active_runs) == 2),
    }
    payload = evaluate_peralert(
        sidecar_rows,
        {
            "shipped": {"status": shipped_status, "runs": shipped_runs},
            "active": {**active_info, "runs": active_runs},
        },
        per_runs,
        g3=g3,
        keep_g2=True,
        sidecar_sha256=side_sha,
        bundle_hashes=bundle_file_hashes(model_dir),
        decision_sha=decision_path_sha(),
        git_head=_git_head(),
        peralert_details=per_details,
        unseen=unseen,
    )
    payload["dataset"] = str(root)
    write_peralert_outputs(payload, EVAL)
    print(f"ACCEPTANCE PER-ALERT: {'PASS' if payload['passed'] else 'FAIL'}", flush=True)
    print(f"wrote {EVAL / 'EVAL_PERALERT.md'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "data" / "e2e" / "dataset"))
    ap.add_argument("--llm-cases", type=int, default=0)
    ap.add_argument("--llm-only", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--md-only", action="store_true", help="rewrite eval/EVAL.md and charts from eval/results.json")
    ap.add_argument("--profiles", action="store_true", help="design 8.4: score the original test split with the profile system")
    ap.add_argument("--peralert", action="store_true", help="design section 7: per-alert acceptance against legacy; does not change e2e_mode")
    ap.add_argument("--write-sidecar", action="store_true", help="write eval/acceptance_peralert.jsonl from the test split and exit")
    a = ap.parse_args()
    if a.write_sidecar:
        return write_peralert_sidecar(Path(a.dataset))
    if a.peralert:
        return score_peralert_acceptance(Path(a.dataset))
    if a.profiles:
        return score_acceptance(Path(a.dataset))
    settings = load_settings({"use_tspulse": False})
    EVAL.mkdir(exist_ok=True)
    res_file = EVAL / "results.json"
    if a.md_only:
        res = json.loads(res_file.read_text(encoding="utf-8"))
        if res.get("llm"):
            res["llm"]["summary"] = llm_summary(res["llm"])
            res_file.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        write_md(res)
        return 0
    if a.llm_only:
        res = json.loads(res_file.read_text(encoding="utf-8"))
        res["llm"] = llm_subset(settings, Path(a.dataset), a.llm_cases or 8)
        res_file.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        write_md(res)
        return 0

    t0 = time.monotonic()
    root = Path(a.dataset)
    rows = load_manifest(root)
    X = cached_features(root, rows, ROOT / "data" / "e2e" / "cache")
    rr = rule_results(root, [r for r in rows if r["split"] != "train"], ROOT / "data" / "e2e" / "cache")
    mdir = Path(settings.e2e_model_dir)
    legacy_enc = legacy_model_dir(mdir, "encoder")
    legacy_base = legacy_model_dir(mdir, "base")
    enc = Encoder(legacy_enc)
    scaler = Scaler.load(legacy_base / "scaler.npz")
    head = Head.load(legacy_base)
    index = SimilarityIndex.load(legacy_base, "index")
    cal = Calibration.load(legacy_base / "calibration.json")
    print("embedding all cases ...")
    E = enc.embed(X)
    Z = scaler(E)
    split = np.array([r["split"] for r in rows])
    sig = np.array([r["signature"] for r in rows])
    ids = [r["case_id"] for r in rows]
    rule_act = np.array([rr.get(i, {}).get("action", "n/a") for i in ids])

    def run(idx_rows: np.ndarray, hd: Head = head, ix: SimilarityIndex = index) -> List[Dict[str, Any]]:
        return [decide(Z[i], hd, ix, cal) for i in idx_rows]

    res: Dict[str, Any] = {"created": datetime.now().isoformat(timespec="seconds"), "card": json.loads((mdir / "card.json").read_text(encoding="utf-8"))}

    # ------------------------------------------------------------------ iid test
    te = np.flatnonzero(split == "test")
    dt = run(te)
    l_sig = [d["signature"] for d in dt]
    l_act = [act(s) for s in l_sig]
    h_sig = [d["head_signature"] for d in dt]
    nn_sig = [d["similar"][0]["signature"] for d in dt]
    t_sig, t_act, r_act = list(sig[te]), [act(s) for s in sig[te]], list(rule_act[te])
    agree = np.array([x == y for x, y in zip(l_act, r_act)])
    l_ok = np.array([x == y for x, y in zip(l_act, t_act)])
    r_ok = np.array([x == y for x, y in zip(r_act, t_act)])
    low = np.array([d["confidence"] < cal.low_confidence or d["novel"] for d in dt])
    needs = (~agree) | low
    cov = np.array([s in RULE_CLASSES for s in t_sig])
    amb = np.array([rows[i].get("ambiguous", False) for i in te])
    res["test"] = {
        "n": len(te),
        "learned_signature_acc": accuracy(t_sig, l_sig),
        "head_only_signature_acc": accuracy(t_sig, h_sig),
        "nearest_neighbour_signature_acc": accuracy(t_sig, nn_sig),
        "learned_action_acc": accuracy(t_act, l_act),
        "rules_action_acc": accuracy(t_act, r_act),
        "rule_classes": {"n": int(cov.sum()), "learned_action_acc": round(float(l_ok[cov].mean()), 4), "rules_action_acc": round(float(r_ok[cov].mean()), 4)},
        "other_classes": {"n": int((~cov).sum()), "learned_action_acc": round(float(l_ok[~cov].mean()), 4), "rules_action_acc": round(float(r_ok[~cov].mean()), 4)},
        "ambiguous": {"n": int(amb.sum()), "learned_action_acc": round(float(l_ok[amb].mean()), 4), "rules_action_acc": round(float(r_ok[amb].mean()), 4)},
        "false_novelty_rate": round(float(np.mean([d["novel"] for d in dt])), 4),
        "shadow": {
            "agreement_rate": round(float(agree.mean()), 4),
            "acc_when_agree": round(float(l_ok[agree].mean()), 4) if agree.any() else None,
            "learned_acc_when_disagree": round(float(l_ok[~agree].mean()), 4) if (~agree).any() else None,
            "rules_acc_when_disagree": round(float(r_ok[~agree].mean()), 4) if (~agree).any() else None,
            "needs_human_rate": round(float(needs.mean()), 4),
            "auto_accept_acc": round(float(l_ok[~needs].mean()), 4) if (~needs).any() else None,
            "errors_caught_by_needs_human": int(((~l_ok) & needs).sum()),
            "errors_total": int((~l_ok).sum()),
        },
        "confusion_signature": confusion(t_sig, l_sig, list(BASE_SIGNATURES)),
        "confusion_action_learned": confusion(t_act, l_act, sorted(set(t_act))),
        "confusion_action_rules": confusion(t_act, r_act, sorted(set(t_act))),
        "per_class": [
            {**pc, "rules_action_acc": round(float(np.mean([r_act[i] == t_act[i] for i in range(len(te)) if t_sig[i] == pc["class"]])), 3),
             "learned_action_acc": round(float(np.mean([l_act[i] == t_act[i] for i in range(len(te)) if t_sig[i] == pc["class"]])), 3)}
            for pc in per_class(t_sig, l_sig, list(BASE_SIGNATURES))
        ],
    }
    print(f"test: learned sig {res['test']['learned_signature_acc']}  action {res['test']['learned_action_acc']}  rules {res['test']['rules_action_acc']}")

    # ------------------------------------------------------------------ variants (+ one-shot)
    va_ = np.flatnonzero(split == "variant")
    dv = run(va_)
    var = {}
    for kind in ("shifted", "load", "magnitude"):
        sel = [k for k, i in enumerate(va_) if rows[i]["variant"] == kind]
        tv = [act(sig[va_[k]]) for k in sel]
        var[kind] = {
            "n": len(sel),
            "learned_signature_acc": accuracy([sig[va_[k]] for k in sel], [dv[k]["signature"] for k in sel]),
            "learned_action_acc": accuracy(tv, [act(dv[k]["signature"]) for k in sel]),
            "rules_action_acc": accuracy(tv, [rule_act[va_[k]] for k in sel]),
            "novel_rate": round(float(np.mean([dv[k]["novel"] for k in sel])), 4),
        }
    # one-shot: label the first case of each (signature, variant) group, score the rest
    groups: Dict[str, List[int]] = {}
    for i in va_:
        groups.setdefault(f"{sig[i]}|{rows[i]['variant']}", []).append(int(i))
    before_ok = after_ok = cited = tot = 0
    per_group = []
    for g, members in sorted(groups.items()):
        first, rest = members[0], members[1:]
        ix = index.merged(SimilarityIndex(Z[[first]], [{"id": ids[first], "signature": sig[first], "action": act(sig[first]), "source": "engineer"}]))
        b = [decide(Z[i], head, index, cal)["signature"] == sig[i] for i in rest]
        dd = [decide(Z[i], head, ix, cal) for i in rest]
        aft = [d["signature"] == sig[i] for d, i in zip(dd, rest)]
        cit = [any(s["id"] == ids[first] for s in d["similar"]) for d in dd]
        before_ok += sum(b)
        after_ok += sum(aft)
        cited += sum(cit)
        tot += len(rest)
        per_group.append({"group": g, "n": len(rest), "before": round(float(np.mean(b)), 3), "after": round(float(np.mean(aft)), 3), "example_cited_top5": round(float(np.mean(cit)), 3)})
    res["variant"] = {"by_kind": var, "one_shot": {"n": tot, "signature_acc_before": round(before_ok / tot, 4), "signature_acc_after_one_label": round(after_ok / tot, 4),
                                                   "labeled_example_in_top5": round(cited / tot, 4), "groups": per_group}}
    print(f"variants: {var}; one-shot {before_ok / tot:.3f} -> {after_ok / tot:.3f}")

    # ------------------------------------------------------------------ unseen fault types: novelty + one-/few-shot
    nn_test = [d["nearest_distance"] for d in dt]

    def kshot_curve(members: List[int], sig_name: str, action: str) -> List[Dict[str, Any]]:
        out = []
        for k in (1, 2, 3, 5, 10):
            lab, rest = members[:k], members[k:]
            ix = index.merged(SimilarityIndex(Z[lab], [{"id": ids[i], "signature": sig_name, "action": action, "source": "engineer"} for i in lab]))
            hd_r = train_head(np.vstack([index.Z, Z[lab]]), [r["signature"] for r in index.records] + [sig_name] * k,
                              np.concatenate([np.ones(len(index)), np.full(k, 3.0)]))
            d_idx = [decide(Z[i], head, ix, cal) for i in rest]
            d_full = [decide(Z[i], hd_r, ix, cal) for i in rest]
            d_te = [decide(Z[i], hd_r, ix, cal) for i in te]
            out.append({
                "labels": k,
                "recognised_index_only": round(float(np.mean([d["signature"] == sig_name for d in d_idx])), 3),
                "recognised_after_retrain": round(float(np.mean([d["signature"] == sig_name for d in d_full])), 3),
                "action_acc_index_only": round(float(np.mean([(action if d["signature"] == sig_name else act(d["signature"])) == action for d in d_idx])), 3),
                "action_acc_after_retrain": round(float(np.mean([(action if d["signature"] == sig_name else act(d["signature"])) == action for d in d_full])), 3),
                "still_novel_after_retrain": round(float(np.mean([d["novel"] for d in d_full])), 3),
                "test_signature_acc_after_retrain": round(float(np.mean([d["signature"] == s_ for d, s_ in zip(d_te, t_sig)])), 4),
            })
        return out

    def unseen_block(sig_name: str) -> Dict[str, Any]:
        mem = [int(i) for i in np.flatnonzero((split == "unseen") & (sig == sig_name))]
        du = [decide(Z[i], head, index, cal) for i in mem]
        action = act(sig_name)
        nn_un = [d["nearest_distance"] for d in du]
        forced = [d["signature"] for d in du if not d["novel"]]
        return {
            "n": len(mem),
            "signature": sig_name,
            "truth_action": action,
            "novelty_detection_rate": round(float(np.mean([d["novel"] for d in du])), 4),
            "forced_into": {str(x): forced.count(x) for x in sorted(set(forced))},
            "learned_actions": {x: [act(d["signature"]) for d in du].count(x) for x in sorted({act(d["signature"]) for d in du})},
            "learned_action_acc": accuracy([action] * len(mem), [act(d["signature"]) for d in du]),
            "rules_action_acc": accuracy([action] * len(mem), list(rule_act[mem])),
            "rules_actions": {str(x): int((rule_act[mem] == x).sum()) for x in sorted(set(rule_act[mem]))},
            "nn_distance_auroc": round(auroc(nn_un, nn_test), 4),
            "nn_distance_median": {"test": round(float(np.median(nn_test)), 4), "unseen": round(float(np.median(nn_un)), 4), "threshold": cal.novelty_threshold},
            "nn_test_all": [round(float(x), 4) for x in nn_test],
            "nn_unseen_all": [round(float(x), 4) for x in nn_un],
            "kshot": kshot_curve(mem, sig_name, action),
            "members": mem,
        }

    res["unseen"] = unseen_block("bearing_vibration")
    res["lookalike"] = unseen_block("tc_harness_fault")
    un = np.array(res["unseen"]["members"] + res["lookalike"]["members"])
    for key in ("unseen", "lookalike"):
        u = res[key]
        print(f"{key} {u['signature']}: novelty {u['novelty_detection_rate']} auroc {u['nn_distance_auroc']} learned {u['learned_action_acc']} rules {u['rules_action_acc']} "
              f"forced {u['forced_into']} kshot {[(k['labels'], k['action_acc_index_only'], k['action_acc_after_retrain']) for k in u['kshot']]}")

    res["residual"] = residual_block(settings, root, rows, ids, Z, head, index, cal)
    rb = res["residual"]
    print(f"residual ({rb.get('mode')}): test false contradictions {rb['test']['false_contradictions']['k']}/{rb['test']['false_contradictions']['n']}; "
          f"error contradictions {rb['test']['error_contradictions']['k']}/{rb['test']['error_contradictions']['n']}; "
          f"sensor-fault cold-spot calls contradicted {rb['sensorfault']['cold_spot_contradicted']['k']}/{rb['sensorfault']['cold_spot_contradicted']['n']}")

    # ------------------------------------------------------------------ simulations
    if not a.skip_sim:
        rng = random.Random(7)
        stream, hold = [], []
        for g, members in sorted(groups.items()):
            for j, i in enumerate(members):
                (stream if j < 5 else hold).append(int(i))
        for key in ("unseen", "lookalike"):
            for j, i in enumerate(res[key]["members"]):
                (stream if j < 25 else hold).append(int(i))
        rng.shuffle(stream)

        def item(i: int) -> Dict[str, Any]:
            return {"case_id": ids[i], "emb": E[i], "signature": sig[i], "action": act(sig[i]), "rules_action": rule_act[i]}

        print("simulation A: site adaptation (variants + 2 unseen fault types), retrain every 5 labels")
        res["sim_site"] = site_adaptation(settings, [item(i) for i in stream], [item(i) for i in hold], every=5,
                                          new_signature_actions={"bearing_vibration": "escalate", "tc_harness_fault": "sensor_fix_replace"}, log=print)
        res["sim_site"]["stream"] = len(stream)
        res["sim_site"]["holdout"] = len(hold)
        print("simulation B: cold start (no labels), fine-tuned vs zero-shot TS Pulse embeddings")
        tr = np.flatnonzero(split == "train")
        vv = np.flatnonzero(split == "val")
        perm = np.random.default_rng(3).permutation(tr)[:240]
        nocal = lambda c: Calibration(novelty_threshold=9.0, match_threshold=c.match_threshold)  # noqa: E731
        res["sim_cold"] = {"fine_tuned": cold_start(Z[perm], list(sig[perm]), Z[te], t_sig, nocal(cal), every=5, max_labels=240)}
        zs = Encoder(settings.tspulse_clf_dir)
        E0 = zs.embed(X)
        s0 = Scaler.fit(E0[tr])
        Z0 = s0(E0)
        cal0 = calibrate(SimilarityIndex(Z0[tr], [{"id": ids[i], "signature": sig[i]} for i in tr]), Z0[vv], list(sig[vv]))
        res["sim_cold"]["zero_shot"] = cold_start(Z0[perm], list(sig[perm]), Z0[te], t_sig, nocal(cal0), every=5, max_labels=240)
        hd0 = train_head(Z0[tr], list(sig[tr]))
        res["zero_shot_encoder"] = {
            "test_signature_acc_full_train": accuracy(t_sig, hd0.predict(Z0[te])),
            "unseen_novelty_auroc": round(auroc([float((1 - Z0[tr] @ Z0[i]).min()) for i in res["unseen"]["members"]], [float((1 - Z0[tr] @ Z0[i]).min()) for i in te]), 4),
        }
        print(f"zero-shot encoder: {res['zero_shot_encoder']}")

    if a.llm_cases:
        res["llm"] = llm_subset(settings, root, a.llm_cases)
    elif res_file.is_file():
        old = json.loads(res_file.read_text(encoding="utf-8"))
        if old.get("llm"):
            res["llm"] = old["llm"]
    res["elapsed_min"] = round((time.monotonic() - t0) / 60, 1)
    res_file.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_md(res)
    print(f"wrote {res_file} and eval/EVAL.md in {res['elapsed_min']} min")
    return 0


# ---------------------------------------------------------------------------- residual cross-check (V2)


def _contradict_rate(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    k = sum(1 for x in recs if x["verdict"] == "contradicts")
    n = len(recs)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None}


def _call_counts(recs: List[Dict[str, Any]], key: str = "pred") -> Dict[str, int]:
    vals = [x[key] for x in recs]
    return {str(v): vals.count(v) for v in sorted(set(vals))}


def residual_block(settings: Any, root: Path, rows: List[Dict[str, Any]], ids: List[str], Z: np.ndarray,
                   head: Head, index: SimilarityIndex, cal: Calibration) -> Dict[str, Any]:
    """Deterministic residual cross-check on held-out cases, plus the residual-novelty effect on the iid test.

    The classifier numbers in the rest of this file are decide() without the residual summary, so they
    stay comparable to eval/results_v1_baseline.json. This block is the V2 addition: does the residual
    pattern support the classifier, and does residual novelty change held-out accuracy.
    """
    from sentinel.e2e import residual as residual_mod
    from sentinel.e2e.dataset import cached_residuals

    held = [r for r in rows if r["split"] in ("test", "unseen", "sensorfault")]
    print(f"residual cross-check on {len(held)} held-out cases ...", flush=True)
    t0 = time.monotonic()
    scored = cached_residuals(root, held, ROOT / "data" / "e2e" / "cache", settings, log=print,
                              thresholds=cal.residual_thresholds or None)
    id_to_i = {cid: i for i, cid in enumerate(ids)}

    def pack(subset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in subset:
            i = id_to_i[r["case_id"]]
            res = scored[r["case_id"]]
            d0 = decide(Z[i], head, index, cal)
            d1 = decide(Z[i], head, index, cal, residual=residual_mod.summary(res))
            xc = residual_mod.cross_check(d0["signature"], res)
            pat = res.get("pattern") or {}
            out.append({
                "case_id": r["case_id"],
                "truth": r["signature"],
                "pred": d0["signature"],
                "pred_residual": d1["signature"],
                "correct": d0["signature"] == r["signature"],
                "novel0": bool(d0["novel"]),
                "novel1": bool(d1["novel"]),
                "verdict": xc["verdict"],
                "points_to": xc.get("points_to"),
                "reason": (xc.get("reason") or "")[:180],
                "sensor_like": bool(pat.get("sensor_like")),
                "ring": pat.get("ring"),
            })
        return out

    def examples(recs: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
        hit = [x for x in recs if x["verdict"] == "contradicts"]
        return [{"case_id": x["case_id"], "truth": x["truth"], "pred": x["pred"], "points_to": x["points_to"], "reason": x["reason"]} for x in hit[:limit]]

    def novelty_effect(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        flipped_ok = [x for x in recs if x["correct"] and x["pred_residual"] == NOVEL and not x["novel0"]]
        flipped_bad = [x for x in recs if (not x["correct"]) and x["pred_residual"] == NOVEL and not x["novel0"]]
        return {
            "action_acc_before": accuracy([act(x["truth"]) for x in recs], [act(x["pred"]) for x in recs]),
            "action_acc_after": accuracy([act(x["truth"]) for x in recs], [act(x["pred_residual"]) for x in recs]),
            "signature_acc_before": accuracy([x["truth"] for x in recs], [x["pred"] for x in recs]),
            "signature_acc_after": accuracy([x["truth"] for x in recs], [x["pred_residual"] for x in recs]),
            "false_novelty_before": round(float(np.mean([x["novel0"] for x in recs])), 4) if recs else None,
            "false_novelty_after": round(float(np.mean([x["novel1"] for x in recs])), 4) if recs else None,
            "correct_flipped_to_novel": len(flipped_ok),
            "errors_flipped_to_novel": len(flipped_bad),
        }

    test = pack([r for r in rows if r["split"] == "test"])
    sensor = pack([r for r in rows if r["split"] == "sensorfault"])
    bearing = pack([r for r in rows if r["split"] == "unseen" and r["signature"] == "bearing_vibration"])
    harness = pack([r for r in rows if r["split"] == "unseen" and r["signature"] == "tc_harness_fault"])
    cold_calls = [x for x in sensor if x["pred"] == "cold_spot_hardware"]
    other_sensor = [x for x in sensor if x["pred"] != "cold_spot_hardware"]
    forced_bearing = [x for x in bearing if not x["novel0"]]
    instab = [x for x in bearing if x["pred"] == "combustion_instability"]
    mode = next((scored[r["case_id"]].get("mode") for r in held if r["case_id"] in scored), None)
    return {
        "mode": mode,
        "n_scored": len(held),
        "seconds": round(time.monotonic() - t0, 1),
        "test": {
            "false_contradictions": _contradict_rate([x for x in test if x["correct"]]),
            "error_contradictions": _contradict_rate([x for x in test if not x["correct"]]),
            "supports": sum(1 for x in test if x["verdict"] == "supports"),
            "contradicts": sum(1 for x in test if x["verdict"] == "contradicts"),
            "inconclusive": sum(1 for x in test if x["verdict"] == "inconclusive"),
            "novelty": novelty_effect(test),
            "examples": examples([x for x in test if not x["correct"]]),
        },
        "sensorfault": {
            "n": len(sensor),
            "truth": "tc_sensor_fault",
            "classifier_calls": _call_counts(sensor),
            "cold_spot_contradicted": _contradict_rate(cold_calls),
            "other_calls_contradicted": _contradict_rate(other_sensor),
            "correct_contradicted": _contradict_rate([x for x in sensor if x["correct"]]),
            "sensor_like_k": sum(1 for x in sensor if x["sensor_like"]),
            "ring": _call_counts(sensor, "ring"),
            "examples": examples(cold_calls) or examples(sensor),
        },
        "bearing": {
            "n": len(bearing),
            "novelty_before": round(float(np.mean([x["novel0"] for x in bearing])), 4) if bearing else None,
            "novelty_after": round(float(np.mean([x["novel1"] for x in bearing])), 4) if bearing else None,
            "forced_into": _call_counts(forced_bearing),
            "forced_contradicted": _contradict_rate(forced_bearing),
            "instability_contradicted": _contradict_rate(instab),
            "examples": examples(instab) or examples(forced_bearing),
        },
        "harness": {
            "n": len(harness),
            "novelty_before": round(float(np.mean([x["novel0"] for x in harness])), 4) if harness else None,
            "novelty_after": round(float(np.mean([x["novel1"] for x in harness])), 4) if harness else None,
            "classifier_calls": _call_counts(harness),
            "sensor_like_k": sum(1 for x in harness if x["sensor_like"]),
            "cold_spot_contradicted": _contradict_rate([x for x in harness if x["pred"] == "cold_spot_hardware"]),
            "examples": examples(harness),
        },
    }


# ---------------------------------------------------------------------------- LLM subset


def llm_subset(settings: Any, root: Path, n: int) -> Dict[str, Any]:
    """Real local LLM (default Gemma 4 12B) on held-out cases through the full learned pipeline."""
    import tempfile

    from sentinel import llm_runtime
    from sentinel.e2e.pipeline import analyze_case

    rows = load_manifest(root)
    pick: List[Dict[str, Any]] = []
    want = ["cold_spot_hardware", "tc_sensor_fault", "combustion_instability", "dynamics_probe_fault", "hets_trip_hardware", "igv_lvdt_drift",
            "fuel_control_oscillation", "normal"]
    for s in want:
        cand = [r for r in rows if r["split"] == "test" and r["signature"] == s and r["notes_hint"] != "none"]
        if cand:
            pick.append(cand[0])
    pick.append(next(r for r in rows if r["split"] == "unseen" and r["signature"] == "bearing_vibration"))
    pick.append(next(r for r in rows if r["split"] == "unseen" and r["signature"] == "tc_harness_fault"))
    chosen = {r["case_id"] for r in pick}
    misleading = [r for r in rows if r["split"] == "test" and str(r["notes_hint"]).startswith("misleading") and r["case_id"] not in chosen]
    if misleading:
        pick.append(misleading[0])
    # One single-TC step (held-out sensor-fault split) so the sample shows the residual contradiction.
    sensor = next((r for r in rows if r["split"] == "sensorfault"), None)
    if sensor is not None and sensor["case_id"] not in {r["case_id"] for r in pick}:
        pick.insert(1, sensor)
    pick = pick[: max(1, n)]
    tmp = Path(tempfile.mkdtemp(prefix="sentinel_llm_eval_"))
    s = load_settings({"data_dir": tmp, "use_tspulse": False})
    # learned-path index / labels are read from the shipped model; cases go to a temp store
    plan = llm_runtime.plan(s)
    out = {"model": plan.model_label, "device": plan.device, "device_reason": plan.reason, "cases": []}
    ex_dir = EVAL / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for r in pick:
        print(f"LLM: {r['case_id']} ...", flush=True)
        t0 = time.monotonic()
        rep = analyze_case(root / r["file"], r["analytic"], r["notes"], settings=s, use_llm=True, source_name=Path(r["file"]).name)
        secs = time.monotonic() - t0
        d, n_ = rep["decision"], rep["narrative"]
        lm = rep["audit"].get("llm") or {}
        rc = d.get("residual_consistency") or {}
        rec = {
            "case_id": r["case_id"], "split": r["split"], "truth_signature": r["signature"], "truth_action": act(r["signature"]), "notes": r["notes"], "notes_hint": r["notes_hint"],
            "classifier_signature": rep["learned"]["signature"], "classifier_action": d.get("classifier_action"), "final_action": d["primary_action"],
            "llm_signature": d.get("llm_signature", rep["learned"]["signature"]) if n_["engine"] == "llm" else None,
            "rules_action": rep["shadow"]["rules"].get("primary_action"), "engine": n_["engine"], "narrative_engine": rep["audit"]["narrative_engine"],
            "checks": n_.get("checks", {}), "needs_human": d["needs_human"], "flags": d["review_flags"], "seconds": round(secs, 1),
            "prompt_tokens": (lm.get("usage") or {}).get("prompt_tokens"), "completion_tokens": (lm.get("usage") or {}).get("completion_tokens"),
            "tok_per_s": lm.get("tok_per_s"), "device": lm.get("device"),
            "residual_det": rc.get("deterministic"), "residual_llm": rc.get("llm"), "residual_final": rc.get("final"),
            "residual_reason": (rc.get("reason") or "")[:180],
        }
        rec["final_correct"] = rec["final_action"] == rec["truth_action"]
        rec["classifier_correct"] = rec["classifier_action"] == rec["truth_action"]
        out["cases"].append(rec)
        src = Path(rep["_path"])
        shutil.copy(src / "report.md", ex_dir / f"{r['case_id']}.md")
        shutil.copy(src / "report.json", ex_dir / f"{r['case_id']}.json")
        print(f"   {rec['engine']} final {rec['final_action']} (truth {rec['truth_action']}, classifier {rec['classifier_action']}) "
              f"residual {rec['residual_det']}/{rec['residual_llm']} {secs:.0f}s", flush=True)
    keep = {f"{r['case_id']}.md" for r in pick} | {f"{r['case_id']}.json" for r in pick}
    for old in ex_dir.iterdir():
        if old.suffix in (".md", ".json") and old.name not in keep:
            old.unlink()
    out["summary"] = llm_summary(out)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def llm_summary(out: Dict[str, Any]) -> Dict[str, Any]:
    seen, c = set(), []
    for x in out["cases"]:  # one entry per case
        if x["case_id"] not in seen:
            seen.add(x["case_id"])
            c.append(x)
    out["cases"] = c
    traced = [x["checks"].get("numbers traced to evidence", "") for x in c]
    num = sum(int(t.split("/")[0]) for t in traced if "/" in t)
    den = sum(int(t.split("/")[1]) for t in traced if "/" in t)
    wrong_unflagged = [x["case_id"] for x in c if not x["final_correct"] and not x["needs_human"]]
    return {
        "n": len(c),
        "valid_json": sum(1 for x in c if x["engine"] == "llm"),
        "final_action_acc": round(float(np.mean([x["final_correct"] for x in c])), 3) if c else None,
        "classifier_action_acc": round(float(np.mean([x["classifier_correct"] for x in c])), 3) if c else None,
        "llm_changed_action": sum(1 for x in c if x["final_action"] != x["classifier_action"]),
        "numbers_traced": f"{num}/{den}",
        "wrong_and_not_flagged": wrong_unflagged,
        "mean_seconds": round(float(np.mean([x["seconds"] for x in c])), 1) if c else None,
        "mean_prompt_tokens": round(float(np.mean([x["prompt_tokens"] for x in c if x.get("prompt_tokens")])), 0) if any(x.get("prompt_tokens") for x in c) else None,
        "mean_completion_tokens": round(float(np.mean([x["completion_tokens"] for x in c if x.get("completion_tokens")])), 0) if any(x.get("completion_tokens") for x in c) else None,
        "residual_det_contradicts": sum(1 for x in c if x.get("residual_det") == "contradicts"),
        "residual_llm_contradicts": sum(1 for x in c if x.get("residual_llm") == "contradicts"),
        "residual_agree": sum(1 for x in c if x.get("residual_det") and x.get("residual_llm") and x["residual_det"] == x["residual_llm"]),
    }


# ---------------------------------------------------------------------------- report


def counts(d: Dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "none"


def pct(x: Any) -> str:
    return "n/a" if x is None else f"{100 * float(x):.1f} %"


def write_md(res: Dict[str, Any]) -> None:
    from charts_svg import line_chart, write_charts

    charts = write_charts(res, EVAL / "charts")
    T, V, U = res["test"], res["variant"], res["unseen"]
    card = res["card"]
    L: List[str] = []
    w = L.append
    w("# Sentinel GT AI — End-to-end learned path: evaluation")
    w("")
    w(f"Generated {res['created']} by `python scripts/eval_e2e.py` (fully local, CPU). Raw numbers: `eval/results.json`.")
    w("")
    w("## Read this first — what these numbers can and cannot tell you")
    w("")
    w("- **All data is synthetic.** It comes from `scripts/generate_scenario_csvs.py --dataset` (8 signatures, varied frames, "
      "resolutions, tag names, operating profiles, noisy notes). The learned model was trained on the *same generator family* "
      "(different random seeds), so iid test accuracy is an upper bound, not a field estimate. The rules were written before "
      "this generator existed and were not tuned to it — a home-field advantage for the learned path.")
    w("- Two signatures (IGV LVDT drift, fuel control oscillation) have **no rule analyzer**; rules can only reach them via the "
      "generic path. Scores are therefore shown separately for the 6 signatures the rules were designed for.")
    w("- The variant and unseen splits are the honest part: shifted can position / ring size, a 4-5x larger frame, and magnitudes "
      "outside the training range, plus a fault type (bearing vibration + lube-oil rise) that never appears in training.")
    w("- Real historian data will contain things this generator does not (sensor dropouts of other kinds, mixed faults, "
      "operating modes, ambient effects). Treat the learned path as **shadow-mode decision support** until it has been "
      "scored on real labeled Tier 2 cases (`app.py e2e export-dataset` → `train_e2e.py --extra`).")
    w("")
    w("## Model")
    w("")
    w(f"- Encoder: {card['encoder']}; base weights `{card['base_weights']}`.")
    w(f"- Data: " + ", ".join(f"{k} {v}" for k, v in card["data"].items()) + " cases.")
    w(f"- Fine-tune: best validation accuracy {card['finetune']['best']['val_acc']:.3f} at epoch {card['finetune']['best']['epoch']}; "
      f"training time {card['train_minutes']} min on CPU.")
    w(f"- Calibration: novelty threshold {card['calibration']['novelty_threshold']} (cosine distance to the nearest labeled case, "
      f"q{int(100 * card['calibration']['novelty_quantile'])} of validation cases), one-shot match threshold {card['calibration']['match_threshold']}.")
    w("")
    w("## 1. Held-out test split (iid, 320 cases)")
    w("")
    w("| Metric | Learned path | Rules |")
    w("|---|---|---|")
    w(f"| **Action accuracy, all 8 signatures** | **{pct(T['learned_action_acc'])}** | {pct(T['rules_action_acc'])} |")
    w(f"| Action accuracy, 6 signatures the rules cover ({T['rule_classes']['n']} cases) | {pct(T['rule_classes']['learned_action_acc'])} | {pct(T['rule_classes']['rules_action_acc'])} |")
    w(f"| Action accuracy, 2 signatures without a rule analyzer ({T['other_classes']['n']} cases) | {pct(T['other_classes']['learned_action_acc'])} | {pct(T['other_classes']['rules_action_acc'])} |")
    w(f"| Action accuracy, ambiguous cases ({T['ambiguous']['n']}) | {pct(T['ambiguous']['learned_action_acc'])} | {pct(T['ambiguous']['rules_action_acc'])} |")
    w(f"| Signature accuracy (learned: head + index + novelty) | {pct(T['learned_signature_acc'])} | — |")
    w(f"| Signature accuracy, classifier head only | {pct(T['head_only_signature_acc'])} | — |")
    w(f"| Signature accuracy, nearest labeled case only | {pct(T['nearest_neighbour_signature_acc'])} | — |")
    w(f"| False novelty rate (known signatures flagged novel) | {pct(T['false_novelty_rate'])} | — |")
    w("")
    w(f"![Action accuracy by signature]({charts['per_class']})")
    w("")
    w("Per signature (action accuracy):")
    w("")
    w("| Signature | n | Learned | Rules |")
    w("|---|---|---|---|")
    for pc in T["per_class"]:
        w(f"| {SHORT.get(pc['class'], pc['class'])} | {pc['n']} | {pct(pc['learned_action_acc'])} | {pct(pc['rules_action_acc'])} |")
    w("")
    w("**Learned signature confusion matrix** (rows = truth):")
    w("")
    w(md_confusion(T["confusion_signature"], SHORT))
    w("")
    w("**Action confusion — learned path:**")
    w("")
    w(md_confusion(T["confusion_action_learned"], SHORT))
    w("")
    w("**Action confusion — rules:**")
    w("")
    w(md_confusion(T["confusion_action_rules"], SHORT))
    w("")
    sh = T["shadow"]
    w("### Shadow mode: what running both buys you")
    w("")
    w(f"- Learned and rule actions agree on **{pct(sh['agreement_rate'])}** of test cases; when they agree the action is right "
      f"**{pct(sh['acc_when_agree'])}** of the time.")
    w(f"- When they disagree, the learned action is right {pct(sh['learned_acc_when_disagree'])} and the rule action "
      f"{pct(sh['rules_acc_when_disagree'])} of the time — disagreement is exactly where a human is needed.")
    w(f"- `needs_human` (disagreement, low confidence or novelty) is raised on {pct(sh['needs_human_rate'])} of cases; "
      f"it catches {sh['errors_caught_by_needs_human']} of the {sh['errors_total']} learned-path errors. Cases that pass "
      f"without a flag are right {pct(sh['auto_accept_acc'])} of the time.")
    w("- That flag rate is the price of shadow mode: much of it comes from the two signatures the rules cannot see and from rule "
      "misses (sensor / probe faults). It is the right default until the learned path has a track record on real cases; the rate "
      "can be lowered later by trusting agreement plus high confidence.")
    w("")
    w("## 2. Variants (held out): shifted can, other frame/load, out-of-range magnitude")
    w("")
    w("| Variant | n | Learned action | Rules action | Learned signature | Flagged novel |")
    w("|---|---|---|---|---|---|")
    for k, v in V["by_kind"].items():
        w(f"| {k} | {v['n']} | {pct(v['learned_action_acc'])} | {pct(v['rules_action_acc'])} | {pct(v['learned_signature_acc'])} | {pct(v['novel_rate'])} |")
    w("")
    o = V["one_shot"]
    same = abs(o["signature_acc_before"] - o["signature_acc_after_one_label"]) < 1e-9
    w(f"**One-shot on variants:** in each of the {len(o['groups'])} (signature, variant) groups the engineer labels **one** case; the other "
      f"{o['n']} are re-scored with that single example in the similarity index (no retrain). Signature accuracy "
      f"{pct(o['signature_acc_before'])} → **{pct(o['signature_acc_after_one_label'])}**; the labeled example is cited among the "
      f"top-5 similar cases for {pct(o['labeled_example_in_top5'])} of them."
      + (" The variants were already handled by the base model (rotation-invariant channels, per-record scaling), so one label "
         "changes little here; one-shot earns its keep on the look-alike fault in section 3." if same else ""))
    w("")
    w("| Group | n | Before | After one label | Example cited |")
    w("|---|---|---|---|---|")
    for g in o["groups"]:
        w(f"| {g['group'].replace('|', ' / ')} | {g['n']} | {pct(g['before'])} | {pct(g['after'])} | {pct(g['example_cited_top5'])} |")
    w("")
    LA = res.get("lookalike") or {}
    w("## 3. Fault types never seen in training")
    w("")
    w("Two held-out fault types test the two ways an unknown fault can show up:")
    w("")
    w(f"**A. Different physics — bearing vibration + lube-oil rise ({U['n']} cases).** Correct handling before anyone has labeled one: "
      f"flag **novel pattern**, set `needs_human`, escalate — never force a known class.")
    w("")
    w(f"- **Novelty detection rate: {pct(U['novelty_detection_rate'])}** (threshold {U['nn_distance_median']['threshold']}; median nearest "
      f"distance test {U['nn_distance_median']['test']} vs unseen {U['nn_distance_median']['unseen']}; AUROC {U['nn_distance_auroc']}).")
    w(f"- Forced into a known class: {counts(U['forced_into'])}. Escalate accuracy: learned {pct(U['learned_action_acc'])}, rules "
      f"{pct(U['rules_action_acc'])} (rule actions: {counts(U['rules_actions'])} — the rules send the unknown analytic name down the generic path, which escalates).")
    w("")
    if charts.get("novelty"):
        w(f"![Nearest-neighbour distances]({charts['novelty']})")
        w("")
    if LA:
        w(f"**B. Look-alike — exhaust TC harness / connector fault ({LA['n']} cases).** Several adjacent TCs drop out together, in steps. It looks "
          f"like a cold sector, so it is *not* novel to the model; the right action is sensor repair, not a borescope.")
        w("")
        w(f"- Before any label: learned action accuracy {pct(LA['learned_action_acc'])} (learned actions: {counts(LA['learned_actions'])}); "
          f"rules {pct(LA['rules_action_acc'])} (rule actions: {counts(LA['rules_actions'])}). Flagged novel: {pct(LA['novelty_detection_rate'])}.")
        w("- This is the case one-shot learning is for: one engineer label of the new signature fixes the action for almost all the "
          "other harness cases (table below). \"Recognised\" stays lower than action accuracy because the remaining cases are called "
          "`tc_sensor_fault`, which carries the same action (sensor repair) — the right outcome, under the older name.")
        w("")
    w("After the engineer labels k cases with the new signature (index only = one-shot retrieval, no retrain; retrain = head retrained):")
    w("")
    w("| Fault type | Labels k | Recognised (index only) | Recognised after retrain | Action accuracy (index only) | Action accuracy after retrain | iid test signature acc after retrain |")
    w("|---|---|---|---|---|---|---|")
    for blk in (U, LA):
        for k in blk.get("kshot", []):
            w(f"| {SHORT.get(blk['signature'], blk['signature'])} | {k['labels']} | {pct(k['recognised_index_only'])} | {pct(k['recognised_after_retrain'])} | "
              f"{pct(k['action_acc_index_only'])} | {pct(k['action_acc_after_retrain'])} | {pct(k['test_signature_acc_after_retrain'])} |")
    w("")
    if res.get("sim_site"):
        S = res["sim_site"]
        c0, c1 = S["curve"][0], S["curve"][-1]
        w("## 4. Scripted labeling simulation (label on the fly, retrain every 5 labels)")
        w("")
        w(f"The shipped model meets site-like data: {S['stream']} variant and unseen-fault (bearing, TC harness) cases arrive one by one through the real "
          f"`Engine.add_label` path (instant index add, background head retrain every 5 labels, versioned heads). After every 5 "
          f"labels both paths are scored on {S['holdout']} held-out site cases.")
        w("")
        w(f"![Labeling simulation]({charts['sim_site']})")
        w("")
        w(f"- Learned action accuracy on held-out site cases: **{pct(c0['learned'])} → {pct(c1['learned'])}** after {c1['labels']} labels "
          f"({S['versions'] - 1} retrained head versions); signature accuracy {pct(c0.get('learned_signature'))} → {pct(c1.get('learned_signature'))} "
          f"(the new fault types become named classes). Rules stay at {pct(c1['rules'])} (they do not learn from labels).")
        pq = S["prequential"]
        if pq:
            first = pq[: max(1, len(pq) // 4)]
            last = pq[-max(1, len(pq) // 4):]
            w(f"- Prequential (prediction made *before* each label): first quarter {pct(np.mean([p['learned_ok'] for p in first]))}, "
              f"last quarter {pct(np.mean([p['learned_ok'] for p in last]))}.")
        ha = S.get("heldout_after") or {}
        if ha:
            w(f"- No forgetting: iid test after the simulation — learned action {pct(ha.get('action_acc'))}, signature {pct(ha.get('signature_acc'))} "
              f"(rules {pct(ha.get('rules_action_acc'))}).")
        w("")
    if res.get("sim_cold"):
        C = res["sim_cold"]
        w("### Cold start: learning from zero labels")
        w("")
        w(f"![Cold start]({charts['sim_cold']})")
        w("")
        ft, zs = C["fine_tuned"], C["zero_shot"]
        pick = [p for p in ft if p["labels"] in (5, 20, 50, 100, 240)]
        w("| Labels | Classes seen | Fine-tuned TS Pulse embedding | Zero-shot TS Pulse embedding |")
        w("|---|---|---|---|")
        for p in pick:
            z = next((q for q in zs if q["labels"] == p["labels"]), None)
            w(f"| {p['labels']} | {p['classes_seen']} | {pct(p['learned_acc'])} | {pct(z['learned_acc']) if z else 'n/a'} |")
        w("")
        zse = res.get("zero_shot_encoder") or {}
        tail = (f" With all training labels, a head on zero-shot embeddings reaches {pct(zse['test_signature_acc_full_train'])} test "
                f"signature accuracy (fine-tuned: {pct(T['head_only_signature_acc'])}); zero-shot novelty AUROC {zse['unseen_novelty_auroc']} "
                f"(fine-tuned {U['nn_distance_auroc']}).") if zse else ""
        w("The fine-tuned encoder has already seen the training labels (its representation is biased toward these signatures), so "
          "its cold-start curve is optimistic; the **zero-shot** TS Pulse curve is the honest picture of learning purely from "
          "labels given in the UI." + tail)
        w("")
    if res.get("llm"):
        Lm = res["llm"]
        s = Lm.get("summary", {})
        w("## 5. Report writer LLM (real runs)")
        w("")
        w(f"{Lm['model']} on **{Lm['device'].upper()}** ({Lm['device_reason']}). {s.get('n')} held-out cases through the full learned "
          f"pipeline (classifier + similar cases + notes + rules shadow → LLM → checks). Reports: `eval/examples/`.")
        w("")
        w(f"- Valid schema-constrained JSON: {s.get('valid_json')}/{s.get('n')}; numbers in the narrative traced to the evidence: "
          f"{s.get('numbers_traced', 'n/a')}; final action accuracy {pct(s.get('final_action_acc'))} (classifier alone "
          f"{pct(s.get('classifier_action_acc'))}); the LLM changed the classifier's action on {s.get('llm_changed_action')} case(s).")
        w(f"- Timing: mean {s.get('mean_seconds')} s per report on this laptop's CPU (model loaded once). "
          f"Mean prompt {s.get('mean_prompt_tokens')} tokens, mean completion {s.get('mean_completion_tokens')} tokens.")
        w(f"- Residual cross-check on this sample: the deterministic check contradicted the classifier on {s.get('residual_det_contradicts')} "
          f"of {s.get('n')}; the LLM set residual_consistency to contradicts on {s.get('residual_llm_contradicts')}; "
          f"the two verdicts agreed on {s.get('residual_agree')}. A contradiction lowers confidence and sets needs_human. "
          f"The saved reports are drafts for review.")
        if s.get("wrong_and_not_flagged"):
            w(f"- **Confidently wrong and not flagged:** {', '.join(s['wrong_and_not_flagged'])} — the learned path and the rules agreed on "
              "the wrong action, and the residual check did not contradict, so nothing raised needs_human.")
        w("")
        w("| Case | Truth action | Classifier | LLM final | Residual det / LLM | Tokens (prompt+completion) | Time |")
        w("|---|---|---|---|---|---|---|")
        for c in Lm["cases"]:
            tok = f"{c.get('prompt_tokens')}+{c.get('completion_tokens')}"
            w(f"| {c['case_id']} | {c['truth_action']} | {c['classifier_action']} | {c['final_action']}{' ✓' if c['final_correct'] else ' ✗'} | "
              f"{c.get('residual_det') or '—'} / {c.get('residual_llm') or '—'} | {tok} | {c['seconds']:.0f} s |")
        w("")
    R = res.get("residual")
    if R:
        def kx(block: Dict[str, Any]) -> str:
            if not block or not block.get("n"):
                return "n/a (0 cases)"
            return f"{block['k']}/{block['n']} ({pct(block['rate'])})"

        w("## Residual cross-check (V2)")
        w("")
        w(f"Second TS Pulse view (reconstruction residuals, mode **{R.get('mode')}**) scored on {R.get('n_scored')} held-out cases "
          f"in {R.get('seconds')} s. The cross-check compares that pattern with the classifier's signature. "
          f"It does not replace the classifier. Sections 1–4 above are the classifier without this view, so they stay comparable "
          f"to `eval/results_v1_baseline.json`.")
        w("")
        base_path = EVAL / "results_v1_baseline.json"
        if base_path.is_file():
            b = json.loads(base_path.read_text(encoding="utf-8"))
            nv = R["test"]["novelty"]
            w("| Metric | V1 baseline | V2 classifier | V2 + residual novelty |")
            w("|---|---|---|---|")
            w(f"| iid action accuracy | {pct(b['test']['learned_action_acc'])} | {pct(T['learned_action_acc'])} | {pct(nv['action_acc_after'])} |")
            w(f"| iid signature accuracy | {pct(b['test']['learned_signature_acc'])} | {pct(T['learned_signature_acc'])} | {pct(nv['signature_acc_after'])} |")
            w(f"| iid false novelty | {pct(b['test']['false_novelty_rate'])} | {pct(T['false_novelty_rate'])} | {pct(nv['false_novelty_after'])} |")
            w(f"| bearing novelty rate | {pct(b['unseen']['novelty_detection_rate'])} | {pct(U['novelty_detection_rate'])} | {pct(R['bearing'].get('novelty_after'))} |")
            w(f"| harness look-alike action accuracy | {pct(b['lookalike']['learned_action_acc'])} | {pct(res['lookalike']['learned_action_acc'])} | — |")
            w("")
            w(f"Residual novelty on the iid test flipped {nv['correct_flipped_to_novel']} correctly classified case(s) to NOVEL and "
              f"{nv['errors_flipped_to_novel']} classifier error(s) to NOVEL "
              f"(validation budget was at most 1 % extra false novelty).")
            w("")
        tf, te_ = R["test"]["false_contradictions"], R["test"]["error_contradictions"]
        sf = R["sensorfault"]
        w(f"- **False contradictions** on correctly classified iid test cases: **{kx(tf)}**.")
        w(f"- Contradictions on iid classifier errors: **{kx(te_)}** "
          f"(supports {R['test']['supports']}, contradicts {R['test']['contradicts']}, inconclusive {R['test']['inconclusive']}).")
        w(f"- **Sensor-fault split** ({sf['n']} single-TC steps, truth `{sf['truth']}`). Classifier calls: {counts(sf['classifier_calls'])}. "
          f"Called cold spot and contradicted: **{kx(sf['cold_spot_contradicted'])}**. "
          f"Other calls contradicted (false, when the call is already a sensor fault): **{kx(sf['other_calls_contradicted'])}**. "
          f"Correct calls contradicted: **{kx(sf['correct_contradicted'])}**. "
          f"Sensor-like residual shape: {sf['sensor_like_k']}/{sf['n']}. Ring pattern: {counts(sf['ring'])}.")
        br, ha = R["bearing"], R["harness"]
        w(f"- Bearing faults forced into a known class and contradicted: **{kx(br['forced_contradicted'])}** "
          f"(forced into {counts(br['forced_into'])}; of those called combustion instability, {kx(br['instability_contradicted'])} contradicted). "
          f"Novelty {pct(br.get('novelty_before'))} → {pct(br.get('novelty_after'))} once the residual view is allowed to flag NOVEL.")
        w(f"- Look-alike harness: sensor-like residual on **{ha['sensor_like_k']}/{ha['n']}**. "
          f"Called cold spot and contradicted: {kx(ha['cold_spot_contradicted'])}. "
          f"Classifier calls: {counts(ha['classifier_calls'])}.")
        shown = False
        for label, block in (("iid errors", R["test"]), ("sensor-fault", sf), ("bearing", br)):
            for ex in (block.get("examples") or [])[:1]:
                if not shown:
                    w("")
                    w("Example contradictions:")
                    w("")
                    shown = True
                w(f"- {label} `{ex['case_id']}`: classifier `{ex['pred']}` (truth `{ex['truth']}`) → {ex['reason']}")
        w("")
    w("## Limitations and next steps")
    w("")
    w("- Synthetic-only training and evaluation (see top). First real step: export reviewed Tier 2 cases, retrain with "
      "`--extra`, and re-run this script on a real held-out set.")
    w("- The novelty threshold is calibrated on synthetic validation data; on real data expect more novelty flags at first — "
      "each labeled case shrinks them (one-shot).")
    w("- The classifier does not read the notes; only the LLM does. Notes that change the answer are surfaced as a review flag.")
    w("- No LoRA: Gemma 4 12B cannot be LoRA-tuned on this CPU-only laptop or on a 6 GB GPU; the LLM learns in-context from "
      "retrieved labeled cases instead. Approved reports are kept for a later LoRA when hardware allows.")
    w("")
    (EVAL / "EVAL.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
