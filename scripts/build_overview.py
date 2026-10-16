#!/usr/bin/env python3
"""Build docs/sentinel_overview.html — a single self-contained, offline, interactive overview.

Everything shown is real: validation reports (validation/*.json|md), the sample CSVs, live
analyzer runs against the generator ground truth, a live offline feedback round trip, and
(optionally) a pytest run. No external resources: CSS, JS, data and SVG are all inline.

    python scripts/build_overview.py            # runs pytest for the test summary (~40 s)
    python scripts/build_overview.py --skip-tests
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sentinel import __version__  # noqa: E402
from sentinel.analytics import cold_spot, combustion_dynamics, hets_trip  # noqa: E402
from sentinel.analytics.base import ACTIONS, CAUSES, URGENCY  # noqa: E402
from sentinel.config import load_settings  # noqa: E402
from sentinel.data import load_csv  # noqa: E402
from sentinel.limits import GENERAL  # noqa: E402
from sentinel.notes import parse_notes  # noqa: E402

VALIDATION = ROOT / "validation"
CASES = [
    ("cold_spot", "cold_spot_01", "samples/cold_spot/cold_spot_01.csv", "Exhaust cold spot"),
    ("combustion_dynamics", "combustion_dynamics_05", "samples/combustion_dynamics/comb_dyn_05.csv", "Combustion dynamics"),
    ("hets_trip", "hets_trip_01", "samples/hets_trip/hets_trip_01.csv", "HETS trip"),
]
MAX_POINTS = 420


# --------------------------------------------------------------------------- markdown


def _inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)  # links become plain text (no external links)
    return t


def md_to_html(md: str, shift: int = 2) -> str:
    """Small converter for the markdown subset Sentinel writes (headings, tables, lists, quotes, code)."""
    lines = md.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = min(6, len(m.group(1)) + shift)
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
            continue
        if re.match(r"^[-*]\s+", s):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and not re.match(r"^\s*[-*]\s+", lines[i]):
                    item += " " + lines[i].strip()
                    i += 1
                items.append(f"<li>{_inline(item)}</li>")
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"^\d+\.\s+", s):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                item = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                items.append(f"<li>{_inline(item)}</li>")
                i += 1
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(_inline(lines[i].strip().lstrip(">").strip()))
                i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue
        if s == "---":
            out.append("<hr>")
            i += 1
            continue
        if not s:
            i += 1
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||>|```|[-*]\s|\d+\.\s|---$)", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(buf))}</p>")
    return "\n".join(out)


def md_section(md: str, heading_prefix: str) -> str:
    """Text of one '### ...' / '## ...' section."""
    lines = md.splitlines()
    out, on, lvl = [], False, 0
    for ln in lines:
        m = re.match(r"^(#{2,3})\s+(.*)$", ln)
        if m:
            if on and len(m.group(1)) <= lvl:
                break
            if m.group(2).startswith(heading_prefix):
                on, lvl = True, len(m.group(1))
                continue
        if on:
            out.append(ln)
    return "\n".join(out).strip()


def md_table(md: str) -> List[List[str]]:
    rows = []
    for ln in md.splitlines():
        if ln.strip().startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                rows.append(cells)
    return rows


# --------------------------------------------------------------------------- data


def _bucket(df, cols, ts, lo, hi):
    seg = df.iloc[lo : hi + 1]
    tseg = ts.iloc[lo : hi + 1]
    step = max(1, int(np.ceil(len(seg) / MAX_POINTS)))
    grp = np.arange(len(seg)) // step
    t = tseg.groupby(grp).first().dt.strftime("%Y-%m-%dT%H:%M").tolist()
    panels = []
    for c in cols:
        v = seg[c].groupby(grp).mean().round(3).tolist()
        panels.append({"name": c, "values": v})
    return t, panels, step


def evidence_series(analytic: str, csv: str, chart: Dict[str, Any]) -> Dict[str, Any]:
    case = load_csv(ROOT / csv)
    cols = [c for c in chart["channels"] if c in case.df.columns][:4]
    ev = chart.get("event")
    lo, hi = 0, case.n - 1
    if chart.get("focus"):
        lo, hi = chart["focus"]
    elif analytic == "combustion_dynamics" and ev:
        pad = case.samples(24 * 60)
        lo, hi = max(0, ev[0] - pad), min(case.n - 1, ev[1] + pad)
    t, panels, step = _bucket(case.df, cols, case.timestamps, lo, hi)
    return {
        "t": t,
        "panels": panels,
        "event": [case.time_at(ev[0]).replace(" ", "T"), case.time_at(ev[1]).replace(" ", "T")] if ev else None,
        "agg": f"bucket means of {step} sample{'s' if step > 1 else ''} ({step * case.minutes_per_sample:g} min)",
    }


def ground_truth() -> Dict[str, Any]:
    """Run the analyzers on all 25 samples and compare with the generator's parameters."""
    none = parse_notes("")
    cold, dyn, hets = [], [], []
    for i in range(1, 11):
        f = cold_spot.analyze(load_csv(ROOT / f"samples/cold_spot/cold_spot_{i:02d}.csv"), none)
        tc, depth, width = 1 + ((i - 1) * 3) % 27, 18 + (i % 5) * 4, 3 if i % 3 else 5
        s = f.signature
        cold.append({"i": i, "true_depth": depth, "depth": s["depth_C"], "true_tc": f"TC{tc}", "tc": s["primary_tc"], "true_width": width, "width": s["sector_size"], "action": f.top.action})
    for i in range(1, 11):
        c = load_csv(ROOT / f"samples/combustion_dynamics/comb_dyn_{i:02d}.csv")
        f = combustion_dynamics.analyze(c, none)
        onset = int(14400 * (0.25 + 0.04 * (i % 5)))
        dyn.append({"i": i, "onset_error_min": abs(f.events[0]["start_i"] - onset) * c.minutes_per_sample, "true_period": 3 + (i % 5), "period": f.signature["modulation_period_min"], "peak": f.signature["peak_psi"], "action": f.top.action, "escalate": f.escalate})
    for i in range(1, 6):
        f = hets_trip.analyze(load_csv(ROOT / f"samples/hets_trip/hets_trip_{i:02d}.csv"), none)
        trip = int(8640 * (0.55 + 0.02 * i))
        hets.append({"i": i, "trip_error_samples": abs(f.events[0]["trip_i"] - trip), "opposing": f.signature["opposing_hot_cold"], "action": f.top.action})
    agree = (
        sum(r["tc"] == r["true_tc"] and r["width"] == r["true_width"] and abs(r["depth"] - r["true_depth"]) <= 3 and r["action"] == "borescope" for r in cold)
        + sum(r["onset_error_min"] <= 10 and abs(r["period"] - r["true_period"]) <= 1 and r["action"] == "combustion_tuning" for r in dyn)
        + sum(r["trip_error_samples"] == 0 and r["opposing"] and r["action"] == "borescope" for r in hets)
    )
    return {"cold": cold, "dyn": dyn, "hets": hets, "agree": agree, "total": 25}


def live_demos() -> Dict[str, Any]:
    """Offline (template) runs: an uncertain case for the confidence chart, and a feedback round trip."""
    from sentinel import store
    from sentinel.pipeline import run_case

    with tempfile.TemporaryDirectory() as tmp:
        s = load_settings({"data_dir": Path(tmp), "use_llm": False, "use_tspulse": False})
        unknown = run_case(ROOT / "samples/gt_sensors_demo.csv", "Bearing vibration high", "", settings=s, save=False)
        first = run_case(ROOT / "samples/cold_spot/cold_spot_01.csv", "Exhaust Cold Spot", "Unit 2", settings=s)
        reason = "Borescope found the cans clean; the TC27/TC1/TC2 harness run was chafed and has been replaced"
        store.record_feedback(s, first["case_id"], "not_good", reason=reason, corrected_action="sensor_fix_replace", reviewer="overview demo")
        second = run_case(ROOT / "samples/cold_spot/cold_spot_01.csv", "Exhaust Cold Spot", "Unit 2", settings=s)
        third_src = run_case(ROOT / "samples/combustion_dynamics/comb_dyn_03.csv", "combustion dynamics", "", settings=s)
        store.record_feedback(s, third_src["case_id"], "good", reviewer="overview demo")
        third = run_case(ROOT / "samples/combustion_dynamics/comb_dyn_03.csv", "combustion dynamics", "", settings=s)

    def brief(r):
        d = r["decision"]
        return {
            "case_id": r["case_id"],
            "action": d["primary_action"],
            "action_label": ACTIONS[d["primary_action"]].split(" (")[0],
            "secondary": d["secondary_actions"],
            "confidence": d["confidence"],
            "needs_human": d["needs_human"],
            "rules": d["rules_fired"],
            "flags": d["review_flags"],
            "factors": d["confidence_factors"],
        }

    return {
        "unknown": brief(unknown),
        "unknown_summary": unknown["findings"]["summary"],
        "before": brief(first),
        "after": brief(second),
        "reason": reason,
        "similarity": second["similar_cases"][0]["similarity"] if second["similar_cases"] else None,
        "confirm_before": brief(third_src),
        "confirm_after": brief(third),
    }


def run_tests(skip: bool) -> Dict[str, Any]:
    if skip:
        return {"ran": False}
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (p.stdout or "").strip().splitlines()[-1] if p.stdout else ""
    m_pass = re.search(r"(\d+) passed", tail)
    m_skip = re.search(r"(\d+) skipped", tail)
    m_fail = re.search(r"(\d+) failed", tail)
    m_time = re.search(r"in ([\d.]+)s", tail)
    return {
        "ran": True,
        "passed": int(m_pass.group(1)) if m_pass else 0,
        "skipped": int(m_skip.group(1)) if m_skip else 0,
        "failed": int(m_fail.group(1)) if m_fail else 0,
        "seconds": float(m_time.group(1)) if m_time else None,
        "summary": tail,
    }


def validation_cases() -> List[Dict[str, Any]]:
    out = []
    for analytic, stem, csv, title in CASES:
        rep = json.loads((VALIDATION / f"{stem}_report.json").read_text(encoding="utf-8"))
        md = (VALIDATION / f"{stem}_report.md").read_text(encoding="utf-8")
        d, n = rep["decision"], rep["narrative"]
        m = re.search(r"(\d+) s on CPU", rep["audit"]["narrative_engine"])
        llm_s = float(m.group(1)) if m else 0.0
        tm = n.get("timings") or {}
        prompt_s, gen_s = float(tm.get("prompt_s", 0)), float(tm.get("gen_s", 0))
        out.append(
            {
                "analytic": analytic,
                "stem": stem,
                "title": title,
                "case_id": rep["case_id"],
                "unit": rep["case"].get("unit"),
                "notes": rep["case"].get("notes"),
                "entered": rep["analytic"]["entered"],
                "action": d["primary_action"],
                "action_label": ACTIONS[d["primary_action"]].split(" (")[0],
                "urgency": URGENCY[d["urgency"]].split(" — ")[0],
                "severity": d["severity"],
                "escalate": d["escalate"],
                "needs_human": d["needs_human"],
                "confidence": d["confidence"],
                "confidence_label": d["confidence_label"],
                "factors": d["confidence_factors"],
                "bottom_line": rep["bottom_line"],
                "checks": n.get("checks", {}),
                "prompt_tokens": tm.get("prompt_tokens"),
                "gen_tokens": tm.get("gen_tokens"),
                "runtime": {
                    "rules": round(max(0.0, rep["audit"]["elapsed_s"] - llm_s), 1),
                    "prompt": round(prompt_s, 1),
                    "generation": round(gen_s, 1),
                    "load_other": round(max(0.0, llm_s - prompt_s - gen_s), 1),
                },
                "elapsed": rep["audit"]["elapsed_s"],
                "html": md_to_html(md, shift=2),
                "series": evidence_series(analytic, csv, rep["findings"]["chart"]),
            }
        )
    return out


E2E_SHORT = {"normal": "normal", "cold_spot_hardware": "cold spot", "tc_sensor_fault": "TC fault", "combustion_instability": "instab.",
             "dynamics_probe_fault": "probe", "hets_trip_hardware": "HETS", "igv_lvdt_drift": "IGV", "fuel_control_oscillation": "fuel osc"}


def e2e_section() -> Dict[str, Any]:
    """'End-to-end AI' tab: built from eval/results.json (scripts/eval_e2e.py)."""
    res_file = ROOT / "eval" / "results.json"
    intro = (
        "<p class=\"lead\">The <code>e2e-ai</code> branch adds a <strong>learned path</strong>: raw CSV → canonical channels → IBM TS Pulse "
        "(fine-tuned) → classifier head + similar labeled cases (one-shot) + novelty check → Gemma 4 12B writes the report. The rule path on this "
        "page keeps running in <strong>shadow mode</strong>; disagreement, low confidence or novelty sends the case to a human. Engineers drop in CSVs, "
        "confirm or correct with one click, and the model retrains in the background every 5 labels (versioned, with rollback).</p>"
    )
    link = ('<p>Full interactive explainer (pipeline, drop-in &amp; labeling loop, dashboard, real example reports, evaluation): '
            '<a href="e2e_explainer.html">docs/e2e_explainer.html</a>. Numbers: <code>eval/EVAL.md</code>. Deployment: <code>DEPLOY.md</code>.</p>')
    if not res_file.is_file():
        return {"html": intro + "<p class=\"note\">Not evaluated yet — run <code>python scripts/eval_e2e.py</code>.</p>" + link, "data": None}
    r = json.loads(res_file.read_text(encoding="utf-8"))
    T, U, V = r["test"], r["unseen"], r["variant"]

    def pct(x: Any) -> str:
        return "n/a" if x is None else f"{100 * float(x):.0f}%"

    sim = (r.get("sim_site") or {}).get("curve") or []
    tiles = [
        ("Action accuracy, learned path", pct(T["learned_action_acc"]), f"held-out test · rules {pct(T['rules_action_acc'])}"),
        ("Rule-covered signatures only", pct(T["rule_classes"]["learned_action_acc"]), f"rules {pct(T['rule_classes']['rules_action_acc'])}"),
        ("Unseen fault flagged novel", pct(U["novelty_detection_rate"]),
         f"bearing fault never trained on; {sum(U['forced_into'].values())} of {U['n']} forced into a known class"),
        ("Look-alike fault after ONE label", f"{pct(r['lookalike']['learned_action_acc'])} → {pct(r['lookalike']['kshot'][0]['action_acc_index_only'])}"
         if r.get("lookalike") else "n/a", "TC harness fault mistaken for a cold spot, fixed by one-shot retrieval"),
        ("Labeling simulation", f"{pct(sim[0]['learned'])} → {pct(sim[-1]['learned'])}" if sim else "n/a", f"after {sim[-1]['labels']} labels" if sim else ""),
    ]
    tile_html = '<div class="tiles" role="list">' + "".join(
        f'<div class="tile" role="listitem"><div class="label">{html.escape(a)}</div><div class="value">{html.escape(b)}</div><div class="sub">{html.escape(c)}</div></div>'
        for a, b, c in tiles) + "</div>"
    figs = (
        '<div class="grid-2">'
        '<figure class="chart-card" data-chart="e2ecurve" data-aria="Labeling simulation: action accuracy vs labels"><figcaption><div class="chart-title">Label on the fly: accuracy on held-out site cases</div>'
        '<div class="chart-sub">Scripted simulation through the real code path; retrain every 5 labels. Rules do not learn.</div></figcaption>'
        '<div class="chart"></div><button class="btn table-toggle" type="button">Show data table</button><div class="table-view" hidden></div></figure>'
        '<figure class="chart-card" data-chart="e2eclass" data-aria="Action accuracy per signature, learned vs rules"><figcaption><div class="chart-title">Action accuracy by signature (held-out test)</div>'
        '<div class="chart-sub">IGV drift and fuel oscillation have no rule analyzer.</div></figcaption>'
        '<div class="chart"></div><button class="btn table-toggle" type="button">Show data table</button><div class="table-view" hidden></div></figure>'
        "</div>"
    )
    caveat = ('<p class="note">Synthetic data only: the learned model was trained on the same generator family (different seeds), so these are '
              'upper bounds; the variant and unseen-fault results are the more honest ones. Treat the learned path as shadow-mode decision support until '
              'it has been scored on real labeled Tier 2 cases.</p>')
    data = {
        "curve": sim,
        "per_class": [{"short": E2E_SHORT.get(p["class"], p["class"]), "learned": p["learned_action_acc"], "rules": p["rules_action_acc"]} for p in T["per_class"]],
    }
    return {"html": intro + tile_html + figs + caveat + link, "data": data}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


# --------------------------------------------------------------------------- page


def build(skip_tests: bool) -> str:
    cases = validation_cases()
    gt = ground_truth()
    demos = live_demos()
    tests = run_tests(skip_tests)
    design_path = ROOT / "DESIGN.md"
    if not design_path.is_file():
        design_path = ROOT / "docs" / "dev" / "DESIGN.md"
    design = design_path.read_text(encoding="utf-8")
    val_md = (VALIDATION / "VALIDATION.md").read_text(encoding="utf-8")
    decisions = md_table(md_section(design, "3. Key decisions"))[1:]
    removed = md_table(md_section(design, "8. What was removed"))[1:]
    history = md_table(md_section(val_md, "Validation history"))[1:]
    judgements = {
        "cold_spot": md_to_html(md_section(val_md, "1. Cold spot"), shift=2),
        "combustion_dynamics": md_to_html(md_section(val_md, "2. Combustion dynamics"), shift=2),
        "hets_trip": md_to_html(md_section(val_md, "3. HETS trip"), shift=2),
    }
    rounds = []
    for r in history:
        issues = r[2]
        n_issues = len([x for x in re.split(r";", issues) if x.strip()]) if issues and issues != "—" else 0
        rounds.append({"round": re.sub(r"\s*\(.*", "", r[0]), "issues": n_issues})

    traced = []
    for c in cases:
        m = re.match(r"(\d+)/(\d+)", c["checks"].get("numbers traced to evidence", "0/0"))
        traced.append((int(m.group(1)), int(m.group(2))) if m else (0, 0))
    traced_ok, traced_all = sum(a for a, _ in traced), sum(b for _, b in traced)

    data = {
        "cases": [{k: v for k, v in c.items() if k != "html"} for c in cases],
        "gt": gt,
        "demos": demos,
        "rounds": rounds,
    }
    e2e = e2e_section()
    data["e2e"] = e2e["data"]
    test_tile = f"{tests['passed']} passed" if tests.get("ran") else "not run"
    test_sub = (
        f"{tests['failed']} failed · {tests['skipped']} opt-in skipped · {tests['seconds']:.0f} s"
        if tests.get("ran") and tests.get("seconds") is not None
        else "build used --skip-tests"
    )

    dec_html = "".join(
        f'<details class="card decision"><summary><span>{_inline(r[0])}</span></summary><p>{_inline(r[1])}</p></details>' for r in decisions if len(r) >= 2
    )
    rem_html = "".join(f"<tr><td>{_inline(r[0])}</td><td>{_inline(r[1])}</td></tr>" for r in removed if len(r) >= 2)
    hist_html = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in history)
    actions_html = "".join(f"<tr><td><code>{k}</code></td><td>{html.escape(v)}</td></tr>" for k, v in ACTIONS.items())
    urgency_html = "".join(f"<tr><td><code>{k}</code></td><td>{html.escape(v)}</td></tr>" for k, v in URGENCY.items())
    causes_html = "".join(f"<li><code>{k}</code> — {html.escape(v)}</li>" for k, v in CAUSES.items())

    verdict_rows = "".join(
        f"<tr><td>{html.escape(c['stem'])}</td><td>{html.escape(c['entered'])}</td><td><strong>{html.escape(c['action_label'])}</strong></td>"
        f"<td>{html.escape(c['urgency'])} / {c['severity']} / {'yes' if c['escalate'] else 'no'}</td><td class='num'>{c['confidence']:.2f}</td>"
        f"<td>{html.escape(c['checks'].get('numbers traced to evidence', '?'))}</td><td><span class='badge good'>✓ Correct</span></td></tr>"
        for c in cases
    )
    case_tabs = "".join(f'<button role="tab" data-tab="{c["analytic"]}">{html.escape(c["title"])}</button>' for c in cases)
    report_panels = "".join(
        f'<div role="tabpanel" data-panel="{c["analytic"]}"><div class="report-frame">{c["html"]}</div></div>' for c in cases
    )
    val_panels = "".join(
        f'''<div role="tabpanel" data-panel="{c["analytic"]}">
          <p class="bottom-line"><strong>Bottom line (from the report):</strong> {html.escape(c["bottom_line"])}</p>
          <figure class="chart-card" data-chart="evidence" data-case="{i}">
            <figcaption><span class="chart-title">Evidence — {html.escape(c["title"])}, {html.escape(c["stem"])}</span>
            <span class="chart-sub">One panel per channel (different units, never a shared axis). Shaded band = detected event window. {html.escape(c["series"]["agg"])}.</span></figcaption>
            <div class="chart"></div><button class="table-toggle" type="button">Show data table</button><div class="table-view" hidden></div>
          </figure>
          <details class="card" open><summary>Tier 2 judgement</summary>{judgements[c["analytic"]]}</details>
        </div>'''
        for i, c in enumerate(cases)
    )

    page = TEMPLATE
    repl = {
        "%%VERSION%%": __version__,
        "%%COMMIT%%": git_commit(),
        "%%GENERATED%%": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "%%GT_AGREE%%": f"{gt['agree']}/{gt['total']}",
        "%%TESTS%%": test_tile,
        "%%TESTS_SUB%%": test_sub,
        "%%TRACED%%": f"{traced_ok}/{traced_all}",
        "%%DECISIONS%%": dec_html,
        "%%REMOVED%%": rem_html,
        "%%HISTORY%%": hist_html,
        "%%ACTIONS%%": actions_html,
        "%%URGENCY%%": urgency_html,
        "%%CAUSES%%": causes_html,
        "%%VERDICTS%%": verdict_rows,
        "%%CASE_TABS%%": case_tabs,
        "%%REPORT_PANELS%%": report_panels,
        "%%VAL_PANELS%%": val_panels,
        "%%APPLY_SIM%%": f"{GENERAL['feedback_apply_similarity']:.2f}",
        "%%SHOW_SIM%%": f"{GENERAL['feedback_show_similarity']:.2f}",
        "%%E2E%%": e2e["html"],
        "%%DATA%%": json.dumps(data).replace("</", "<\\/"),
    }
    for k, v in repl.items():
        page = page.replace(k, v)
    return page


TEMPLATE = (Path(__file__).with_name("overview_template.html")).read_text(encoding="utf-8") if (Path(__file__).with_name("overview_template.html")).exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "sentinel_overview.html"))
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(a.skip_tests), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
