#!/usr/bin/env python3
"""Build docs/sentinel_v2_explainer.html — one offline page for the V2 app.

Reads the evaluation, the example reports, the table of contents in sentinel_gt_ai.py,
and one real Gemma prompt (built by the writer, model not called). No external URLs.

    python scripts/build_v2_explainer.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from build_overview import md_to_html  # noqa: E402

OUT = ROOT / "docs" / "sentinel_v2_explainer.html"
CHARTS = ("per_class.svg", "sim_site.svg", "novelty.svg")


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pct(x: object) -> str:
    if x is None:
        return "n/a"
    return f"{100 * float(x):.1f}%"


def kx(block: dict) -> str:
    if not block or not block.get("n"):
        return "n/a"
    return f"{block['k']}/{block['n']} ({pct(block['rate'])})"


def toc_rows() -> list[tuple[str, str, str]]:
    """Line, section name, and the one-line why from the top of sentinel_gt_ai.py."""
    rows = []
    for line in (ROOT / "sentinel_gt_ai.py").read_text(encoding="utf-8").splitlines():
        # Line numbers are right-aligned in five columns, so the leading space count varies (3 to 5 digits).
        m = re.match(r"^\s*(\d+)\s+(sentinel[\w.]*)\s+—\s+(.*)$", line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3)))
    if not rows:
        raise SystemExit("could not read the table of contents in sentinel_gt_ai.py")
    return rows


def chart_svgs() -> str:
    parts = []
    folder = ROOT / "eval" / "charts"
    for name in CHARTS:
        path = folder / name
        if not path.is_file():
            continue
        svg = path.read_text(encoding="utf-8").strip()
        if svg.startswith("<?xml"):
            svg = svg.split("?>", 1)[1].strip()
        parts.append(f'<figure class="chart">{svg}<figcaption>{esc(name)}</figcaption></figure>')
    return "\n".join(parts) or "<p>Charts appear after <code>python scripts/eval_e2e.py</code>.</p>"


def comparison_table(res: dict) -> str:
    residual = res.get("residual") or {}
    test = res.get("test") or {}
    base_path = ROOT / "eval" / "results_v1_baseline.json"
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.is_file() else {}
    bt = base.get("test") or {}
    bu = base.get("unseen") or {}
    novelty = (residual.get("test") or {}).get("novelty") or {}
    rows = [
        ("iid action accuracy", pct(bt.get("learned_action_acc")), pct(test.get("learned_action_acc")), pct(novelty.get("action_acc_after"))),
        ("iid signature accuracy", pct(bt.get("learned_signature_acc")), pct(test.get("learned_signature_acc")), pct(novelty.get("signature_acc_after"))),
        ("iid false novelty", pct(bt.get("false_novelty_rate")), pct(test.get("false_novelty_rate")), pct(novelty.get("false_novelty_after"))),
        ("bearing novelty rate", pct(bu.get("novelty_detection_rate")), pct((res.get("unseen") or {}).get("novelty_detection_rate")), pct((residual.get("bearing") or {}).get("novelty_after"))),
    ]
    body = "".join(
        f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>" for a, b, c, d in rows
    )
    return (
        "<table><thead><tr><th>Metric</th><th>V1 baseline</th><th>V2 classifier</th><th>V2 + residual novelty</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def residual_notes(res: dict) -> str:
    r = res.get("residual")
    if not r:
        return "<p>Residual cross-check numbers are written by <code>python scripts/eval_e2e.py</code> into eval/results.json.</p>"
    sf, br, ha = r["sensorfault"], r["bearing"], r["harness"]
    nv = r["test"]["novelty"]
    return f"""
    <ul>
      <li>Residual mode on this run: <strong>{esc(r.get("mode"))}</strong> ({esc(r.get("n_scored"))} held-out cases, {esc(r.get("seconds"))} s).</li>
      <li>False contradictions on correctly classified iid cases: <strong>{esc(kx(r["test"]["false_contradictions"]))}</strong>.</li>
      <li>Contradictions on iid classifier errors: <strong>{esc(kx(r["test"]["error_contradictions"]))}</strong>.</li>
      <li>Sensor-fault split ({esc(sf["n"])} single-thermocouple steps, truth <code>tc_sensor_fault</code>). Called a cold spot and contradicted: <strong>{esc(kx(sf["cold_spot_contradicted"]))}</strong>. Other calls contradicted: {esc(kx(sf["other_calls_contradicted"]))}. Sensor-like shape: {esc(sf["sensor_like_k"])}/{esc(sf["n"])}.</li>
      <li>Bearing faults forced into a known class and contradicted: <strong>{esc(kx(br["forced_contradicted"]))}</strong>. Called combustion instability and contradicted: {esc(kx(br["instability_contradicted"]))}.</li>
      <li>Look-alike harness cases with a sensor-like residual: <strong>{esc(ha["sensor_like_k"])}/{esc(ha["n"])}</strong>.</li>
      <li>Residual novelty flipped {esc(nv["correct_flipped_to_novel"])} correct iid case(s) and {esc(nv["errors_flipped_to_novel"])} error(s) to novel.</li>
    </ul>"""


def llm_table(res: dict) -> str:
    cases = ((res.get("llm") or {}).get("cases")) or []
    if not cases:
        return "<p>Example timings appear after <code>python scripts/eval_e2e.py --llm-cases 5</code>.</p>"
    rows = []
    for c in cases:
        mark = "yes" if c.get("final_correct") else "no"
        rows.append(
            "<tr>"
            f"<td><code>{esc(c.get('case_id'))}</code></td>"
            f"<td>{esc(c.get('truth_action'))}</td>"
            f"<td>{esc(c.get('classifier_action'))}</td>"
            f"<td>{esc(c.get('final_action'))} ({mark})</td>"
            f"<td>{esc(c.get('residual_det'))} / {esc(c.get('residual_llm'))}</td>"
            f"<td>{esc(c.get('prompt_tokens'))}+{esc(c.get('completion_tokens'))}</td>"
            f"<td>{esc(c.get('seconds'))} s</td>"
            "</tr>"
        )
    summary = (res.get("llm") or {}).get("summary") or {}
    head = (
        f"<p>{esc((res.get('llm') or {}).get('model'))} on {esc((res.get('llm') or {}).get('device'))}. "
        f"Mean {esc(summary.get('mean_seconds'))} s, mean prompt {esc(summary.get('mean_prompt_tokens'))} tokens, "
        f"mean completion {esc(summary.get('mean_completion_tokens'))} tokens. "
        f"Final action matched ground truth on {esc(pct(summary.get('final_action_acc')))} of this sample. "
        f"Deterministic contradictions: {esc(summary.get('residual_det_contradicts'))}. "
        f"LLM contradictions: {esc(summary.get('residual_llm_contradicts'))}. "
        f"The two verdicts agreed on {esc(summary.get('residual_agree'))}.</p>"
    )
    return head + (
        "<table><thead><tr><th>Case</th><th>Truth</th><th>Classifier</th><th>Final</th>"
        "<th>Residual det / LLM</th><th>Tokens</th><th>Time</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def report_tabs() -> str:
    files = sorted((ROOT / "eval" / "examples").glob("*.md"))
    if not files:
        return "<p>No example reports yet.</p>"
    buttons, panels = [], []
    for i, path in enumerate(files):
        buttons.append(f'<button type="button" class="tab{" on" if i == 0 else ""}" data-tab="rep-{i}">{esc(path.stem)}</button>')
        hidden = "" if i == 0 else " hidden"
        panels.append(f'<div class="panel{hidden}" id="rep-{i}"><article class="report">{md_to_html(path.read_text(encoding="utf-8"))}</article></div>')
    return f'<div class="tabs">{"".join(buttons)}</div>{"".join(panels)}'


def real_prompt() -> tuple[str, str, str]:
    """One prompt the writer would send to Gemma, from a held-out case. Does not call the model."""
    from sentinel.config import load_settings
    from sentinel.e2e.dataset import load_manifest
    from sentinel.e2e.pipeline import analyze_case
    from sentinel.e2e.writer import SYSTEM, build_prompt

    root = ROOT / "data" / "e2e" / "dataset"
    rows = load_manifest(root)
    row = next((r for r in rows if r["split"] == "sensorfault"), None)
    if row is None:
        row = next(r for r in rows if r["split"] == "test" and r["signature"] == "cold_spot_hardware")
    with tempfile.TemporaryDirectory(prefix="sentinel_v2_prompt_") as tmp:
        settings = load_settings({"data_dir": tmp, "use_llm": False})
        report = analyze_case(root / row["file"], row["analytic"], row["notes"], settings=settings, use_llm=False, save=False, source_name=Path(row["file"]).name)
        from sentinel.e2e.pipeline import get_engine

        user = build_prompt(report, get_engine(settings).taxonomy())
    title = f"{row['case_id']} ({row['split']}, truth {row['signature']})"
    return title, SYSTEM.strip(), user


def pattern_list() -> str:
    from sentinel.e2e.residual import PATTERN_TEXT

    items = "".join(f"<li><code>{esc(k)}</code> — {esc(v)}</li>" for k, v in PATTERN_TEXT.items())
    return f"<ul>{items}</ul>"


def assert_offline(page: str) -> None:
    if re.search(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", page):
        raise SystemExit("explainer has an external src or href")
    if "<link" in page or "@import" in page or re.search(r"<script[^>]*\bsrc=", page):
        raise SystemExit("explainer loads an external stylesheet or script")
    if re.search(r"Qwen|llm_profile", page):
        raise SystemExit("explainer still mentions a second model")
    for sec in ("north", "pipeline", "prompt", "teach", "dashboard", "reports", "eval", "deploy", "map"):
        if f'id="{sec}"' not in page:
            raise SystemExit(f"missing section {sec}")


def build() -> str:
    res = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    title, system, user = real_prompt()
    toc = "\n".join(
        f"<tr><td>{esc(line)}</td><td><code>{esc(name)}</code></td><td>{esc(why)}</td></tr>" for line, name, why in toc_rows()
    )
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel GT AI V2</title>
<style>
:root {{
  --bg: #f6f5f1; --ink: #1c1b19; --muted: #5c5a55; --line: #d9d6cc;
  --card: #fffcf7; --accent: #0f6e56; --accent-2: #8a3b12; --code: #efece4;
}}
body.dark {{
  --bg: #171614; --ink: #f3f0e8; --muted: #b7b2a6; --line: #3a3833;
  --card: #221f1c; --accent: #7dcea0; --accent-2: #e8a87c; --code: #2c2925;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 16px/1.5 "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--ink); }}
a {{ color: var(--accent); }}
header {{ padding: 1.2rem 1.4rem 0.4rem; border-bottom: 1px solid var(--line); }}
header h1 {{ margin: 0.2rem 0; font-size: 1.6rem; }}
.eyebrow {{ letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.75rem; color: var(--muted); }}
nav {{ display: flex; flex-wrap: wrap; gap: 0.4rem 0.8rem; padding: 0.6rem 1.4rem 0.8rem; position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--line); z-index: 2; }}
nav a {{ text-decoration: none; }}
main {{ max-width: 980px; margin: 0 auto; padding: 1rem 1.2rem 3rem; }}
section {{ margin: 1.6rem 0 2.2rem; }}
h2 {{ font-size: 1.25rem; margin-bottom: 0.4rem; }}
.card, .report, pre, table {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; }}
.card {{ padding: 0.8rem 1rem; margin: 0.6rem 0; }}
ol.flow {{ padding-left: 1.2rem; }}
ol.flow li {{ margin: 0.35rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 0.35rem 0.5rem; vertical-align: top; }}
pre {{ padding: 0.8rem 1rem; overflow: auto; white-space: pre-wrap; font: 0.82rem/1.45 ui-monospace, Consolas, monospace; background: var(--code); }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0.6rem 0; }}
.tab {{ border: 1px solid var(--line); background: var(--card); color: var(--ink); border-radius: 999px; padding: 0.25rem 0.7rem; cursor: pointer; }}
.tab.on {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
body.dark .tab.on {{ color: #14211b; }}
.hidden {{ display: none; }}
.chart svg {{ width: 100%; height: auto; }}
figure {{ margin: 0.8rem 0; }}
figcaption {{ color: var(--muted); font-size: 0.8rem; }}
.pipe {{ display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; margin: 0.8rem 0; }}
.pipe span {{ background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 0.35rem 0.55rem; }}
.pipe i {{ color: var(--accent-2); font-style: normal; }}
button.ghost {{ margin-left: 0.6rem; border: 1px solid var(--line); background: transparent; color: var(--ink); border-radius: 6px; padding: 0.2rem 0.55rem; cursor: pointer; }}
code {{ font-family: ui-monospace, Consolas, monospace; font-size: 0.9em; }}
</style>
</head>
<body>
<header>
  <div class="eyebrow">Sentinel GT AI · local · human in the loop</div>
  <h1>One app for a Tier 2 engineer</h1>
  <p>Generated {esc(generated)}. Single file, no network. Rebuild with <code>python scripts/build_v2_explainer.py</code>.</p>
</header>
<nav>
  <a href="#north">Workflow</a>
  <a href="#pipeline">Pipeline</a>
  <a href="#prompt">What Gemma sees</a>
  <a href="#teach">Teach and retrain</a>
  <a href="#dashboard">Dashboard</a>
  <a href="#reports">Reports</a>
  <a href="#eval">Evaluation</a>
  <a href="#deploy">Deploy</a>
  <a href="#map">The file</a>
  <button type="button" class="ghost" id="theme">Dark</button>
</nav>
<main>

<section id="north">
  <h2>North star</h2>
  <p>A Tier 2 engineer stays in one local app. They drop in a historian CSV, the analytic name and their notes, and get a draft site report with the evidence beside it. They review, edit, and approve or correct it. The human makes the final call. They can also teach the app and retrain it here. Review and inference do not use the network.</p>
  <div class="card">
    <ol class="flow">
      <li><strong>Drop in.</strong> CSV, analytic name, notes.</li>
      <li><strong>Draft.</strong> Classifier, residual view, similar labeled cases, rule shadow, process steps, then Gemma 4 12B (or the template if the model fails its checks).</li>
      <li><strong>Review.</strong> The report stays a draft until the engineer marks it. A disagreement, a low confidence, a novel pattern, or a residual contradiction asks for a person.</li>
      <li><strong>Teach.</strong> Label the case and say why, add a process document, name a new signature, add a frame layout, or add an alert type.</li>
      <li><strong>Retrain.</strong> Quick update in seconds, or a full TS Pulse fine-tune. Keep the new version only when held-out accuracy is not worse, including each frame and alert slice that has enough cases. Roll back in one click.</li>
    </ol>
  </div>
</section>

<section id="pipeline">
  <h2>Pipeline</h2>
  <div class="pipe" aria-label="Pipeline from CSV to the review gate">
    <span>CSV</span><i>→</i>
    <span>features</span><i>→</i>
    <span>TS Pulse classifier and embedding</span><i>+</i>
    <span>TS Pulse residuals</span><i>→</i>
    <span>similarity and novelty</span><i>→</i>
    <span>rules shadow</span><i>→</i>
    <span>process knowledge</span><i>→</i>
    <span>Gemma prompt</span><i>→</i>
    <span>checks</span><i>→</i>
    <span>report</span><i>→</i>
    <span>review gate</span>
  </div>
  <div class="card"><h3>Features</h3><p>Fourteen canonical channels, 512 points (the whole record and a zoom). The encoder was fine-tuned on this layout. The frame models are 6B, 7EA, 7FA, 7HA, 9E, 9FA, 9HA, and Other / custom. Each model points at a tag layout in <code>LAYOUTS</code> plus a typical exhaust thermocouple count. That layout (or one added from Teach) puts the frame's exhaust and process tags onto those channels. Ring tags are ordered by the full sensor number, so TTXD_10 sits after TTXD_9. The case carries a frame model, from <code>--frame</code>, the dropdown, or the column names. Other / custom picks a layout or defines tags. Old layout ids on stored cases are mapped when the file is loaded.</p></div>
  <div class="card"><h3>Classifier, similarity, novelty</h3><p>The fine-tuned TS Pulse encoder writes a 672-number embedding. A full retrain appends a small frame one-hot so the head, the similarity index, and the novelty radius can tell layouts apart. A frame gets its own novelty radius once enough validation cases exist; otherwise the shared radius is used. The similarity index cites the nearest labeled cases, including one the engineer just taught. If the nearest case is farther than the radius, the signature is novel and the action is escalate. One close engineer-labeled case can override the head.</p></div>
  <div class="card"><h3>Residuals</h3><p>A second TS Pulse (reconstruction) scores each channel against its own baseline. It records which channel moved first, the size in physical units, whether neighbours followed, and whether the shape is a step, a freeze, spikes, or a sector. If those weights are missing, a robust-z score is used instead. <code>doctor</code> says which mode is active.</p></div>
  <div class="card"><h3>Cross-check</h3><p>Each known signature has an expected residual pattern. The deterministic check returns supports, contradicts, or inconclusive. A single thermocouple stepping down while its neighbours stay flat contradicts a cold-sector signature and points at a sensor fault. A contradiction lowers confidence, is written into the concerns, and sets <code>needs_human</code>. Gemma is asked for the same verdict. If the two disagree, that disagreement is itself a review flag.</p>
  {pattern_list()}
  <div class="card"><h3>Rules, knowledge, checks</h3><p>The rule analyzers still run, in shadow. Their action is shown next to the learned action. Process maps (and documents added on the Teach page) are retrieved for the analytic, the signature, and whatever the residual check points to. Gemma must walk those steps. Every number in the narrative has to appear in the prompt. A novel case is never forced into a known class.</p></div>
</section>

<section id="prompt">
  <h2>What Gemma sees</h2>
  <p>This is a real prompt for held-out case <code>{esc(title)}</code>. The writer built it from the case. Gemma was not called to produce this block. The system text is the instruction. The user text is the evidence.</p>
  <h3>System</h3>
  <pre>{esc(system)}</pre>
  <h3>User</h3>
  <pre>{esc(user)}</pre>
  <p>How the cross-check uses it: the user text has a residual view (onset order and neighbour behaviour) and a catalogue line for the expected pattern of the classifier's signature. Gemma sets <code>residual_consistency</code> to supports, contradicts, or inconclusive, and names the deciding fact. The app compares that with the deterministic verdict computed before the prompt was sent.</p>
</section>

<section id="teach">
  <h2>Teach the app, then retrain</h2>
  <div class="card"><h3>Teach page</h3>
    <ul>
      <li><strong>Cases.</strong> A labeled CSV plus a one-sentence why. The why is stored with the case and can be cited in a later prompt.</li>
      <li><strong>Documents.</strong> Markdown, text, or PDF. Sections are retrieved with the process maps.</li>
      <li><strong>Signatures.</strong> Name a new class, its action, and the residual pattern you expect.</li>
      <li><strong>Frames.</strong> Add a frame. Exhaust tags and process tags map onto the shared canonical channels. The shipped models are 6B, 7EA, 7FA, 7HA, 9E, 9FA, 9HA, and Other / custom. Each one points at a tag layout. A frame added here is stored with Teach, appears in the dropdown, and returns if that version is rolled forward.</li>
      <li><strong>Alert types.</strong> A new alert is a signature plus a process document. An alert with no built-in analyzer uses the generic fallback (learned model, similarity index, and that document). The report states which path ran.</li>
      <li><strong>Manage.</strong> See what was taught and roll a teach-state version back.</li>
    </ul>
  </div>
  <div class="card"><h3>Retrain TS Pulse</h3>
    <p><strong>Quick update</strong> refits the classifier head, rebuilds the similarity index from labeled cases, and recalibrates novelty. It does not change the encoder. On a laptop this is seconds once embeddings exist.</p>
    <p><strong>Full retrain</strong> fine-tunes TS Pulse on the synthetic set plus engineer cases, then runs the quick steps. On CPU that is on the order of 15–20 minutes. The job is a background process with a progress file, so leaving the page does not stop it.</p>
    <p>Every result is a new version. It becomes the active model only when held-out action accuracy and signature accuracy are not worse than the current version, and no frame or alert slice with enough cases got worse. Force promote overrides that gate, and the reason is written to the train log. The Teach page and the report audit show the slice table. <strong>Roll back one version</strong> or restore any earlier version from the table. The same commands exist on the command line:</p>
    <pre>python sentinel_gt_ai.py train --quick --device auto
python sentinel_gt_ai.py train --full --device auto
python sentinel_gt_ai.py train --full --device cpu --force</pre>
    <p><code>--device auto</code> uses the GPU when CUDA is available and retries the fine-tune on CPU if the GPU is missing or runs out of memory. <code>gpu</code> and <code>cpu</code> skip the choice.</p>
  </div>
</section>

<section id="dashboard">
  <h2>Learning dashboard</h2>
  <p>The Learning dashboard page plots held-out accuracy, agreement with the rules, novelty, and how many labels have come in. The curve below is the scripted site-adaptation drill in the evaluation: cases arrive, the engineer label is applied, and the head retrains every five labels. Rules do not move. The learned line does.</p>
  {chart_svgs()}
</section>

<section id="reports">
  <h2>Example reports</h2>
  <p>These are real Gemma 4 12B reports on held-out cases, saved by <code>scripts/eval_e2e.py</code>. Each one is still a draft until an engineer marks it.</p>
  {report_tabs()}
  <h3>This sample</h3>
  {llm_table(res)}
</section>

<section id="eval">
  <h2>Before and after</h2>
  <p>All of this data is synthetic, from the built-in generator. Iid accuracy is an upper bound, not a field estimate. The classifier column is the same decision used in the V1 baseline (embedding, head, index). The last column lets residual novelty change a borderline or low-confidence call. Full write-up: <code>eval/EVAL.md</code>.</p>
  <p>A later CPU re-fine-tune, after exhaust tags were sorted by the full sensor number, scored 89.69% action and signature accuracy on the same 320 cases (was 90.31%). The promotion gate blocked on that drop and on two alert slices. <code>--force</code> promoted that version and wrote the reason to the train log. Frame slices and the command are in <code>eval/frame_retrain.json</code>.</p>
  {comparison_table(res)}
  {residual_notes(res)}
</section>

<section id="deploy">
  <h2>Deploying to the Dell</h2>
  <ol class="flow">
    <li>Copy <code>sentinel_gt_ai.py</code> (or this repository) onto the machine.</li>
    <li><code>python sentinel_gt_ai.py setup</code>. Offline, point it at a USB copy: <code>python sentinel_gt_ai.py setup --models-from E:\\models</code>.</li>
    <li><code>python sentinel_gt_ai.py</code>.</li>
  </ol>
  <p><code>python sentinel_gt_ai.py doctor</code> checks the GPU, the llama.cpp build, Gemma, both TS Pulse snapshots, and the fine-tuned encoder. A missing model file prints <code>python sentinel_gt_ai.py setup</code>. A missing encoder prints <code>python sentinel_gt_ai.py train --full --device auto</code>.</p>
  <p>The target Dell has a 6 GB NVIDIA GPU. <code>device: auto</code> offloads as many Gemma layers as free memory allows (about 28 of 48 when roughly 5.5 GB is free) and runs the rest on CPU. If the GPU init fails or runs out of memory, the report finishes on CPU and the audit says so. The build laptop used for these numbers is CPU-only. TS Pulse embedding is about a second or less per case. A Gemma report on that CPU is a few minutes; the table in the reports section has the measured times for this run.</p>
</section>

<section id="map">
  <h2>Where to edit sentinel_gt_ai.py</h2>
  <p>The whole app is one file. Each block under a banner is one module. The line numbers are the ones in the file. Edit the block in place. Data stays outside: <code>models/</code>, <code>knowledge/</code>, <code>data/</code>, <code>settings.local.json</code>. Tunable defaults live in the CONFIG section (<code>sentinel.config</code>).</p>
  <table>
    <thead><tr><th>Line</th><th>Section</th><th>What it is for</th></tr></thead>
    <tbody>
    {toc}
    </tbody>
  </table>
</section>

</main>
<script>
(function () {{
  var theme = document.getElementById("theme");
  theme.addEventListener("click", function () {{
    document.body.classList.toggle("dark");
    theme.textContent = document.body.classList.contains("dark") ? "Light" : "Dark";
  }});
  document.querySelectorAll(".tab").forEach(function (btn) {{
    btn.addEventListener("click", function () {{
      var id = btn.getAttribute("data-tab");
      document.querySelectorAll(".tab").forEach(function (b) {{ b.classList.toggle("on", b === btn); }});
      document.querySelectorAll(".panel").forEach(function (p) {{ p.classList.toggle("hidden", p.id !== id); }});
    }});
  }});
}})();
</script>
</body>
</html>
"""
    assert_offline(page)
    return page


def main() -> int:
    page = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
