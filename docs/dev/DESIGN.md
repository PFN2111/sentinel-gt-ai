# Sentinel GT AI — Design

Owner: the engineer (Tier 2 gas turbine remote monitoring engineer). Status: v1 design, written before implementation.

## 1. End goal

A drop-in tool for a **Tier 2 remote monitoring engineer**. Tier 1 opens a case from an analytic; the
engineer drops in:

1. a CSV of the relevant historian data,
2. the analytic name (free text, e.g. `Exhaust Cold Spot`, `comb dyn high`, `HETS trip`),
3. free-text notes (unit, operator notes, prior history),

and Sentinel returns a professional **Tier 2 site report** — what happened, why, the specific
corrective action, evidence, and confidence — as a **DRAFT**. The engineer marks it
**Good / Not good** (+ optional reason / corrected action). Feedback is stored locally and
changes future reports.

Everything runs on the local machine. No network calls at runtime.

## 2. Architecture

```
CSV + analytic name + notes
        │
        ▼
 [1] data.py         load CSV, parse timestamps, map channel roles, data-quality checks
        │
        ▼
 [2] analytics/      resolve analytic name → analyzer (cold_spot | combustion_dynamics |
                     hets_trip | generic). Each analyzer is deterministic signal processing
                     that returns Findings: events, signature numbers, evidence items,
                     ranked cause hypotheses with the rule that supports each.
        │
        ├──► [3] anomaly.py   generic anomaly scan (robust z-score always; TS Pulse
        │                     reconstruction error if weights + tsfm load). Used only to
        │                     corroborate the event window and to detect NOVEL anomalies
        │                     in channels the analytic does not explain.
        │
        ├──► [4] store.py     similar past cases that carry engineer feedback
        │                     (structured signature similarity, same analytic)
        ▼
 [5] triage.py       DECISION LAYER (rules, no LLM): severity, cause category, primary
                     action, secondary actions, urgency, escalate, needs_human, review
                     flags, confidence with an itemised factor breakdown, rule IDs fired.
                     Applies engineer feedback (corrections / confirmations).
        │
        ▼
 [6] llm.py          OPTIONAL narrative: Granite 4.1 8B GGUF via the local llama.cpp
                     binary (llama-completion.exe, subprocess, JSON-schema constrained).
                     Gets facts + decision + process-map excerpt + similar feedback cases.
                     Writes what happened / why / action rationale, plus a boolean
                     supports_action (false → review flag). Cannot change the action.
                     Site instructions stay deterministic (vetted analyzer wording).
                     Output is checked: JSON valid, action named, numbers traceable.
                     No GGUF / disabled / failure → template narrative (always available).
        │
        ▼
 [7] report.py       report.json (machine-readable, full audit) + report.md (site report)
        │
        ▼
 [8] store.py        data/cases/<case_id>/{report.json, report.md, input.csv, llm_*.txt}
                     feedback → data/feedback.jsonl + case status update (human gate)
```

UI: one **Streamlit** page (local only) + a **CLI** (`python app.py run|feedback|cases|status|ui`).

## 3. Key decisions and why

| Decision | Why |
|---|---|
| **Deterministic triage layer before the LLM (yes, added).** | An 8B Q4 model on CPU is not reliable at choosing a corrective action (smoke test: it described an exhaust cold spot as "moisture condensation"). Remote monitoring actions must be consistent and auditable. Rules grounded in the process maps decide; the LLM explains. Every conclusion carries a rule ID. |
| **Analytic-specific analyzers** for cold_spot, combustion_dynamics, hets_trip; a **generic** analyzer for anything else. | The three scoped analytics have well-defined physics (TC ring sector, pulsation amplitude/coupling, pre-trip spread build-up). The generic path still runs anomaly scan + all three signature probes, but is always flagged `needs_human`. |
| **Sensor-vs-process discrimination is explicit** in each analyzer. | The most common Tier 2 mistake is a borescope call on a bad thermocouple (or a tuning call on a failed dynamics probe). Each analyzer tests adjacency, step-vs-ramp onset, flatline/saturation and cross-channel coupling before choosing a process cause. |
| **LLM via llama.cpp CLI subprocess** (`llama-completion.exe`, existing binaries), JSON-schema constrained, temp 0, fixed seed. | Already proven on this machine with this GGUF; no python-binding build issues; grammar-constrained JSON makes parsing robust; deterministic. Model reloads per report (~17 s) — acceptable for a one-report-per-case workflow and keeps the process model simple. |
| **LLM writes prose only; site instructions stay deterministic.** | Validation showed Granite occasionally garbles step lists ("after the borescope… schedule the borescope") and writes "No concerns" into a free-text concern field. Prose is where the LLM adds value; instructions a technician acts on keep the vetted analyzer wording, and disagreement is a boolean, not parsed text. |
| **Template narrative is a first-class path**, not an error path. | Brief requires offline operation without the GGUF. The template uses the same findings, so the structured content (action, evidence, confidence) is identical with or without the LLM. |
| **TS Pulse kept, but only as corroboration/novelty**, with robust z-score always computed. | TS Pulse (`main` = hybrid-allhead, the AD variant) is a 1M-param zero-shot model — fine for "is this window abnormal?", not for root cause. The base used only the last 512 rows per channel, so it missed events in long CSVs; replaced with sliding windows over the full series. If torch/tsfm/weights are missing, z-score is used and the report says so. |
| **ChromaDB + sentence-transformers removed**; replaced by structured similarity for cases + BM25 for knowledge text. | Corpus is tiny (a few process maps, tens–hundreds of cases). Structured signature similarity (same analytic, similar depth/width/onset/amplitude…) finds *relevant* engineer feedback far better than text embeddings, is explainable (score shown), and drops a heavy dependency that has its own telemetry. |
| **Feedback changes outcomes deterministically** (not only via prompt). | "Not good + corrected action" on a highly similar case (similarity ≥ 0.80) re-routes the primary action for the new case, with the rule action kept as "alternative" and a review flag. "Good" on similar cases raises confidence. All feedback examples (with reasons) also go into the LLM prompt. Everything is logged in JSONL — the raw material for preference tuning later (`python app.py export-prefs`). |
| **Human review gate.** | Every report is `DRAFT — pending Tier 2 review` until the engineer marks it. Good → `APPROVED`; Not good → `REJECTED` (+ corrected action shown as the engineer's final action). Status is in both report.md and report.json. |
| **Streamlit** for the single UI. | Drag-and-drop CSV, markdown rendering and feedback buttons with very little code. Configured local-only: `server.address=localhost`, `browser.gatherUsageStats=false`, headless (no telemetry, nothing leaves the machine). |
| **Settings**: env vars > `settings.local.json` (git-ignored) > defaults. | `GT_GGUF_PATH` points at the existing GGUF (no download, ever). `SENTINEL_LLAMA_BIN_DIR`, `SENTINEL_TSPULSE_DIR` likewise point at existing assets. |
| **Thresholds in one place** (`sentinel/limits.py`). | Site limits differ by frame/unit; a Tier 2 engineer must be able to see and change them. Every threshold used is echoed into the audit section. |

## 4. Analyzer logic (summary)

**cold_spot** (TC ring `TC1..TCn`, falls back to zone EGTs):
deviation of each TC from the row ring-median, minus its own long-run baseline deviation →
"excess cold". Event = sustained excess cold below threshold. Characterise: primary TC,
sector members, ring adjacency (wrap-around), depth, onset ramp (step vs gradual), duration,
still active at end of data, spread vs baseline, fuel/load/vibration coupling.
- isolated single TC / step onset / flatline / implausible values → **instrumentation** →
  *sensor fix/replace*
- ≥2 adjacent TCs, gradual onset, fuel compensation → **combustion (fuel nozzle / combustor
  hardware in the upstream can(s))** → *borescope* (+ verify sector TCs first, review fuel
  nozzle/purge; combustion tuning check if recurrent)
- no event found → alert **not confirmed** → *monitor only* + needs_human

**combustion_dynamics** (`comb_dyn*` channels):
baseline median/MAD, rolling envelope, event = sustained elevation above alert limit.
Characterise: peak / mean amplitude vs limits, duration, envelope modulation period,
coupling (correlation with EGT spread and vibration), load/IGV/fuel movement at onset,
probe health (flatline, saturation, uncoupled spikes).
- probe fault signature → *sensor fix/replace* (dynamics probe/cable)
- coupled, above alert → **combustion instability** → *combustion tuning*; above alarm or
  rising → plus load-reduction guidance and escalation; prolonged above high limit with
  hardware indicators → plus *borescope* (liner/transition piece)
- below alert / short → *monitor only*

**hets_trip** (zones `EGT_Z*` or TC ring, `trip_active`, load):
trip detection (trip flag edge, else load collapse), pre-trip window spread build-up (rate,
duration), zone pattern at peak (opposing hot + cold = true circumferential imbalance vs one
zone moving alone = instrument), step check, concurrent vib/dynamics, post-trip restart and
spread after restart.
- single-zone / step → **instrumentation** → *sensor fix/replace* before restart
- true imbalance with gradual build → **combustion hardware / fuel nozzle** → *borescope*
  + *escalate* (engineering hold / restart only with tighter spread watch)
- no trip found → needs_human (data window may not contain the event)

**generic** (unknown analytic): anomaly scan + all signature probes; tentative action from
the best-matching probe (if any) else *escalate*; always `needs_human`, confidence capped.

Actuator/control consistency check (all analyzers): IGV vs load and fuel vs load residuals;
an abnormal residual adds *LVDT calibration* or *control adjustment* as a secondary action.

## 5. Action taxonomy

`sensor_fix_replace`, `borescope`, `lvdt_calibration`, `combustion_tuning`,
`control_adjustment`, `monitor_only`, `escalate` — each report has one **primary** action,
optional **secondary** actions, an **urgency** (`immediate`, `before_restart`,
`next_opportunity`, `routine`) and an **escalate** flag.

## 6. Confidence and uncertainty

Confidence (0–1) is an additive, itemised score: signature strength of the best hypothesis,
margin over the runner-up, data sufficiency (required channels, pre-event baseline, rows,
resolution, missing data), corroboration (anomaly scan overlap, secondary channels), feedback
from similar cases. Unknown analytic caps at 0.40. Label: High ≥ 0.75, Medium ≥ 0.50, Low.

`needs_human = true` (extra scrutiny beyond the normal review gate) when: unknown analytic,
insufficient data, analytic not confirmed by the data, data matches a different analytic,
novel anomalies in unexplained channels, conflicting hypotheses, low confidence, LLM concern,
or action changed by feedback.

## 7. Report format

`report.md` (site report) and `report.json` (same content, machine-readable):

1. Header — status banner (DRAFT / APPROVED / REJECTED), case ID, unit, analytic (entered →
   resolved), data window, generated time.
2. **Summary** (bottom line up front: event, action, urgency, confidence).
3. **Recommended corrective action** — primary, secondary/pre-checks, escalation, operating
   / restart guidance, site instructions.
4. **What happened** — narrative + event timeline table.
5. **Why — engineering assessment** — cause + alternatives considered and why rejected.
6. **Evidence** — table: finding | value | source (rule ID / model).
7. **Confidence & uncertainty** — score, factor breakdown, review flags, data gaps.
8. **Similar past cases** — feedback-informed examples and how they were used.
9. **Audit trail** — analyzer + version, thresholds, anomaly engine, LLM model/binary/params
   (or "template"), knowledge sections used, rules fired, input SHA-256, app version.
10. **Engineer review** — verdict, reason, corrected action, reviewer, time.
11. Disclaimer (decision support only).

## 8. What was removed from the earlier prototype and why

| Removed | Why |
|---|---|
| `src/download.py`, auto `pip install`, HF / GitHub downloads | On-prem constraint: the app must never fetch anything. Existing GGUF/TS Pulse/llama.cpp are referenced by path. |
| ChromaDB, sentence-transformers, `chroma_db/` | See §3 — replaced with structured similarity + BM25; removes telemetry risk and heavy deps. |
| Textual TUI (`src/tui_app.py`) and `textual`/`rich` deps | One UI only: Streamlit. |
| TS Pulse classification "dual-head scaffold" | Untrained head producing meaningless labels — misleading in an auditable report. |
| Alerts vs Trips/Event mode switch | Replaced by analytic-driven analyzers. |
| Free-form 4-section LLM output + reflection pass | Replaced by deterministic decision + JSON-constrained narrative + validation checks. |
| `llama-cpp-python` dependency | Not needed; the official llama.cpp CPU binary is used. |
| Old tests, `CLAUDE_REBUILD_BRIEF.md`, `saved_cases/`, `logs/` | Superseded by new tests and `data/`. |

Kept: knowledge process maps (RAG context), scenario CSVs + generator (test ground truth),
`gt_sensors_demo.csv` (short generic file → used to test the insufficient-data path).

## 9. Testing and validation

- pytest over **every** sample CSV (10 cold spot, 10 dynamics, 5 HETS) against the generator's
  ground truth (cold TC location, trip time, event presence) and expected actions.
- Synthetic fault injections (single-TC step, flatlined dynamics probe) → sensor action.
- Offline mode (no GGUF), unknown analytic, insufficient data, feedback round trip (Not good
  + correction changes the next report; Good raises confidence), CLI, Streamlit smoke test.
- End-to-end validation with the real GGUF on one sample per analytic → `validation/`, judged
  in `validation/VALIDATION.md`.
