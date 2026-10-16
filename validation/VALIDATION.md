# Sentinel GT AI — End-to-end validation

Date: 2026-10-12. Machine: Windows 11, 16 logical CPUs, 15.7 GB RAM, CPU only.

**Method.** Each report was produced by the CLI in normal mode with the real model:
Granite 4.1 8B `granite-4.1-8b-Q4_K_M.gguf` via local `llama-completion.exe` (llama.cpp b9947,
temp 0, seed 42, JSON-schema constrained). TS Pulse was enabled, and so were Tier 1-style
notes. I judged each report the way a Tier 2 engineer would, against the scenario each sample
was generated from (`samples/SCENARIO_README.md`, ground truth in
`scripts/generate_scenario_csvs.py`) and the site process maps in `knowledge/`.

```
python app.py run --csv samples/<analytic>/<file>.csv --analytic "<name>" --notes "<notes>" --out validation
```

| Final report | Analytic entered | Scenario (ground truth) |
|---|---|---|
| [cold_spot_01_report.md](cold_spot_01_report.md) | `Exhaust Cold Spot` | cold sector centred on **TC1**, 3 TCs wide (wraps the ring: TC27-TC1-TC2), depth 22 C, 6 h ramp, about 6 days, then clears; small fuel rise |
| [combustion_dynamics_05_report.md](combustion_dynamics_05_report.md) | `Combustion Dynamics High` | 2.05 psi-class pulsation burst starting 2026-03-05 12:00, 3 min amplitude beat, EGT flutter + vibration coupling |
| [hets_trip_01_report.md](hets_trip_01_report.md) | `HETS Trip` | spread builds about 3 h (Z1 hot, Z3 cold), trip 2025-09-04 04:40, load collapse, restart ramp over about 24 h |

## Verdict summary (final round 4)

| Sample | Primary action | Urgency / severity / escalate | Confidence | Narrative checks | Tier 2 verdict |
|---|---|---|---|---|---|
| cold_spot_01 | **Borescope** (cans upstream of TC27/TC1/TC2) | next opportunity / medium / no | 0.90 High | JSON ok, action named, 18/18 numbers traced | **Correct, sensible** |
| combustion_dynamics_05 | **Combustion tuning** | prompt / high / **yes** | 0.92 High | JSON ok, action named, 14/14 numbers traced | **Correct, sensible** |
| hets_trip_01 | **Borescope** | prompt / high / **yes** | 0.90 High | JSON ok, action named, 10/11 numbers traced ("90%" = 0.90) | **Correct, sensible** (one wording edit) |

All three recommend the action the scenario calls for. None recommends the classic wrong call:
replacing a thermocouple for a real combustion sector, or tuning on a suspect probe. Each one
gives an explicit, evidence-based reason for rejecting the instrument hypothesis.

## Per-report judgement

### 1. Cold spot — `cold_spot_01_report.md`

**What it found:** a cold sector at TC1, TC2, TC27, correctly treated as one adjacent arc across
the ring wrap-around. TC1 was 22.6 C below normal (truth: 22 C). Onset was gradual, 290 min from
10% to 90% depth (truth: 6 h linear ramp). The event lasted 6.0 days and cleared before the end
of the data (truth: 864 samples = 6.0 days). Spread went from 5.7 to 26.7 C, and fuel flow ran
+0.70% above the load model. TS Pulse and the z-score scan both flag the same window.

**Action check (process map: "true cold sector / nozzle imbalance → site combustion procedure";
"sensor-only → TC verification"):**
- Primary is **borescope of the cans upstream of the sector**, after verifying the sector TCs in
  the controller and mapping through the swirl chart, plus a check of fuel-nozzle passages,
  purge and manifold pressures. That is the right call for three adjacent TCs cooling together
  on a gradual onset with fuel compensation.
- Urgency is next planned opportunity, because the event self-cleared, with a recurrence watch
  that brings the inspection forward. A Tier 2 engineer would choose the same. Not escalated is
  reasonable at medium severity with spread well below the 40 C precursor level.
- The alternative (TC/harness fault, rule support 0.15) is rejected for the right reason: one
  failed TC cannot cool three neighbours.

**Narrative:** accurate and grounded. In round 2 it had said "gradual onset over 6 days", which
was fixed (see history).

### 2. Combustion dynamics — `combustion_dynamics_05_report.md`

**What it found:** onset 2026-03-05 12:02 (truth 12:00). Peak 2.76 psi, mean 1.48 psi against
a 0.35 psi baseline. 12.2 h above baseline, 197 min above the 2.0 psi alarm, 0 min above 3.0 psi.
Amplitude beat about 3 min (truth: 3 min). Coupled with vibration (r = 0.91) and spread
(r = 0.59). Probe health plausible. Load steady before onset (1.4%). The operator humming
report was used as supporting evidence.

**Action check (process map: "above limit or rising → load reduction per procedure; notify
combustion SME"):**
- Primary is **combustion tuning** (fuel splits / DLN schedule), urgency prompt, **escalate to
  the combustion SME**. Guidance: reduce load or change mode if it exceeds 2.0 psi again, and
  avoid fast ramps and fuel transfers in this band.
- Site checks are the ones a Tier 2 engineer asks for before tuning: pull the high-speed
  spectrum (tone and leading cans), review fuel splits, fuel gas temperature and Wobbe, and
  confirm probe health against redundant probes.
- Borescope is correctly *not* recommended. The time above 3 psi is 0 min, so there is no
  sustained hardware-risk exposure.
- The report honestly says that 1-min historian amplitude cannot resolve the acoustic tone.

### 3. HETS trip — `hets_trip_01_report.md`

**What it found:** the trip at 2025-09-04 04:40 (exact). Spread built from 2.4 C to 59.5 C over
2.3 h (24.5 C/h), gradual, not a step. At peak, Z1 was +31.9 C and Z3 −27.6 C, opposing zones and
therefore a true circumferential imbalance. Load was steady before the rise. Vibration and
dynamics rose only at the trip, so they are a consequence of the trip, not a precursor. The unit
was back at load 16 h later with spread 0.8 C. The first-out (HETS) was taken from the notes and
confirmed.

**Action check (process map: "after hardware suspicion: engineering hold until inspection plan
agreed"; "after sensor proof: monitored restart"):**
- Primary is **borescope** of the cans mapped from the hot and cold zones (fuel nozzles, liners,
  transition pieces, crossfire tubes), **escalated**, urgency prompt. Before that: confirm the
  first-out/SOE and loop-check the TCs in the hot and cold zones. Both thermocouple and
  operational-transient alternatives are rejected with the right reasons.
- The unit had already been restarted without inspection. The report says so explicitly and asks
  the combustion SME to confirm continued operation is acceptable until the borescope, and to
  set a removal-from-service spread limit. That is exactly the Tier 2 point to make here.
- **One edit a reviewer would make:** the LLM rationale says "necessitate immediate action",
  while the decided urgency is *prompt (within days)*. The structured urgency field is correct.
  The prose word is too strong. This is caught at the DRAFT review gate, and is listed as a next
  step for an automatic check.

## Validation history (fix → rerun)

| Round | Result | Issues found by Tier 2 review | Fix |
|---|---|---|---|
| 1 ([round1/](round1/)) | 3/3 actions correct | "No TC or harness work" read as TC work (raised the sensor hypothesis); unrelated process-map sections in the LLM context; unit shown as "GT 3"; HETS lacked an SME decision on running after an uninspected restart | Negation-aware notes parser; deterministic per-analytic knowledge; unit naming; HETS guidance |
| 2 ([round2/](round2/)) | 3/3 correct, all sensible | Cold spot narrative: "gradual onset over 6 days" (6 days was the event duration) | Prompt rule: keep durations attached to what they measure; ramp stated in hours |
| 3 ([round3/](round3/)) | 3/3 correct; cold spot narrative fixed | HETS: Granite wrote "No concerns…" in the concern field, which counted as a concern (false review flag); one garbled site instruction | Schema: boolean `supports_action`; null-concern normalisation; site instructions kept deterministic |
| 4 (final, this folder) | **3/3 correct and sensible** | HETS rationale says "immediate" against urgency "prompt" (wording only) | Left to the review gate; noted as next step |

## Feedback loop on the validated case

On a copy of the validation store, the cold spot report was marked **Not good** with a
corrected action and reason:

```
python app.py feedback --case SGT-20261012-155038-56a4 --verdict not_good --corrected-action sensor_fix_replace \
  --reason "Borescope at last outage found the cans clean; TC27/TC1/TC2 share a harness run that was found chafed - harness replaced"
→ FEEDBACK  not_good  STATUS REJECTED  FINAL_ACTION sensor_fix_replace
python app.py run --csv samples/cold_spot/cold_spot_01.csv --analytic "Exhaust Cold Spot" --no-llm
→ ACTION sensor_fix_replace   SECONDARY borescope   NEEDS_HUMAN True
  ⚠ Primary action changed from borescope to sensor_fix_replace based on engineer feedback on similar
    case SGT-20261012-155038-56a4 (similarity 1.00) — confirm it applies here.
```

The verdict is appended to `data/feedback.jsonl`. The correction and its reason also appear in
the next LLM prompt as an example. Dissimilar cases, and cases of other analytics, are not
affected; pytest covers both.

## Automated test evidence

`python -m pytest`: **87 passed**, plus 1 opt-in real-Granite test. That test was also run here
(`SENTINEL_RUN_LLM_TESTS=1`) and **passed in 148 s**. Coverage:
- All 25 sample CSVs checked against generator ground truth: cold TC, sector width, adjacency,
  depth, onset; dynamics onset, modulation period, coupling; HETS trip time, hot/cold zones,
  build-up.
- Injected instrument faults route to `sensor_fix_replace`: a single-TC step, a frozen TC, a
  saturated dynamics probe, and a single-zone step before a trip.
- Offline/no-GGUF template mode; unknown analytic (generic, needs_human, confidence ≤ 0.40);
  short data; missing channels (escalate); wrong analytic for the data (cross-check); novel
  lube-oil anomaly flagged; notes history (recent tuning) escalates.
- Feedback: correction overrides, approval confirms, rejection lowers confidence, scoped by
  analytic and similarity, preference export.
- CLI subprocess round trip; headless Streamlit test of the review gate.

## Performance

About 15 s per report without the LLM (CSV + TS Pulse + rules). About 2.5–3.3 min with Granite
on CPU: model load ~15 s, prompt about 1.4–1.7k tokens at about 25 tok/s, about 300–450 output
tokens at about 5 tok/s.

## Limits of this validation

- The samples are synthetic and clean. Each scenario has one fault, little noise and no real
  sensor dropouts, so confidence scores (0.90–0.92) are higher than real plant data will
  usually produce. Thresholds in `sentinel/limits.py` are generic and need site/frame values.
- Swirl mapping from TC to combustor can is frame-specific, so the report asks site to apply the
  unit swirl chart rather than naming cans.
- Only one LLM sample per analytic was judged by hand. The other 22 samples were validated on
  the deterministic decision layer (pytest).
