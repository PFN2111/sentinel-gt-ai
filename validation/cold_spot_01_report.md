# Tier 2 Site Report — Exhaust cold spot

> **DRAFT — pending Tier 2 engineer review.** Not for release to site until marked Good.

| Case | Unit | Analytic (entered → resolved) | Data window | Generated |
|---|---|---|---|---|
| SGT-20261012-155038-56a4 | Unit 2 | Exhaust Cold Spot → `cold_spot` | 2025-11-01 00:00 → 2025-12-30 23:50 (8640 rows, 10.0 min) | 2026-10-12T15:50:38 |

## 1. Summary

**Bottom line:** Cold sector TC1, TC2, TC27 (3 TCs, adjacent), TC1 22.6 C below normal, gradual onset at 2025-11-23 19:50, 6.0 days, cleared; spread 5.7 -> 26.7 C. Recommended: Borescope inspection — next planned outage / shutdown opportunity. Confidence 0.90 (High).

- **Recommended action:** Borescope inspection (combustion / hot gas path hardware)
- **Urgency:** Next planned outage / shutdown opportunity
- **Severity:** Medium · **Confidence:** 0.90 (High) · **Escalate:** No · **Needs extra human scrutiny:** No
- **Most likely cause:** True cold sector from the combustor can(s) upstream of TC1, TC2, TC27 (fuel nozzle restriction / combustion hardware) *(Combustion hardware / fuel nozzle issue)*

## 2. Recommended corrective action

**Primary:** Borescope inspection (combustion / hot gas path hardware) — Next planned outage / shutdown opportunity.

**Rationale:** A borescope inspection of the combustion/hot gas path hardware, focusing on the fuel nozzles and transition pieces upstream of the affected TCs, is recommended. This action is warranted by the high confidence (0.90) in the combustion hardware cause and the medium severity of the event. The inspection should be scheduled during the next planned outage or shutdown opportunity to address the potential fuel nozzle restriction promptly.

**Site instructions:**

1. Verify TC1, TC2, TC27 readings in the controller (and spare/redundant TCs if fitted) before opening the unit.
2. Apply the unit swirl chart to map TC1, TC2, TC27 back to the upstream combustor can(s); inspect those fuel nozzles, liners and transition pieces.
3. Check fuel-nozzle passages, purge and manifold pressures for that can (restriction / coking).

**Operating guidance:**

- Event cleared on its own — trend spread and the sector TCs for recurrence; recurrence brings the inspection forward.

**Escalation:** Not required on current evidence.

## 3. What happened

On 2025-11-23 19:50, an exhaust cold sector involving TC1, TC2, and TC27 began, with TC1 dropping 22.6°C below normal. The sector peaked on 2025-11-24 17:20 with a spread of ~27°C. The event lasted 6.0 days, ending on 2025-11-29 19:00 when the sector returned to normal. Fuel flow increased by +0.70% during the event, while load, vibration, and combustion dynamics remained unchanged.

| Time | Event |
|---|---|
| 2025-11-23 19:50 | TC1 begins to run cold; sector TC1, TC2, TC27 follows |
| 2025-11-24 17:20 | Deepest point: TC1 22.6 C below normal, spread ~27 C |
| 2025-11-29 19:00 | Sector back to normal (event duration 6.0 days) |

## 4. Why — engineering assessment

The simultaneous cooling of three adjacent thermocouples indicates a real cold gas sector originating from the combustor upstream of TC1, TC2, and TC27, likely due to a fuel nozzle restriction or combustion hardware issue. The gradual onset over hours, consistent with progressive fuel-nozzle restriction, and the slight increase in fuel flow (compensating for reduced heat release) support this cause. The alternative hypothesis of a thermocouple/harness fault was rejected because a single failed TC cannot affect three adjacent probes, and the gradual onset contradicts a step change expected from sensor issues.

**Alternatives considered:**

- Thermocouple / harness fault on TC1 → Sensor fix / replace (rule support 0.15). Not preferred: 3 TCs are cold together — one failed TC cannot do that; gradual onset over hours, not a step.

## 5. Evidence

| # | Finding | Value | Source |
|---|---|---|---|
| 1 | Cold sector | TC1, TC2, TC27 (TC ring, 3 of 27) (TCs >= 30% of primary depth) | CS-R2 |
| 2 | Primary cold TC / depth | TC1, 22.6 C below its normal (90th pct over event) | CS-R2 |
| 3 | Sector contiguity | adjacent arc | CS-R3 |
| 4 | Onset | gradual onset ramp: 290 min (4.8 h) from 10% to 90% depth; max single-sample jump 2.2 C | CS-R4 |
| 5 | Event window | 2025-11-23 19:50 -> 2025-11-29 19:00 (6.0 days) | CS-R1 |
| 6 | Status at end of data | cleared before end of data | CS-R1 |
| 7 | Exhaust spread | baseline 5.7 C -> 26.7 C during event (95th pct) | CS-R5 |
| 8 | Fuel flow vs load model | +0.70% during event (positive = control compensating for lost heat) | CS-R6 |
| 9 | Load change during event | +0.00 MW | CS-R6 |
| 10 | Vibration change during event | +0.00 mm/s (vib_DE_mm_s) | CS-R6 |
| 11 | Combustion dynamics change | +0.00 psi (comb_dyn_psi) | CS-R6 |
| 12 | TC1 signal health | normal noise, plausible values | CS-R7 |
| 13 | Anomaly scan (robust z) | EGT_spread_C /z/≤24.4 8520 min; TC1 /z/≤15.0 8460 min; TC2 /z/≤11.2 8410 min; TC27 /z/≤10.9 8410 min | GEN-Z |
| 14 | TS Pulse reconstruction anomalies | EGT_spread_C 2025-11-24 01:00→2025-11-29 21:20 | AN-TSP |

## 6. Confidence and uncertainty

**Confidence 0.90 (High).** Itemised:

| Factor | Effect |
|---|---|
| rule support for 'combustion_hardware' hypothesis = 0.85 | base +0.77 |
| clear margin over the alternative (0.70) | +0.05 |
| TS Pulse independently flags the event window | +0.05 |
| robust z-score scan flags the event window | +0.03 |

## 7. Similar past cases (engineer feedback)

No reviewed similar cases yet. Feedback on this report will be used for future cases.

## 8. Audit trail

- **Analyzer:** `cold_spot/1.0` — analytic resolved by: matched cold spot keywords.
- **Rules fired:** `CS-R2` Sector: TCs colder than max(30% of primary depth, 4.0 C) belong to the cold sector; `CS-R3` Contiguity: >= 2 adjacent TCs (ring wrap-around) = real cold gas sector; a single or scattered TC = instrument suspicion; `CS-R4` Onset: 10-90% depth in <= 20 min or one jump >= 60% of depth = step (instrument); slower = gradual (combustion); `CS-R6` Coupling: fuel flow above the load model by >= 0.4%, load loss, vibration rise support a real heat-release deficit; `AN-TSP` TS Pulse zero-shot reconstruction error flags the event window (corroboration only, +0.05 confidence); `AN-Z` Robust z-score scan flags the event window (corroboration only, +0.03 confidence)
- **Thresholds:** `{'cold_excess_C': 8.0, 'min_event_minutes': 60, 'sector_member_fraction': 0.3, 'step_onset_minutes': 20, 'high_depth_C': 25.0, 'spread_alarm_C': 40.0, 'fuel_comp_fraction': 0.004}`
- **Anomaly engines:** robust z-score (|z| > 6.0, ≥ 30 min); TS Pulse: ok.
- **Narrative:** Granite 4.1 8B GGUF `granite-4.1-8b-Q4_K_M.gguf` via `llama-completion.exe` (llama.cpp b9947), temp 0, seed 42, JSON-schema constrained, 1554 prompt tokens, 351 generated, 146 s on CPU (8 threads). Narrative only — action and evidence come from the rules engine.
- **LLM output checks:** json: valid (schema-constrained); action named in rationale: yes; numbers traced to evidence: 18/18
- **Knowledge used:** process_map_cold_spot.md § Typical plant signatures; process_map_cold_spot.md § Decision flow; process_map_exhaust_spread.md § Decision flow
- **Input:** `cold_spot_01.csv` SHA-256 `84cb25eb714ef60a…`, 8640 rows × 39 channels.
- **Notes interpreted:** unit = Unit 2, frame = 7FA.03, flag: outage
- **App:** Sentinel GT AI 1.0.0; fully local, no network calls; elapsed 160.9 s.

## 9. Engineer review

Status: **DRAFT**. Mark Good / Not good in the UI, or `python app.py feedback --case SGT-20261012-155038-56a4 --verdict good|not_good [--reason ...] [--corrected-action ...]`.

**Notes supplied with the case:**

> Unit 2 (7FA.03) at baseload. Tier 1 opened the case from the Exhaust Cold Spot analytic (spread advisory). No alarms other than the spread advisory. Last combustion inspection March 2025. No TC or harness work since that outage.

---
*Engineering decision support generated locally by Sentinel GT AI. Not an OEM-certified procedure or a protection system. Site procedures and OEM limits take precedence. Draft until reviewed by a Tier 2 engineer.*
