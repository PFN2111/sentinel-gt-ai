# Tier 2 Site Report — HETS trip (high exhaust temperature spread)

> **DRAFT — pending Tier 2 engineer review.** Not for release to site until marked Good.

| Case | Unit | Analytic (entered → resolved) | Data window | Generated |
|---|---|---|---|---|
| SGT-20261012-152232-58e4 | GT 1 | HETS Trip → `hets_trip` | 2025-08-01 00:00 → 2025-09-29 23:50 (8640 rows, 10.0 min) | 2026-10-12T15:22:32 |

## 1. Summary

**Bottom line:** Trip at 2025-09-04 04:40 after spread built from 2.4 to 59.5 C over 2.3 h (gradual), opposing hot/cold zones; restarted 2025-09-04 21:10, post-restart spread 0.8 C. Recommended: Borescope inspection — prompt. Escalate to engineering. Confidence 0.90 (High).

- **Recommended action:** Borescope inspection (combustion / hot gas path hardware)
- **Urgency:** Prompt — within days, or at the next shutdown if sooner
- **Severity:** High · **Confidence:** 0.90 (High) · **Escalate:** Yes · **Needs extra human scrutiny:** No
- **Most likely cause:** Combustion hardware / fuel nozzle fault causing circumferential imbalance (hot and cold sectors) *(Combustion hardware / fuel nozzle issue)*

## 2. Recommended corrective action

**Primary:** Borescope inspection (combustion / hot gas path hardware) — Prompt — within days, or at the next shutdown if sooner.

**Rationale:** A combustion hardware/fuel nozzle fault is the most plausible cause of the circumferential imbalance. Prompt borescope inspection is required to identify and rectify the fault, given the high severity and the need to prevent recurrence before the next shutdown.

**Site instructions:**

1. Confirm the first-out and SOE is HETS/exhaust spread protection, not flameout, overspeed, or vibration.
2. Perform a loop check of the exhaust thermocouples in the hot and cold zones to ensure instrumentation integrity.
3. Map the hot/cold zones through the swirl chart to the combustor cans and conduct a borescope inspection of those cans, focusing on fuel nozzles, liners, transition pieces, and crossfire tubes.
4. Review fuel nozzle/manifold pressures and gas valve positions for the pre-trip hours to verify normal operation.
5. Monitor the exhaust temperature spread closely after restart and schedule the borescope inspection at the earliest feasible opportunity.

**Operating guidance:**

- Unit was restarted at 2025-09-04 21:10; spread after restart 0.8 C. Keep a tight spread watch and schedule the borescope at the earliest opportunity.

**Escalation:** Yes — notify the combustion / engineering SME with this report.

## 3. What happened

On 2025-09-04 at 04:40, GT1 (7FA.03) tripped due to a high exhaust temperature spread (HETS) of 59.5 °C, building gradually from 2.4 °C over 2.3 hours. The hot/cold zones were EGT_Z1_C (+31.9 °C) and EGT_Z3_C (-27.6 °C), indicating a circumferential imbalance. Load and fuel dropped at the trip. The unit was restarted at 2025-09-04 21:10, 16 hours later, with a spread of 0.8 °C.

| Time | Event |
|---|---|
| 2025-08-27 22:40 | Normal spread ~2.4 C |
| 2025-09-04 02:20 | Spread starts to build |
| 2025-09-04 04:40 | Peak pre-trip spread 59.5 C; hot EGT_Z1_C / cold EGT_Z3_C |
| 2025-09-04 04:40 | Trip (trip flag 'trip_active'); load and fuel collapse |
| 2025-09-04 21:10 | Back at load; spread 0.8 C |

## 4. Why — engineering assessment

The opposing hot and cold zones suggest a real circumferential combustion hardware/fuel nozzle fault causing the imbalance. The gradual multi-hour build-up rules out a single bad thermocouple or a transient operational event. No significant load or IGV movement preceded the rise, and vibration/dynamics spikes were present but secondary to the temperature pattern.

**Alternatives considered:**

- Exhaust thermocouple / harness fault fooling the spread protection → Sensor fix / replace (rule support 0.10). Not preferred: hot and cold zones moved in opposite directions — a single bad TC cannot do that; gradual multi-hour build-up.
- Operational transient (load swing / fuel transfer) driving the spread → Control adjustment (rule support 0.10). Not preferred: no load or IGV movement before the rise.

## 5. Evidence

| # | Finding | Value | Source |
|---|---|---|---|
| 1 | Trip | 2025-09-04 04:40 (trip flag 'trip_active'); 1 trip(s) in file | HT-R1 |
| 2 | First-out (notes) | high exhaust temperature spread (confirms exhaust spread protection) | HT-R1 |
| 3 | Spread build-up (to trip) | 2.4 C -> 59.5 C over 2.3 h (24.5 C/h) | HT-R2 |
| 4 | Rise character | gradual build-up (step if <= 20 min) | HT-R2 |
| 5 | Zone deviations at peak | EGT_Z1_C +31.9, EGT_Z2_C +1.2, EGT_Z3_C -27.6, EGT_Z4_C -1.2 C (vs median of zones, baseline offsets removed) | HT-R3 |
| 6 | Pattern | opposing hot and cold zones (true circumferential imbalance) | HT-R3 |
| 7 | Load movement before/during rise | 1.6% of normal | HT-R4 |
| 8 | Vibration | baseline 3.06, pre-trip max 3.19, at trip 6.88 mm/s (vib_DE_mm_s) | HT-R5 |
| 9 | Dynamics | baseline 0.35, pre-trip max 0.47, at trip 1.82 psi (comb_dyn_psi) | HT-R5 |
| 10 | Restart | back to >=80% load at 2025-09-04 21:10 (16.0 h after trip) | HT-R6 |
| 11 | Spread after restart | 0.8 C (first 6 h at load) | HT-R6 |
| 12 | Anomaly scan (robust z) | EGT_spread_C /z/≤45.4 150 min; load_MW /z/≤19.1 850 min; EGT_Z1_C /z/≤17.9 110 min; EGT_Z3_C /z/≤14.8 110 min | GEN-Z |
| 13 | TS Pulse reconstruction anomalies | load_MW 2025-09-03 21:10→2025-09-06 15:00; IGV_deg 2025-09-04 02:50→2025-09-06 21:40; fuel_flow_kg_s 2025-09-04 02:40→2025-09-05 05:10; EGT_spread_C 2025-09-04 01:50→2025-09-04 06:40 | AN-TSP |

## 6. Confidence and uncertainty

**Confidence 0.90 (High).** Itemised:

| Factor | Effect |
|---|---|
| rule support for 'combustion_hardware' hypothesis = 0.85 | base +0.77 |
| clear margin over the alternative (0.75) | +0.05 |
| TS Pulse independently flags the event window | +0.05 |
| robust z-score scan flags the event window | +0.03 |

## 7. Similar past cases (engineer feedback)

No reviewed similar cases yet. Feedback on this report will be used for future cases.

## 8. Audit trail

- **Analyzer:** `hets_trip/1.0` — analytic resolved by: exact analytic id.
- **Rules fired:** `HT-R2` Build-up: spread rise >= 15.0 C over baseline in the 6 h before trip; completed in <= 20 min = step (instrument); `HT-R3` Zone pattern at peak (median reference, baseline offsets removed): zones >= 8.0 C hot AND cold = true circumferential imbalance; one zone alone = instrument; `HT-R6` Restart: load back >= 80%; spread after restart > baseline + 10.0 C = fault still present (critical); `AN-TSP` TS Pulse zero-shot reconstruction error flags the event window (corroboration only, +0.05 confidence); `AN-Z` Robust z-score scan flags the event window (corroboration only, +0.03 confidence)
- **Thresholds:** `{'load_collapse_fraction': 0.5, 'pretrip_hours': 6.0, 'spread_rise_C': 15.0, 'step_rise_minutes': 20, 'zone_dev_C': 8.0, 'restart_spread_margin_C': 10.0}`
- **Anomaly engines:** robust z-score (|z| > 6.0, ≥ 30 min); TS Pulse: ok.
- **Narrative:** Granite 4.1 8B GGUF `granite-4.1-8b-Q4_K_M.gguf` via `llama-completion.exe` (llama.cpp b9947), temp 0, seed 42, JSON-schema constrained, 1646 prompt tokens, 432 generated, 167 s on CPU (8 threads). Narrative only — action and evidence come from the rules engine.
- **LLM output checks:** json: valid (schema-constrained); action named in rationale: yes; numbers traced to evidence: 11/11
- **Knowledge used:** process_map_hets_trip.md § Typical plant signatures; process_map_hets_trip.md § Decision flow; process_map_cold_spot.md § Purpose; process_map_vibration.md § Triggers
- **Input:** `hets_trip_01.csv` SHA-256 `b2dfc44556ac2a37…`, 8640 rows × 16 channels.
- **Notes interpreted:** unit = GT 1, frame = 7FA.03, first-out = high exhaust temperature spread, flag: restarted
- **Other signatures probed:** cold_spot: Cold sector EGT_Z3_C (1 TC, single), EGT_Z3_C 21.0 C below normal, gradual onset at 2025-09-04 01:50, 2.7 h, cleared; spread 2.4 -> 50.3 C.
- **App:** Sentinel GT AI 1.0.0; fully local, no network calls; elapsed 183.7 s.

## 9. Engineer review

Status: **DRAFT**. Mark Good / Not good in the UI, or `python app.py feedback --case SGT-20261012-152232-58e4 --verdict good|not_good [--reason ...] [--corrected-action ...]`.

**Notes supplied with the case:**

> GT1 (7FA.03) tripped. First-out: high exhaust temperature spread (HETS). Operations restarted the unit after a walk-down found nothing obvious. No TC maintenance in the last 6 months.

---
*Engineering decision support generated locally by Sentinel GT AI. Not an OEM-certified procedure or a protection system. Site procedures and OEM limits take precedence. Draft until reviewed by a Tier 2 engineer.*
