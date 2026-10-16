# Tier 2 Site Report — Combustion dynamics

> **DRAFT — pending Tier 2 engineer review.** Not for release to site until marked Good.

| Case | Unit | Analytic (entered → resolved) | Data window | Generated |
|---|---|---|---|---|
| SGT-20261012-151935-46d1 | GT 3 | Combustion Dynamics High → `combustion_dynamics` | 2026-03-03 00:00 → 2026-03-12 23:59 (14400 rows, 1.0 min) | 2026-10-12T15:19:35 |

## 1. Summary

**Bottom line:** Dynamics event on comb_dyn_psi: peak 2.76 psi, mean 1.48 psi (baseline 0.35), 12.2 h from 2026-03-05 12:02; 197 min above alarm; coupled with spread/vibration; ended. Recommended: Combustion tuning — prompt. Escalate to engineering. Confidence 0.92 (High).

- **Recommended action:** Combustion tuning (fuel splits / DLN schedule)
- **Urgency:** Prompt — within days, or at the next shutdown if sooner
- **Severity:** High · **Confidence:** 0.92 (High) · **Escalate:** Yes · **Needs extra human scrutiny:** No
- **Most likely cause:** Combustion instability — insufficient dynamics margin at this operating point (fuel splits / DLN schedule) *(Combustion instability (dynamics / tuning margin))*

## 2. Recommended corrective action

**Primary:** Combustion tuning (fuel splits / DLN schedule) — Prompt — within days, or at the next shutdown if sooner.

**Rationale:** Prompt combustion tuning (adjusting fuel splits and DLN schedule) is required to restore dynamics margin and prevent potential liner or basket damage. The urgency is high due to the prolonged duration above the alarm level and the high confidence (0.92) in the diagnosis.

**Site instructions:**

1. Pull the high-speed dynamics monitor spectrum for the event to identify the dominant tone (in Hz) and the affected cans.
2. Review fuel splits, fuel gas temperature, and composition (Wobbe index) at the onset of the event against the tuning baseline.
3. Confirm the dynamics probe health by comparing redundant probes or cans before proceeding with tuning.
4. If dynamics exceed the 2.0 psi alarm level again, reduce load or change mode per site procedure and notify the combustion SME immediately.
5. Avoid fast load ramps and fuel transfers in this load band until the combustion is tuned.

**Operating guidance:**

- Dynamics exceeded the 2.0 psi alarm level for 197 min: if it recurs, reduce load / change mode per site procedure and notify the combustion SME.
- Avoid fast load ramps and fuel transfers in this load band until tuned.

**Escalation:** Yes — notify the combustion / engineering SME with this report.

## 3. What happened

On 2026-03-05 at 12:02, combustion dynamics (comb_dyn_psi) exceeded the alert limit, peaking at 2.76 psi with a mean of 1.48 psi, deviating from the baseline of 0.35 psi. The event lasted 12.2 hours, with 197 minutes above the 2.0 psi alarm threshold. The dynamics were coupled with increased exhaust spread (+1.0°C) and vibration (+0.28 mm/s). A minor load change (1.4%) occurred an hour before the onset. The dynamics probe health was deemed plausible, and the event decayed by 2026-03-06 00:16.

| Time | Event |
|---|---|
| 2026-03-05 12:02 | Dynamics rise above baseline and cross the alert limit |
| 2026-03-05 18:00 | Peak envelope 1.83 psi (raw 99th pct 2.76 psi) |
| 2026-03-06 00:16 | Dynamics return below alert |

## 4. Why — engineering assessment

The high combustion dynamics indicate insufficient dynamics margin at the current operating point, likely due to fuel splits or DLN schedule tuning issues. The strong correlation (r = 0.91) between dynamics and vibration confirms real pressure oscillations in the flame. The alternative hypothesis of a dynamics probe fault was rejected because the amplitude was mirrored by exhaust spread and vibration, indicating genuine combustion pulsing rather than sensor error.

**Alternatives considered:**

- Dynamics probe / cable fault on comb_dyn_psi → Sensor fix / replace (rule support 0.10). Not preferred: amplitude is mirrored by exhaust spread / vibration — the combustion is really pulsing.

## 5. Evidence

| # | Finding | Value | Source |
|---|---|---|---|
| 1 | Dynamics baseline | 0.35 psi (robust sigma 0.053) on comb_dyn_psi | CD-R1 |
| 2 | Event window | 2026-03-05 12:02 -> 2026-03-06 00:16 (12.2 h) | CD-R2 |
| 3 | Peak / mean amplitude | 2.76 / 1.48 psi (baseline 0.35) (99th pct / mean of raw samples) | CD-R2 |
| 4 | Time above limits | alert 1.0 psi: 511 min; alarm 2.0 psi: 197 min; high 3.0 psi: 0 min (raw historian samples above each limit) | CD-R3 |
| 5 | Envelope modulation period | ~3 min (amplitude beating, not the acoustic tone) | CD-R4 |
| 6 | Coupling | r(dyn, spread) = 0.59; r(dyn, vibration) = 0.91 (coupled if r >= 0.4) | CD-R5 |
| 7 | Exhaust spread during event | +1.0 C vs normal | CD-R5 |
| 8 | Vibration during event | +0.28 mm/s (vib_DE_mm_s) | CD-R5 |
| 9 | Load movement in hour before onset | 1.4% of normal load | CD-R6 |
| 10 | Trend inside event | decayed / ended | CD-R2 |
| 11 | comb_dyn_psi probe health | plausible, varies with process | CD-R7 |
| 12 | Anomaly scan (robust z) | comb_dyn_psi /z/≤23.6 730 min | GEN-Z |
| 13 | TS Pulse reconstruction anomalies | comb_dyn_psi 2026-03-05 11:53→2026-03-06 01:44 | AN-TSP |

## 6. Confidence and uncertainty

**Confidence 0.92 (High).** Itemised:

| Factor | Effect |
|---|---|
| rule support for 'combustion_instability' hypothesis = 0.89 | base +0.79 |
| clear margin over the alternative (0.79) | +0.05 |
| TS Pulse independently flags the event window | +0.05 |
| robust z-score scan flags the event window | +0.03 |

**Data gaps / limitations:**

- Historian amplitude only: the acoustic tone frequency (Hz) needs the high-speed dynamics monitor spectrum.

## 7. Similar past cases (engineer feedback)

No reviewed similar cases yet. Feedback on this report will be used for future cases.

## 8. Audit trail

- **Analyzer:** `combustion_dynamics/1.0` — analytic resolved by: matched dynamics keywords.
- **Rules fired:** `CD-R2` Event: 10 min envelope > baseline + 6 sigma, with >= 30 min of samples above the 1.0 psi alert; `CD-R3` Limits: alert 1.0 / alarm 2.0 / high 3.0 psi; >= 30 min above alarm = high severity + SME notification; >= 6.0 h above high = hardware risk (borescope); `CD-R5` Coupling: r >= 0.4 with exhaust spread or vibration = real combustion pulsation; uncoupled = probe suspicion; `AN-TSP` TS Pulse zero-shot reconstruction error flags the event window (corroboration only, +0.05 confidence); `AN-Z` Robust z-score scan flags the event window (corroboration only, +0.03 confidence)
- **Thresholds:** `{'alert_psi': 1.0, 'alarm_psi': 2.0, 'high_psi': 3.0, 'min_event_minutes': 30, 'envelope_minutes': 10, 'coupling_r': 0.4, 'hardware_risk_hours': 6.0}`
- **Anomaly engines:** robust z-score (|z| > 6.0, ≥ 30 min); TS Pulse: ok.
- **Narrative:** Granite 4.1 8B GGUF `granite-4.1-8b-Q4_K_M.gguf` via `llama-completion.exe` (llama.cpp b9947), temp 0, seed 42, JSON-schema constrained, 1471 prompt tokens, 438 generated, 159 s on CPU (8 threads). Narrative only — action and evidence come from the rules engine.
- **LLM output checks:** json: valid (schema-constrained); action named in rationale: yes; numbers traced to evidence: 13/13
- **Knowledge used:** process_map_combustion_dynamics.md § Typical plant signatures; process_map_combustion_dynamics.md § Decision flow; gt_overview.md § Trips/Event focus; process_map_cold_spot.md § Decision flow
- **Input:** `comb_dyn_05.csv` SHA-256 `83b16e05215b1580…`, 14400 rows × 16 channels.
- **Notes interpreted:** unit = GT 3, frame = 7FA.04, flag: load_change, flag: operator_reports_noise
- **App:** Sentinel GT AI 1.0.0; fully local, no network calls; elapsed 181.9 s.

## 9. Engineer review

Status: **DRAFT**. Mark Good / Not good in the UI, or `python app.py feedback --case SGT-20261012-151935-46d1 --verdict good|not_good [--reason ...] [--corrected-action ...]`.

**Notes supplied with the case:**

> GT3 (7FA.04) at baseload. Combustion Dynamics High analytic fired overnight. Operators reported a humming tone from the combustion compartment. No load changes or fuel transfers reported.

---
*Engineering decision support generated locally by Sentinel GT AI. Not an OEM-certified procedure or a protection system. Site procedures and OEM limits take precedence. Draft until reviewed by a Tier 2 engineer.*
