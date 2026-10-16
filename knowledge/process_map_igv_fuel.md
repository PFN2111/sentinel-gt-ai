# Process Map — IGV and Fuel Control

## Decision flow
1. Compare IGV feedback with the IGV reference at the same load.
2. If feedback drifts and the reference holds, treat it as position feedback.
3. If fuel flow and load swing together, treat it as control oscillation.
4. A single noisy sample with no swing is not a control fault.

## Quick checks
- IGV position versus load across the record
- Fuel flow period, if a swing is present
- Recent actuator or valve work noted by the site
