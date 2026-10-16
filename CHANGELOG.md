# Changelog

## Unreleased

Public update prepared from the per-alert work, with a privacy pass and install docs. Not pushed.

- The local dashboard is NiceGUI. Header tabs: Analyze, Teach, Models, Settings. The theme is near-black with a mint accent. Streamlit is no longer a dependency.
- Analyze is a three-column dashboard: case input and pipeline, signature / confidence / novelty plus tag charts, and the short site report with nearest cases and a Good / Not good / Edit review gate. The full document opens from Full report.
- The server binds to 127.0.0.1 port 8501. Vue, Quasar, and ECharts are served from the installed package.
- Optional desktop window: `python sentinel_gt_ai.py --native` after `python -m pip install pywebview`.
- Analyze shows the upload, alert type, frame, notes, tag plots with a mint residual window, signature, confidence, novelty, nearest past cases, and a printable site report.
- Teach edits alert types, frames, and labels. Retrain and rollback stay on Models, including the not-worse promote gate.
- Settings changes the CPU / GPU device only when Save device is pressed, and runs the doctor check on demand.
- Per-alert dynamic tag profiles. Each alert type has its own tag list, head, similarity index, and novelty calibration.
- Alert types are unlimited. Seven ship with the app. Add one from a sample CSV header (`alert-types new`) or from the Teach page.
- Frame dropdown: 6B, 7EA, 7FA, 7HA, 9E, 9FA, 9HA, and Other / custom. A frame added on Teach joins that list.
- Teach page: labeled cases with a why, process documents, signatures, and alert types. Retrain and model rollback are on Models.
- Retrain promotes a version only when held-out action and signature accuracy are not worse, and no large frame or alert slice got worse. Force promote is logged. Rollback is one click.
- CPU fallback when the GPU build is missing or runs out of memory. Review and inference stay on this computer.
- Docs for a fresh Windows install with an empty case library: `docs/INSTALL.md`, `docs/USER_GUIDE.md`, `docs/ARCHITECTURE.md`.
- Privacy pass on the tree this update is built from. See `PRIVACY_CHECK.md`.

Synthetic held-out result recorded with this work: per-alert signature accuracy 95.3% (305/320) versus 90.0% (288/320) for the legacy head, false novelty 0.6% (2/320). The rows are synthetic. The default `e2e_mode` remains `legacy` because two truth-signature slices were slightly lower.

## Earlier public history

These commits are already on the public repository. This update does not rewrite them.

- Modern UI, real frame models, review fixes.
- First publish of the app as one local file.
