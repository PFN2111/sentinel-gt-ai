# Models (all paths relative to the app folder)

`install.ps1` fills this folder (download once from Hugging Face, or copy from an offline folder with
`-ModelsFrom`), and verifies SHA-256. Nothing is downloaded at runtime.

| File / folder | Size | Used for | Source |
|---|---|---|---|
| `gemma-4-12b-it-Q4_K_M.gguf` | 7.1 GB | report writer — the one LLM (without it, reports use the deterministic template) | `unsloth/gemma-4-12b-it-GGUF` |
| `tspulse/clf/` | 1.9 MB | TS Pulse r1 `tspulse-block-dualhead-512-p16-r1` — base weights the learned path was fine-tuned from | `ibm-granite/granite-timeseries-tspulse-r1` |
| `tspulse/ad/` | 4.4 MB | TS Pulse r1 `main` (`TSPulseForReconstruction`) — the residual feed of the learned path (second TS Pulse view) and anomaly corroboration in the rule path | `ibm-granite/granite-timeseries-tspulse-r1` |
| `e2e/` | ~7 MB | **the learned model** (committed to git): fine-tuned TS Pulse encoder, classifier head, similarity index, calibration, held-out set, model card | `scripts/train_e2e.py` |

The large files are git-ignored. `"gguf_path"` in `settings.local.json` points at the Gemma file if it lives
elsewhere.
