# Architecture

Sentinel GT AI is one local program. `sentinel_gt_ai.py` holds every section, with a table of contents and the engineering comments. The `sentinel/` package is a set of import hooks: importing `sentinel.profiles` compiles that section out of the single file. Edit the section in `sentinel_gt_ai.py`. Data stays outside the file: `models/`, `knowledge/`, `data/`, `settings.local.json`.

Nothing in the request path contacts the network. `setup` and `install.ps1` are the only downloaders.

## Request path

```
historian CSV
  + alert name
  + notes
  + frame (dropdown or an id you added)
        |
        v
tag profile for that alert type     sentinel.profiles
frame tag map                       sentinel.data / sentinel.config
        |
        v
canonical window (512 points)       sentinel.e2e.features
TS Pulse encoder                    sentinel.e2e.encoder
head + similarity index + novelty   sentinel.e2e.model
residual cross-check                sentinel.e2e.residual
rule shadow                         sentinel.analytics.*  sentinel.rules
process-map retrieval               sentinel.knowledge
        |
        v
decision + draft report             sentinel.e2e.engine / pipeline / report
language-model narrative            sentinel.e2e.writer  sentinel.llm_runtime
        |
        v
case folder under data/cases        sentinel.store
engineer label                      sentinel.e2e.labels / sentinel.e2e.teach
retrain, gate, rollback             sentinel.e2e.train / train_job
```

`sentinel.e2e.pipeline` is the learned path: features, decide, rules in shadow, report, save. `sentinel.pipeline` is the rule path used by the single-case page. `sentinel.triage` turns rule findings into one action, an urgency, and the reasons a person must look.

## Two learned modes

`e2e_mode` in settings:

- **`legacy`** (the default). One encoder and one head on a fixed 14-channel window. This is the bundle under `models/e2e/legacy` and the active line the UI uses until you change the setting.
- **`peralert`**. The same encoder family, but the head, the similarity index, and the novelty calibration are keyed by alert type. Channel order is the slot order in that type's profile, not the fixed 14. Set `e2e_mode` to `peralert` or export `SENTINEL_E2E_MODE=peralert`.

A stored value of `profiles` is read as `legacy`. The default stays `legacy` because the synthetic acceptance gate still fails two truth-signature slices, even though the overall synthetic score is higher. See the README for the numbers.

## Alert types

An alert type is a schema-2 JSON profile (`sentinel.profiles`).

- **Id and name**, plus aliases. Routing is exact: the alert name matches the canonical name, the id, or an alias. Keywords do not select a type.
- **Slots.** Each slot is a logical tag (context, process, flag, ring, bank, ignore) with a role, a unit, and alias patterns. Every slot is expected. Coverage is part of the confidence cap.
- **Signatures** the type is allowed to name, and the residual pattern each signature should show.
- **Analyzer** key: `cold_spot`, `combustion_dynamics`, `hets_trip`, or `generic`. A type you add uses `generic` unless you point it at one of the built-in analyzers.
- **Thresholds** for the novelty quantile and the neighbor vote, inside allowed ranges.

Seven types are built in: `exhaust_cold_spot`, `combustion_dynamics`, `exhaust_spread_trip`, `igv_fuel_control`, `bearing_vibration`, `inlet_filter_dp`, `lube_oil_system`. A file with the same id under `profiles_dir` (default `data/e2e/profiles`) replaces the built-in. A bad file is skipped and reported by `alert-types check`.

`alert-types new` drafts a profile from a CSV header (`draft_alert_type`). The new type is cold until it has labels. There is no cap on how many site files you add, beyond the id rules (lowercase, 2 to 48 characters).

`sentinel.e2e.migrate` can assign stored labels and cases onto these profiles. It runs when `e2e_mode` is `peralert`. The legacy label log is not rewritten.

## Frames

Frame models are 6B, 7EA, 7FA, 7HA, 9E, 9FA, and 9HA, plus Other / custom. A frame points at a tag layout: exhaust ring or zones, and the process tags (load, fuel, IGV, and the rest). `apply_frame` maps the file's columns onto the canonical channels. Ring and zone tags sort by the full sensor number.

Frames you add are teach records. They show up in the same dropdown as the shipped list and come back if you roll that teach version forward.

## Models on disk

| Path | Role | In git? |
|---|---|---|
| `models/e2e/` | Fine-tuned encoder, legacy head, per-alert heads, indexes, held-out arrays, model card | Yes (small) |
| `models/tspulse/clf/` | TS Pulse block the encoder was fine-tuned from | No, install downloads it |
| `models/tspulse/ad/` | TS Pulse reconstruction for the residual view and the rule-path anomaly scan | No, install downloads it |
| `models/gemma-4-12b-it-Q4_K_M.gguf` | Report writer | No, install downloads it |

`sentinel.llm_runtime` loads Gemma through `llama-cpp-python`. GPU offload when that build loads, otherwise CPU, otherwise the template in `sentinel.report`. `residual_mode=auto` uses TS Pulse reconstruction when the anomaly snapshot loads, and a robust-z residual otherwise.

Device selection (`auto`, `gpu`, `cpu`) is one setting for the language model and for `train --full`. The encoder fine-tune retries on CPU when the GPU runs out of memory.

## Teach, retrain, rollback

`sentinel.e2e.teach` versions labeled cases, documents, signatures, frames, and alert types. Every change appends a teach version. Rollback restores that snapshot and rebuilds the similarity index.

`sentinel.e2e.labels` is the label log and the model-version folder. `v0000` is the shipped head.

`sentinel.e2e.train` refits a head, rebuilds the index, recalibrates novelty, and scores the held-out set. Promotion requires action and signature accuracy not worse than the active version, and no frame or alert slice with enough rows to have dropped. `--force` promotes anyway and records the reason. `train --quick --alert-type ID` refits that type only. `train --full` fine-tunes the shared encoder, then runs the quick steps.

`sentinel.e2e.train_job` runs that work in a subprocess so the dashboard does not kill it. Progress is a file under `data/e2e/jobs`.

## Knowledge and the narrative

`knowledge/*.md` holds the shipped process maps. Documents you add are `knowledge/user/*.md` with a tag header (alert types, signatures). Retrieval prefers a document tagged for this alert type or signature, then a lexical search. Another alert type's map is left out. The learned path asks the language model to walk those steps against the residual and the classifier, and to cross-check them. The model does not invent the technician action. The action comes from the decision layer.

## UI and CLI

`sentinel.ui` is the NiceGUI dashboard: Analyze, Teach, Models, and Settings. Importing the module does not draw the app and does not register routes. `serve()` starts the server on 127.0.0.1. Vue, Quasar, and ECharts are served from the installed package.

`sentinel.cli` is `run`, `doctor`, `train`, `setup`, `alert-types`, `e2e`, and the UI launcher. `sentinel.bootstrap` is `setup`: venv, pinned installs, model download or `--models-from`, starter knowledge.

## What is stored where

| Location | Contents | Git |
|---|---|---|
| `data/cases/<id>/` | Input CSV, report, prompt, raw model output | Ignored |
| `data/feedback.jsonl` | Review log | Ignored |
| `data/e2e/` | Labels, teach versions, jobs, site profiles, retrained heads | Ignored |
| `knowledge/user/` | Documents added from Teach | Ignored |
| `settings.local.json` | Device and relative model paths | Ignored |

A new clone has none of those. The synthetic sample CSVs in `samples/` are the only historian files in git.
