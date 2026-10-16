# User guide

Start the app from the repository folder:

```powershell
.venv\Scripts\python sentinel_gt_ai.py
```

The dashboard is a local NiceGUI app. The address is http://127.0.0.1:8501. Nothing else on the network can open it. `--port 8502` uses another port. `--headless` starts the server without opening a browser. `--native` opens a desktop window. That window needs the optional package:

```powershell
python -m pip install pywebview
python sentinel_gt_ai.py --native
```

Vue, Quasar, and the chart library are files inside the installed NiceGUI package. The dashboard does not load them from the internet. You can unplug the network after `setup` has finished and the pages still draw. Ctrl+C stops the server. `python app.py` is the same entry.

The header is the navigation: Analyze, Teach, Models, Settings. The right side of the header reads Local, the device, and the active model version. The limit of the tool is on the Analyze pipeline card and on Settings: decision support only, not a certified procedure.

The reviewer box is a label stored with your review. It stays on this machine.

## Screenshots

![Analyze, before a run](screenshots/analyze.png)

![Analyze, with a draft site report](screenshots/analyze_case.png)

![Teach: alert types, frames, and labels](screenshots/teach.png)

![Models: versions, retrain, and rollback](screenshots/models.png)

![Settings: device and doctor](screenshots/settings.png)

## Analyze

This is the page that opens first.

1. Upload the historian CSV. It needs a timestamp column and the sensor channels.
2. Pick the alert type. The list is the alert types this site has taught, plus the names the app already knows (`Exhaust Cold Spot`, `Combustion Dynamics High`, `HETS Trip`, and the others).
3. Pick the frame. **Detect from tags** leaves the choice to the file. The shipped frames are 6B, 7EA, 7FA, 7HA, 9E, 9FA, and 9HA. A frame you add on Teach shows up here. **Other / custom** opens the tag form: a layout, the exhaust ring in sensor order, the zone tags, and process tags written as `role=tag` (one per line).
4. Write notes: unit, operator notes, prior history.
5. Leave **Also write the local model site report** off unless you want Gemma 4 12B to write the narrative. On a CPU that takes several minutes. Off, the report uses the deterministic template and is usually ready in well under a minute.
6. Press **Run analysis**.

The draft opens on the same page, in three columns:

- **Case input** and a pipeline checklist stay on the left.
- The middle column shows the signature, confidence, and novelty. Novelty reads Novel, Known, or Not scored. Novel is red. Known is mint. Under that, one plot per tag that the report uses as evidence, up to eight. The mint band is the residual window stored on the report (the learned zoom, a rule event, or a residual onset).
- The right column is the short site report (what happened, why, recommended action), a table of nearest past cases, and the review gate.

**Full report** opens the document-style writeup. Print from that view: the header and the review form stay off the printed page.

**Good** stores a Good verdict. **Not good** and **Edit** open the correction: Good or Not good, a corrected action, a reason, and your name as the reviewer, then **Submit review**. Good becomes APPROVED. Not good becomes REJECTED and keeps the corrected action. The verdict is stored under `data/` and is used the next time a similar case is drafted.

Recent cases on the input panel open a stored draft. They do not analyze the file again.

## Teach

Three blocks, all local.

**Alert types.** The table lists each type, its aliases, and its tag list. Add alert type saves a new one. A sample CSV is optional and is used only to draft the tag list. An empty process document is stored as "no document", not as a blank file that later retrieval treats as text.

**Frames.** The line under the heading is every frame name the Analyze dropdown can use. Add frame writes a name, a short description, the exhaust ring, the zone tags, and process tags as `role=tag`.

**Labels.** The table is cases an engineer has already labeled. Add labeled case uploads one CSV, picks a signature and an action, and requires a one-sentence why. The learned model loads when you press the button, not when you open the page. The reviewer string is the one you last typed on a review form, or blank.

## Models

The table is every version the registry has: which one is active, whether it was shipped or trained, when it was created, what triggered it, held-out action accuracy, held-out signature accuracy, and whether it was promoted.

**Retrain** starts a quick update or a full retrain on auto, gpu, or cpu. The new version becomes the active model only when held-out action accuracy and signature accuracy are not worse, and no large frame or alert slice got worse. **Force promote** overrides that gate and the reason is written on the job. The page prints the promoted flag the trainer stored. It does not decide the gate a second time.

**Roll back** makes the previous version active. **Activate** makes the version you picked active. Both load the learned model when you press the button.

While a job is running the page shows its message and refreshes when the job finishes. A finished job that was not promoted stays on disk and stays inactive.

The shell is the same gate:

```powershell
.venv\Scripts\python sentinel_gt_ai.py train --quick
.venv\Scripts\python sentinel_gt_ai.py train --full --device cpu
.venv\Scripts\python sentinel_gt_ai.py train --quick --force
```

`--full` fine-tunes the shared encoder and does not take `--alert-type`. Quick update refits the classifier head, rebuilds the similarity index, and recalibrates novelty. Full retrain is about 15–20 minutes on CPU. `auto` and `gpu` use the GPU when CUDA works and fall back to CPU if it does not or runs out of memory.

Rolling back the model does not delete your CSVs.

## Settings

**Runtime device** is auto, gpu, or cpu. Opening the page only reads the current choice. **Save device** writes `device` into `settings.local.json`. Until you save, the choice is not stored.

**Doctor** runs the same check as `python sentinel_gt_ai.py doctor`. Include the LLM generate test only when you want a real generation from the local model. The output is shown as text. It is not treated as HTML.

## Add an alert type from the shell

Seven alert types ship with the app: exhaust cold spot, combustion dynamics, high exhaust spread trip, IGV / fuel control, bearing vibration, inlet filter differential pressure, and lube oil temperature. Each one is a JSON profile: a fixed tag list, the alert names that mean this type, and the signatures it can call.

A name that matches none of them does not silently pick a different type.

```powershell
.venv\Scripts\python sentinel_gt_ai.py alert-types list
.venv\Scripts\python sentinel_gt_ai.py alert-types new --name "Fuel gas heater" --sample samples\your_export.csv --frame 7FA --signatures "normal,heater_fault"
.venv\Scripts\python sentinel_gt_ai.py alert-types show fuel_gas_heater
.venv\Scripts\python sentinel_gt_ai.py alert-types add-alias --type fuel_gas_heater --name "FG heater trouble"
```

`--sample` is a CSV whose header becomes the proposed tag list. `--frame` attaches that frame's tag aliases. The new type is saved under `data/e2e/profiles/` and starts cold: no labeled cases, so it will not outvote a trained head until you label some. The command records the teach version.

Check for refused files and name clashes:

```powershell
.venv\Scripts\python sentinel_gt_ai.py alert-types check
```

To score with one head per alert type, set `"e2e_mode": "peralert"` in `settings.local.json` (or `SENTINEL_E2E_MODE=peralert`) and restart. `legacy` keeps the single 14-channel head. A fresh install stays on `legacy`.

Retrain one alert type after you have labels for it:

```powershell
.venv\Scripts\python sentinel_gt_ai.py train --quick --alert-type fuel_gas_heater
.venv\Scripts\python sentinel_gt_ai.py train --quick --all
```

## Label from the shell

Three places accept a label:

- **Analyze**, the review on a draft (above).
- **Teach → Add labeled case.**
- The shell:

```powershell
.venv\Scripts\python sentinel_gt_ai.py e2e label --case SGT-... --signature tc_sensor_fault --verdict good --note "neighbors flat"
```

A new signature name creates a class. The classifier recognizes it through similarity as soon as one case uses it, and as its own class after the next retrain.

Each saved label is in the similarity index immediately. The head retrains in the background after every few labels (`e2e_retrain_every` in settings, default 5).

## What a report is

Every report is a **draft** until you mark it. The primary action is one of: sensor fix or replace, borescope, LVDT calibration, combustion tuning, control adjustment, monitor only, escalate. Urgency and an escalate flag sit beside it. The audit section names the model, the thresholds, and the input hash.

A Good verdict becomes APPROVED. Not good becomes REJECTED and stores the corrected action. Similar future cases reuse that verdict: a correction on a closely similar case (similarity ≥ 0.80) changes the recommended action, and an approval raises confidence. Feedback also goes into the LLM prompt as examples.

This is decision support. It is not a trip system and it does not replace site procedures.

## Command line

The same install still has the CLI. `python sentinel_gt_ai.py doctor`, `train`, `run`, `feedback`, and `e2e` do not start the dashboard. `python sentinel_gt_ai.py ui` is the dashboard, and it is also what a bare `python sentinel_gt_ai.py` launches.
