# Install on a new Windows laptop

This is a from-scratch install. You do not need a previous case library, a previous virtual environment, or the model files. The install script creates the environment and downloads the weights. Cases start empty. You teach the app after it runs.

Tested with Python 3.11 on Windows 11. An NVIDIA GPU is optional.

## 1. Python 3.11

Install Python 3.11 from python.org. On the installer, tick **Add python.exe to PATH**.

Check:

```powershell
py -3.11 --version
```

You want `Python 3.11.x`. The app warns if the virtual environment is some other version.

## 2. Get the code

```powershell
git clone https://github.com/2111gt/sentinel-gt-ai.git
cd sentinel-gt-ai
```

Or copy the folder onto the machine. The folder can live anywhere. Settings use paths relative to that folder.

## 3. Run the install script

Online, with a GPU if `nvidia-smi` works, otherwise CPU:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

What it does:

1. Creates `.venv` with Python 3.11 and installs the pinned packages, including a CPU build of PyTorch. TS Pulse and the learned path use that CPU build.
2. Installs `llama-cpp-python` 0.3.36. If NVIDIA is present it installs the CUDA 12.4 wheel and the CUDA runtime libraries from pip. If that wheel does not load, it installs the CPU wheel. The install does not fail just because of the GPU. `-Cpu` forces the CPU wheel even when a GPU is present.
3. Downloads **Gemma 4 12B** (`gemma-4-12b-it-Q4_K_M.gguf`, about 7.1 GB) and the **TS Pulse** snapshots (`models/tspulse/clf` and `models/tspulse/ad`, a few megabytes) from Hugging Face, and checks SHA-256.
4. Copies `settings.example.json` to `settings.local.json` if you do not already have one.
5. Runs `python sentinel_gt_ai.py doctor`.

The learned model under `models/e2e` ships in the repository (fine-tuned encoder, heads, similarity indexes). It is not downloaded. If that folder is missing, doctor tells you to run `train --full`.

### Switches

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -Cpu
powershell -ExecutionPolicy Bypass -File install.ps1 -SkipModels
powershell -ExecutionPolicy Bypass -File install.ps1 -Python C:\Path\to\python.exe
```

Offline models: copy a working `models` folder to a USB stick, then:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -ModelsFrom E:\models
```

Files are copied instead of downloaded and still hash-checked. Python packages still need network, or a wheel cache you prepared with `pip download`.

The same steps exist as `python sentinel_gt_ai.py setup` and `python sentinel_gt_ai.py setup --models-from E:\models`.

## 4. Doctor

```powershell
.venv\Scripts\python sentinel_gt_ai.py doctor
.venv\Scripts\python sentinel_gt_ai.py doctor --verify
.venv\Scripts\python sentinel_gt_ai.py --device cpu doctor
```

Doctor reports the GPU name and VRAM when a GPU is present, whether `llama-cpp-python` is a CUDA or CPU build, the active device, how many layers it plans to offload, each model file, and a short generate. It ends with `RESULT: READY` when the stack can write a report. Without the Gemma file the app still runs and uses the template narrative. `--verify` also hashes the Gemma file, which is slow.

## 5. GPU and CPU

| Setting | Behavior |
|---|---|
| `auto` (default) | GPU when the CUDA build loads, otherwise CPU. A GPU that runs out of memory retries on CPU. |
| `cpu` | Never offload. Use this on a machine with no NVIDIA GPU, or when you want a predictable run. |
| `gpu` | Prefer the GPU. Still falls back to CPU if the build cannot load or runs out of memory. |

Set it in `settings.local.json` (`"device": "cpu"`) or for one command:

```powershell
.venv\Scripts\python sentinel_gt_ai.py --device cpu
```

The Settings tab has the same Device control for the session.

You do not install the CUDA toolkit. A current NVIDIA driver is enough for the CUDA wheel. CPU-only laptops skip that wheel.

## 6. Start with an empty case library

`data/` is not in git. After install the case list is empty: no prior labels, no prior reports, no site documents. That is what you want on a new laptop. Do not copy a `data` folder from another machine if those cases are plant exports.

Built-in alert types and process maps are in the code and in `knowledge/`. The shipped model under `models/e2e` can score a CSV before you have labeled anything. Similarity matches stay empty until you teach it.

Synthetic samples, safe to click through:

```powershell
.venv\Scripts\python sentinel_gt_ai.py
```

The dashboard is NiceGUI at http://127.0.0.1:8501 and listens on localhost only. Open **Analyze**, choose a frame, and upload a file from `samples/cold_spot/`. Leave the case library as the only place your own files are stored.

## 7. Teach it

On **Teach**:

1. Add a labeled case from a CSV you already understand. The one-sentence why is what later reports see when this case is a neighbor.
2. Add an alert type when Tier 1 raises a name the seven built-ins do not cover. A sample CSV is optional and is used only to draft the tag list. From a shell, the same draft comes from a header:

```powershell
.venv\Scripts\python sentinel_gt_ai.py alert-types new --name "Inlet filter high dp" --sample samples\your_export.csv --frame 7FA
```

3. Add a frame if this unit's tag names are not in the dropdown.
4. Retrain on **Models** when you want the head itself to learn the new labels. Quick update is seconds. Full retrain is on the order of 15–20 minutes on CPU. The new version is kept only when held-out accuracy is not worse, unless you tick force promote.

Model rollback is on **Models**. See [USER_GUIDE.md](USER_GUIDE.md).

## 8. Tests

```powershell
.venv\Scripts\python -m pytest -m "not slow"
```

Tests marked `slow` load TS Pulse weights or run a fine-tune. They need the TS Pulse files from step 3.
