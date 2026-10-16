<#
.SYNOPSIS
  Sentinel GT AI (e2e-ai) - one-step install on a Windows PC, GPU or CPU-only.

.DESCRIPTION
  1. Creates .venv (Python 3.11) next to this script and installs requirements (CPU torch:
     TS Pulse and the learned path always run on CPU).
  2. Installs llama-cpp-python: the CUDA 12.4 wheel (+ CUDA runtime DLLs from pip) when an NVIDIA
     GPU is present, otherwise the CPU wheel. If the CUDA build cannot load, it falls back to the
     CPU wheel automatically - install never fails just because of the GPU.
  3. Puts the models into .\models (relative to the app folder): either copied from an offline
     folder (-ModelsFrom, e.g. a USB stick with a copy of another install's models folder) or
     downloaded once from Hugging Face. Every large file is SHA-256 verified.
  4. Creates settings.local.json (relative paths) and runs `python sentinel_gt_ai.py doctor`.
     The dashboard is NiceGUI at http://127.0.0.1:8501 (localhost only). Vue, Quasar and
     ECharts are served from the installed package. --native opens a desktop window when
     pywebview is installed.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install.ps1
  powershell -ExecutionPolicy Bypass -File install.ps1 -ModelsFrom E:\sentinel_models
  powershell -ExecutionPolicy Bypass -File install.ps1 -Cpu
#>
[CmdletBinding()]
param(
    [string]$ModelsFrom = "",      # offline models folder to copy from (same layout as .\models)
    [switch]$Cpu,                  # force the CPU build of llama-cpp-python even if a GPU is present
    [switch]$SkipModels,           # do not download / copy models
    [string]$Python = ""           # path to python.exe 3.11 (default: py -3.11, then python)
)
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $App
$LlamaVersion = "0.3.36"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    [ok] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    [warn] $msg" -ForegroundColor Yellow }

# Model files: relative path, source URL, SHA-256, required?
$Models = @(
    @{ Path = "models/gemma-4-12b-it-Q4_K_M.gguf"; Url = "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf";
       Sha = "0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42"; Kind = "llm" },
    @{ Path = "models/tspulse/clf/model.safetensors"; Url = "https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1/resolve/tspulse-block-dualhead-512-p16-r1/model.safetensors";
       Sha = "71fddf719ebe8e8c64eaec5769620935af48263dc35e4ce6f3f88f7708e973e1"; Kind = "tspulse" },
    @{ Path = "models/tspulse/clf/config.json"; Url = "https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1/resolve/tspulse-block-dualhead-512-p16-r1/config.json";
       Sha = ""; Kind = "tspulse" },
    @{ Path = "models/tspulse/ad/model.safetensors"; Url = "https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1/resolve/main/model.safetensors";
       Sha = "57fa03b67d1473a7253ac37801b24c391d1fa931395db986743e2f59e556b9ac"; Kind = "tspulse" },
    @{ Path = "models/tspulse/ad/config.json"; Url = "https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1/resolve/main/config.json";
       Sha = ""; Kind = "tspulse" }
)

# ---------------------------------------------------------------- 1. Python + venv
Step "Python 3.11 virtual environment"
$venvPy = Join-Path $App ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    if ($Python) { & $Python -m venv .venv }
    elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3.11 -m venv .venv }
    else { & python -m venv .venv }
    if (-not (Test-Path $venvPy)) { throw "could not create .venv - install Python 3.11 (python.org) and re-run" }
}
$ver = & $venvPy -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($ver -ne "3.11") { Warn "venv Python is $ver (tested with 3.11)" } else { Ok "Python $ver in .venv" }
& $venvPy -m pip install --upgrade pip --quiet

Step "Python packages (CPU torch for TS Pulse)"
& $venvPy -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
& $venvPy -m pip install --quiet -r requirements.txt
Ok "requirements installed"
Ok "NiceGUI UI included (Vue, Quasar and ECharts are local; no CDN)"

# ---------------------------------------------------------------- 2. llama-cpp-python (GPU or CPU)
Step "LLM runtime (llama-cpp-python $LlamaVersion)"
$gpu = $null
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi -and -not $Cpu) {
    try { $gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1) } catch { $gpu = $null }
}
$cpuWheel = { & $venvPy -m pip install --quiet --force-reinstall --no-deps "llama-cpp-python==$LlamaVersion" --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
              & $venvPy -m pip install --quiet "llama-cpp-python==$LlamaVersion" --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu }
if ($gpu) {
    Ok "NVIDIA GPU: $gpu - installing the CUDA 12.4 build"
    & $venvPy -m pip install --quiet "nvidia-cuda-runtime-cu12==12.4.*" "nvidia-cublas-cu12==12.4.*"
    & $venvPy -m pip install --quiet --force-reinstall --no-deps "llama-cpp-python==$LlamaVersion" --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    & $venvPy -m pip install --quiet "llama-cpp-python==$LlamaVersion" --only-binary llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    $probe = & $venvPy -c "import sys; sys.path.insert(0, '.'); from sentinel import llm_runtime; print(llm_runtime.llama_cpp_info())" 2>&1
    if ("$probe" -match "'gpu_offload': True") { Ok "CUDA build loads (GPU offload supported)" }
    else { Warn "CUDA build did not load ($probe) - falling back to the CPU build"; & $cpuWheel; $gpu = $null }
} else {
    if ($Cpu) { Ok "CPU build requested (-Cpu)" } else { Ok "no NVIDIA GPU found - installing the CPU build" }
    & $cpuWheel
}

# ---------------------------------------------------------------- 3. models
function Get-Sha($p) {
    # .NET directly: Get-FileHash can be unavailable when Windows PowerShell inherits a PowerShell 7 module path
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $fs = [System.IO.File]::OpenRead((Resolve-Path $p))
    try { return ([System.BitConverter]::ToString($sha.ComputeHash($fs)) -replace '-', '').ToLower() } finally { $fs.Dispose(); $sha.Dispose() }
}
if (-not $SkipModels) {
    Step "Models into .\models (relative to the app folder)"
    foreach ($m in $Models) {
        $dst = Join-Path $App $m.Path
        New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
        if ((Test-Path $dst) -and ((-not $m.Sha) -or ((Get-Sha $dst) -eq $m.Sha))) { Ok "$($m.Path) present"; continue }
        $copied = $false
        if ($ModelsFrom) {
            $rel = $m.Path.Substring("models/".Length)
            $src = Join-Path $ModelsFrom $rel
            if (Test-Path $src) { Copy-Item $src $dst -Force; $copied = $true; Ok "copied $rel from $ModelsFrom" }
        }
        if (-not $copied) {
            Write-Host "    downloading $($m.Url)"
            & curl.exe -L --fail --retry 5 -C - -o $dst $m.Url
            if ($LASTEXITCODE -ne 0) { throw "download failed: $($m.Url)" }
        }
        if ($m.Sha) {
            $h = Get-Sha $dst
            if ($h -ne $m.Sha) { throw "SHA-256 mismatch for $($m.Path): $h (expected $($m.Sha)) - delete it and re-run" }
            Ok "$($m.Path) SHA-256 verified"
        }
    }
    if (-not (Test-Path (Join-Path $App "models/e2e/encoder/model.safetensors"))) {
        if ($ModelsFrom -and (Test-Path (Join-Path $ModelsFrom "e2e"))) { Copy-Item -Recurse -Force (Join-Path $ModelsFrom "e2e") (Join-Path $App "models/") ; Ok "copied learned model (models/e2e)" }
        else { Warn "learned model models/e2e missing - it ships with the repository; or run: .venv\Scripts\python sentinel_gt_ai.py train --full --device auto" }
    } else { Ok "learned model models/e2e present (fine-tuned TS Pulse + head + index)" }
}

# ---------------------------------------------------------------- 4. settings + doctor
Step "Settings"
if (-not (Test-Path "settings.local.json")) { Copy-Item settings.example.json settings.local.json; Ok "created settings.local.json (relative paths, device auto)" }
else { Ok "settings.local.json kept" }

Step "Doctor"
& $venvPy sentinel_gt_ai.py doctor
Write-Host "`nStart the app:  .venv\Scripts\python sentinel_gt_ai.py        (NiceGUI at http://127.0.0.1:8501, localhost only)" -ForegroundColor Cyan
Write-Host "Desktop window: .venv\Scripts\python sentinel_gt_ai.py --native   (optional: pip install pywebview)" -ForegroundColor Cyan
Write-Host "Force CPU:      .venv\Scripts\python sentinel_gt_ai.py --device cpu" -ForegroundColor Cyan
