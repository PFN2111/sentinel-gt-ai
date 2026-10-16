"""The single file is the app: imports, setup from an empty folder, doctor."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from conftest import ROOT


def _env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_single_file_load_all_exports_the_package_api():
    code = (
        "import sentinel_gt_ai as app\n"
        "app.load_all()\n"
        "import sentinel.config as cfg\n"
        "from sentinel.e2e.train import not_worse\n"
        "from sentinel.bootstrap import starter_documents, FIX_TRAIN\n"
        "docs = starter_documents()\n"
        "assert 'gt_overview.md' in docs and 'process_map_cold_spot.md' in docs\n"
        "assert not_worse({'action_acc': 1, 'signature_acc': 1}, {'action_acc': 1, 'signature_acc': 1})\n"
        "assert cfg.ROOT == app.APP_ROOT\n"
        "assert 'train --full' in FIX_TRAIN\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_env(),
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-2000:] + proc.stdout[-500:]
    assert "OK" in proc.stdout


def test_setup_from_a_folder_that_only_has_the_single_file(tmp_path):
    """Copy the file alone, hardlink models, skip pip, and doctor must say READY."""
    clean = tmp_path / "plant"
    clean.mkdir()
    shutil.copyfile(ROOT / "sentinel_gt_ai.py", clean / "sentinel_gt_ai.py")
    env = _env()
    setup = subprocess.run(
        [sys.executable, "sentinel_gt_ai.py", "setup", "--models-from", str(ROOT / "models"), "--skip-pip", "--no-download"],
        cwd=str(clean),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )
    assert setup.returncode == 0, setup.stdout[-2000:] + setup.stderr[-2000:]
    assert (clean / "knowledge" / "gt_overview.md").is_file()
    assert (clean / "settings.local.json").is_file()
    assert (clean / "models" / "e2e" / "encoder" / "model.safetensors").is_file()
    # A second setup must not replace the knowledge file the engineer edited.
    overview = clean / "knowledge" / "gt_overview.md"
    overview.write_text("local plant notes\n", encoding="utf-8")
    again = subprocess.run(
        [sys.executable, "sentinel_gt_ai.py", "setup", "--models-from", str(ROOT / "models"), "--skip-pip", "--no-download"],
        cwd=str(clean),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )
    assert again.returncode == 0, again.stderr[-1500:]
    assert overview.read_text(encoding="utf-8") == "local plant notes\n"
    doctor = subprocess.run(
        [sys.executable, "sentinel_gt_ai.py", "--device", "cpu", "doctor", "--no-generate"],
        cwd=str(clean),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )
    out = doctor.stdout + doctor.stderr
    assert doctor.returncode == 0, out[-2000:]
    assert "ACTIVE:" in out and "CPU" in out and "RESULT: READY" in out


def test_table_of_contents_points_at_every_section():
    """An engineer navigates the single file by the TOC, and the V2 explainer's section map is built from it.
    Every entry must name the line of its ``# <<<<<<< SECTION`` banner, and every section must be listed."""
    import re

    lines = (ROOT / "sentinel_gt_ai.py").read_text(encoding="utf-8").splitlines()
    sections = {line.split()[3]: n for n, line in enumerate(lines, 1) if line.startswith("# <<<<<<< SECTION ")}
    toc = {}
    for line in lines[lines.index("TABLE OF CONTENTS") + 1:]:
        m = re.match(r"^\s*(\d+)\s+(sentinel[\w.]*)\s+—", line)
        if not m:
            break
        toc[m.group(2)] = int(m.group(1))
    assert toc == sections, "TOC is stale: run python scripts/build_single_file.py --toc, then python scripts/build_v2_explainer.py"
    html = (ROOT / "docs" / "sentinel_v2_explainer.html").read_text(encoding="utf-8")
    for qual in sections:
        assert f"<code>{qual}</code>" in html, qual
