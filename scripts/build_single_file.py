#!/usr/bin/env python3
"""Build ``sentinel_gt_ai.py`` — one readable file that is the whole app.

The package modules stay the source that this script reads. Each module is copied in
verbatim (comments included) under an ``if False`` block so the file is real Python an
engineer can edit. A loader at the top compiles those blocks back into ``sentinel.*`` modules.
``from __future__ import annotations`` cannot sit inside ``if False`` (SyntaxError), so
those lines are stashed as a marker comment and restored when the block is compiled.

    python scripts/build_single_file.py
    python scripts/build_single_file.py --shims   # point the package at the single file
    python scripts/build_single_file.py --toc     # after hand edits: renumber the TOC only

Do not run ``--shims`` until ``sentinel_gt_ai.py`` imports cleanly. A shimmed package
cannot be rebuilt; restore it from git first.
"""

from __future__ import annotations

import ast
import json
import py_compile
import sys
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sentinel_gt_ai.py"

# Lower runs earlier when several modules are ready. Topological order still wins.
PRIORITY = {
    "sentinel": 0,
    "sentinel.config": 1,
    "sentinel.limits": 10,
    "sentinel.notes": 11,
    "sentinel.data": 12,
    "sentinel.analytics.base": 20,
    "sentinel.analytics.cold_spot": 21,
    "sentinel.analytics.combustion_dynamics": 22,
    "sentinel.analytics.hets_trip": 23,
    "sentinel.analytics.generic": 24,
    "sentinel.analytics": 25,
    "sentinel.knowledge": 30,
    "sentinel.anomaly": 31,
    "sentinel.charts": 32,
    "sentinel.rules": 33,
    "sentinel.triage": 34,
    "sentinel.report": 35,
    "sentinel.store": 36,
    "sentinel.llm": 37,
    "sentinel.llm_runtime": 38,
    "sentinel.pipeline": 39,
    "sentinel.e2e": 40,
    "sentinel.e2e.features": 41,
    "sentinel.e2e.taxonomy": 42,
    "sentinel.e2e.model": 43,
    "sentinel.e2e.encoder": 44,
    "sentinel.e2e.residual": 45,
    "sentinel.e2e.labels": 46,
    "sentinel.e2e.dataset": 47,
    "sentinel.e2e.train": 48,
    "sentinel.e2e.evaluation": 49,
    "sentinel.e2e.engine": 50,
    "sentinel.e2e.writer": 51,
    "sentinel.e2e.report": 52,
    "sentinel.e2e.simulate": 53,
    "sentinel.e2e.dashboard": 54,
    "sentinel.e2e.teach": 55,
    "sentinel.e2e.pipeline": 56,
    "sentinel.e2e.train_job": 57,
    "sentinel.bootstrap": 60,
    "sentinel.cli": 70,
    "sentinel.ui": 80,
    "sentinel.generator": 90,
}

WHY = {
    "sentinel": "Package identity: version and app name. Submodules hang off this.",
    "sentinel.config": "CONFIG — every tunable setting. Defaults live on class Settings. settings.local.json and SENTINEL_* environment variables override them. Nothing here contacts the network.",
    "sentinel.limits": "Fixed engineering thresholds the rule path compares sensor readings against.",
    "sentinel.notes": "Pull unit id, first-out text and free-text hints out of the engineer's notes.",
    "sentinel.data": "Read a historian CSV into canonical roles. Ring and zone tags sort by full sensor number; each frame model points at a tag layout.",
    "sentinel.analytics.base": "Shared evidence types and the robust statistics every analyzer uses.",
    "sentinel.analytics.cold_spot": "Rule-path cold-sector check: a contiguous arc of the exhaust ring, not one noisy thermocouple.",
    "sentinel.analytics.combustion_dynamics": "Rule-path dynamics check: pulsation that is real versus a dead probe.",
    "sentinel.analytics.hets_trip": "Rule-path trip check: first-out spread trip versus a sensor step that only looks like one.",
    "sentinel.analytics.generic": "Fallback when the analytic name matches no dedicated rule set. It still reports evidence and asks for a human.",
    "sentinel.analytics": "Resolve the free-text analytic name Tier 1 typed to one analyzer.",
    "sentinel.knowledge": "Retrieve process-map sections for the prompt. Learned 'why' notes from Teach sit beside the shipped maps.",
    "sentinel.anomaly": "Rule-path anomaly scan: robust z always, TS Pulse reconstruction when that snapshot loads.",
    "sentinel.charts": "Evidence charts and the residual view. Altair keeps the heatmap contract; the UI plots use plotly (matplotlib, then Altair) with one local palette. No remote assets.",
    "sentinel.rules": "Shadow rule names the learned path stores next to its own decision.",
    "sentinel.triage": "Turn analyzer findings into one action, an urgency and the reasons a human must look.",
    "sentinel.report": "Deterministic template narrative. Used when the LLM is off, missing, or fails its checks.",
    "sentinel.store": "Cases, feedback and the review gate, as files under data/. No database.",
    "sentinel.llm": "Prompt and JSON schema for the rule-path narrative, plus the number check on the model's reply.",
    "sentinel.llm_runtime": "The one LLM: Gemma 4 12B Q4_K_M. GPU offload when it loads, else CPU, else the template.",
    "sentinel.pipeline": "Rule path end to end: CSV, notes, analyzer, triage, template or LLM, save.",
    "sentinel.e2e": "Learned path package marker.",
    "sentinel.e2e.features": "Canonical 14-channel window (512 points) the encoder was fine-tuned on. Do not reorder it.",
    "sentinel.e2e.taxonomy": "Signature ids the head can name, plus NOVEL, which is never trained as a class.",
    "sentinel.e2e.model": "Light head, similarity index and novelty calibration on frozen embeddings.",
    "sentinel.e2e.encoder": "TS Pulse fine-tune and embed. device auto|gpu|cpu, and a CPU retry when the GPU runs out of memory.",
    "sentinel.e2e.residual": "Second TS Pulse view: reconstruction residual, robust-z fallback, cross-check against the classifier.",
    "sentinel.e2e.labels": "Engineer labels, metric log, and the version folder (v0000 is the shipped head).",
    "sentinel.e2e.dataset": "Load the synthetic manifest and import or export engineer-labeled CSV folders.",
    "sentinel.e2e.train": "Quick-update pieces: refit the head, recalibrate novelty, score held-out, promote only when not worse.",
    "sentinel.e2e.evaluation": "Cached rule-path scores and the confusion tables the eval report uses.",
    "sentinel.e2e.engine": "Load one version (head, and the encoder that version belongs to) and run decide.",
    "sentinel.e2e.writer": "What the LLM sees on the learned path: evidence, residual, process steps, and the cross-check.",
    "sentinel.e2e.report": "Markdown site report for a learned-path case, including the residual evidence rows.",
    "sentinel.e2e.simulate": "Site-adaptation drill used by tests: label, retrain the head, measure the lift.",
    "sentinel.e2e.dashboard": "Learning-dashboard series: held-out accuracy, rolling agreement, label counts.",
    "sentinel.e2e.teach": "Teach the app: labeled cases with a why, process documents, signatures, version rollback.",
    "sentinel.e2e.pipeline": "Learned path end to end: features, decide, rules in shadow, report, save.",
    "sentinel.e2e.train_job": "Retrain in a subprocess so a Streamlit rerun does not kill it. Progress lives in data/e2e/jobs.",
    "sentinel.bootstrap": "setup: venv, pinned installs, model download or --models-from, starter knowledge, optional train.",
    "sentinel.cli": "Command line: run, doctor, train, setup, and the NiceGUI launcher.",
    "sentinel.ui": "NiceGUI pages: Analyze, Teach, Models, Settings. serve() starts the server; importing this module does not.",
    "sentinel.generator": "Built-in synthetic historian CSVs. setup --train uses these when the encoder is missing.",
}


def discover() -> List[Tuple[str, str, Path]]:
    """(module name, repo-relative path, absolute path) for every module the file must contain."""
    found: List[Tuple[str, str, Path]] = []
    for path in sorted((REPO / "sentinel").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.endswith("/__init__.py"):
            qual = rel[: -len("/__init__.py")].replace("/", ".")
        elif rel.endswith("__init__.py"):
            qual = "sentinel"
        else:
            qual = rel[:-3].replace("/", ".")
        found.append((qual, rel, path))
    gen = REPO / "scripts" / "generate_scenario_csvs.py"
    found.append(("sentinel.generator", "scripts/generate_scenario_csvs.py", gen))
    return found


def _qual_from_parts(parts: Sequence[str]) -> str:
    return ".".join(p for p in parts if p)


def module_deps(qual: str, src: str, known: Set[str]) -> Set[str]:
    """Module-level ``sentinel`` imports. Imports inside functions are lazy and not edges."""
    tree = ast.parse(src)
    is_pkg = qual in {"sentinel", "sentinel.e2e", "sentinel.analytics"} or qual.count(".") == 0
    # A package's own name is the qual; a module's package is the parent.
    base_parts = qual.split(".") if is_pkg else qual.split(".")[:-1]
    deps: Set[str] = set()

    def add(name: str) -> None:
        if name in known and name != qual:
            deps.add(name)

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and (node.module == "sentinel" or node.module.startswith("sentinel.")):
                if node.module == "sentinel":
                    add("sentinel")
                    for alias in node.names:
                        add(f"sentinel.{alias.name}")
                else:
                    add(node.module)
                    for alias in node.names:
                        add(f"{node.module}.{alias.name}")
            elif node.level:
                parts = list(base_parts[: len(base_parts) - (node.level - 1)])
                if node.module:
                    parts.extend(node.module.split("."))
                    add(_qual_from_parts(parts))
                    for alias in node.names:
                        add(_qual_from_parts(parts + [alias.name]))
                else:
                    for alias in node.names:
                        add(_qual_from_parts(list(parts) + [alias.name]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sentinel" or alias.name.startswith("sentinel."):
                    add(alias.name)
    return deps


def topo(items: Sequence[Tuple[str, str, Path]], sources: Dict[str, str]) -> List[str]:
    known = {qual for qual, _rel, _path in items}
    graph: Dict[str, Set[str]] = {qual: module_deps(qual, sources[qual], known) for qual, _r, _p in items}
    ready = TopologicalSorter(graph)
    try:
        ready.prepare()
    except CycleError as exc:
        raise SystemExit(f"import cycle in the package:\n{exc}") from exc
    ordered: List[str] = []
    while ready.is_active():
        batch = list(ready.get_ready())
        batch.sort(key=lambda name: (PRIORITY.get(name, 50), name))
        for name in batch:
            ordered.append(name)
            ready.done(name)
    return ordered


def add_docstrings(src: str) -> str:
    """Give every function and class a docstring so the single file meets the brief.

    Existing docstrings stay. A one-line ``def`` is left alone and reported; expanding it
    would drop the only copy of the signature.
    """
    tree = ast.parse(src)
    lines = src.splitlines()
    inserts: List[Tuple[int, str]] = []
    skipped: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if ast.get_docstring(node) is not None or not node.body:
            continue
        first = node.body[0]
        if first.lineno == node.lineno:
            skipped.append(f"line {node.lineno} {node.name}")
            continue
        indent = " " * first.col_offset
        kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
        inserts.append((first.lineno, f'{indent}"""{kind} {node.name}."""'))
    for lineno, text in sorted(inserts, key=lambda item: item[0], reverse=True):
        lines.insert(lineno - 1, text)
    if ast.get_docstring(tree) is None:
        lines.insert(0, '"""Section of Sentinel GT AI."""')
    if skipped:
        print("  one-line defs left without a new docstring: " + ", ".join(skipped))
    body = "\n".join(lines)
    return body + ("\n" if src.endswith("\n") or not src.endswith("\n") else "")


def stash_future(src: str) -> str:
    """Hide ``from __future__`` so the ``if False`` block parses inside the bundle."""
    out = []
    for line in src.splitlines():
        if line.strip() == "from __future__ import annotations":
            out.append("# @@future@@ from __future__ import annotations")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def embed_bootstrap(src: str) -> str:
    """Put starter knowledge, requirements and the example settings into the bootstrap section."""
    docs = {path.name: path.read_text(encoding="utf-8") for path in sorted((REPO / "knowledge").glob("*.md"))}
    req = (REPO / "requirements.txt").read_text(encoding="utf-8")
    settings = (REPO / "settings.example.json").read_text(encoding="utf-8")
    src = src.replace(
        "_EMBEDDED_DOCS: Dict[str, str] = {}  # @@EMBED_DOCS@@",
        "_EMBEDDED_DOCS: Dict[str, str] = json.loads(" + json.dumps(json.dumps(docs)) + ")",
    )
    src = src.replace(
        '_EMBEDDED_REQUIREMENTS = ""  # @@EMBED_REQ@@',
        "_EMBEDDED_REQUIREMENTS = json.loads(" + json.dumps(json.dumps(req)) + ")",
    )
    src = src.replace(
        '_EMBEDDED_SETTINGS = ""  # @@EMBED_SETTINGS@@',
        "_EMBEDDED_SETTINGS = json.loads(" + json.dumps(json.dumps(settings)) + ")",
    )
    if "@@EMBED_" in src:
        raise SystemExit("bootstrap.py embed tokens were not replaced")
    return src


def indent_block(src: str, spaces: int = 4) -> str:
    pad = " " * spaces
    out = []
    for line in src.splitlines():
        out.append(pad + line if line.strip() else "")
    return "\n".join(out) + "\n"


def render_order(order: Sequence[str], rels: Dict[str, str]) -> str:
    rows = ",\n".join(f'    ("{qual}", "{rels[qual]}")' for qual in order)
    return "ORDER = [\n" + rows + "\n]\n"


LOADER = r'''
from __future__ import annotations

import sys
import types
from pathlib import Path

# The file lives in the app folder. Module __file__ values point at the original
# package paths (sentinel/config.py and so on) so ROOT = parents[1] still means
# this folder, including on a machine where those paths are not real files yet.
APP_FILE = Path(__file__).resolve()
APP_ROOT = APP_FILE.parent

# @@ORDER@@

_CACHE = {}
_DONE = {}
_ALL = False


def _section_table():
    """Read the blocks under the banners. Line numbers match this file, so tracebacks do too."""
    if _CACHE:
        return _CACHE
    lines = APP_FILE.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("# <<<<<<< SECTION "):
            parts = line.split()
            qual, rel = parts[3], parts[4]
            index += 1
            if index < len(lines) and lines[index].startswith("if False"):
                index += 1
            start = index + 1
            body = []
            while index < len(lines) and not lines[index].startswith("# >>>>>>> END "):
                raw = lines[index]
                if raw.startswith("    "):
                    raw = raw[4:]
                elif raw.strip() == "":
                    raw = ""
                body.append(raw)
                index += 1
            source = "\n".join(body) + "\n"
            source = source.replace(
                "# @@future@@ from __future__ import annotations",
                "from __future__ import annotations",
            )
            _CACHE[qual] = (source, start, rel)
        index += 1
    return _CACHE


def _prepare(ns, qual, rel):
    """Set __file__ / __package__ only when importlib has not already set them (a real shim)."""
    virtual = APP_ROOT / rel
    ns.setdefault("__file__", str(virtual))
    if rel.endswith("__init__.py"):
        ns.setdefault("__package__", qual)
        ns.setdefault("__path__", [str(virtual.parent)])
    else:
        ns.setdefault("__package__", qual.rpartition(".")[0])
    ns.setdefault("__name__", qual)


def _attach(qual, mod):
    parent, _, child = qual.rpartition(".")
    if parent and parent in sys.modules and mod is not None:
        setattr(sys.modules[parent], child, mod)


def exec_section(qual, namespace=None):
    """Compile one section into ``namespace`` (a shim's globals) or into a new module.

    The compiled code's filename is this file and the source is padded so a traceback
    line number is the line in ``sentinel_gt_ai.py``. Safe to call twice.
    """
    if qual in _DONE and namespace is None:
        return _DONE[qual]
    table = _section_table()
    if qual not in table:
        raise ImportError(f"{qual} is not a section of {APP_FILE.name}")
    source, start, rel = table[qual]
    created = None
    if namespace is None:
        mod = sys.modules.get(qual)
        if mod is None:
            mod = types.ModuleType(qual)
            sys.modules[qual] = mod
            created = mod
        namespace = mod.__dict__
        _prepare(namespace, qual, rel)
        _attach(qual, mod)
    else:
        _prepare(namespace, qual, rel)
        mod = sys.modules.get(qual)
        if mod is None:
            # python -m runs the shim as __main__. Keep that dict, and also publish the
            # real module name so a relative import inside the child can find siblings.
            mod = types.ModuleType(qual)
            mod.__dict__.update(namespace)
            sys.modules[qual] = mod
            _attach(qual, mod)
    if qual in _DONE and _DONE[qual].__dict__ is namespace:
        return _DONE[qual]
    _DONE[qual] = sys.modules.get(qual) or created
    padded = ("\n" * (start - 1)) + source
    code = compile(padded, str(APP_FILE), "exec", dont_inherit=True)
    try:
        exec(code, namespace)  # noqa: S102 — this is the single-file module loader
    except Exception:
        _DONE.pop(qual, None)
        if created is not None:
            sys.modules.pop(qual, None)
        raise
    return _DONE.get(qual) or sys.modules.get(qual)


def load_all(skip=()):
    """Install every section. Used by the command line and by Streamlit, not by a shim import.

    The CLI skips the UI section so ``doctor`` does not import NiceGUI. The ui command
    loads it and calls ``serve()``.
    """
    global _ALL
    if _ALL:
        return
    skip = set(skip)
    # Parents first so a child can be attached. Exec order is the topological ORDER.
    for qual, rel in sorted(ORDER, key=lambda item: item[0].count(".")):
        if qual in skip or qual in _DONE or qual in sys.modules:
            continue
        mod = types.ModuleType(qual)
        _prepare(mod.__dict__, qual, rel)
        # Force our virtual path. setdefault would keep a stray entry.
        mod.__file__ = str(APP_ROOT / rel)
        sys.modules[qual] = mod
        _attach(qual, mod)
    for qual, _rel in ORDER:
        if qual in skip:
            continue
        exec_section(qual)
    cfg = sys.modules["sentinel.config"]
    cli = sys.modules["sentinel.cli"]
    globals()["load_settings"] = cfg.load_settings
    globals()["Settings"] = cfg.Settings
    globals()["ROOT"] = cfg.ROOT
    globals()["main"] = cli.main
    if not skip:
        _ALL = True


def __getattr__(name):
    """``from sentinel_gt_ai import load_settings`` loads the sections on first use.

    Dunder lookups such as ``__path__`` must fail immediately. The import system probes
    ``__path__`` to see whether this file is a package; loading every section in response
    would put modules into ``sys.modules`` before ``python -m`` can run them.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    load_all()
    if name in globals():
        return globals()[name]
    raise AttributeError(name)


if __name__ == "__main__":
    # Doctor, train and setup do not draw widgets. The ui command loads the NiceGUI
    # section itself. Leaving it unloaded here keeps NiceGUI out of every other command.
    load_all(skip=("sentinel.ui",))
    raise SystemExit(sys.modules["sentinel.cli"].main())
'''


def header(toc: str) -> str:
    return f'''#!/usr/bin/env python3
"""Sentinel GT AI — one local file for a human-in-the-loop Tier 2 engineer.

Drop a historian CSV, the analytic name and your notes. The app drafts a site report
with evidence. You review it, correct it, and teach the app. Retrain stays on this computer.
Review and inference do not use the network. ``setup`` is the only command that downloads.

    python sentinel_gt_ai.py setup
    python sentinel_gt_ai.py
    python sentinel_gt_ai.py doctor
    python sentinel_gt_ai.py train --quick --device auto

The blocks under each banner are the package modules, comments included. They sit in
``if False`` so this file stays valid Python; the loader compiles each block into the
``sentinel.*`` module it came from. Edit a block in place. Data stays outside the file:
models/, knowledge/, data/, settings.local.json.

TABLE OF CONTENTS
{toc}
"""
'''


def build() -> str:
    items = discover()
    rels = {qual: rel for qual, rel, _path in items}
    sources: Dict[str, str] = {}
    for qual, rel, path in items:
        src = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if "exec_section(" in src:
            raise SystemExit(f"{rel} is already a shim. Restore the package from git before rebuilding.")
        if qual == "sentinel.bootstrap":
            src = embed_bootstrap(src)
        sources[qual] = src
    order = topo(items, sources)
    print("section order:")
    for qual in order:
        print(f"  {qual}")
    parts = ["%%HEADER%%\n", LOADER.replace("# @@ORDER@@\n", render_order(order, rels))]
    for qual in order:
        src = add_docstrings(sources[qual])
        src = stash_future(src)
        # The restored source must still compile on its own (future import back in place).
        check = src.replace("# @@future@@ from __future__ import annotations", "from __future__ import annotations")
        try:
            compile(check, rels[qual], "exec", dont_inherit=True)
        except SyntaxError as exc:
            raise SystemExit(f"section {qual} does not compile: {exc}") from exc
        why = WHY.get(qual, qual)
        parts.append(
            "\n# =============================================================================\n"
            f"# {qual}\n"
            f"# {why}\n"
            "# =============================================================================\n"
            f"# <<<<<<< SECTION {qual} {rels[qual]}\n"
            f"if False:  # compiled by exec_section({qual!r}); not executed as part of this module\n"
            f"{indent_block(src)}"
            f"# >>>>>>> END {qual}\n"
        )
    text = "".join(parts)
    # Fill the TOC with the real line numbers of each banner.
    lines = text.splitlines()
    toc_lines = []
    for number, line in enumerate(lines, 1):
        if line.startswith("# <<<<<<< SECTION "):
            qual = line.split()[3]
            toc_lines.append(f"{number:5d}  {qual} — {WHY.get(qual, '')}")
    toc = "\n".join(toc_lines)
    text = text.replace("%%HEADER%%\n", header(toc), 1)
    if "%%HEADER%%" in text or "@@ORDER@@" in text or "@@EMBED_" in text or "@@future@@" in text.split("if False")[0]:
        # @@future@@ is expected inside the if False blocks only.
        pass
    if "%%HEADER%%" in text or "@@ORDER@@" in text or "@@EMBED_" in text:
        raise SystemExit("unfilled placeholder in sentinel_gt_ai.py")
    return text


SHIM = '''"""Loaded from sentinel_gt_ai.py. This file is the import hook; edit the section in that file."""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _parent in _here.parents:
    if (_parent / "sentinel_gt_ai.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
else:
    raise ImportError("sentinel_gt_ai.py was not found above {{here}}".format(here=_here))

from sentinel_gt_ai import exec_section

exec_section({qual!r}, globals())
'''


def write_shims(order_rels: Dict[str, str]) -> None:
    """Replace package modules with loaders. The generator script stays; it is not a package file."""
    for qual, rel in order_rels.items():
        if qual == "sentinel.generator":
            continue
        path = REPO / rel
        path.write_text(SHIM.format(qual=qual), encoding="utf-8", newline="\n")
        print(f"shim {rel}")


def refresh_toc(path: Path = OUT) -> int:
    """Renumber the TABLE OF CONTENTS so each entry names the line of its SECTION banner.

    The single file is now edited by hand (the package is shims), so a full rebuild is not possible.
    This touches only the number column of the TOC; the line count stays the same, so one pass is enough.
    """
    import re

    raw = path.read_bytes()
    crlf = b"\r\n" in raw
    lines = raw.decode("utf-8").replace("\r\n", "\n").split("\n")
    sections = {line.split()[3]: n for n, line in enumerate(lines, 1) if line.startswith("# <<<<<<< SECTION ")}
    start = lines.index("TABLE OF CONTENTS")
    changed = 0
    for j in range(start + 1, len(lines)):
        m = re.match(r"^\s*(\d+)\s+(sentinel[\w.]*)( — .*)$", lines[j])
        if not m:
            break
        new = f"{sections[m.group(2)]:5d}  {m.group(2)}{m.group(3)}"
        changed += new != lines[j]
        lines[j] = new
    out = "\n".join(lines)
    if crlf:
        out = out.replace("\n", "\r\n")
    path.write_bytes(out.encode("utf-8"))
    print(f"TOC: {changed} line number(s) updated in {path.name}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--toc" in args:
        return refresh_toc()
    text = build()
    OUT.write_text(text, encoding="utf-8", newline="\n")
    py_compile.compile(str(OUT), doraise=True)
    print(f"wrote {OUT} ({text.count(chr(10))} lines)")
    if "--shims" in args:
        # Rebuild the map the same way build() did. A second discover is cheap.
        items = discover()
        rels = {qual: rel for qual, rel, _path in items}
        # ORDER in the written file is the source of truth for names.
        order = []
        for line in text.splitlines():
            if line.startswith("# <<<<<<< SECTION "):
                order.append(line.split()[3])
        write_shims({qual: rels[qual] for qual in order})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
