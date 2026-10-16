"""Neutral branding: no OEM company name, abbreviation, or department name in the deliverable.

Scans every tracked text file (code, prompts, templates, docs, generated HTML, eval reports, samples,
tests) plus the generated docs, case-insensitively with word boundaries. Binary files (model weights,
indexes) are skipped. The banned vocabulary is assembled from pieces so this file is not itself a hit.
Briefs, progress notes, and the privacy checklist are excluded: they quote the search vocabulary.
"""

from __future__ import annotations

import re
import subprocess

from conftest import ROOT


def _j(*parts: str) -> str:
    return "".join(parts)


def banned_pattern() -> re.Pattern[str]:
    """OEM company, department, and internal monitoring-tool names."""
    company = _j("G", "E")
    later = _j("Ver", "nova")
    full = _j("General ", "Electric")
    dept = _j("M", "&", "D")
    dept_html = _j("M", "&amp;", "D")
    phrase = "monitoring (and|&) diagnostics"
    mark = _j("Mark ", "VI")
    system = _j("System ", "One")
    signal = _j("Smart", "Signal")
    onsite = _j("O", "S", "M")
    return re.compile(
        "|".join(
            [
                r"\b" + company + r"\b",
                later,
                full,
                dept,
                dept_html,
                phrase,
                mark + r"e?",
                system,
                signal,
                r"\b" + onsite + r"\b",
            ]
        ),
        re.I,
    )


EXCLUDED = re.compile(
    r"^(E2E_BRIEF.*\.md|PROGRESS.*\.md|tests/test_branding\.py|PRIVACY_CHECK\.md)$"
)


def _files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    gen = [p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.html")] + [
        p.relative_to(ROOT).as_posix() for p in (ROOT / "eval").glob("*.md")
    ]
    return sorted(set(out) | set(gen))


def test_pattern_matches_terms_but_not_ordinary_words():
    banned = banned_pattern()
    bad = [
        _j("G", "E"),
        _j("g", "e"),
        "a " + _j("G", "E") + " unit",
        _j("Ver", "nova"),
        _j("general ", "electric"),
        _j("M", "&", "D"),
        _j("m", "&amp;", "d"),
        _j("Monitoring and ", "Diagnostics"),
        _j("monitoring & ", "diagnostics"),
        _j("Mark ", "VIe"),
        _j("Mark ", "VI"),
        _j("System ", "One"),
        _j("Smart", "Signal"),
        _j("O", "S", "M"),
    ]
    for item in bad:
        assert banned.search(item), item
    for ok in (
        "gear",
        "age",
        "image",
        "GEMMA",
        "large",
        "range",
        "TTXD_GE1x",
        "monitoring, diagnostics",
        "cosmos",
        "mark",
        "signal",
    ):
        assert not banned.search(ok), ok


def test_no_banned_terms_in_names_or_text():
    banned = banned_pattern()
    hits = []
    for rel in _files():
        if EXCLUDED.match(rel):
            continue
        if banned.search(rel):
            hits.append(f"{rel}: file name")
        p = ROOT / rel
        if not p.is_file():
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary (weights, npz)
        text = data.decode("utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            m = banned.search(line)
            if m:
                hits.append(f"{rel}:{i}: {line.strip()[:120]}")
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if banned.search(branch):
        hits.append(f"branch name {branch}")
    assert not hits, "\n".join(hits[:40])
