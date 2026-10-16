"""docs/sentinel_overview.html builder: self-contained, offline, complete."""

from __future__ import annotations

import importlib.util
import json
import re

from conftest import ROOT


def _builder():
    spec = importlib.util.spec_from_file_location("build_overview", ROOT / "scripts" / "build_overview.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_overview_is_self_contained_and_complete():
    page = _builder().build(skip_tests=True)
    assert not re.findall(r"%%[A-Z_]+%%", page), "unfilled placeholders"
    # no external resources of any kind
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:https?:)?//""", page)
    assert "<link" not in page and "@import" not in page and not re.search(r"url\(\s*['\"]?https?:", page)
    assert len(re.findall(r"<script[^>]*\bsrc=", page)) == 0
    for sec in ("overview", "pipeline", "decisions", "triage", "report", "validation", "feedback", "run", "limits", "e2e"):
        assert f'<section id="{sec}"' in page
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S).group(1).replace("<\\/", "</"))
    assert len(data["cases"]) == 3 and data["gt"]["agree"] == data["gt"]["total"] == 25
    assert data["demos"]["after"]["action"] == "sensor_fix_replace" and "FB-OVERRIDE" in data["demos"]["after"]["rules"]
    assert all(c["series"]["panels"] for c in data["cases"])
    # a real validation report is embedded, rendered
    assert "Tier 2 Site Report" in page and "DRAFT — pending Tier 2 engineer review" in page
    # End-to-end AI tab: links the offline explainer; charts fed from eval/results.json when evaluated
    assert 'href="#e2e"' in page and 'href="e2e_explainer.html"' in page
    if (ROOT / "eval" / "results.json").is_file():
        assert data["e2e"]["curve"] and len(data["e2e"]["per_class"]) == 8
