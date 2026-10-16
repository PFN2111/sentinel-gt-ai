"""Smoke checks for the NiceGUI dashboard: local assets and the four pages."""

from __future__ import annotations

from conftest import ROOT
from ui_support import browser, run

_HOSTS = ("jsdelivr", "unpkg.com", "googleapis", "fonts.gstatic", "cdnjs")


def _ui_source() -> str:
    text = (ROOT / "sentinel_gt_ai.py").read_text(encoding="utf-8")
    start = text.index("# <<<<<<< SECTION sentinel.ui ")
    start = text.index("\n", start) + 1
    end = text.index("# >>>>>>> END sentinel.ui", start)
    return text[start:end]


def test_ui_section_is_local_nicegui():
    source = _ui_source()
    assert "import streamlit" not in source
    assert "ui.echart" in source
    assert "127.0.0.1" in source
    for host in _HOSTS:
        assert host not in source


def test_pages_render_and_stay_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SENTINEL_USE_LLM", "0")
    monkeypatch.setenv("SENTINEL_USE_TSPULSE", "0")
    settings_file = ROOT / "settings.local.json"
    before = settings_file.read_bytes() if settings_file.is_file() else None

    async def body():
        async with browser() as user:
            response = await user.http_client.get("/")
            assert response.status_code == 200
            page = response.text.lower()
            assert "/_nicegui/" in page
            for host in _HOSTS:
                assert host not in page
            await user.open("/")
            await user.should_see("Run analysis")
            await user.open("/teach")
            await user.should_see("Alert types")
            await user.should_see("Frames")
            await user.should_see("Labels")
            await user.should_see("Add labeled case")
            await user.open("/models")
            await user.should_see("Model versions")
            await user.should_see("Retrain")
            await user.should_see("Roll back")
            await user.open("/settings")
            await user.should_see("Runtime device")
            await user.should_see("Doctor")

    run(body)
    after = settings_file.read_bytes() if settings_file.is_file() else None
    assert after == before
