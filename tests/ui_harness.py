"""NiceGUI test entry. Registers the four pages, then calls ui.run.

The user simulation executes this file as __main__. The page wrappers report
their module as tests.ui_pages so the simulation cleanup does not unload the
sentinel package. Routes are not registered when this module is imported.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nicegui import ui

import sentinel.ui as pages


def _page(draw, *, takes_case: bool = False):
    """Wrap a draw function. __module__ is set before the page decorator stores the function."""
    if takes_case:
        def wrapper(case: str = "") -> None:
            pages.shell(lambda: draw(case))
    else:
        def wrapper() -> None:
            pages.shell(draw)

    wrapper.__module__ = "tests.ui_pages"
    return wrapper


def main() -> None:
    ui.page("/", title="Sentinel GT AI", dark=True, response_timeout=120.0)(
        _page(pages.draw_analyze, takes_case=True)
    )
    ui.page("/teach", title="Sentinel GT AI", dark=True, response_timeout=120.0)(_page(pages.draw_teach))
    ui.page("/models", title="Sentinel GT AI", dark=True, response_timeout=120.0)(_page(pages.draw_models))
    ui.page("/settings", title="Sentinel GT AI", dark=True, response_timeout=120.0)(_page(pages.draw_settings))
    # Returns immediately under the user simulation. storage_secret is required only so the
    # simulated session middleware can start; the dashboard itself does not use browser storage.
    ui.run(
        host="127.0.0.1",
        port=8501,
        title="Sentinel GT AI",
        dark=True,
        reload=False,
        show=False,
        fastapi_docs=False,
        prod_js=True,
        storage_secret="simulated",
    )


if __name__ == "__main__":
    main()
