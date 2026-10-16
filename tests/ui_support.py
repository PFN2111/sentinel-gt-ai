"""Drive the NiceGUI dashboard from pytest without a browser plugin.

The simulation's HTTP client defaults to a 5 second timeout. Building a case
page reads the stored CSV, so the client used here waits 120 seconds.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from nicegui.testing import user_simulation

HARNESS = Path(__file__).with_name("ui_harness.py")


@asynccontextmanager
async def browser():
    real_init = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("timeout", httpx.Timeout(120.0))
        return real_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = init  # type: ignore[method-assign]
    try:
        async with user_simulation(main_file=HARNESS) as user:
            yield user
    finally:
        httpx.AsyncClient.__init__ = real_init  # type: ignore[method-assign]


def run(body) -> None:
    """Run one async test body. pytest-asyncio is not required."""
    asyncio.run(body())


def set_value(user, marker: str, value: str) -> None:
    """Set one marked widget. Call this after the page has rendered."""
    with user:
        element = next(iter(user.find(marker=marker).elements))
        element.value = value
