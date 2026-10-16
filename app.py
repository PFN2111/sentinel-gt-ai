#!/usr/bin/env python3
"""Sentinel GT AI entry point.

    python app.py                      # local web UI (NiceGUI, localhost only)
    python app.py run --csv X.csv --analytic "cold spot" --notes "Unit 2 ..."
    python app.py feedback --case SGT-... --verdict good
    python app.py --help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
