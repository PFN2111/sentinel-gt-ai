"""Loaded from sentinel_gt_ai.py. This file is the import hook; edit the section in that file."""
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
    raise ImportError("sentinel_gt_ai.py was not found above {here}".format(here=_here))

from sentinel_gt_ai import exec_section

exec_section('sentinel.analytics.base', globals())
