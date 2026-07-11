"""Deterministic controller tests.

Importing this package puts src/ and the project root on sys.path so test
modules can import the simulator modules exactly as main.py does. Run with:

    py -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _path in (str(_SRC), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
