"""Deprecated prompt-context module.

All LLM-facing prose now lives in prompts.py.  Keep this file only so old imports
fail gracefully during transition.
"""

from __future__ import annotations

from prompts import render_option_card, render_options

__all__ = ["render_option_card", "render_options"]
