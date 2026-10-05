"""Shared paper-figure resolution settings.

Viewer screenshots and matplotlib saves used by Figure 1 / 2 / 5 should pull
from here so everything stays at the same pixel budget.

Baseline was 1600×900 @ ~140 dpi plots; current scale is 8× that.
"""

from __future__ import annotations

# Taichi / GGUI screenshot size (px)
VIEWER_WIDTH = 12800
VIEWER_HEIGHT = 7200

# Matplotlib savefig dpi (~8× the former typical 140)
MPL_DPI = 1120


def viewer_window_config(width: int = VIEWER_WIDTH, height: int = VIEWER_HEIGHT) -> dict:
    return {"window": {"width": int(width), "height": int(height)}}
