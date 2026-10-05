"""MP4 writer for viewer / RGB streams."""

from __future__ import annotations

import numpy as np

from src.utils.video_io import write_rgb_mp4


def test_write_rgb_mp4(tmp_path):
    frames = [np.full((36, 48, 3), c, dtype=np.uint8) for c in (40, 80, 120, 160)]
    out = write_rgb_mp4(frames, tmp_path / "t.mp4", fps=10.0, scale=1, max_width=64)
    assert out.exists() and out.stat().st_size > 0
