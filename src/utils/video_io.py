"""MP4 writing through the ``ffmpeg`` binary (no Python video dependency)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame)
    if f.ndim == 2:
        f = np.repeat(f[..., None], 3, axis=2)
    f = f[..., :3]
    if f.dtype != np.uint8:
        f = (np.clip(f.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(f)


class RGBVideoWriter:
    """Stream (H, W, 3) frames (uint8, or float in [0, 1]) into an H.264 MP4.

    ``scale`` upsamples with nearest neighbour (small camera frames stay crisp);
    ``max_width`` then caps the width. Output sides are rounded to even numbers.
    """

    def __init__(self, path: str | Path, fps: float = 30.0, scale: int = 1,
                 max_width: int | None = None, crf: int = 18) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.scale = max(1, int(scale))
        self.max_width = None if max_width is None else int(max_width)
        self.crf = int(crf)
        self.n_frames = 0
        self._proc: subprocess.Popen | None = None
        self._size: tuple[int, int] | None = None

    def _open(self, w: int, h: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH (brew install ffmpeg)")
        ow, oh = w * self.scale, h * self.scale
        if self.max_width is not None and ow > self.max_width:
            oh, ow = round(oh * self.max_width / ow), self.max_width
        ow, oh = max(2, ow - ow % 2), max(2, oh - oh % 2)
        flags = "neighbor" if self.scale > 1 else "area"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{self.fps:.6f}", "-i", "-",
             "-vf", f"scale={ow}:{oh}:flags={flags}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(self.crf), "-movflags", "+faststart",
             str(self.path)],
            stdin=subprocess.PIPE,
        )
        self._size = (w, h)

    def write(self, frame: np.ndarray) -> None:
        f = _to_uint8(frame)
        h, w = f.shape[:2]
        if self._proc is None:
            self._open(w, h)
        elif (w, h) != self._size:
            raise ValueError(f"frame size {w}x{h} != first frame {self._size[0]}x{self._size[1]}")
        self._proc.stdin.write(f.tobytes())
        self.n_frames += 1

    def close(self) -> Path:
        if self._proc is not None:
            self._proc.stdin.close()
            if self._proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
            self._proc = None
        return self.path

    def __enter__(self) -> "RGBVideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_rgb_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: float = 30.0,
                  scale: int = 1, max_width: int | None = None) -> Path:
    with RGBVideoWriter(path, fps=fps, scale=scale, max_width=max_width) as w:
        for f in frames:
            w.write(f)
    return w.path
