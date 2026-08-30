"""Encode BGR frames to a real H.264 MP4, independent of this box's OpenCV build.

This box's OpenCV/system FFmpeg has no libx264 -- only a hardware
`h264_v4l2m2m` encoder that fails without a v4l2 device (see
docs/handoff.md / repo memory "Video output"). `cv2.VideoWriter` can only
write VP8/webm here, which VS Code's preview plays but Telegram and most
ntfy-viewing clients treat as a generic file rather than an inline video.

`imageio_ffmpeg` bundles its own static, GPL FFmpeg build with a real
libx264, completely independent of the system/OpenCV one. This pipes raw
BGR24 frames to that binary over stdin and asks for yuv420p + faststart --
the combination that plays natively in VS Code, Telegram, and a phone or
browser opening an ntfy attachment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


class Mp4Writer:
    """Write BGR uint8 frames to `out_path` as an H.264 mp4."""

    def __init__(self, out_path: str, *, fps: float, width: int, height: int) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # libx264 requires even dimensions for yuv420p; pad rather than crop.
        self.width = width + (width % 2)
        self.height = height + (height % 2)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self._proc = subprocess.Popen(
            [
                ffmpeg_exe,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                out_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            padded = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
            padded[:h, :w] = frame
            frame = padded
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        self._proc.stdin.close()
        stderr = self._proc.stderr.read()
        code = self._proc.wait()
        if code != 0:
            detail = stderr.decode(errors="replace")
            raise RuntimeError(f"ffmpeg encode failed (exit {code}): {detail}")
