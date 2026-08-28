"""Extract one reference frame from a downloaded clip as an image.

Phase 0c has no zone editor yet (that's Phase 2 step 24, a browser UI). Until
then, drawing a fence polyline into config/cameras.yaml means looking at a
still frame from that camera, picking a handful of points along the fence
line by eye, and converting their pixel coordinates to width/height fractions
(0.0-1.0) -- normalised coordinates are resolution-independent.

Run:
    uv run python scripts/extract_frame.py data/history/cam05/18269.mp4 \
        --out data/reports/cam05_frame.png
    # --frame-index N to pick a different moment (default: middle frame)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def extract_frame(video_path: str, *, frame_index: int | None = None) -> np.ndarray:
    """Read one frame from video_path. Defaults to the clip's middle frame."""
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        index = frame_index if frame_index is not None else max(total // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {index} from {video_path}")
        return frame
    finally:
        cap.release()


def main() -> None:  # pragma: no cover - thin CLI wrapper around extract_frame
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("--out", required=True, help="Output image path (.png/.jpg)")
    parser.add_argument("--frame-index", type=int, default=None)
    args = parser.parse_args()

    frame = extract_frame(args.video_path, frame_index=args.frame_index)
    height, width = frame.shape[:2]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, frame)
    print(f"Wrote {args.out} ({width}x{height})")


if __name__ == "__main__":
    main()
