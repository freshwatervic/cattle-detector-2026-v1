"""Extract still frames from a drone video, for labelling or testing.

Usage:
    cow-extract-frames --source flight1.mp4 --out frames/ --every 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract(source: Path, out_dir: Path, every_seconds: float, max_frames: int | None = None) -> int:
    """Save one frame every `every_seconds`. Returns the number of frames saved."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps * every_seconds), 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            cv2.imwrite(str(out_dir / f"{source.stem}_frame{idx:06d}.jpg"), frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
    cap.release()
    print(f"Extracted {saved} frames from {source.name} (every {every_seconds}s) -> {out_dir}")
    return saved


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract frames from a video.")
    ap.add_argument("--source", required=True, help="Video file")
    ap.add_argument("--out", default="frames", help="Output directory")
    ap.add_argument("--every", type=float, default=1.0, help="Seconds between frames")
    ap.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames")
    args = ap.parse_args(argv)
    extract(Path(args.source), Path(args.out), args.every, args.max_frames)


if __name__ == "__main__":
    main()
