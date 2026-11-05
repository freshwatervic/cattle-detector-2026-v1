"""Small helpers shared by the detection and re-identification modules."""

from __future__ import annotations

from pathlib import Path

# Where the trained detector is expected by default. `cow-train` writes to
# runs/detect/cow_model/weights/best.pt; copy that file here (or download it
# from the GitHub release) so every command works without extra flags.
DEFAULT_WEIGHTS = Path("weights/best.pt")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def load_detector(weights):
    """Load a YOLO detector, with a readable error if the weights file is missing.

    A bare model name such as "yolov8s.pt" is passed straight to Ultralytics,
    which downloads it on first use. Anything with a folder in the path must
    already exist.
    """
    from ultralytics import YOLO  # imported here so tests / light imports stay fast

    weights = Path(weights)
    if weights.parent != Path(".") and not weights.exists():
        raise FileNotFoundError(
            f"Detector weights not found: {weights}\n"
            "Train them with `cow-train`, or download best.pt from the GitHub "
            "release and place it at weights/best.pt."
        )
    return YOLO(str(weights))
