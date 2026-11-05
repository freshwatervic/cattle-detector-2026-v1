"""Tiled animal detection for drone photos and video.

Drone frames are large (often 4000+ px wide) while a YOLO model works at about
640 px. Shrinking a whole frame to that size turns a cow into a handful of
pixels, so instead we cut the frame into overlapping tiles, detect on each tile
at native resolution, shift the boxes back into full-frame coordinates and merge
duplicates with non-maximum suppression (NMS).

Usage:
    cow-detect --source photo.jpg --out results/
    cow-detect --source flight.mp4 --out results/ --frame-skip 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .common import DEFAULT_WEIGHTS, IMAGE_EXTS, VIDEO_EXTS, load_detector

# COCO class ids that are animals. Only used for a *COCO* model (80 classes), so
# a run over farmland isn't full of "car" / "person" noise.
COCO_ANIMAL_CLASSES = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}
COCO_NUM_CLASSES = 80

TILE_SIZE = 640      # YOLO's native training resolution
TILE_OVERLAP = 0.2   # 20% overlap so animals on tile borders aren't cut in half


def class_labels(model) -> dict[int, str]:
    """Return {class_id: label} for the classes we should keep.

    A COCO model is filtered down to animals. A fine-tuned model (e.g. our
    single-class cow detector, where cow == class 0) keeps all of its classes.
    """
    names = dict(model.names)
    return COCO_ANIMAL_CLASSES if len(names) == COCO_NUM_CLASSES else names


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile start offsets along one axis, always ending flush with the far edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_image(img, tile_size: int = TILE_SIZE, overlap: float = TILE_OVERLAP):
    """Split an image into overlapping tiles: [(tile, x_offset, y_offset), ...].

    Every pixel is covered by at least one tile, including the bottom-right corner.
    """
    h, w = img.shape[:2]
    stride = max(int(tile_size * (1 - overlap)), 1)
    return [
        (img[y:y + tile_size, x:x + tile_size], x, y)
        for y in _axis_starts(h, tile_size, stride)
        for x in _axis_starts(w, tile_size, stride)
    ]


def iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_merge(detections, iou_thresh: float = 0.5):
    """Drop duplicate boxes (the same cow seen in two overlapping tiles).

    Keeps the highest-confidence box and discards any box overlapping it by
    more than `iou_thresh`.
    """
    keep = []
    for det in sorted(detections, key=lambda d: d["conf"], reverse=True):
        if all(iou(det["xyxy"], k["xyxy"]) < iou_thresh for k in keep):
            keep.append(det)
    return keep


def detect_on_image(model, img, conf_thresh: float = 0.25):
    """Run tiled detection on one BGR image.

    Returns [{"xyxy": (x1, y1, x2, y2), "conf": float, "label": str}, ...] in
    full-image pixel coordinates.
    """
    labels = class_labels(model)
    detections = []
    for tile, x_off, y_off in tile_image(img):
        for result in model.predict(tile, verbose=False, conf=conf_thresh):
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in labels:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "xyxy": (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off),
                    "conf": float(box.conf[0]),
                    "label": labels[cls_id],
                })
    return nms_merge(detections)


def draw_detections(img, detections):
    """Return a copy of `img` with a labelled green box per detection."""
    out = img.copy()
    thickness = max(2, round(max(out.shape[:2]) / 1000))
    for d in detections:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(out, f"{d['label']} {d['conf']:.2f}", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, (0, 255, 0), thickness)
    return out


def process_image(model, src: Path, out_dir: Path, conf: float = 0.25):
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Could not read image: {src}")
    detections = detect_on_image(model, img, conf)
    out_path = out_dir / f"annotated_{src.name}"
    cv2.imwrite(str(out_path), draw_detections(img, detections))
    print(f"[{src.name}] {len(detections)} detection(s) -> {out_path}")
    for d in detections:
        print(f"    {d['label']:10s} conf={d['conf']:.2f} box={tuple(round(v) for v in d['xyxy'])}")
    return detections


def process_video(model, src: Path, out_dir: Path, frame_skip: int = 5, conf: float = 0.25):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_path = out_dir / f"annotated_{src.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    frame_idx, total = 0, 0
    detections = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_skip == 0:          # detect every Nth frame, reuse boxes in between
            detections = detect_on_image(model, frame, conf)
            total += len(detections)
        writer.write(draw_detections(frame, detections))
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[{src.name}] {frame_idx} frames, {total} detection-events -> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Detect animals in a drone photo or video (tiled).")
    ap.add_argument("--source", required=True, help="Image or video file")
    ap.add_argument("--out", default="results", help="Output directory")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help="YOLO weights (default: weights/best.pt; use yolov8s.pt for the plain COCO model)")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    ap.add_argument("--frame-skip", type=int, default=5, help="Detect every N frames (video only)")
    args = ap.parse_args(argv)

    src, out_dir = Path(args.source), Path(args.out)
    if src.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
        raise SystemExit(f"Unsupported file type: {src.suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_detector(args.weights)

    if src.suffix.lower() in IMAGE_EXTS:
        process_image(model, src, out_dir, args.conf)
    else:
        process_video(model, src, out_dir, args.frame_skip, args.conf)


if __name__ == "__main__":
    main()
