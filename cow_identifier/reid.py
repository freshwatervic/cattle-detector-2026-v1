"""Identify INDIVIDUAL cows by matching drone detections against a gallery.

Stage 2 of the pipeline (stage 1 is the detector):

  1. the detector finds and crops every cow in a drone image
  2. each crop becomes a "fingerprint" (an embedding vector)
  3. the fingerprint is compared with a gallery of known cows; the closest wins

The gallery is one folder per cow; the folder name is the cow's name:

    looking_for/
        Bella/   bella_left.jpg  bella_back.jpg
        Greta/   greta_1.jpg     greta_2.jpg

Several photos per cow are averaged into one fingerprint (more photos = more robust).

HONESTY NOTE
------------
The fingerprint network is a generic ImageNet ResNet50, NOT trained on cows. It
may or may not separate your cows, especially across the ground-photo vs
top-down-drone viewpoint gap. That is why every result carries the full ranking
and the margin between 1st and 2nd place: a clear gap means it works, bunched
scores mean it is guessing and you need top-down reference photos.

Usage:
    cow-reid --image photos/drone.jpg --gallery looking_for --out identified.jpg
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .common import DEFAULT_WEIGHTS, IMAGE_EXTS, load_detector

UNCERTAIN = "uncertain"
INPUT_SIZE = 224   # ResNet50's expected input


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _preprocess():
    # torchvision is imported lazily: `import cow_identifier` stays light and the
    # pure-numpy logic below can be tested without PyTorch installed.
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def open_rgb(path) -> Image.Image:
    """Open an image as RGB with its EXIF rotation applied.

    Phone photos are often stored sideways with an orientation flag. Without
    applying it, a reference photo would be fingerprinted rotated 90 degrees.
    """
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def load_embedder(pretrained: bool = True):
    """ResNet50 with the classifier removed: any image -> 2048-number fingerprint.

    `pretrained=False` skips the ImageNet download (only useful for plumbing tests).
    """
    from torchvision import models
    import torch

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    return model.eval()


def embed_pil(model, pil_img: Image.Image) -> np.ndarray:
    """Fingerprint of one PIL image, scaled to unit length (so dot product = cosine similarity)."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        x = _preprocess()(pil_img.convert("RGB")).unsqueeze(0)
        feat = F.normalize(model(x), dim=1)
    return feat.squeeze(0).numpy()


# --------------------------------------------------------------------------- #
# Gallery + matching
# --------------------------------------------------------------------------- #
def build_gallery(model, looking_for_dir, verbose: bool = True) -> dict:
    """One averaged fingerprint per cow folder.

    Returns {cow_name: {"embedding": vector, "n_photos": int}}.
    """
    root = Path(looking_for_dir)
    if not root.exists():
        raise FileNotFoundError(f"Gallery folder not found: {root}")

    gallery = {}
    for cow_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        vectors = [embed_pil(model, open_rgb(p))
                   for p in sorted(cow_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        if not vectors:
            continue
        mean = np.mean(vectors, axis=0)
        gallery[cow_dir.name] = {"embedding": mean / (np.linalg.norm(mean) + 1e-9),
                                 "n_photos": len(vectors)}
        if verbose:
            print(f"  gallery: {cow_dir.name:20s} ({len(vectors)} photo(s))")

    if not gallery:
        raise ValueError(f"No cow sub-folders with images found in {root}. "
                         "Expected one folder per cow, each containing photos.")
    return gallery


def rank_matches(query: np.ndarray, gallery: dict) -> list[tuple[str, float]]:
    """Gallery cows sorted by cosine similarity to `query`, best first."""
    ranked = [(name, float(np.dot(query, data["embedding"]))) for name, data in gallery.items()]
    return sorted(ranked, key=lambda t: t[1], reverse=True)


def score_margin(ranked) -> float:
    """Gap between the best and second-best score. A one-cow gallery has no rival (inf)."""
    return ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")


def decide_label(ranked, match_threshold: float = 0.0, min_margin: float = 0.0) -> str:
    """Name of the best match, or 'uncertain' if its score or its lead is too small."""
    name, score = ranked[0]
    if score < match_threshold or score_margin(ranked) < min_margin:
        return UNCERTAIN
    return name


def match_crop(model, crop: Image.Image, gallery: dict):
    """Rank every gallery cow against one cow crop."""
    return rank_matches(embed_pil(model, crop), gallery)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def identify_cows_in_image(detector, embedder, gallery, image_path,
                           conf: float = 0.4, match_threshold: float = 0.0,
                           min_margin: float = 0.0):
    """Detect every cow in `image_path` and match each against the gallery.

    `detector` may be a loaded YOLO model or a path to weights.
    Returns one dict per cow:
        {"box": (x1, y1, x2, y2), "best_match": name | 'uncertain',
         "score": float, "margin": float, "ranked": [(name, score), ...]}
    Leave both thresholds at 0 to always get a name; raise them once you have seen
    real scores to flag weak matches as 'uncertain'.
    """
    if isinstance(detector, (str, Path)):
        detector = load_detector(detector)

    image = open_rgb(image_path)
    # Detect on the same (EXIF-corrected) pixels we crop from, so boxes line up.
    boxes = detector.predict(image, conf=conf, verbose=False)[0].boxes

    found = []
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        ranked = match_crop(embedder, image.crop((x1, y1, x2, y2)), gallery)
        found.append({
            "box": (x1, y1, x2, y2),
            "best_match": decide_label(ranked, match_threshold, min_margin),
            "score": ranked[0][1],
            "margin": score_margin(ranked),
            "ranked": ranked,
        })
    return found


def draw_identifications(image_bgr, found):
    """Copy of the image with a box + 'Name score' per cow (orange if uncertain)."""
    out = image_bgr.copy()
    thickness = max(2, round(max(out.shape[:2]) / 1000))
    for cow in found:
        x1, y1, x2, y2 = cow["box"]
        colour = (0, 165, 255) if cow["best_match"] == UNCERTAIN else (0, 255, 0)   # BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(out, f'{cow["best_match"]} {cow["score"]:.2f}', (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, colour, thickness)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Identify individual cows in a drone image.")
    ap.add_argument("--image", required=True, help="Drone image to analyse")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Trained cow detector (best.pt)")
    ap.add_argument("--gallery", default="looking_for", help="Folder with one sub-folder per known cow")
    ap.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    ap.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity to accept a match")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="Minimum lead of 1st over 2nd place to accept a match")
    ap.add_argument("--out", default=None, help="Optional path to save the annotated image")
    args = ap.parse_args(argv)

    print("Loading fingerprint model (downloads ResNet50 the first time)...")
    embedder = load_embedder()
    print("Building gallery from", args.gallery)
    gallery = build_gallery(embedder, args.gallery)

    print(f"\nAnalysing {args.image} ...")
    found = identify_cows_in_image(args.weights, embedder, gallery, args.image,
                                   args.conf, args.threshold, args.min_margin)
    print(f"\n{len(found)} cow(s) detected:\n")
    for i, cow in enumerate(found, 1):
        top3 = ", ".join(f"{n}:{s:.2f}" for n, s in cow["ranked"][:3])
        print(f"  Cow #{i} at {cow['box']}")
        print(f"      best match: {cow['best_match']}  (score {cow['score']:.3f}, margin {cow['margin']:.3f})")
        print(f"      top matches: {top3}")

    if args.out:
        image = cv2.imread(args.image)
        cv2.imwrite(args.out, draw_identifications(image, found))
        print(f"\nAnnotated image saved to {args.out}")


if __name__ == "__main__":
    main()
