"""Build a YOLO training set for cows from the public WAID aerial dataset.

WAID (Wildlife Aerial Images from Drone) has thousands of labelled top-down
drone images. A model pretrained on COCO has mostly seen animals from the side,
which is why it calls aerial cows "bear" or "dog". We therefore:

  * train on WAID's `cattle` images, relabelled as a single class `cow`;
  * keep real-world images of your own as a separate TEST set, so the numbers
    answer "how well does this work on MY field?" rather than flattering the
    model by testing on images it trained on.

Output layout (standard YOLO format):

    dataset/
        data.yaml
        train/{images,labels}   <- WAID train
        valid/{images,labels}   <- WAID valid (used while training)
        test/{images,labels}    <- your own images, or WAID test if you give none

Usage:
    git clone https://github.com/xiaohuicui/WAID.git
    cow-prepare-data --waid ./WAID --out ./dataset
    cow-prepare-data --waid ./WAID --my-data ./my_labelled_images --out ./dataset
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Class order of WAID's label files (index == class id).
WAID_CLASSES = ["sheep", "cattle", "seal", "camelus", "kiang", "zebra"]
COW_SOURCE_CLASSES = {"cattle"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def read_yolo_label(path: Path):
    """Return [(class_id, x, y, w, h), ...] from a YOLO label file."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append((int(parts[0]), *[float(v) for v in parts[1:5]]))
    return rows


def write_yolo_label(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))


def find_image_for_label(label_path: Path, image_dirs):
    """Find the image that belongs to a label file (same stem, any common extension)."""
    for directory in image_dirs:
        for ext in IMAGE_SUFFIXES:
            candidate = directory / f"{label_path.stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def remap_row(row, keep_all: bool):
    """Map a WAID label row to our class scheme, or return None to drop it."""
    cls_id = row[0]
    if keep_all:
        return row
    name = WAID_CLASSES[cls_id] if 0 <= cls_id < len(WAID_CLASSES) else None
    return (0, *row[1:]) if name in COW_SOURCE_CLASSES else None


def process_waid_split(waid_root: Path, split: str, out_root: Path, dst_split: str, keep_all: bool) -> int:
    """Copy one WAID split into the output dataset. Returns the number of images kept."""
    # The repo nests as WAID/WAID/{images,labels}; accept the flat layout too.
    for base in (waid_root / "WAID", waid_root):
        label_dir, image_dir = base / "labels" / split, base / "images" / split
        if label_dir.exists():
            break
    else:
        print(f"  [warn] WAID '{split}' labels not found under {waid_root} - skipping")
        return 0

    out_img, out_lbl = out_root / dst_split / "images", out_root / dst_split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    kept = 0
    for label_path in label_dir.glob("*.txt"):
        rows = [r for r in (remap_row(r, keep_all) for r in read_yolo_label(label_path)) if r]
        if not rows:
            continue  # image has none of our target classes
        img_path = find_image_for_label(label_path, [image_dir])
        if img_path is None:
            continue
        shutil.copy(img_path, out_img / img_path.name)
        write_yolo_label(out_lbl / label_path.name, rows)
        kept += 1
    return kept


def process_my_data(my_root: Path, out_root: Path, keep_all: bool) -> int:
    """Copy your own labelled images (YOLO format) into the test split."""
    ignore = {"classes.txt", "notes.txt", "README.txt"}
    label_files = [p for p in my_root.rglob("*.txt") if p.name not in ignore]
    if not label_files:
        print(f"  [warn] no label .txt files found under {my_root}")
        return 0

    out_img, out_lbl = out_root / "test" / "images", out_root / "test" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    image_dirs = list({p.parent for ext in ("jpg", "jpeg", "png") for p in my_root.rglob(f"*.{ext}")})

    kept = 0
    for label_path in label_files:
        rows = read_yolo_label(label_path)
        if not keep_all:
            rows = [(0, *r[1:]) for r in rows]   # your images are assumed to be cows already
        if not rows:
            continue
        img_path = find_image_for_label(label_path, image_dirs)
        if img_path is None:
            continue
        shutil.copy(img_path, out_img / img_path.name)
        write_yolo_label(out_lbl / label_path.name, rows)
        kept += 1
    return kept


def write_data_yaml(out_root: Path, keep_all: bool) -> None:
    names = WAID_CLASSES if keep_all else ["cow"]
    (out_root / "data.yaml").write_text(
        f"train: {(out_root / 'train' / 'images').resolve()}\n"
        f"val: {(out_root / 'valid' / 'images').resolve()}\n"
        f"test: {(out_root / 'test' / 'images').resolve()}\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )


def _clear_previous_build(out_root: Path) -> None:
    """Remove an earlier dataset build, but never an arbitrary folder.

    Deleting `--out` blindly would wipe whatever the user pointed it at (say, `.`),
    so we only delete a folder that is empty or already looks like one of our builds.
    """
    if not out_root.exists():
        return
    if any(out_root.iterdir()) and not (out_root / "data.yaml").exists():
        raise SystemExit(
            f"Refusing to overwrite {out_root}: it isn't empty and doesn't look like a "
            "previous dataset build (no data.yaml). Pick another --out or remove it yourself."
        )
    print(f"Output {out_root} already holds a dataset - rebuilding it.")
    shutil.rmtree(out_root)


def build_dataset(waid: Path, out: Path, my_data: Path | None = None, keep_all: bool = False) -> dict:
    """Build the dataset and return image counts per split."""
    waid, out = Path(waid), Path(out)
    _clear_previous_build(out)

    counts = {
        "train": process_waid_split(waid, "train", out, "train", keep_all),
        "valid": process_waid_split(waid, "valid", out, "valid", keep_all),
        "test": 0,
    }
    if my_data:
        counts["test"] = process_my_data(Path(my_data), out, keep_all)
    if counts["test"] == 0:   # no own images: fall back to WAID's test split
        counts["test"] = process_waid_split(waid, "test", out, "test", keep_all)

    write_data_yaml(out, keep_all)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a YOLO cow dataset from WAID.")
    ap.add_argument("--waid", required=True, help="Path to the cloned WAID repo")
    ap.add_argument("--my-data", default=None, help="Your own labelled images (YOLO format) -> test split")
    ap.add_argument("--out", default="dataset", help="Output dataset folder")
    ap.add_argument("--keep-all-classes", action="store_true",
                    help="Keep all 6 WAID species instead of collapsing to one 'cow' class")
    args = ap.parse_args(argv)

    counts = build_dataset(Path(args.waid), Path(args.out), args.my_data, args.keep_all_classes)
    print("Images per split:", counts)
    print(f"\nDone. Train with:\n    cow-train --data {Path(args.out) / 'data.yaml'}")


if __name__ == "__main__":
    main()
