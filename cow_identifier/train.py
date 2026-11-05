"""Fine-tune a YOLOv8 detector on the cow dataset.

Starts from the COCO-pretrained weights, so the model keeps its general sense of
animal shapes and only has to learn what cows look like from above.

Usage:
    cow-train --data dataset/data.yaml --epochs 20
    cow-train --data dataset/data.yaml --epochs 20 --device mps   # Apple Silicon GPU

The best weights are written to runs/detect/<name>/weights/best.pt.
"""

from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8 on the cow dataset.")
    ap.add_argument("--data", default="dataset/data.yaml", help="data.yaml from cow-prepare-data")
    ap.add_argument("--base-model", default="yolov8s.pt", help="Starting weights (downloaded if missing)")
    ap.add_argument("--epochs", type=int, default=20, help="Training epochs")
    ap.add_argument("--imgsz", type=int, default=640, help="Training image size")
    ap.add_argument("--batch", type=int, default=8, help="Batch size (lower it if you run out of memory)")
    ap.add_argument("--name", default="cow_model", help="Run name (output folder under runs/detect/)")
    ap.add_argument("--device", default=None, help="'cpu', 'mps' (Apple Silicon), '0' (first GPU); default: auto")
    args = ap.parse_args(argv)

    from ultralytics import YOLO  # heavy import, so only when actually training

    model = YOLO(args.base_model)
    print(f"Fine-tuning {args.base_model} on {args.data} for {args.epochs} epochs...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        patience=15,   # stop early if validation stops improving
    )

    print(f"\nDone. Weights: runs/detect/{args.name}/weights/best.pt")
    print("Copy that file to weights/best.pt so the other commands find it.")


if __name__ == "__main__":
    main()
