from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _default_data_yaml() -> str:
    local = Path(__file__).with_name("data.yaml")
    if local.exists():
        return str(local)
    # Fallback to your previous absolute path (works if it still exists)
    return r"C:\Users\Focus Computer\Downloads\shoplifting.v1i.yolov8\data.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description="Train a YOLOv8 model for shoplifter detection.")
    p.add_argument("--data", default=_default_data_yaml(), help="Path to YOLO data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="Base model or pretrained weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cpu", help="cpu, 0, 0,1,2,3 etc.")
    p.add_argument("--project", default=str(Path(__file__).with_name("output")))
    p.add_argument("--name", default="train")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        # Augmentations (kept reasonable for CCTV/shop scenes)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,   # vertical flips usually unrealistic for CCTV
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,    # too much mixup can hurt box localization
    )


if __name__ == "__main__":
    main()