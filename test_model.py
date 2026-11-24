from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


ALIASES = {
    "suspito": "suspicious",
    "suspito.": "suspicious",
    "suspeito": "suspicious",
    "suspeitos": "suspicious",
    "suspicious": "suspicious",
    "shoplifter": "shoplifter",
}


def _canonical_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    return ALIASES.get(key, raw)


def _result_to_log_row(r: Any) -> Dict[str, Any]:
    names = getattr(r, "names", {}) or {}
    boxes = getattr(r, "boxes", None)
    out: Dict[str, Any] = {
        "source": str(getattr(r, "path", "")),
        "shape": list(getattr(r, "orig_shape", []) or []),
        "detections": [],
    }
    if boxes is None:
        return out

    xyxy = boxes.xyxy.tolist() if getattr(boxes, "xyxy", None) is not None else []
    conf = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
    cls = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []

    for i in range(len(xyxy)):
        class_id = int(cls[i]) if i < len(cls) else -1
        raw_name = names.get(class_id, str(class_id))
        out["detections"].append(
            {
                "class_id": class_id,
                "class_name": _canonical_label(str(raw_name)),
                "confidence": float(conf[i]) if i < len(conf) else None,
                "xyxy": [float(v) for v in xyxy[i]],
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run inference and save outputs + detection logs.",
        epilog="Live preview uses OpenCV (cv2.imshow). Use --no-show for a faster run with no window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--weights",
        default=str(
            Path(__file__).with_name("runs")
            / "detect"
            / "train8"
            / "weights"
            / "best.pt"
        ),
        help="Path to trained YOLO weights.",
    )
    p.add_argument(
        "--source",
        default=r"C:\Users\Focus Computer\Downloads\demo.mp4",
        help="Path to an image/video, a webcam index, or a media URL.",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument(
        "--show",
        "--view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Live OpenCV window during inference (default: on). Use --no-show or --no-view to disable.",
    )
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--project", default=str(Path(__file__).with_name("output")))
    p.add_argument("--name", default="detect")
    args = p.parse_args()

    if (
        not Path(args.source).exists()
        and not args.source.isdecimal()
        and "://" not in args.source
    ):
        p.error(
            f"Source does not exist: {args.source!r}. "
            "Pass an existing image or video with --source."
        )

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        save=args.save,
        project=args.project,
        name=args.name,
    )

    # Save a structured log alongside Ultralytics saved media.
    log_dir = Path(args.project) / "detection_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"detections_{stamp}.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(_result_to_log_row(r), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()