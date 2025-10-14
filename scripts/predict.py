#!/usr/bin/env python
"""Simple inference helper around Ultralytics YOLOv8."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on images or videos.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained weights (e.g. best.pt).")
    parser.add_argument("--source", type=str, required=True, help="Path to image/video or directory of inputs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu, 0, 0,1, etc.).")
    parser.add_argument("--save-txt", action="store_true", help="Save predictions to *.txt files.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in output txt files.")
    parser.add_argument("--save", type=Path, default=None, help="Optional directory to copy prediction artifacts.")
    parser.add_argument("--show", action="store_true", help="Display predictions in a window (if supported).")
    parser.add_argument("--stream", action="store_true", help="Stream results via generator mode.")
    return parser.parse_args()


def run_prediction(args: argparse.Namespace) -> Dict[str, Any]:
    model = YOLO(str(args.weights))

    predict_kwargs = dict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save is not None,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        stream=args.stream,
    )

    if args.save is not None:
        Path(args.save).mkdir(parents=True, exist_ok=True)
        predict_kwargs["project"] = str(args.save)
        predict_kwargs["name"] = "predict"
        predict_kwargs["exist_ok"] = True

    results = model.predict(**predict_kwargs)

    summary = {
        "num_predictions": len(results),
        "weights": str(args.weights),
        "source": args.source,
        "save_dir": str(getattr(results, "save_dir", args.save)) if hasattr(results, "save_dir") else str(args.save),
    }

    first_boxes = getattr(results[0], "boxes", None)
    if first_boxes is not None and first_boxes.data is not None:
        summary["mean_conf"] = float(first_boxes.conf.mean().item()) if hasattr(first_boxes, "conf") else None

    return summary


def main() -> None:
    args = parse_args()
    summary = run_prediction(args)

    print("Inference complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
