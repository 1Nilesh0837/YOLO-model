#!/usr/bin/env python
"""Evaluate a YOLOv8 checkpoint and generate simple plots."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model and create plots.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to YOLOv8 weights (best.pt).")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--save", type=Path, default=Path("plots"), help="Directory to save plots.")
    return parser.parse_args()


def run_validation(args: argparse.Namespace) -> Dict[str, float]:
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=True,
    )

    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def parse_training_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_file = run_dir / "results.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Could not find results.csv in {run_dir}")
    return pd.read_csv(metrics_file)


def plot_metrics(df: pd.DataFrame, save_dir: Path) -> List[Path]:
    plots: List[Path] = []
    save_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        ("train/box_loss", "train_box_loss.png", "Train Box Loss"),
        ("train/obj_loss", "train_obj_loss.png", "Train Objectness Loss"),
        ("metrics/precision(B)", "precision.png", "Precision"),
        ("metrics/recall(B)", "recall.png", "Recall"),
        ("metrics/mAP50(B)", "map50.png", "mAP@0.50"),
        ("metrics/mAP50-95(B)", "map50_95.png", "mAP@0.50:0.95"),
    ]

    if "val/box_loss" in df.columns:
        metric_columns.append(("val/box_loss", "val_box_loss.png", "Validation Box Loss"))

    for column, filename, title in metric_columns:
        if column not in df.columns:
            continue
        plt.figure()
        plt.plot(df.index, df[column])
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(column)
        plt.grid(True, linestyle="--", alpha=0.4)
        outfile = save_dir / filename
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close()
        plots.append(outfile)

    return plots


def main() -> None:
    args = parse_args()
    run_dir = Path(args.weights).resolve().parent
    save_dir = args.save

    metrics = run_validation(args)
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    try:
        df = parse_training_metrics(run_dir)
    except FileNotFoundError as err:
        print(err)
        return

    plot_paths = plot_metrics(df, save_dir)
    print("Saved plots:")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
