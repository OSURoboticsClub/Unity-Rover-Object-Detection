#!/usr/bin/env python3
"""
YOLO Hyperparameter Tuning Script
Usage: python tune_yolo.py --model yolo11n.pt --data dataset/data.yaml [options]
"""

import argparse
import os
import json
import torch
from datetime import datetime
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tune a YOLO model")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt file)")
    parser.add_argument("--data", required=True, help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per trial (default: 30)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of tuning iterations (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", default=None, help="Device: 0, cpu, etc. (auto-detected if omitted)")
    parser.add_argument("--project", default="runs/tune", help="Output project directory (default: runs/tune)")
    parser.add_argument("--name", default=None, help="Run name (auto-generated if omitted)")
    parser.add_argument("--optimizer", default="AdamW", choices=["AdamW", "Adam", "SGD", "auto"],
                        help="Optimizer (default: AdamW)")
    parser.add_argument("--space", default=None, help="Path to JSON file defining custom search space")
    return parser.parse_args()


def load_search_space(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def detect_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    device = detect_device(args.device)
    run_name = args.name or f"tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.data}")
    print(f"Device:     {device}")
    print(f"Epochs/trial: {args.epochs}  |  Iterations: {args.iterations}")
    print(f"Output:     {args.project}/{run_name}")
    print()

    model = YOLO(args.model)

    tune_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        optimizer=args.optimizer,
        project=args.project,
        name=run_name,
        plots=True,
        save=True,
        val=True,
    )

    if args.space:
        tune_kwargs["space"] = load_search_space(args.space)
        print(f"Custom search space loaded from {args.space}")

    results = model.tune(**tune_kwargs)

    print("\nTuning complete.")
    best_path = os.path.join(args.project, run_name, "best_hyperparameters.yaml")
    if os.path.exists(best_path):
        print(f"Best hyperparameters saved to: {best_path}")

    return results


if __name__ == "__main__":
    main()
