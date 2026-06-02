#!/usr/bin/env python3
"""
YOLO Training Script with robust augmentation for angle, distance, lighting, and occlusion.
Usage (example): 
source ~/hpc-share/venvs/rover_env/bin/activate
python train_yolo.py --model yolo26n.pt --data ../dataset/dataset_BOTTLE/data.yaml --name "BOTTLE-150EP" --project final_train --wandb
srun --account=eecs --partition=gpu --cpus-per-task=16 --mem=64G --gres=gpu:2 --pty bash
    

"""

import argparse
import os
import random
import numpy as np
#import albumentations as A
import torch
import ultralytics.data.augment as aug_module
from datetime import datetime
from ultralytics import YOLO
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model with robust augmentation")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model checkpoint (default: yolo11n.pt)")
    parser.add_argument("--data", default="dataset_all/data.yaml", help="Dataset YAML (default: dataset_all/data.yaml)")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs (default: 200)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640). Use 512 or 416 on 8GB VRAM.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (default: 8). Mosaic x4 multiplies effective load; 16+ OOMs on 8GB.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 4). High values spike RAM with mosaic.")
    parser.add_argument("--device", default=None, help="Device: 0, cpu, etc. (auto-detected if omitted)")
    parser.add_argument("--project", default="runs/train", help="Output project directory (default: runs/train)")
    parser.add_argument("--name", default=None, help="Run name (auto-generated if omitted)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="rover-object-detection")
    parser.add_argument("--wandb_entity", default=None, help="W&B entity/team (optional)")
    return parser.parse_args()


def detect_device(requested):
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    device = detect_device(args.device)
    run_name = args.name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    best_weights = None

    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.data}")
    print(f"Device:   {device}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch}  (effective load = batch × 4 with mosaic)")
    print(f"ImgSz:    {args.imgsz}")
    print(f"Output:   {args.project}/{run_name}")
    print()

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_NAME"] = run_name
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        # Make sure ultralytics wandb integration is on
        from ultralytics import settings
        settings.update({"wandb": True})


    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=[0,1],
        project=args.project,
        name=run_name,
        patience=50,
        optimizer="AdamW",
        amp=True,
        cos_lr=True, 
        warmup_epochs=5,        # prevent early high-LR damage
        warmup_bias_lr=0.01,
        lr0=0.001,              # lower initial LR for AdamW (default 0.01 is SGD-tuned)
        lrf=0.01,               # final LR ratio — don't decay too aggressively
        weight_decay=0.0005,


        # NOTE Change as needed 
        degrees=30.0,
        scale=0.5,
        perspective=0.001,
        translate=0.2,
        shear=2,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        erasing=0.1,
        close_mosaic=20,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        plots=True,
        save=True,
        val=True,
    )

    model.val()
    best_weights = f"{args.project}/{run_name}/weights/best.pt"
    print(f"\n[INFO] Training complete. Best weights: {best_weights}")


if __name__ == "__main__":
    main()
