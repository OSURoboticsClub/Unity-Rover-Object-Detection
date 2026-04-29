#!/usr/bin/env python3
"""
YOLO Training Script with robust augmentation for angle, distance, lighting, and occlusion.
Usage: python train_yolo.py --model yolo11n.pt --data dataset_all/data.yaml [options]
"""

import argparse
import os
import random
import numpy as np
import albumentations as A
import torch
import ultralytics.data.augment as aug_module
from datetime import datetime
from ultralytics import YOLO


class RobustAlbumentations:
    """Augmentation pipeline targeting angle variation, small/far objects, low light, and occlusion."""

    def __init__(self, p: float = 1.0, transforms=None) -> None:
        self.p = p
        # contains_spatial=True tells Ultralytics to pass bboxes into the transform
        self.contains_spatial = True

        T = [
            # --- Lighting: biased toward dark/dim to handle low-light failure cases ---
            A.RandomBrightnessContrast(
                brightness_limit=(-0.55, 0.2),   # heavily skewed dark
                contrast_limit=0.45,
                p=0.65,
            ),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=35, p=0.5),
            A.RandomGamma(gamma_limit=(35, 190), p=0.45),  # wide range: near-black to washed-out
            A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.3),

            # --- Heavy shadows: simulates one-sided lighting, dusk, or shade ---
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),               # full-frame shadows, not just bottom half
                num_shadows_limit=(1, 5),
                shadow_dimension=6,
                shadow_intensity_range=(0.4, 0.92),    # near-black intensity
                p=0.55,
            ),

            # --- Blur & noise (motion, out-of-focus, sensor) ---
            A.OneOf([
                A.Blur(blur_limit=7, p=1.0),
                A.MotionBlur(blur_limit=13, p=1.0),
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            ], p=0.45),
            A.GaussNoise(std_range=(0.04, 0.22), p=0.35),
            A.ImageCompression(quality_range=(35, 100), p=0.2),

            # --- Downscale: simulates far-away / low-resolution capture ---
            # Bottle appears small and blurry when detected from a distance
            A.Downscale(scale_range=(0.2, 0.65), p=0.4),

            # --- Occlusion / partial visibility ---
            A.CoarseDropout(
                num_holes_range=(1, 10),
                hole_height_range=(15, 90),
                hole_width_range=(15, 90),
                fill=0,
                p=0.4,
            ),

            # --- Angle & perspective variation ---
            A.Perspective(scale=(0.05, 0.22), keep_size=True, p=0.45),
            # Affine handles rotation (viewed from side/above) and scale variation
            A.Affine(
                rotate=(-50, 50),
                scale=(0.55, 1.45),
                p=0.5,
            ),

            # --- Weather & environmental obscurity ---
            A.RandomFog(fog_coef_range=(0.05, 0.4), alpha_coef=0.12, p=0.22),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_radius=140,
                angle_range=(0, 1),
                p=0.1,
            ),
            A.RandomRain(
                slant_range=(-15, 15),
                drop_length=18,
                drop_width=1,
                brightness_coefficient=0.85,
                p=0.12,
            ),

            # --- Grayscale / thermal-cam edge case ---
            A.ToGray(p=0.05),
        ]

        self.transform = A.Compose(
            T,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def __call__(self, labels: dict) -> dict:
        if self.transform is None or random.random() > self.p:
            return labels

        im = labels["img"]
        if im.shape[2] != 3:
            return labels

        cls = labels["cls"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes

            flat_cls = np.array(cls).flatten()
            new = self.transform(image=im, bboxes=bboxes, class_labels=flat_cls)

            if len(new["class_labels"]) > 0:
                labels["img"] = new["image"]
                new_cls = np.array(new["class_labels"])
                # ultralytics expects [N, 1] in newer versions; preserve original ndim
                if np.array(cls).ndim == 2:
                    new_cls = new_cls.reshape(-1, 1)
                labels["cls"] = new_cls
                bboxes = np.array(new["bboxes"], dtype=np.float32)
            labels["instances"].update(bboxes=bboxes)

        return labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model with robust augmentation")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model checkpoint (default: yolo11n.pt)")
    parser.add_argument("--data", default="dataset_all/data.yaml", help="Dataset YAML (default: dataset_all/data.yaml)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640). Use 512 or 416 on 8GB VRAM.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8). Mosaic x4 multiplies effective load; 16+ OOMs on 8GB.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers (default: 4). High values spike RAM with mosaic.")
    parser.add_argument("--device", default=None, help="Device: 0, cpu, etc. (auto-detected if omitted)")
    parser.add_argument("--project", default="runs/train", help="Output project directory (default: runs/train)")
    parser.add_argument("--name", default=None, help="Run name (auto-generated if omitted)")
    return parser.parse_args()


def detect_device(requested):
    if requested is not None:
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()
    device = detect_device(args.device)
    run_name = args.name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.data}")
    print(f"Device:   {device}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch}  (effective load = batch × 4 with mosaic)")
    print(f"ImgSz:    {args.imgsz}")
    print(f"Output:   {args.project}/{run_name}")
    print()

    # Patch before YOLO initializes its dataset pipeline
    aug_module.Albumentations = RobustAlbumentations
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=run_name,
        amp=True,            # fp16 mixed precision: cuts VRAM ~40-50%
        # --- Built-in geometric augmentations ---
        # NOTE: degrees/scale/perspective are intentionally disabled here.
        # RobustAlbumentations already applies A.Affine (rotate, scale) and
        # A.Perspective. Running both pipelines is redundant augmentation and
        # wastes CPU budget that increases effective batch prep time.
        degrees=0.0,
        scale=0.0,
        perspective=0.0,
        translate=0.15,
        shear=8,
        flipud=0.15,
        fliplr=0.5,
        # --- Mosaic / mix augmentations ---
        # mosaic=1.0 creates a 2×imgsz canvas from 4 images per sample — the
        # single biggest VRAM multiplier. keep it but trim the extras that also
        # load additional images per step.
        mosaic=1.0,
        close_mosaic=10,     # disable mosaic for final 10 epochs (default, made explicit)
        mixup=0.1,           # reduced from 0.2 — each mixup sample loads a second image
        copy_paste=0.1,      # reduced from 0.15 — same reason
        # --- HSV color jitter ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        # --- Output ---
        plots=True,
        save=True,
        val=True,
    )

    model.val()
    print(f"\nTraining complete. Best weights: {args.project}/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()
