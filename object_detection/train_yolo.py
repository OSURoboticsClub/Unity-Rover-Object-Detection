import os
import random
import numpy as np
import albumentations as A
import ultralytics.data.augment as aug_module
from ultralytics import YOLO


class RobustAlbumentations:
    """Custom Albumentations pipeline targeting partial occlusion, angle variation, and lighting changes."""

    def __init__(self, p: float = 1.0) -> None:
        self.p = p
        # contains_spatial=True tells Ultralytics to pass bboxes into the transform
        self.contains_spatial = True

        T = [
            # --- Lighting & color ---
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.55),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=35, val_shift_limit=25, p=0.45),
            A.RandomGamma(gamma_limit=(60, 140), p=0.3),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.25),

            # --- Blur & noise (motion, focus, sensor noise) ---
            A.OneOf([
                A.Blur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=9, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            A.GaussNoise(std_range=(0.04, 0.15), p=0.25),
            A.ImageCompression(quality_range=(60, 100), p=0.15),

            # --- Occlusion / partial obscurity ---
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                shadow_intensity_range=(0.3, 0.7),
                p=0.35,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(10, 50),
                hole_width_range=(10, 50),
                fill=0,
                p=0.25,
            ),

            # --- Perspective / angle changes ---
            A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.3),

            # --- Weather & environment ---
            A.RandomFog(fog_coef_range=(0.05, 0.25), alpha_coef=0.1, p=0.15),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_radius=120,
                angle_range=(0, 1),
                p=0.08,
            ),
            A.RandomRain(
                slant_range=(-10, 10),
                drop_length=15,
                drop_width=1,
                brightness_coefficient=0.9,
                p=0.1,
            ),

            # --- Grayscale edge case ---
            A.ToGray(p=0.02),
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

            new = self.transform(image=im, bboxes=bboxes, class_labels=cls)

            if len(new["class_labels"]) > 0:
                labels["img"] = new["image"]
                labels["cls"] = np.array(new["class_labels"])
                bboxes = np.array(new["bboxes"], dtype=np.float32)
            labels["instances"].update(bboxes=bboxes)

        return labels


# Patch before YOLO initializes its dataset pipeline
aug_module.Albumentations = RobustAlbumentations

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

model = YOLO("yolo11n.pt")

train_results = model.train(
    data="dataset_all/data.yaml",
    epochs=100,
    imgsz=640,
    device="cpu",
    # Built-in geometric augmentations
    degrees=15,           # random rotation ±15°
    translate=0.1,        # random translation ±10%
    scale=0.5,            # random scale 50–150%
    shear=5,              # random shear ±5°
    perspective=0.0005,   # slight perspective distortion
    flipud=0.1,           # vertical flip 10%
    fliplr=0.5,           # horizontal flip 50%
    # Mosaic / mix augmentations
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    # HSV color jitter (complements Albumentations color augments)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)

metrics = model.val()

results = model("mallet1.jpg")
results[0].show()

path = model.export(format="onnx")
