#!/usr/bin/env python3
"""
augment_dataset.py — Augment YOLOv8-OBB dataset using albumentations.

Run on HOST machine (not in container):
    pip install albumentations opencv-python
    python3 augment_dataset.py --input /path/to/worlds/dataset --factor 3

For each original training image, generates <factor> augmented copies.
Validation images are NOT augmented (correct ML practice).

Augmentations chosen for sim-to-real robustness:
  1. RandomBrightnessContrast  — Gazebo renders ideal lighting; real setups vary
  2. GaussNoise               — camera sensor noise
  3. MotionBlur               — camera/robot vibration
  4. SafeRotate ±5°           — imperfect camera mounting
  5. HueSaturationValue       — colour temperature variations

NOTE: YOLO OBB annotations use 4-corner format (8 normalised coords per box).
      albumentations handles keypoint-based augmentation — we treat each corner
      as a keypoint, augment, then re-assemble into OBB lines.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    raise SystemExit(
        "albumentations not found.\n"
        "Install:  pip install albumentations"
    )


# ── Compatibility helper ──────────────────────────────────────────────────────

def _gauss_noise() -> A.BasicTransform:
    """
    GaussNoise with albumentations 1.x / 2.x compatibility.
      1.x API: var_limit=(lo, hi)   — variance range in pixel units [0,255]
      2.x API: std_range=(lo, hi)   — std-dev range normalised [0, 1]
    """
    ver = tuple(int(x) for x in A.__version__.split(".")[:2])
    if ver >= (2, 0):
        return A.GaussNoise(std_range=(5.0 / 255, 25.0 / 255), p=0.5)
    else:
        # var = std² in pixel units; std ≈ 5–25 → var ≈ 25–625
        return A.GaussNoise(var_limit=(25.0, 625.0), p=0.5)


# ── Augmentation pipeline ─────────────────────────────────────────────────────

def build_pipeline(seed: int) -> A.Compose:
    """
    Build albumentations pipeline with keypoint support.
    Keypoints = OBB corners → transformed together with the image.
    """
    return A.Compose(
        [
            # ── Photometric ────────────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.30,
                contrast_limit=0.30,
                p=0.8,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5,
            ),
            _gauss_noise(),
            # ── Blur (motion / focus) ──────────────────────────────────────
            # blur_limit=3 → kernel size up to 3px (1–3px as specified)
            A.MotionBlur(blur_limit=3, p=0.3),
            # ── Geometric (small — camera is fixed overhead) ──────────────
            A.SafeRotate(
                limit=5,          # ±5 degrees — imperfect camera mount
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",          # (x, y) in pixels
            remove_invisible=False,
        ),
        seed=seed,
    )


# ── YOLO OBB I/O ──────────────────────────────────────────────────────────────

def read_label(path: Path, img_w: int, img_h: int
               ) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    Parse YOLO OBB label file.
    Returns list of (class_id, [(x,y), (x,y), (x,y), (x,y)]) in pixels.
    """
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        corners = [
            (coords[i] * img_w, coords[i + 1] * img_h)
            for i in range(0, 8, 2)
        ]
        boxes.append((class_id, corners))
    return boxes


def write_label(path: Path,
                boxes: List[Tuple[int, List[Tuple[float, float]]]],
                img_w: int, img_h: int) -> None:
    """Write YOLO OBB label file from (class_id, corners_px) list."""
    lines = []
    for class_id, corners in boxes:
        pts = " ".join(
            f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in corners
        )
        lines.append(f"{class_id} {pts}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# ── Per-image augmentation ────────────────────────────────────────────────────

def augment_one(img: np.ndarray,
                boxes: List[Tuple[int, List[Tuple[float, float]]]],
                pipeline: A.Compose,
                ) -> Tuple[np.ndarray, List[Tuple[int, List[Tuple[float, float]]]]]:
    """
    Apply pipeline to image and its OBB boxes.
    Returns (augmented_image, augmented_boxes).
    """
    # Flatten all corners into a single keypoints list,
    # keeping track of which box each corner belongs to.
    keypoints = []
    for _, corners in boxes:
        keypoints.extend(corners)

    result = pipeline(image=img, keypoints=keypoints)
    aug_img   = result["image"]
    aug_kpts  = result["keypoints"]

    # Rebuild boxes from augmented keypoints (4 corners per box)
    aug_boxes = []
    idx = 0
    for class_id, _ in boxes:
        new_corners = list(aug_kpts[idx: idx + 4])
        # Clamp to image bounds
        h, w = aug_img.shape[:2]
        new_corners = [
            (max(0.0, min(float(x), w - 1)),
             max(0.0, min(float(y), h - 1)))
            for x, y in new_corners
        ]
        aug_boxes.append((class_id, new_corners))
        idx += 4

    return aug_img, aug_boxes


# ── Main ──────────────────────────────────────────────────────────────────────

def augment_dataset(input_dir: Path, factor: int, seed: int) -> None:
    train_img_dir = input_dir / "images" / "train"
    train_lbl_dir = input_dir / "labels" / "train"

    if not train_img_dir.exists():
        raise SystemExit(f"Train images not found: {train_img_dir}")

    images = sorted(train_img_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"No .png files found in {train_img_dir}")

    print(f"Found {len(images)} training images → generating ×{factor} augmented copies")

    rng = random.Random(seed)
    generated = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: cannot read {img_path.name} — skipping")
            continue

        h, w = img.shape[:2]
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        boxes = read_label(lbl_path, w, h)

        for k in range(factor):
            aug_seed = rng.randint(0, 2 ** 31)
            pipeline = build_pipeline(aug_seed)
            aug_img, aug_boxes = augment_one(img, boxes, pipeline)

            stem = f"{img_path.stem}_aug{k:02d}"
            out_img = train_img_dir / f"{stem}.png"
            out_lbl = train_lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(out_img), aug_img)
            write_label(out_lbl, aug_boxes, w, h)
            generated += 1

        if generated % 200 == 0:
            print(f"  Generated {generated} augmented images …")

    total = len(list(train_img_dir.glob("*.png")))
    print(f"\nDone!  augmented={generated}  total train images={total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment YOLOv8-OBB training dataset with albumentations"
    )
    parser.add_argument(
        "--input",  type=str, default="./dataset",
        help="Path to dataset root (must contain images/train, labels/train)",
    )
    parser.add_argument(
        "--factor", type=int, default=3,
        help="Number of augmented copies per original image (default: 3)",
    )
    parser.add_argument(
        "--seed",   type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    augment_dataset(Path(args.input), args.factor, args.seed)


if __name__ == "__main__":
    main()
