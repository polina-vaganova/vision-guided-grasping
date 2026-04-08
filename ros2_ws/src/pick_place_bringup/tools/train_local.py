#!/usr/bin/env python3
"""
train_local.py — YOLOv8-OBB training on macOS (MPS / CPU).

Usage (Python 3.12 recommended):
    python3 train_local.py

After training, best.pt is copied to:
    /Users/polina/Documents/Stankin_3/Diplom/worlds/models/yolo_boxes.pt
"""

import os
import shutil
from pathlib import Path

# ── MPS stability env vars (must be set before torch import) ──────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

from ultralytics import YOLO  # noqa: E402  (import after env setup)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_YAML = Path(
    "/Users/polina/Documents/Stankin_3/Diplom/worlds/dataset/dataset.yaml"
)
RUNS_DIR     = Path("/Users/polina/Documents/Stankin_3/Diplom/runs")
RUN_NAME     = "boxes_obb"
DEST_MODEL   = Path(
    "/Users/polina/Documents/Stankin_3/Diplom/worlds/models/yolo_boxes.pt"
)

assert DATASET_YAML.exists(), f"dataset.yaml not found: {DATASET_YAML}"

# ── Training ──────────────────────────────────────────────────────────────────
model = YOLO("yolov8n-obb.pt")

model.train(
    data    = str(DATASET_YAML),
    epochs  = 50,
    imgsz   = 416,
    batch   = 16,         # safe for 16 GB unified memory
    device  = "mps",      # Apple Silicon GPU; falls back to cpu automatically
    workers = 0,          # required for MPS stability (no fork)
    amp     = False,      # disable AMP — avoids MPS precision issues
    project = str(RUNS_DIR),
    name    = RUN_NAME,
    patience= 15,
    # Augmentation (light — heavy aug is already done by augment_dataset.py)
    hsv_h   = 0.010,
    hsv_s   = 0.2,
    hsv_v   = 0.15,
    degrees = 2.0,
    flipud  = 0.5,
    fliplr  = 0.5,
    mosaic  = 0.3,
    plots   = True,
)

# ── Copy best.pt to models dir ────────────────────────────────────────────────
best_src = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
if best_src.exists():
    DEST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_src, DEST_MODEL)
    print(f"\nSaved: {DEST_MODEL} ✓")
else:
    print(f"\nWARNING: best.pt not found at {best_src}")
    print("Check runs/ directory manually.")
