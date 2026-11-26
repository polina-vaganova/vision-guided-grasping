#!/usr/bin/env python3
"""
infer_gazebo_line.py — прогоняет best.pt на кадре с линией коробок из Gazebo.
Перед инференсом конвертирует RGB→grayscale-3ch (как в yolo_detector.py).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "runs/obb/stereo_v1/weights/best.pt"

model = YOLO(str(MODEL))

for cam in ("left", "right"):
    src = ROOT / f"inference_demo/gazebo_line_{cam}.png"
    if not src.exists():
        print(f"нет {src}"); continue

    # RGB → grayscale-3ch (идентично inference-пайплайну yolo_detector.py)
    img_bgr = cv2.imread(str(src))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    gray_path = ROOT / f"inference_demo/gazebo_line_{cam}_gray.png"
    cv2.imwrite(str(gray_path), gray3)

    results = model.predict(gray3, conf=0.5, save=False, verbose=False, imgsz=640)
    r = results[0]
    n_det = len(r.obb) if r.obb is not None else 0
    annotated = r.plot()
    out_path = ROOT / f"inference_demo/gazebo_line_{cam}_detected.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"[{cam}] {n_det} детекций → {out_path.name}")
