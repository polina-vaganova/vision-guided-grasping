#!/usr/bin/env python3
"""
demo_inference.py — прогоняет best.pt на 4 «новых» кадрах:
  1-3. Три случайных кадра из test split (модель их не видела при обучении)
  4.   Синтетический кадр: коробки в линию на реальном фоне сцены
Сохраняет аннотированные картинки с bbox-ами и confidence.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Важно: torch до cv2, иначе конфликт libomp на macOS
import torch
torch.set_num_threads(1)

import random
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "obb" / "stereo_v1" / "weights" / "best.pt"
TEST_DIR = ROOT / "dataset_stereo_final" / "images" / "test"
OUT_DIR = ROOT / "inference_demo"
OUT_DIR.mkdir(exist_ok=True)

random.seed(7)

# ── 1-3. Три случайных тестовых кадра ────────────────────────────────────────
test_images = sorted(TEST_DIR.glob("*.png"))
random_tests = random.sample(test_images, 3)
print(f"Выбраны 3 test-кадра:")
for p in random_tests:
    print(f"  {p.name}")


# ── 4. Синтетический кадр «коробки в линию» ──────────────────────────────────
# Основа — существующий test-кадр, на его столешницу накладываем 5 коробок в ряд.
# Находим кадр с минимальным числом существующих коробок, чтобы меньше пересечений.
base_path = random.choice(test_images)
base = cv2.imread(str(base_path))
synth = base.copy()

# Table region in image coords (approximate, from camera pose):
#   центр стола ~ (180, 200), экстенты ~ ±120 по X, ±80 по Y
# Рисуем линию из 5 прямоугольников.
line_y = 220
line_x_start = 100
spacing = 50
for i in range(5):
    cx = line_x_start + i * spacing
    cy = line_y
    w = 40 + (i % 2) * 8
    h = 28 + (i % 3) * 4
    angle = (i - 2) * 4  # небольшие повороты
    rect = ((cx, cy), (w, h), angle)
    corners = cv2.boxPoints(rect).astype(np.int32)
    # Заливка: разные оттенки серого (имитируют grayscale от разных цветов)
    g = 80 + i * 28
    cv2.fillPoly(synth, [corners], (g, g, g))
    # Тёмная окантовка
    cv2.polylines(synth, [corners], True, (35, 35, 35), 2)

synth_path = OUT_DIR / "scene_synthetic_line.png"
cv2.imwrite(str(synth_path), synth)
print(f"Синтетический кадр: {synth_path.name}")


# ── Инференс ─────────────────────────────────────────────────────────────────
all_images = random_tests + [synth_path]
model = YOLO(str(MODEL_PATH))

print(f"\nИнференс через {MODEL_PATH.name}:")
for i, img_path in enumerate(all_images, 1):
    results = model.predict(str(img_path), conf=0.5, save=False, verbose=False, imgsz=640)
    r = results[0]
    n_det = len(r.obb) if r.obb is not None else 0
    annotated = r.plot()  # OpenCV BGR image with boxes + confidence
    out_name = f"detection_{i}_{img_path.stem}.png"
    out_path = OUT_DIR / out_name
    cv2.imwrite(str(out_path), annotated)
    print(f"  [{i}] {img_path.name:40s} → {n_det} детекций → {out_name}")

print(f"\nГотово. Смотри в {OUT_DIR}/")
