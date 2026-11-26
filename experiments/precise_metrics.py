#!/usr/bin/env python3
"""precise_metrics.py — полные float-значения метрик best.pt на val и test."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "runs/obb/stereo_v1/weights/best.pt"
YAML = ROOT / "dataset_stereo_final/dataset.yaml"

model = YOLO(str(MODEL))

for split in ("val", "test"):
    print(f"\n========== best.pt on {split.upper()} ==========")
    m = model.val(data=str(YAML), split=split, plots=False, verbose=False,
                  name=f"_tmp_precise_{split}", exist_ok=True)
    # m.box предоставляет ВСЕ метрики в полной float-точности
    print(f"mAP50:      {m.box.map50!r}")
    print(f"mAP50-95:   {m.box.map!r}")
    print(f"Precision:  {m.box.mp!r}")
    print(f"Recall:     {m.box.mr!r}")
    # F1 — производная от P и R
    p, r = m.box.mp, m.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"F1 (calc):  {f1!r}")
    # Per-class (тут 1 класс box, но видно как метод называется)
    print(f"Всего объектов в GT:      {int(m.box.nc)} класс(а), tp+fp: нижнее см. ниже")
    # Численные tp/fp/fn доступны через confusion matrix
    if hasattr(m, "confusion_matrix") and m.confusion_matrix is not None:
        cm = m.confusion_matrix.matrix
        # cm[i][j] = GT class i predicted as class j
        # For 1 class + background: 2x2, but stored as (nc+1) x (nc+1)
        print(f"Confusion matrix:")
        print(cm)
