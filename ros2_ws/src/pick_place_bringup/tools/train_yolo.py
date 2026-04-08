#!/usr/bin/env python3
"""
train_yolo.py — Fine-tune YOLOv8n-OBB on collected Gazebo dataset.

Run on Google Colab (GPU) or locally:
    pip install ultralytics
    python3 train_yolo.py --data /path/to/dataset/dataset.yaml

After training, best weights are saved to runs/obb/train/weights/best.pt
Copy best.pt to the container:
    cp runs/obb/train/weights/best.pt /path/to/worlds/models/yolo_boxes.pt

Then update yolo_params.yaml:
    model_path: /worlds/models/yolo_boxes.pt
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not found.  pip install ultralytics")


def train(data_yaml: str, epochs: int, imgsz: int, batch: int,
          device: str, project: str) -> None:

    print(f"Loading YOLOv8n-OBB pretrained weights …")
    model = YOLO("yolov8n-obb.pt")

    print(f"Starting training: epochs={epochs}  imgsz={imgsz}  "
          f"batch={batch}  device={device}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name="train",
        exist_ok=True,
        # ── Augmentation (albumentations handles heavy augmentation
        #    in augment_dataset.py; keep YOLO built-in augmentation light)
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=3.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,       # mosaic at 50% of batches
        mixup=0.0,        # off — boxes have specific geometry
        copy_paste=0.0,
        # ── Training settings
        patience=20,      # early stopping
        save_period=10,
        val=True,
        plots=True,
    )

    best = Path(project) / "train" / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best weights: {best}")
    print(f"\nTo use in simulation, copy to container and update yolo_params.yaml:")
    print(f"  model_path: /worlds/models/yolo_boxes.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n-OBB on Gazebo dataset")
    parser.add_argument("--data",    default="./dataset/dataset.yaml",
                        help="Path to dataset.yaml")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16,
                        help="Batch size (-1 = auto)")
    parser.add_argument("--device",  default="0",
                        help="Device: '0' (GPU), 'cpu', or 'mps' (Apple Silicon)")
    parser.add_argument("--project", default="runs/obb",
                        help="Output directory for training results")
    args = parser.parse_args()
    train(args.data, args.epochs, args.imgsz, args.batch, args.device, args.project)


if __name__ == "__main__":
    main()
