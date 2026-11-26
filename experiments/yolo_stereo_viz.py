#!/usr/bin/env python3
"""
yolo_stereo_viz.py — снимает по кадру с обеих камер, прогоняет YOLO
и сохраняет аннотированный стерео-пару для слайда 8.

Запуск (из хоста):
  docker exec ros2-gazebo bash -c \
    'source /opt/ros/jazzy/setup.bash && python3 /experiments/yolo_stereo_viz.py'
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path("/worlds/models/yolo_boxes.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "runs" / "obb" / "stereo_v1" / "weights" / "best.pt"

OUT_LEFT = ROOT / "presentation_assets" / "yolo_left.png"
OUT_RIGHT = ROOT / "presentation_assets" / "yolo_right.png"


def main():
    from ultralytics import YOLO
    print(f"[init] loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    rclpy.init()
    node = Node("yolo_stereo_viz")
    frames = {}

    def make_cb(key):
        def cb(msg):
            if key in frames:
                return
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == "rgb8":
                frames[key] = arr.copy()   # RGB
            else:  # bgr8 → reorder
                frames[key] = arr[:, :, ::-1].copy()
            print(f"[{key}] got frame {msg.width}×{msg.height}")
        return cb

    node.create_subscription(Image, "/camera_left/image_raw", make_cb("left"), 10)
    node.create_subscription(Image, "/camera_right/image_raw", make_cb("right"), 10)

    deadline = time.time() + 15
    print("Ждём кадры …")
    while time.time() < deadline and len(frames) < 2:
        rclpy.spin_once(node, timeout_sec=0.2)

    if len(frames) < 2:
        print(f"ERROR: получено только {len(frames)}/2 кадров")
        sys.exit(1)

    node.destroy_node()
    rclpy.shutdown()

    import cv2
    for key, rgb in frames.items():
        # Преобразуем в grayscale-3ch (модель обучена на ч/б)
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        frame_bw = np.stack([gray, gray, gray], axis=-1)
        results = model(frame_bw, conf=0.70, verbose=False)
        annotated_rgb = results[0].plot()          # RGB
        annotated_bgr = annotated_rgb[:, :, ::-1]  # для cv2.imwrite
        out = OUT_LEFT if key == "left" else OUT_RIGHT
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated_bgr)
        n_det = len(results[0].obb) if results[0].obb is not None else 0
        print(f"[{key}] {n_det} detection(s) → {out}")

    print("OK")


if __name__ == "__main__":
    main()
