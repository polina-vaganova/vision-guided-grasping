#!/usr/bin/env python3
"""
line_capture.py — выстраивает 6 коробок в линию в Gazebo и захватывает кадр
с левой стерео-камеры. Запускать ВНУТРИ контейнера ros2-gazebo.

Usage (host):
  docker exec ros2-gazebo bash -c 'source /opt/ros/jazzy/setup.bash && python3 /experiments/line_capture.py'
"""
import math
import struct
import time
import sys

import gz.transport13 as _gz
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

WORLD = "pick_and_place"
TABLE_Z = 0.780
OUT_LEFT = "/experiments/inference_demo/gazebo_line_left.png"
OUT_RIGHT = "/experiments/inference_demo/gazebo_line_right.png"

# (name, height) — z_center = TABLE_Z + h/2 + 0.002
BOXES = [
    ("box_01", 0.03),
    ("box_02", 0.04),
    ("box_03", 0.05),
    ("box_04", 0.06),
    ("box_05", 0.07),
    ("box_06", 0.07),
]

# Линия по оси X, Y=0 (середина стола)
X_START = -0.28
X_END = 0.28
Y_LINE = 0.0


def _varint(v):
    buf = []
    while v > 0x7F:
        buf.append((v & 0x7F) | 0x80); v >>= 7
    buf.append(v)
    return bytes(buf)


def _pb_str(f, s):
    b = s.encode()
    return _varint((f << 3) | 2) + _varint(len(b)) + b


def _pb_dbl(f, v):
    return _varint((f << 3) | 1) + struct.pack("<d", v)


def _pb_emb(f, b):
    return _varint((f << 3) | 2) + _varint(len(b)) + b


def build_pose(name, x, y, z, qx=0, qy=0, qz=0, qw=1.0):
    pos = _pb_dbl(2, x) + _pb_dbl(3, y) + _pb_dbl(4, z)
    ori = _pb_dbl(2, qx) + _pb_dbl(3, qy) + _pb_dbl(4, qz) + _pb_dbl(5, qw)
    return _pb_str(2, name) + _pb_emb(4, pos) + _pb_emb(5, ori)


def main():
    gz_node = _gz.Node()
    service = f"/world/{WORLD}/set_pose"

    # 1. Расставляем коробки в линию
    n = len(BOXES)
    positions = [X_START + i * (X_END - X_START) / (n - 1) for i in range(n)]
    print("Размещаем коробки в линию:")
    for (name, h), x in zip(BOXES, positions):
        z = TABLE_Z + h / 2 + 0.002
        payload = build_pose(name, x, Y_LINE, z)
        # fire-and-forget
        gz_node.request_raw(service, payload, "gz.msgs.Pose", "gz.msgs.Boolean", 200)
        print(f"  {name}: x={x:+.3f} y={Y_LINE:+.3f} z={z:.3f}")
    time.sleep(2.0)  # даём физике стабилизироваться

    # 2. Захватываем один кадр с каждой камеры
    rclpy.init()
    node = Node("line_capture")

    frames = {}

    def make_cb(key, save_path):
        def cb(msg):
            if key in frames:
                return
            frames[key] = (msg, save_path)
            print(f"[{key}] got frame {msg.width}×{msg.height} encoding={msg.encoding}")
        return cb

    node.create_subscription(Image, "/camera_left/image_raw", make_cb("left", OUT_LEFT), 10)
    node.create_subscription(Image, "/camera_right/image_raw", make_cb("right", OUT_RIGHT), 10)

    print("Ждём кадры…")
    deadline = time.time() + 15
    while time.time() < deadline and len(frames) < 2:
        rclpy.spin_once(node, timeout_sec=0.2)

    if not frames:
        print("ERROR: кадры не получены за 15 сек")
        sys.exit(1)

    # 3. Сохраняем как PNG
    import numpy as np
    import cv2
    for key, (msg, path) in frames.items():
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            arr = arr[:, :, ::-1]  # RGB -> BGR для cv2
        cv2.imwrite(path, arr)
        print(f"[{key}] saved → {path}")

    node.destroy_node()
    rclpy.shutdown()
    print("OK")


if __name__ == "__main__":
    main()
