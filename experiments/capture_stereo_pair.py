#!/usr/bin/env python3
"""
capture_stereo_pair.py — снимает по одному кадру с левой и правой камер.
Без вмешательства в сцену (коробки остаются там, где расставлены).

Запуск (из хоста):
  docker exec ros2-gazebo bash -c \
    'source /opt/ros/jazzy/setup.bash && python3 /experiments/capture_stereo_pair.py'
"""
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

OUT_LEFT = "/experiments/presentation_assets/scene_left.png"
OUT_RIGHT = "/experiments/presentation_assets/scene_right.png"


def main():
    rclpy.init()
    node = Node("capture_stereo_pair")

    frames = {}

    def make_cb(key, save_path):
        def cb(msg):
            if key in frames:
                return
            frames[key] = (msg, save_path)
            print(f"[{key}] got frame {msg.width}×{msg.height} encoding={msg.encoding}")
        return cb

    node.create_subscription(Image, "/camera_left/image_raw",
                             make_cb("left", OUT_LEFT), 10)
    node.create_subscription(Image, "/camera_right/image_raw",
                             make_cb("right", OUT_RIGHT), 10)

    print("Ждём кадры …")
    deadline = time.time() + 15
    while time.time() < deadline and len(frames) < 2:
        rclpy.spin_once(node, timeout_sec=0.2)

    if len(frames) < 2:
        print(f"ERROR: получено только {len(frames)}/2 кадров за 15 сек")
        sys.exit(1)

    import numpy as np
    import cv2
    for key, (msg, path) in frames.items():
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            arr = arr[:, :, ::-1]
        cv2.imwrite(path, arr)
        print(f"[{key}] saved → {path}")

    node.destroy_node()
    rclpy.shutdown()
    print("OK")


if __name__ == "__main__":
    main()
